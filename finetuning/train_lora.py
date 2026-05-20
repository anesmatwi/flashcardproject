import os
import json
import torch
from torch.nn import MarginRankingLoss
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- Configured Paths ---
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
MODELS_DIR = os.getenv("MODELS_DIR", "../models")

DATA_PATH = os.path.join(OUTPUT_DIR, "train_data.jsonl")
LORA_OUTPUT_DIR = os.path.join(MODELS_DIR, "lora_adapters")
os.makedirs(LORA_OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "BioMistral/BioMistral-7B-SLERP"
TASK = "explanation"
BATCH_SIZE = 1
ACCUM_STEPS = 16
MARGIN = 0.3
EPOCHS = 3
CONTRASTIVE_WEIGHT = 2.0
LEARNING_RATE = 2e-5

print("🔹 Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
print("✅ Models and tokenizer loaded.")

def load_and_format_data():
    with open(DATA_PATH, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    correct_key = "explanation"
    wrong_key = "wrong_explanation"
    
    filtered_data = []
    for r in data:
        if r.get("question") and r.get("answer") and r.get(correct_key) and r.get(wrong_key):
            filtered_data.append({
                "question": r["question"],
                "answer": r["answer"],
                "correct": r[correct_key],
                "wrong": r[wrong_key]
            })
            
    print(f"✅ Loaded {len(filtered_data)} valid records from '{DATA_PATH}'.")
    return train_test_split(filtered_data, test_size=0.1, random_state=42)

def tokenize_batch(example):
    correct_text = f"Question: {example['question']}\nAnswer: {example['answer']}\nExplanation: {example['correct']}"
    wrong_text = f"Question: {example['question']}\nAnswer: {example['answer']}\nExplanation: {example['wrong']}"
    
    correct = tokenizer(correct_text, padding="max_length", truncation=True, max_length=512)
    wrong = tokenizer(wrong_text, padding="max_length", truncation=True, max_length=512)
    
    return {
        "input_ids_correct": correct["input_ids"],
        "attention_mask_correct": correct["attention_mask"],
        "input_ids_wrong": wrong["input_ids"],
        "attention_mask_wrong": wrong["attention_mask"]
    }

def collate_fn(batch):
    correct = {"input_ids": torch.tensor([ex["input_ids_correct"] for ex in batch]),
               "attention_mask": torch.tensor([ex["attention_mask_correct"] for ex in batch])}
    wrong = {"input_ids": torch.tensor([ex["input_ids_wrong"] for ex in batch]),
             "attention_mask": torch.tensor([ex["attention_mask_wrong"] for ex in batch])}
    return correct, wrong

def evaluate(lora_model, val_loader, loss_fn):
    lora_model.eval()
    total_loss = 0
    with torch.no_grad():
        for correct_batch, wrong_batch in val_loader:
            input_ids_c = correct_batch["input_ids"].to(lora_model.device)
            attn_c = correct_batch["attention_mask"].to(lora_model.device)
            input_ids_w = wrong_batch["input_ids"].to(lora_model.device)
            attn_w = wrong_batch["attention_mask"].to(lora_model.device)
            labels = input_ids_c.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            with autocast():
                lm_loss = lora_model(input_ids=input_ids_c, attention_mask=attn_c, labels=labels).loss
                c_out = lora_model.base_model.model(input_ids=input_ids_c, attention_mask=attn_c, output_hidden_states=True)
                w_out = lora_model.base_model.model(input_ids=input_ids_w, attention_mask=attn_w, output_hidden_states=True)
                c_emb = c_out.hidden_states[-1][:, 0]
                w_emb = w_out.hidden_states[-1][:, 0]
                c_norm = torch.nn.functional.normalize(c_emb, p=2, dim=1)
                w_norm = torch.nn.functional.normalize(w_emb, p=2, dim=1)
                c_score = (c_norm * c_norm).sum(dim=1)
                w_score = (c_norm * w_norm).sum(dim=1)
                target = torch.ones_like(c_score)
                contrastive_loss = loss_fn(c_score, w_score, target)
                loss = lm_loss + CONTRASTIVE_WEIGHT * contrastive_loss
            total_loss += loss.item()
    
    lora_model.train()
    return total_loss / len(val_loader)

if __name__ == "__main__":
    print(f"\n🚀 Initiating LoRA training for: {TASK}")
    
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()

    train_data, val_data = load_and_format_data()
    train_ds = Dataset.from_list(train_data).map(tokenize_batch, batched=False)
    val_ds = Dataset.from_list(val_data).map(tokenize_batch, batched=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=2)

    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=LEARNING_RATE)
    contrastive_loss_fn = MarginRankingLoss(margin=MARGIN)
    scaler = GradScaler()
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        lora_model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, (correct_batch, wrong_batch) in enumerate(pbar):
            input_ids_c = correct_batch["input_ids"].to(lora_model.device)
            attn_c = correct_batch["attention_mask"].to(lora_model.device)
            input_ids_w = wrong_batch["input_ids"].to(lora_model.device)
            attn_w = wrong_batch["attention_mask"].to(lora_model.device)
            labels = input_ids_c.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            with autocast():
                lm_loss = lora_model(input_ids=input_ids_c, attention_mask=attn_c, labels=labels).loss
                c_out = lora_model.base_model.model(input_ids=input_ids_c, attention_mask=attn_c, output_hidden_states=True)
                w_out = lora_model.base_model.model(input_ids=input_ids_w, attention_mask=attn_w, output_hidden_states=True)
                c_emb = c_out.hidden_states[-1][:, 0]
                w_emb = w_out.hidden_states[-1][:, 0]
                c_norm = torch.nn.functional.normalize(c_emb, p=2, dim=1)
                w_norm = torch.nn.functional.normalize(w_emb, p=2, dim=1)
                c_score = (c_norm * c_norm).sum(dim=1)
                w_score = (c_norm * w_norm).sum(dim=1)
                target = torch.ones_like(c_score)
                contrastive_loss = contrastive_loss_fn(c_score, w_score, target)
                
                loss = lm_loss + CONTRASTIVE_WEIGHT * contrastive_loss
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix({"loss": total_train_loss / (step + 1)})

        val_loss = evaluate(lora_model, val_loader, contrastive_loss_fn)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {(total_train_loss / len(train_loader)):.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(LORA_OUTPUT_DIR, f"{TASK}_adapter_final")
            lora_model.save_pretrained(save_path)
            print(f"✅ Saved improved model to {save_path}")

    print(f"🎉 Training for '{TASK}' complete.")
