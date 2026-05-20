import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import gc
import pandas as pd
import textstat
import nltk
from dotenv import load_dotenv

load_dotenv()

# --- Configured Paths ---
DATA_DIR = os.getenv("DATA_DIR", "./data")
MODELS_DIR = os.getenv("MODELS_DIR", "../models")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")

# === NLTK Package Check ===
try:
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('tokenizers/punkt.zip')
except LookupError:
    print("Downloading 'punkt' and 'wordnet' from NLTK...")
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("✅ NLTK packages downloaded.")

# === Config ===
MODEL_NAME = "BioMistral/BioMistral-7B-SLERP"
ANSWER_LORA_PATH = os.path.join(MODELS_DIR, "lora_adapters", "answer_adapter_hybrid_best")
EXPLANATION_LORA_PATH = os.path.join(MODELS_DIR, "lora_adapters", "explanation_adapter_final")
DATA_PATH = os.path.join(DATA_DIR, "golden_test_set.jsonl")
MAX_NEW_TOKENS = 256
LOG_N_SAMPLES = 5

ANSWER_PROMPT_TEMPLATE = "{question}\n"
EXPLANATION_PROMPT_TEMPLATE = "{question}\n{answer}\n"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

def load_model(lora_path=None):
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", quantization_config=bnb_config
    )
    if lora_path:
        print(f"Loading LoRA adapter from: {lora_path}")
        return PeftModel.from_pretrained(base_model, lora_path)
    return base_model

eval_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
print(f"✅ Evaluating on {len(eval_dataset)} samples.")

print("Loading evaluation metrics (ROUGE, BLEU, BERTScore)...")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bert_score")

def evaluate_model(model, model_name, field_name):
    preds, refs = [], []
    print(f"\n🔍 Evaluating MODEL: '{model_name}' on FIELD: '{field_name}'")
    model.eval()

    for i, item in enumerate(tqdm(eval_dataset, desc=f"Generating for {model_name}")):
        if field_name == "answer":
            prompt = ANSWER_PROMPT_TEMPLATE.format(question=item["question"].strip())
            expected = item.get("answer", "").strip()
        else:
            prompt = EXPLANATION_PROMPT_TEMPLATE.format(
                question=item["question"].strip(), answer=item["answer"].strip()
            )
            expected = item.get("explanation", "").strip()

        if not expected: continue

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()

        preds.append(generated)
        refs.append(expected)

    print("  Calculating metrics...")
    bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])["bleu"]
    rouge_score = rouge.compute(predictions=preds, references=refs)['rougeL']
    bert_score_f1 = bertscore.compute(predictions=preds, references=refs, lang="en")['f1']
    readability_scores = [textstat.flesch_kincaid_grade(p) for p in preds]

    results = {
        "Model": model_name,
        "Field": field_name,
        "BERTScore-F1": sum(bert_score_f1) / len(bert_score_f1) if bert_score_f1 else 0,
        "ROUGE-L": rouge_score,
        "BLEU": bleu_score,
        "Readability (FK Grade)": sum(readability_scores) / len(readability_scores) if readability_scores else 0
    }
    return results

if __name__ == "__main__":
    all_results = []

    model = load_model(ANSWER_LORA_PATH)
    all_results.append(evaluate_model(model, "Fine-tuned LoRA", "answer"))
    del model; gc.collect(); torch.cuda.empty_cache()

    model = load_model(EXPLANATION_LORA_PATH)
    all_results.append(evaluate_model(model, "Fine-tuned LoRA", "explanation"))
    del model; gc.collect(); torch.cuda.empty_cache()

    model = load_model()
    all_results.append(evaluate_model(model, "Base Model", "answer"))
    all_results.append(evaluate_model(model, "Base Model", "explanation"))
    del model; gc.collect(); torch.cuda.empty_cache()

    print("\n--- 📊 FINAL BALANCED RESULTS SUMMARY ---")
    results_df = pd.DataFrame(all_results)
    pivot_df = results_df.pivot(index='Field', columns='Model', values=['BERTScore-F1', 'ROUGE-L', 'BLEU', 'Readability (FK Grade)'])
    print(pivot_df)

    output_csv = os.path.join(OUTPUT_DIR, "final_balanced_summary.csv")
    pivot_df.to_csv(output_csv)
    print(f"\n✅ Final summary saved to '{output_csv}'")
