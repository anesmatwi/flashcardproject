import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv()

# --- Configured Paths ---
DATA_DIR = os.getenv("DATA_DIR", "./data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./vector_store")

BASE_MODEL = "BioMistral/BioMistral-7B-SLERP"
VALIDATED_FLASHCARDS = os.path.join(OUTPUT_DIR, "validated_flashcards.jsonl")
NEW_FINETUNE_DATA_FILE = os.path.join(OUTPUT_DIR, "finetuning_data_v2.jsonl")
EMBED_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
CHUNKS_FOLDER = os.path.join(DATA_DIR, "allchunks")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🔹 Loading base model and RAG components...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)
print("✅ Base model loaded.")

encoder = SentenceTransformer(EMBED_MODEL, device="cpu") 
index = faiss.read_index(os.path.join(VECTOR_STORE_DIR, "faiss_index.index"))
with open(os.path.join(VECTOR_STORE_DIR, "id_map.json")) as f:
    id_map = json.load(f)

chunk_map = {}
for fname in os.listdir(CHUNKS_FOLDER):
    if fname.endswith(".jsonl"):
        with open(os.path.join(CHUNKS_FOLDER, fname)) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("id") and obj.get("text"):
                        chunk_map[obj["id"]] = obj["text"]
                except json.JSONDecodeError:
                    continue
print("✅ RAG components loaded.")

def search_rag(query: str, k=2) -> str:
    embedding = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    _, I = index.search(embedding, k)
    chunks = [chunk_map.get(id_map.get(str(idx))) for idx in I[0] if id_map.get(str(idx)) in chunk_map]
    return "\n---\n".join(filter(None, chunks))

print(f"🔹 Loading flashcards from {VALIDATED_FLASHCARDS}...")
with open(VALIDATED_FLASHCARDS) as f:
    all_flashcards = [json.loads(line) for line in f]

processed_questions = set()
if os.path.exists(NEW_FINETUNE_DATA_FILE):
    print(f"✅ Found existing output file. Loading processed flashcards to resume...")
    with open(NEW_FINETUNE_DATA_FILE, 'r') as f_in:
        for line in f_in:
            try:
                processed_questions.add(json.loads(line)['question'])
            except (json.JSONDecodeError, KeyError):
                continue

cards_to_process = [card for card in all_flashcards if card.get("question") not in processed_questions]
num_already_processed = len(all_flashcards) - len(cards_to_process)

print(f"✅ Resuming. {num_already_processed} of {len(all_flashcards)} flashcards already processed. {len(cards_to_process)} remaining.")

with open(NEW_FINETUNE_DATA_FILE, "a") as f_out:
    with tqdm(total=len(all_flashcards), initial=num_already_processed, desc="Generating Explanations") as pbar:
        for card in cards_to_process:
            if "question" not in card or "answer" not in card:
                pbar.write(f"\n⚠️ WARNING: Skipping malformed flashcard. Content: {card}")
                continue

            question = card["question"]
            correct_answer = card["answer"]
            
            context = search_rag(f"{question} {correct_answer}")

            prompt = f"""
            [INST] You are a pathology expert. Using ONLY the information in the provided CONTEXT, write a clear and factually accurate explanation for the following question and answer pair. **Do not simply repeat or rephrase the answer.** Add background, context, or mechanisms that clarify *why* the answer is correct.

            CONTEXT:
            {context}

            QUESTION: {question}
            ANSWER: {correct_answer}
            [/INST]
            EXPLANATION:
            """
            
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                output = base_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            explanation = tokenizer.decode(output[0], skip_special_tokens=True).split("EXPLANATION:")[-1].strip()
            
            card['explanation'] = explanation
            f_out.write(json.dumps(card) + "\n")
            f_out.flush() 
            pbar.update(1) 

            del inputs
            del output
            torch.cuda.empty_cache()

print(f"\n🎉 Generation complete. Results saved to '{NEW_FINETUNE_DATA_FILE}'")
