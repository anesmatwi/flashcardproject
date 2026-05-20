import os
import json
import hashlib
from pathlib import Path
from tqdm import tqdm
import logging
from llama_cpp import Llama
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configured Paths ---
DATA_DIR = os.getenv("DATA_DIR", "./data")
MODELS_DIR = os.getenv("MODELS_DIR", "../models")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "BioMistral-7B-Zephyr-Beta-SLERP.Q4_K_M.gguf")
CHUNKS_DIR = Path(DATA_DIR) / "allchunks"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "flashcards.jsonl")
ERROR_LOG = os.path.join(OUTPUT_DIR, "errors.log")
MALFORMED_LOG = os.path.join(OUTPUT_DIR, "malformed_outputs.log")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")

llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=8, n_gpu_layers=40, verbose=False)
logging.basicConfig(filename=ERROR_LOG, level=logging.ERROR)

def hash_chunk(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()

def generate_flashcards(text: str) -> list:
    prompt = f"""<|system|>
You are a medical education AI that generates helpful and varied flashcards for pathology board exam preparation.

<|user|>
From the following text, generate 3 flashcards in JSON format. Vary the types across:
- "definition"
- "question_answer"
- "clinical_case"
- "mechanism"
- "fact"
- "mnemonic" (only if appropriate — do not force it)

Each flashcard should follow this format:
{{
  "type": "definition" | "question_answer" | "clinical_case" | "mechanism" | "fact" | "mnemonic",
  "question": "...",
  "answer": "..."
}}

Text:
{text}

Output:
[
  {{
    "type": "...",
    "question": "...",
    "answer": "..."
  }},
  ...
]
<|assistant|>
"""
    try:
        response = llm(prompt, stop=["</s>"], temperature=0.7, max_tokens=1024)
        output = response["choices"][0]["text"].strip()
        flashcards = json.loads(output)
        if isinstance(flashcards, list):
            return flashcards
        raise ValueError("Output was not a list")
    except Exception as e:
        raise ValueError(f"Malformed output: {output[:500]}") from e

def load_checkpoint():
    return json.load(open(CHECKPOINT_FILE)) if os.path.exists(CHECKPOINT_FILE) else {}

def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)

def append_to_jsonl(path, items):
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def process_chunks():
    checkpoint = load_checkpoint()
    malformed_log = open(MALFORMED_LOG, "a", encoding="utf-8")

    for chunk_file in tqdm(list(CHUNKS_DIR.glob("*.jsonl"))):
        book_key = str(chunk_file.name)
        if book_key not in checkpoint:
            checkpoint[book_key] = []

        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)["text"]
                chunk_id = hash_chunk(chunk)

                if chunk_id in checkpoint[book_key]:
                    continue

                try:
                    flashcards = generate_flashcards(chunk)
                    append_to_jsonl(OUTPUT_FILE, flashcards)
                    checkpoint[book_key].append(chunk_id)
                    save_checkpoint(checkpoint)
                except Exception as e:
                    logging.error(f"{chunk_file.name} (chunk {chunk_id}): ❌ {e}")
                    malformed_log.write(chunk[:500] + "\n---\n")

    malformed_log.close()
    save_checkpoint(checkpoint)

if __name__ == "__main__":
    process_chunks()
