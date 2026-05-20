import os
import json
import re
import time
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- Configured Paths ---
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
MODELS_DIR = os.getenv("MODELS_DIR", "../models")

INPUT_FILE = os.path.join(OUTPUT_DIR, "finetuning_data_v3_paragraphs.jsonl")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "final_training_data_with_negatives.jsonl")
SAMPLE_EVERY_N_ITEMS = 100
LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, "BioMistral-7B-Zephyr-Beta-SLERP.Q4_K_M.gguf")

def generate_llm_output(prompt: str) -> str:
    from llama_cpp import Llama
    model = Llama(model_path=LOCAL_MODEL_PATH, n_ctx=2048, verbose=False)
    output = model(prompt, max_tokens=512, stop=["\n\n", "Flashcard Question:", "[INST]"])
    return output['choices'][0]['text'].strip()

def build_prompt(flashcard: dict) -> str:
    question = flashcard.get("question", "")
    correct_answer = flashcard.get("answer", "")
    return f"""
[INST] You are a pathology expert creating training data.

Given the following question and its correct answer, perform two tasks:
1.  Generate a **plausible but incorrect alternative answer**. This should be a common misconception or a related but incorrect concept.
2.  Generate an **explanation for why the alternative is wrong**. This explanation should briefly **clarify the correct concept** and state why the wrong answer is a mistake.

Output a JSON object with the following format only:

{{
  "wrong_answer": "...",
  "wrong_explanation": "..."
}}

**Flashcard Question:**
{question}

**Correct Answer:**
{correct_answer}
[/INST]
"""

def parse_llm_output(output: str) -> dict:
    try:
        match = re.search(r'\{.*?\}', output, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(output)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from output: {output}") from e

if __name__ == "__main__":
    print(f"🔹 Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        all_flashcards = [json.loads(line) for line in f]
    
    processed_questions = set()
    if os.path.exists(OUTPUT_FILE):
        print("✅ Found existing output file. Resuming...")
        with open(OUTPUT_FILE, 'r') as f_in:
            for line in f_in:
                try:
                    processed_questions.add(json.loads(line)['question'])
                except (json.JSONDecodeError, KeyError):
                    continue
    
    cards_to_process = [card for card in all_flashcards if card.get("question") not in processed_questions]
    num_already_processed = len(all_flashcards) - len(cards_to_process)
    print(f"✅ {num_already_processed} cards already have negative samples. {len(cards_to_process)} remaining.")

    with open(OUTPUT_FILE, "a") as f_out:
        with tqdm(total=len(all_flashcards), initial=num_already_processed, desc="Generating Negative Samples") as pbar:
            for i, card in enumerate(cards_to_process):
                result = None
                for attempt in range(3):
                    try:
                        prompt = build_prompt(card)
                        output = generate_llm_output(prompt)
                        parsed = parse_llm_output(output)
                        
                        card["wrong_answer"] = parsed["wrong_answer"]
                        card["wrong_explanation"] = parsed["wrong_explanation"]
                        result = card
                        break
                    except Exception as e:
                        pbar.write(f"\n[Attempt {attempt+1}/3] Failed for Q: \"{card.get('question', 'N/A')[:50]}...\" | Error: {e}")
                        time.sleep(1)
                
                if result:
                    f_out.write(json.dumps(result) + "\n")
                    f_out.flush()

                    if i == 0 or (i + 1) % SAMPLE_EVERY_N_ITEMS == 0:
                        pbar.write("\n" + "="*50)
                        pbar.write(f"SAMPLE OUTPUT FOR ITEM #{num_already_processed + i + 1}")
                        pbar.write(f"QUESTION: {result['question']}")
                        pbar.write(f"CORRECT ANSWER: {result['answer']}")
                        pbar.write(f"--- CORRECT EXPLANATION (PARAGRAPH) ---")
                        pbar.write(result.get('explanation', 'N/A'))
                        pbar.write(f"--- GENERATED WRONG ANSWER ---")
                        pbar.write(result['wrong_answer'])
                        pbar.write(f"--- GENERATED WRONG EXPLANATION ---")
                        pbar.write(result['wrong_explanation'])
                        pbar.write("="*50 + "\n")
                
                pbar.update(1)

    print(f"\n🎉 Generation of negative samples complete. Final dataset is at '{OUTPUT_FILE}'")
