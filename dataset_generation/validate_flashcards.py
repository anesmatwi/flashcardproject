import os
import json
import time
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is missing.")

genai.configure(api_key=api_key)

# --- Configured Paths ---
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "gemini-1.5-pro"
INPUT_FILE = os.path.join(OUTPUT_DIR, "flashcards.jsonl")
OUTPUT_FILE_VALID = os.path.join(OUTPUT_DIR, "validated_flashcards.jsonl")
OUTPUT_FILE_REJECTED = os.path.join(OUTPUT_DIR, "rejected_flashcards.jsonl")
LOG_FILE = os.path.join(OUTPUT_DIR, "validation_debug.log")

model = genai.GenerativeModel(MODEL_NAME)

def validate_flashcard(question, answer, max_retries=3, delay=3):
    prompt = f"""
You are a pathology flashcard validator for medical board exams.

Return only a JSON object with:
- "valid": true or false
- "reason": short explanation
- "score": float from 0.0 to 1.0

Example:
{{
  "valid": true,
  "reason": "High-yield fact about molecular pathology.",
  "score": 0.95
}}

Now evaluate this flashcard:

Q: {question}
A: {answer}
Output JSON only:
"""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()

            first_brace = raw.find('{')
            last_brace = raw.rfind('}')
            if first_brace != -1 and last_brace != -1:
                json_str = raw[first_brace:last_brace+1]
                return json.loads(json_str)

            raise ValueError("Could not find valid JSON in Gemini response.")

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                return {
                    "valid": False,
                    "reason": f"Validation error: {str(e)}",
                    "score": 0.0
                }

if __name__ == "__main__":
    with open(INPUT_FILE, "r") as infile, \
         open(OUTPUT_FILE_VALID, "w") as valid_outfile, \
         open(OUTPUT_FILE_REJECTED, "w") as rejected_outfile, \
         open(LOG_FILE, "w") as log_file:

        lines = infile.readlines()

        for line in tqdm(lines, desc="Validating flashcards"):
            try:
                card = json.loads(line)
                question = card.get("question", "").strip()
                answer = card.get("answer", "").strip()

                result = validate_flashcard(question, answer)
                card["validation"] = result

                log_file.write(f"Q: {question}\nA: {answer}\nValidation: {result}\n\n")

                if result["valid"]:
                    valid_outfile.write(json.dumps(card) + "\n")
                else:
                    rejected_outfile.write(json.dumps(card) + "\n")

            except Exception as e:
                log_file.write(f"ERROR processing line: {str(e)}\n{line}\n\n")
