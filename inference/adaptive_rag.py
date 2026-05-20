import os
import json
import torch
import faiss
import re
import gc
import numpy as np
import random
import requests
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from Bio import Entrez
from serpapi import GoogleSearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configured Paths ---
DATA_DIR = os.getenv("DATA_DIR", "./data")
MODELS_DIR = os.getenv("MODELS_DIR", "../models")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./vector_store")

# === 1. Configuration ===
BASE_MODEL_ID = "BioMistral/BioMistral-7B-SLERP"
EXPLANATION_LORA_PATH = os.path.join(MODELS_DIR, "lora_adapters", "explanation_adapter_final")
ANSWER_LORA_PATH = os.path.join(MODELS_DIR, "lora_adapters", "answer_adapter_hybrid_best")

TEST_SET_PATH = os.path.join(DATA_DIR, "golden_test_set.jsonl")
OUTPUT_BASELINE = os.path.join(OUTPUT_DIR, "generated_with_baseline_no_lora.jsonl")
OUTPUT_LORA_ONLY = os.path.join(OUTPUT_DIR, "generated_with_baseline_lora.jsonl")
OUTPUT_RAG_LORA = os.path.join(OUTPUT_DIR, "generated_with_rag_lora.jsonl")

EMBED_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
VECTOR_STORE_FOLDER = VECTOR_STORE_DIR
CHUNKS_FOLDER = os.path.join(DATA_DIR, "allchunks")

# Secure API Keys
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if PUBMED_EMAIL:
    Entrez.email = PUBMED_EMAIL
else:
    print("⚠️ WARNING: PUBMED_EMAIL not set. Entrez queries may be restricted.")

TRUSTED_DOMAINS = ["mayoclinic.org", "clevelandclinic.org", "hopkinsmedicine.org", "msdmanuals.com", "medscape.com", "cancer.gov", "cdc.gov", "nih.gov", "webmd.com"]
EXTRACTOR_SERVER_URL = "http://localhost:8000"

EXPLANATION_PROMPT_LORA = """
[INST]
Here is a question and its answer. Your task is to write a detailed explanation that provides additional details, mechanisms, or clinical context.

CRITICAL RULE: Do not repeat information already present in the answer.

QUESTION: {question}
ANSWER: {answer}
[/INST]
EXPLANATION:
"""

RAG_SYNTHESIZER_PROMPT_BASELINE = """
TASK: You are a medical expert. Using the CONTEXT provided, write a detailed explanation for the ANSWER to the QUESTION.

CRITICAL RULE: Do not repeat information already present in the ORIGINAL ANSWER. Your goal is to add new, supplementary information about underlying mechanisms, causes, or clinical context found in the provided CONTEXT.

CONTEXT:
{context}

QUESTION: {question}

ORIGINAL ANSWER:
{answer}

EXPLANATION:
"""

NUM_SAMPLES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = -0.50
REPETITION_THRESHOLD = 0.65
GENERATION_CONFIG = {"do_sample": True, "temperature": 0.6, "top_p": 0.95}
GENERATION_CONFIG_CHAINED = {
    "do_sample": True, "temperature": 0.6, "top_p": 0.95,
    "max_new_tokens": 256,
    "max_total_length": 1024
}

# === 2. Helper Functions ===
def retry_request(max_tries=3, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(max_tries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    print(f"⚠️ API call failed ({i+1}/{max_tries}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            print(f"🚨 API call failed after {max_tries} tries. Returning default value.")
            if "verify" in func.__name__: return {"is_verified": False, "verified_context": "", "context_length": 0}
            if "expand" in func.__name__: return [args[0]]
            if "rewrite" in func.__name__: return args[0]
            if "broaden" in func.__name__: return args[0]
            return "Error: Could not connect to extractor server."
        return wrapper
    return decorator

def final_cleanup(text: str) -> str:
    if "QUESTION:" in text:
        text = text.split("QUESTION:")[0].strip()
    text = re.sub(r'\[/INST\]|\[INST\]', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'^(PARAGRAPH:|KEY FACTS:|ANSWER:|EXPLANATION:)', '', text, flags=re.IGNORECASE).strip()
    return text

def calculate_confidence(generation_output) -> float:
    scores = generation_output.scores
    generated_ids = generation_output.sequences[:, -len(scores):]
    gen_logprobs = []
    for i, score in enumerate(scores):
        token_id = generated_ids[0, i]
        log_prob = torch.log_softmax(score, dim=-1)[0, token_id].item()
        gen_logprobs.append(log_prob)
    return np.mean(gen_logprobs) if gen_logprobs else -float('inf')

def calculate_repetition_score(answer: str, explanation: str) -> float:
    if not isinstance(answer, str) or not isinstance(explanation, str): return 0.0
    answer_words = set(answer.lower().split())
    explanation_words = set(explanation.lower().split())
    if not answer_words or not explanation_words: return 0.0
    intersection = len(answer_words.intersection(explanation_words))
    union = len(answer_words.union(explanation_words))
    return intersection / union

def generate_with_confidence(model, tokenizer, prompt: str) -> tuple[str, float]:
    config = {"max_new_tokens": 512, **GENERATION_CONFIG}
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generation_output = model.generate(
            **inputs, **config, pad_token_id=tokenizer.eos_token_id,
            output_scores=True, return_dict_in_generate=True
        )
    newly_generated_text = tokenizer.decode(generation_output.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    confidence = calculate_confidence(generation_output)
    return final_cleanup(newly_generated_text), confidence

def generate_chained(model, tokenizer, prompt: str) -> str:
    full_text = ""
    current_prompt = prompt
    chunk_size = GENERATION_CONFIG_CHAINED['max_new_tokens']
    max_total_length = GENERATION_CONFIG_CHAINED['max_total_length']
    for _ in range(max_total_length // chunk_size):
        inputs = tokenizer(current_prompt, return_tensors="pt", return_attention_mask=True).to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=chunk_size,
                do_sample=GENERATION_CONFIG_CHAINED['do_sample'],
                temperature=GENERATION_CONFIG_CHAINED['temperature'],
                top_p=GENERATION_CONFIG_CHAINED['top_p'],
                pad_token_id=tokenizer.eos_token_id
            )
        newly_generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        new_text_chunk = tokenizer.decode(newly_generated_ids, skip_special_tokens=True)
        full_text += new_text_chunk
        if output_ids[0][-1] == tokenizer.eos_token_id:
            break
        current_prompt = prompt + full_text
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()
    return final_cleanup(full_text)

@retry_request()
def rewrite_query_with_llm(question: str) -> str:
    response = requests.post(f"{EXTRACTOR_SERVER_URL}/rewrite-query", json={"question": question}, timeout=60)
    response.raise_for_status()
    return response.json().get("rewritten_question", question)

@retry_request()
def broaden_query_with_llm(question: str) -> str:
    response = requests.post(f"{EXTRACTOR_SERVER_URL}/broaden-query", json={"question": question}, timeout=60)
    response.raise_for_status()
    return response.json().get("broader_query", question)

@retry_request()
def expand_query_with_llm(question: str) -> list:
    response = requests.post(f"{EXTRACTOR_SERVER_URL}/expand-query", json={"question": question}, timeout=60)
    response.raise_for_status()
    return response.json().get("queries", [question])

@retry_request()
def verify_context(context: str, question: str) -> dict:
    response = requests.post(f"{EXTRACTOR_SERVER_URL}/verify-context", json={"context": context, "question": question}, timeout=120)
    response.raise_for_status()
    return response.json()

def clean_passage(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def query_internal_index(encoder, index, id_map, chunk_map, query: str, k=5) -> list:
    embedding = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    _, I = index.search(embedding, k)
    return [chunk_map.get(id_map.get(str(idx))) for idx in I[0] if id_map.get(str(idx)) in chunk_map]

def query_pubmed(query: str, k=3) -> list:
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=k, sort="relevance")
        record = Entrez.read(handle)
        ids = record["IdList"]
        if not ids: return []
        fetch = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
        abstracts = fetch.read().split("\n\n")
        return [clean_passage(a) for a in abstracts if a]
    except Exception as e:
        print(f"⚠️ PubMed search failed: {e}")
        return []

def query_targeted_web_search(query: str, k=3) -> list:
    if not SERPAPI_API_KEY: return []
    try:
        site_query = " OR ".join([f"site:{domain}" for domain in TRUSTED_DOMAINS])
        full_query = f'"{query}" ({site_query})'
        params = {"api_key": SERPAPI_API_KEY, "engine": "google", "q": full_query, "num": k}
        search = GoogleSearch(params)
        results = search.get_dict()
        return [f"{res.get('title', '')}: {res.get('snippet', '')}" for res in results.get('organic_results', [])]
    except Exception as e:
        print(f"⚠️ Targeted Web Search failed: {e}")
        return []

def rerank_passages(encoder, query: str, passages: list, k=5) -> list:
    if not passages: return []
    query_emb = encoder.encode(query, convert_to_tensor=True)
    passage_embs = encoder.encode(passages, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, passage_embs)[0]
    pairs = sorted(zip(scores.cpu().numpy(), passages), reverse=True, key=lambda x: x[0])
    sorted_passages = [p[1] for p in pairs[:k]]
    if len(sorted_passages) > 1:
        best_passage = sorted_passages.pop(0)
        sorted_passages.append(best_passage)
    return sorted_passages

def _execute_search(queries: list, answer: str, encoder, index, id_map, chunk_map) -> list:
    all_passages = []
    for query in queries:
        augmented_query = f"{query} {answer}"
        print(f"   Augmented search query: \"{augmented_query[:75]}...\"")
        all_passages.extend(query_internal_index(encoder, index, id_map, chunk_map, augmented_query))
        all_passages.extend(query_pubmed(augmented_query))
        all_passages.extend(query_targeted_web_search(augmented_query))
    return list(dict.fromkeys(filter(None, all_passages)))

def perform_rag_retrieval(encoder, index, id_map, chunk_map, question: str, answer: str) -> str:
    print("🧠 Step 1/4: Rewriting query...")
    rewritten_question = rewrite_query_with_llm(question)
    if rewritten_question != question:
        print(f"   Rewritten to: {rewritten_question}")
    print("🧠 Step 2/4: Expanding query...")
    specific_queries = expand_query_with_llm(rewritten_question)
    print(f"   Generated Queries: {specific_queries}")
    print("🧠 Step 3/4: Performing first search pass...")
    unique_passages = _execute_search(specific_queries, answer, encoder, index, id_map, chunk_map)
    if not unique_passages:
        print("⚠️ First search pass failed. Broadening query...")
        broader_query = broaden_query_with_llm(rewritten_question)
        print(f"   Broader Query: {broader_query}")
        print("🧠 Step 4/4: Performing second search pass...")
        unique_passages = _execute_search([broader_query], answer, encoder, index, id_map, chunk_map)
    if not unique_passages:
        print("   Retrieval failed after all attempts.")
        return ""

    print(f"   Retrieved {len(unique_passages)} unique passages. Reranking and selecting top 7...")
    ranked_passages = rerank_passages(encoder, f"{question} {answer}", unique_passages, k=7)
    
    return "\n---\n".join(ranked_passages)

# === 3. Main Execution ===
if __name__ == "__main__":
    if not SERPAPI_API_KEY:
        print("⚠️ WARNING: SERPAPI_API_KEY not set. Web Search will be skipped.")
    print("🔹 Loading non-LLM components...")
    encoder = SentenceTransformer(EMBED_MODEL, device="cpu")
    index = faiss.read_index(os.path.join(VECTOR_STORE_FOLDER, "faiss_index.index"))
    with open(os.path.join(VECTOR_STORE_FOLDER, "id_map.json")) as f: id_map = json.load(f)
    chunk_map = {}
    for fname in os.listdir(CHUNKS_FOLDER):
        if fname.endswith(".jsonl"):
            with open(os.path.join(CHUNKS_FOLDER, fname)) as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if obj.get("id") and obj.get("text"): chunk_map[obj["id"]] = obj["text"]
                    except json.JSONDecodeError: continue
    print("✅ RAG components loaded.")
    print("🔹 Loading BioMistral and LoRA adapters to GPU...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto")
    lora_model = PeftModel.from_pretrained(base_model, ANSWER_LORA_PATH, adapter_name="answer")
    lora_model.load_adapter(EXPLANATION_LORA_PATH, adapter_name="explanation")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print("✅ Main models loaded.")
    with open(TEST_SET_PATH, 'r') as f: all_test_data = [json.loads(line) for line in f if line.strip()]
    random.seed(42)
    test_data = random.sample(all_test_data, min(NUM_SAMPLES, len(all_test_data)))
    print(f"✅ Created a random sample of {len(test_data)} test items.")
    with open(OUTPUT_BASELINE, 'w') as f_base, \
         open(OUTPUT_LORA_ONLY, 'w') as f_lora, \
         open(OUTPUT_RAG_LORA, 'w') as f_rag:
        pbar = tqdm(test_data, desc="Generating Samples")
        for card in pbar:
            question = card['question']
            
            # --- STEP 1: BASELINE GENERATION ---
            pbar.set_description(f"Q: \"{question[:20]}...\" | 1. Baseline")
            with lora_model.disable_adapter():
                baseline_answer, _ = generate_with_confidence(lora_model, tokenizer, f"QUESTION: {question}\nANSWER:")
                baseline_explanation = generate_chained(lora_model, tokenizer, f"QUESTION: {question}\nANSWER: {baseline_answer}\nEXPLANATION:")
            f_base.write(json.dumps({ "question": question, "answer": baseline_answer, "explanation": baseline_explanation }) + "\n")

            # --- STEP 2: LORA-ONLY GENERATION (TWO-STEP) ---
            pbar.set_description(f"Q: \"{question[:20]}...\" | 2. LoRA-Only")
            lora_model.set_adapter("answer")
            lora_answer, _ = generate_with_confidence(lora_model, tokenizer, f"[INST] QUESTION: {question}\nANSWER: [/INST]")
            lora_model.set_adapter("explanation")
            lora_explanation_prompt = EXPLANATION_PROMPT_LORA.format(question=question, answer=lora_answer)
            lora_explanation = generate_chained(lora_model, tokenizer, lora_explanation_prompt)
            f_lora.write(json.dumps({ "question": question, "answer": lora_answer, "explanation": lora_explanation }) + "\n")
            
            # --- STEP 3: ADAPTIVE RAG (FINAL PIPELINE) ---
            pbar.set_description(f"Q: \"{question[:20]}...\" | 3. RAG Check")
            temp_lora_explanation, confidence = generate_with_confidence(lora_model, tokenizer, f"[INST] QUESTION: {question}\nANSWER: {lora_answer}\nEXPLANATION: [/INST]")
            repetition_score = calculate_repetition_score(lora_answer, temp_lora_explanation)
            
            final_answer = lora_answer
            final_explanation = lora_explanation 
            rag_triggered = False
            trigger_reason = "High Confidence / Low Repetition"
            context = ""
            retrieval_verified = False

            if confidence < CONFIDENCE_THRESHOLD or repetition_score > REPETITION_THRESHOLD:
                rag_triggered = True
                trigger_reason = f"Low Confidence ({confidence:.2f})" if confidence < CONFIDENCE_THRESHOLD else f"High Repetition ({repetition_score:.2f})"
                pbar.set_description(f"RAG Triggered. Synthesizing...")
                context = perform_rag_retrieval(encoder, index, id_map, chunk_map, question, final_answer)
                
                verification_result = verify_context(context, question)
                
                min_word_count = 25 
                if context and verification_result.get("is_verified") and verification_result.get("context_length", 0) > min_word_count:
                    retrieval_verified = True
                    verified_context = verification_result.get("verified_context", "")
                    pbar.set_description(f"Context VERIFIED ({verification_result['context_length']} words). Synthesizing...")
                    synthesizer_prompt = RAG_SYNTHESIZER_PROMPT_BASELINE.format(
                        context=verified_context, 
                        question=question, 
                        answer=final_answer
                    )
                    with lora_model.disable_adapter():
                        final_explanation = generate_chained(lora_model, tokenizer, synthesizer_prompt)
                else:
                    pbar.set_description(f"Context REJECTED (insufficient). Falling back to BASELINE model.")
                    fallback_prompt = EXPLANATION_PROMPT_LORA.format(question=question, answer=final_answer)
                    with lora_model.disable_adapter():
                        final_explanation = generate_chained(lora_model, tokenizer, fallback_prompt)
            else:
                pbar.set_description(f"RAG Skipped (Conf: {confidence:.2f}, Rep: {repetition_score:.2f})")
            
            f_rag.write(json.dumps({
                "question": question, "answer": final_answer, "explanation": final_explanation,
                "rag_triggered": rag_triggered, "trigger_reason": trigger_reason,
                "retrieval_verified": retrieval_verified,
                "confidence_score": confidence, "repetition_score": repetition_score, "context": context
            }) + "\n")
            
            f_base.flush(); f_lora.flush(); f_rag.flush()
            gc.collect()
            torch.cuda.empty_cache()
            
    print(f"\n🎉 All generations complete.")
