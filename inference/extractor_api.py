"""
A FastAPI server that uses a CPU-based LLM (Phi-3) to perform
advanced tasks for a RAG pipeline: rewriting, expansion, broadening, 
verification, and pruning.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import json

# === Configuration ===
EXTRACTOR_MODEL_PATH = "./Phi-3-mini-4k-instruct-q4.gguf"
MODEL_CONTEXT_WINDOW = 4096
PROMPT_BUFFER = 256

print("🔹 Loading Extractor model (Phi-3-mini) onto CPU...")
extractor_model = Llama(model_path=EXTRACTOR_MODEL_PATH, n_ctx=MODEL_CONTEXT_WINDOW, n_gpu_layers=0, verbose=False)
print("✅ Extractor model loaded on CPU.")

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

class VerificationRequest(BaseModel):
    context: str
    question: str

class PruningRequest(BaseModel):
    chunk: str
    question: str

def _truncate_context(prompt_template: str, context: str) -> str:
    header_tokens = extractor_model.tokenize(prompt_template.replace("{context_placeholder}", "").encode("utf-8"))
    available_space = MODEL_CONTEXT_WINDOW - len(header_tokens) - PROMPT_BUFFER
    context_tokens = extractor_model.tokenize(context.encode("utf-8"))
    
    if len(context_tokens) > available_space:
        print(f"⚠️ Context length ({len(context_tokens)} tokens) exceeds safe limit. Truncating.")
        truncated_tokens = context_tokens[:available_space]
        return extractor_model.detokenize(truncated_tokens).decode("utf-8", errors="ignore")
    
    return context

@app.post("/rewrite-query")
def rewrite_query(request: QuestionRequest):
    prompt = f"""<|user|>
Analyze the medical question. If it contains non-standard or misspelled terms, rewrite it using standard terminology. If it's already standard, return the original question verbatim. **Output ONLY the final question string.**

QUESTION: "What are the features of metrioid tumors?"
REWRITTEN QUESTION: "What are the features of endometrioid carcinoma?"

QUESTION: "What is the glymphatic system?"
REWRITTEN QUESTION: "What is the glymphatic system?"

QUESTION: {request.question}
REWRITTEN QUESTION:<|end|>
<|assistant|>
"""
    output = extractor_model(prompt, max_tokens=100, stop=["<|end|>", "\n"])
    rewritten_question = output['choices'][0]['text'].strip()
    return {"rewritten_question": rewritten_question}

@app.post("/expand-query")
def expand_query(request: QuestionRequest):
    prompt = f"""<|user|>
You are a helpful research assistant. Rewrite the user's question into a JSON list of 3 diverse, related search queries. **Output ONLY the JSON list.**

QUESTION: Is there a distinction made between changes occurring during the early stages of chronic rejection and those indicating advanced or irreversible disease?
QUERIES:
["early vs advanced chronic organ rejection stages", "histological features of irreversible transplant rejection", "Banff classification for chronic rejection progression"]

QUESTION: {request.question}
QUERIES:<|end|>
<|assistant|>
"""
    output = extractor_model(prompt, max_tokens=150, stop=["<|end|>"])
    response_text = output['choices'][0]['text'].strip()
    try:
        queries_list = json.loads(response_text)
        if request.question not in queries_list:
            queries_list.insert(0, request.question)
        return {"queries": queries_list}
    except json.JSONDecodeError:
        return {"queries": [request.question]}

@app.post("/broaden-query")
def broaden_query(request: QuestionRequest):
    prompt = f"""<|user|>
The following search query returned no results. Your task is to generate a single, broader, more general query that is related to the original topic but more likely to find documents.

FAILED QUERY: "histology of metrioid tumors"
BROADER QUERY: "rare endometrial cancer subtypes histology"

FAILED QUERY: "extraskeletal myxoid chondrosarcoma NR4A3 gene fusion"
BROADER QUERY: "extraskeletal myxoid chondrosarcoma genetics"

FAILED QUERY: {request.question}
BROADER QUERY:<|end|>
<|assistant|>
"""
    output = extractor_model(prompt, max_tokens=100, stop=["<|end|>", "\n"])
    broader_query = output['choices'][0]['text'].strip()
    return {"broader_query": broader_query}

@app.post("/verify-context")
def verify_context(request: VerificationRequest):
    prompt_template = """<|user|>
Read the provided CONTEXT and QUESTION. Your task is to extract only the specific sentences from the CONTEXT that directly answer the QUESTION. If no sentences are directly relevant, return an empty string.

CONTEXT: {context_placeholder}

QUESTION: {request.question}
<|end|>
<|assistant|>"""
    
    safe_context = _truncate_context(prompt_template, request.context)
    final_prompt = prompt_template.format(context_placeholder=safe_context, request=request)
    
    output = extractor_model(final_prompt, max_tokens=512, stop=["<|end|>"])
    response_text = output['choices'][0]['text'].strip()
    
    is_verified = bool(response_text)
    context_length = len(response_text.split()) 
    
    return {"is_verified": is_verified, "verified_context": response_text, "context_length": context_length}
