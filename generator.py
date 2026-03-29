import requests
from config import OLLAMA_BASE, LLM_GENERAL, LLM_REASON

RAG_PROMPT = """You are a precise assistant. Answer the question using ONLY \
the provided context. If the context does not contain enough information, \
say so clearly — do not fabricate.

Context:
{context}

Question: {question}

Answer:"""

def _call_ollama(model: str, prompt: str, temperature: float = 0.1) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model":       model,
            "prompt":      prompt,
            "stream":      False,
            "temperature": temperature,
            "options": {
                "num_ctx":      8192,
                "num_predict":  1024,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()

def generate(question: str, context_docs: list[dict], use_reasoning: bool = False) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {d['source']}]\n{d['text']}"
        for d in context_docs
    )
    prompt = RAG_PROMPT.format(context=context, question=question)
    model  = LLM_REASON if use_reasoning else LLM_GENERAL
    return _call_ollama(model, prompt)