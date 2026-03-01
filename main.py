from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from src.vectorstore import FaissVectorStore
from src.retriever import ImprovedRetriever

# ── API Key ──
SARVAM_API_KEY = "sk_3mqt47me_KcY8X0qeJ4WGmPFmZpfdvA0b"

# ── Sarvam LLM ──
class SarvamLLM:
    def invoke(self, prompt_list):
        resp = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {SARVAM_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sarvam-m",
                "messages": [{"role": "user", "content": prompt_list[0]}],
                "max_tokens": 1024,
                "temperature": 0.1,
                "top_p": 0.85,
                "repetition_penalty": 1.1
            }
        ).json()
        class Result:
            content = resp['choices'][0]['message']['content'] if 'choices' in resp else f"பிழை: {resp}"
        return Result()

# ── Global Variables ──
store = None
retriever = None

# ── Lifespan ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, retriever
    print("[INFO] Loading RAG pipeline...")
    store = FaissVectorStore(
        persist_dir="faiss_store",
        embedding_model="sentence-transformers/LaBSE"
    )
    store.load()
    llm = SarvamLLM()
    retriever = ImprovedRetriever(store=store, llm=llm)
    print("[INFO] RAG pipeline ready!")
    yield
    print("[INFO] Shutting down!")

# ── FastAPI App ──
app = FastAPI(
    title="Tamil RAG API",
    description="Tamil Wikipedia RAG System using Sarvam AI",
    version="1.0.0",
    lifespan=lifespan
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Request/Response Models ──
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: list

# ── Endpoints ──
@app.get("/")
def home():
    return {
        "message": "Tamil RAG API is running!",
        "model": "Sarvam-M",
        "embedding": "LaBSE"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "sarvam-m",
        "embedding": "LaBSE"
    }

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty!"
        )
    result = retriever.query(request.question, top_k=request.top_k)
    return QueryResponse(
        answer=result['answer'],
        sources=result['sources']
    )

@app.post("/clear")
def clear_history():
    retriever.clear_history()
    return {"message": "வரலாறு அழிக்கப்பட்டது!"}

# ── Run ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)