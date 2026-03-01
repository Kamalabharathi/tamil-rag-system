# 🔍 Tamil Wikipedia RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for Tamil Wikipedia, powered by **Sarvam AI** for Tamil language understanding.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)
![LaBSE](https://img.shields.io/badge/Embedding-LaBSE-orange)
![Sarvam AI](https://img.shields.io/badge/LLM-Sarvam--M-purple)

---

## 🏗️ Architecture

```
Tamil Wikipedia Data
        ↓
LaBSE Embeddings (multilingual)
        ↓
FAISS Vector Store (318K vectors)
        ↓
CrossEncoder Reranking
        ↓
CAG Memory (context-aware)
        ↓
Sarvam-M (Tamil LLM)
        ↓
Tamil Answer! 🎯
```

---

## ✨ Features

- **Tamil-first RAG** — Answers only in Tamil
- **LaBSE Embeddings** — 109-language multilingual model
- **FAISS Vector Store** — 318,498 Tamil Wikipedia chunks
- **CrossEncoder Reranking** — Better context selection
- **CAG Memory** — Context-aware conversation history
- **Sarvam-M LLM** — Best Indic language model
- **FastAPI Backend** — Production REST API
- **Streamlit UI** — Beautiful chat interface
- **Hallucination Prevention** — Strict context-grounded answers

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Embedding | `sentence-transformers/LaBSE` |
| Vector DB | `FAISS` |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| LLM | `Sarvam-M` (Sarvam AI) |
| Backend | `FastAPI` |
| Frontend | `Streamlit` |
| Language | `Python 3.10+` |

---

## 📊 Dataset

- **Source:** Tamil Wikipedia
- **Chunks:** 318,498 text chunks
- **Chunk Size:** 450 tokens (LaBSE optimized)
- **Overlap:** 90 tokens
- **Embedding Dim:** 768

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Kamalabharathi/tamil-rag-system.git
cd tamil-rag-system
```

### 2. Create virtual environment
```bash
python -m venv tamil-rag-env
.\tamil-rag-env\Scripts\Activate.ps1  # Windows
source tamil-rag-env/bin/activate      # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
Create `.env` file:
```
SARVAM_API_KEY=your-sarvam-key
GROQ_API_KEY=your-groq-key
```

### 5. Run FastAPI backend
```bash
uvicorn main:app --reload
```

### 6. Run Streamlit UI
```bash
streamlit run app.py
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/query` | Ask Tamil question |
| POST | `/clear` | Clear chat history |

### Example request:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "தமிழ் இலக்கியம் பற்றி கூறுக", "top_k": 3}'
```

### Example response:
```json
{
  "answer": "தமிழ் இலக்கியம் பல நூற்றாண்டுகள் பழமையானது...",
  "sources": ["சங்க இலக்கியம்...", "தொல்காப்பியம்..."]
}
```

---

## 🔬 LLM Comparison Results

| Model | Tamil Quality | Factual Accuracy | Status |
|-------|--------------|-----------------|--------|
| Sarvam-M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Winner |
| Groq Llama 3.1 | ⭐⭐⭐ | ⭐⭐⭐ | ✅ Baseline |
| OpenRouter | ❌ | ❌ | ❌ Rate limited |

**Winner: Sarvam-M** — Best Indic language understanding!

---

## 📁 Project Structure

```
tamil-rag-system/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      ← Tamil data loading
│   ├── embedding.py        ← LaBSE embeddings
│   ├── vectorstore.py      ← FAISS vector store
│   └── retriever.py        ← RAG pipeline + prompt
├── main.py                 ← FastAPI backend
├── app.py                  ← Streamlit UI
├── search.py               ← LLM comparison
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔑 Key Findings

1. **LaBSE > Tamil BERT** for Tamil embeddings
2. **Sarvam-M > Groq Llama** for Tamil generation
3. **FAISS only > BM25 hybrid** for Tamil retrieval
4. **Tamil prompt > English prompt** for Tamil models
5. **Hallucination detected** for specific fact queries

---

## 📝 RAG Pipeline Parameters

```python
# Embedding
chunk_size = 450          # LaBSE token limit
chunk_overlap = 90
normalize_embeddings = True

# Retrieval
top_k_faiss = 30          # Initial FAISS results
top_k_rerank = 3          # After CrossEncoder
score_threshold = 0       # Filter low scores

# Generation (Sarvam-M)
temperature = 0.1         # Low for factual accuracy
max_tokens = 1024
top_p = 0.85
repetition_penalty = 1.1
```

---

## 👤 Author

**Kamalabharathi**
- GitHub: [@Kamalabharathi](https://github.com/Kamalabharathi)

---

## 📄 License

MIT License
