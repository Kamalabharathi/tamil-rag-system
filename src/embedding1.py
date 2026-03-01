from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_text_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "sentence-transformers/LaBSE", chunk_size: int = 450, chunk_overlap: int = 90):
        """
        Tamil-optimized RAG embedding pipeline with LaBSE.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded LaBSE embedding model: {model_name}")
        print(f"[INFO] Max sequence length: 512 tokens | Using chunk_size={chunk_size}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """FIXED: Tamil-friendly chunking - works with Document list directly."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "।", ". ", "? ", "! ", " ", ""]
        )
        # ✅ RecursiveCharacterTextSplitter.split_documents() accepts Document list directly
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} docs → {len(chunks)} chunks")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """Generate normalized LaBSE embeddings."""
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} Tamil chunks...")
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

# Usage
if __name__ == "__main__":
    docs = load_text_documents(r"C:\Users\naray\Downloads\wiki_data\Clean wiki_part1")
    emb_pipe = EmbeddingPipeline()
    
    # GPU support
    try:
        emb_pipe.model = emb_pipe.model.to('cuda')
        print("[INFO] Model moved to CUDA")
    except:
        print("[INFO] Using CPU")
    
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    
    print(f"✅ COMPLETE! Chunks: {len(chunks)}, Shape: {embeddings.shape}")
