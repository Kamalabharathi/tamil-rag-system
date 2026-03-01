from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import TextLoader

def load_text_documents(data_dir: str) -> List[Any]:
    """
    Load all .txt files from the data directory into LangChain document format.
    """
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")

    documents = []

    txt_files = list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] Found {len(txt_files)} TXT files: {[str(f) for f in txt_files]}")

    for txt_file in txt_files:
        print(f"[DEBUG] Loading TXT: {txt_file}")
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")  # Important for Tamil text
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} TXT docs from {txt_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load TXT {txt_file}: {e}")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents

# Example usage with your folder path
if __name__ == "__main__":
    folder_path = r"C:\Users\naray\Downloads\tamil-rag-system\Clean wiki_part1"
    docs = load_text_documents(folder_path)
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)