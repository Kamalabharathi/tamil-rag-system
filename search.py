import requests
from src.vectorstore import FaissVectorStore
from src.retriever import ImprovedRetriever

SARVAM_API_KEY = "Your_api_key"

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
                "max_tokens": 1500,
                "temperature": 0.1,
                "top_p": 0.85,
                "repetition_penalty": 1.1
            }
        ).json()
        class Result:
            content = resp['choices'][0]['message']['content'] if 'choices' in resp else f"பிழை: {resp}"
        return Result()

if __name__ == "__main__":
    store = FaissVectorStore(
        persist_dir="faiss_store",
        embedding_model="sentence-transformers/LaBSE"
    )
    store.load()
    
    llm = SarvamLLM()
    retriever = ImprovedRetriever(store=store, llm=llm)

    test_queries = [
        "தமிழ் சினிமா வரலாறு",
        "தமிழ்நாட்டின் வரலாறு",
        "தமிழ் இலக்கியம்"
    ]

    for query in test_queries:
        print(f"\n📌 {query}")
        result = retriever.query(query)
        print(f"💬 {result['answer']}")
        print("-"*40)
