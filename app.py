import streamlit as st
import requests

# ── Config ──
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="தமிழ் RAG",
    page_icon="🔍",
    layout="centered"
)

# ── Header ──
st.title("🔍 தமிழ் RAG System")
st.caption("Tamil Wikipedia Question Answering System | Powered by Sarvam AI")

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of sources", 1, 5, 3)
    
    if st.button("🗑️ Clear History"):
        requests.post(f"{API_URL}/clear")
        st.session_state.messages = []
        st.success("History cleared!")
    
    st.divider()
    st.markdown("**Model:** Sarvam-M")
    st.markdown("**Embedding:** LaBSE")
    st.markdown("**Vector DB:** FAISS")
    st.markdown("**Reranker:** CrossEncoder")

# ── Chat History ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display chat history ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Sources"):
                for i, src in enumerate(msg["sources"]):
                    st.markdown(f"**{i+1}.** {src}...")

# ── Chat input ──
if query := st.chat_input("தமிழில் கேள்வி கேளுங்கள்..."):
    
    # Show user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Call FastAPI
    with st.chat_message("assistant"):
        with st.spinner("யோசிக்கிறேன்..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": query, "top_k": top_k}
                )
                result = response.json()
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)
                with st.expander("📚 Sources"):
                    for i, src in enumerate(sources):
                        st.markdown(f"**{i+1}.** {src}...")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"பிழை: {str(e)}")