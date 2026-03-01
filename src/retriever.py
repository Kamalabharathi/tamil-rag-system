from sentence_transformers import CrossEncoder

PROMPT_TEMPLATE = """உங்கள் பணி: தமிழ் RAG நிபுணராக செயல்படுதல்.

கட்டுப்பாடான விதிகள் (இவற்றை கண்டிப்பாக பின்பற்றவும்):
1. கீழே கொடுக்கப்பட்ட சூழல் (context) மட்டுமே பயன்படுத்தவும்
2. உங்கள் சொந்த அறிவை பயன்படுத்த வேண்டாம்
3. சூழலில் இல்லாத தகவலை கூறவே கூறாதீர்கள்
4. தமிழில் மட்டும் பதில் அளிக்கவும்
5. துல்லியமாகவும் முழுமையாகவும் பதில் அளிக்கவும்
6. தகவல் சூழலில் இல்லையெனில் கண்டிப்பாக இதை மட்டும் கூறவும்:
   "இந்த சூழலில் தேவையான தகவல் இல்லை"

⚠️ எச்சரிக்கை: சூழலுக்கு வெளியே தகவல் கூறினால் தவறான பதிலாகும்!

=== சூழல் (இதை மட்டும் பயன்படுத்தவும்) ===
{context}

=== கேள்வி ===
{question}

=== தமிழ் பதில் (சூழலில் இருந்து மட்டும்) ===
"""

VERIFY_TEMPLATE = """Read the context and answer below carefully.

Context: {context}
Question: {question}
Answer: {answer}

Is the answer based ONLY on the given context?
Reply with only ONE word: YES or NO
"""

class ImprovedRetriever:
    def __init__(self, store, llm,
                 reranker_model='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'):
        self.store = store
        self.llm = llm
        self.reranker = CrossEncoder(reranker_model)
        self.chat_history = []
        print("✅ ImprovedRetriever ready!")

    def verify_answer(self, query, context, answer):
        """Verify if answer is grounded in context."""
        try:
            verify_prompt = VERIFY_TEMPLATE.format(
                context=context[:1000],  # limit for token saving
                question=query,
                answer=answer
            )
            response = self.llm.invoke([verify_prompt])
            verdict = response.content.strip().upper()
            print(f"[INFO] Verification verdict: {verdict}")
            return "YES" in verdict
        except:
            return True  # if verification fails, trust the answer

    def query(self, query, top_k=3):
        try:
            if not query or len(query.strip()) == 0:
                return {"answer": "தயவுசெய்து ஒரு கேள்வி கேளுங்கள்!", "sources": []}

            # FAISS search
            initial_results = self.store.query(query, top_k=30)

            if not initial_results:
                return {"answer": "மன்னிக்கவும், தொடர்புடைய தகவல்கள் கிடைக்கவில்லை!", "sources": []}

            # Rerank
            pairs = [[query, r['metadata']['text']] for r in initial_results]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(scores, initial_results), reverse=True)

            # Filter low score chunks
            filtered = [(score, r) for score, r in ranked if score > 0]
            if not filtered:
                filtered = ranked

            # Top results
            top_results = filtered[:top_k]
            context = "\n\n".join([r['metadata']['text'] for _, r in top_results])
            sources = [r['metadata']['text'][:100] for _, r in top_results]

            # CAG history in Tamil
            history_text = ""
            if self.chat_history:
                last = self.chat_history[-1]
                history_text = f"முந்தைய கேள்வி: {last['query']}\nமுந்தைய பதில்: {last['answer'][:100]}"

            # Add history to context
            if history_text:
                full_context = f"முந்தைய உரையாடல்:\n{history_text}\n\n{context}"
            else:
                full_context = context

            # Tamil prompt
            prompt = PROMPT_TEMPLATE.format(
                context=full_context,
                question=query
            )

            response = self.llm.invoke([prompt])
            answer = response.content

            # ── Verification step ──
            is_grounded = self.verify_answer(query, context, answer)
            if not is_grounded:
                print("[WARNING] Hallucination detected! Replacing answer.")
                answer = "இந்த சூழலில் தேவையான தகவல் இல்லை"

            self.chat_history.append({"query": query, "answer": answer})

            return {"answer": answer, "sources": sources}

        except Exception as e:
            return {"answer": f"பிழை: {str(e)}", "sources": []}

    def clear_history(self):
        self.chat_history = []
        print("✅ History cleared!")