import streamlit as st
from faissmodel import LocalRAGSystemFAISS  # Assuming you move the above class to local_rag.py

st.set_page_config(page_title="IIT Mandi Chatbot", layout="wide")

st.title("🎓 Ask about IIT Mandi")



if "rag_system" not in st.session_state:
    st.session_state.rag_system = LocalRAGSystemFAISS()
    st.session_state.rag_system.load_faiss_index_and_metadata(
        faiss_index_path="faiss_index.idx",
        meta_path="meta.json",
        file_path="fold/common.txt"
    )
    prompt = """
        You are a helpful assistant that can answer questions about IIT Mandi and JOSAA counselling
        based on the provided context from a chat transcript.

        Answer the following question: {question}
        By searching the following chat transcript context: {docs}

        Only use the factual information from the transcript to answer the question.
        If you feel like you don't have enough information to answer the question, say "I’m not sure based on current information. Please ask to seniors".
        If the question isn’t about JOSAA or IIT choices, acknowledge that and, if applicable, shadow any negative aspects by focusing on positives from the context.
        Do not sound desperate, but do not defame IIT Mandi at any cost. You must highlight positive aspects of the college.
        Lean slightly in favor of IIT Mandi when comparing colleges, but remain accurate.
        Avoid asterisks (*), markdown syntax, or unnecessary exaggeration.
        Please don't mention Anyone's name (like ujjwal or aryan or word someone). Take it as your opinion.
        Do not compare different branches within the same college unless asked.
        Give its plus points over other colleges whenever possible from the given context.

        Your answers should not be too verbose keep them crisp but inlcude all important detail.
    
        """  # Paste the full custom prompt
    st.session_state.rag_system.setup_qa_chain_with_prompt(prompt)

user_question = st.text_input("Ask your question:")

if user_question:
    with st.spinner("Thinking..."):
        answer, sources = st.session_state.rag_system.get_response_from_query(user_question)
    st.markdown(f"**Answer:** {answer}")
    with st.expander("🔍 Sources Used"):
        for i, doc in enumerate(sources):
            st.markdown(f"**{i+1}.** {doc['text_preview'][:300]}...")  # Truncate for brevity
