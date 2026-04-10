import streamlit as st
import os
from dotenv import load_dotenv

# --- IMPORT CORE ---
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- IMPORT CHAINS (VERSI TERBARU) ---
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- IMPORT HUGGING FACE (API MODE) ---
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

st.set_page_config(page_title="Pro PDF AI - Hugging Face Mode", layout="wide")

@st.cache_resource
def get_vector_db(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_path)
        all_docs.extend(loader.load())
        os.remove(temp_path)

    # Chunking yang pas agar tidak kepanjangan
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    
    # PAKAI HUGGING FACE INFERENCE API (GRATIS & STABIL)
    # Pastikan masukkan HF_TOKEN di Streamlit Secrets
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )
    
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_db.as_retriever(search_kwargs={"k": 3})

# --- SIDEBAR ---
with st.sidebar:
    st.title("📂 Dokumen")
    uploaded_files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
    if st.button("🗑️ Hapus Chat"):
        st.session_state.chat_history = []
        st.rerun()
    st.divider()
    st.info("Brain: Llama 3.3 (Groq)\nMemory: MiniLM (Hugging Face API)")

# --- MAIN UI ---
st.title("💬 Chat AI + Sitasi (HF Mode)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    retriever = get_vector_db(uploaded_files)
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))

    # Chain Logic
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rangkum riwayat obrolan menjadi pertanyaan mandiri."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Jawab pertanyaan berdasarkan konteks: {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    # Render Chat
    for message in st.session_state.chat_history:
        with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
            st.markdown(message.content)

    if user_query := st.chat_input("Tanyakan sesuatu..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.spinner("Mencari jawaban..."):
            response = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
            answer = response["answer"]
            sources = response.get("context", [])

            with st.chat_message("assistant"):
                st.markdown(answer)
                if sources:
                    with st.expander("📌 Lihat Referensi"):
                        for i, doc in enumerate(sources):
                            fname = os.path.basename(doc.metadata.get('source', 'PDF'))
                            page = doc.metadata.get('page', 0) + 1
                            st.write(f"**{fname}** - Hal. {page}")
                            st.caption(f"_{doc.page_content[:200]}..._")

            st.session_state.chat_history.extend([HumanMessage(content=user_query), AIMessage(content=answer)])
else:
    st.warning("Silakan upload PDF di sidebar.")