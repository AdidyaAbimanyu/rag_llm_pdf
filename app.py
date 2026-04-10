import streamlit as st
import os
from dotenv import load_dotenv

# --- IMPORT LANGCHAIN VERSI TERBARU (MODULAR) ---
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Folder baru untuk Chains
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

st.set_page_config(page_title="Pro PDF AI", layout="wide")

# --- PROSES DOKUMEN ---
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    
    # Gunakan model ini, paling jarang kena 404
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_db.as_retriever(search_kwargs={"k": 3})

# --- UI SIDEBAR ---
with st.sidebar:
    st.title("📂 Dokumen")
    uploaded_files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
    if st.button("Hapus Chat"):
        st.session_state.chat_history = []
        st.rerun()

st.title("💬 Chat AI PDF")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    retriever = get_vector_db(uploaded_files)
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))

    # Chain untuk history
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rangkum chat menjadi pertanyaan mandiri."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Chain untuk jawaban
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Jawab berdasarkan konteks: {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    # Chat UI
    for message in st.session_state.chat_history:
        with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
            st.markdown(message.content)

    if user_query := st.chat_input("Tanya dokumen..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.spinner("Berpikir..."):
            response = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
            answer = response["answer"]
            sources = response.get("context", [])

            with st.chat_message("assistant"):
                st.markdown(answer)
                if sources:
                    with st.expander("Sitasi"):
                        for i, doc in enumerate(sources):
                            st.write(f"**{doc.metadata.get('source')}** (Hal. {doc.metadata.get('page')+1})")

            st.session_state.chat_history.extend([HumanMessage(content=user_query), AIMessage(content=answer)])
else:
    st.warning("Upload PDF dulu.")