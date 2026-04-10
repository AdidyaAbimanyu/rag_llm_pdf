import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

st.set_page_config(page_title="Pro PDF AI", layout="wide")

with st.sidebar:
    st.title("📂 Manajemen Dokumen")
    uploaded_files = st.file_uploader(
        "Upload satu atau banyak PDF", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("Hapus Riwayat Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.info("Model: Llama 3.3 70B\nEngine: Groq Cloud")

st.title("💬 Chat dengan Dokumenmu")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
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
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    retriever = vector_db.as_retriever()

    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rangkum riwayat obrolan menjadi pertanyaan mandiri."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Jawab pertanyaan berdasarkan dokumen yang diunggah: {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    for message in st.session_state.chat_history:
        with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
            st.markdown(message.content)

    if user_query := st.chat_input("Tanyakan sesuatu tentang dokumenmu..."):
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Mencari jawaban..."):
            response = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
            answer = response["answer"]
            
            with st.chat_message("assistant"):
                st.markdown(answer)
            
            st.session_state.chat_history.extend([HumanMessage(content=user_query), AIMessage(content=answer)])
else:
    st.warning("Silakan upload minimal satu file PDF di sidebar untuk memulai.")