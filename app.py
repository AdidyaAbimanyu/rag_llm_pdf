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

# --- IMPORT CHAINS (MODULAR) ---
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Chat PDF", page_icon="📄")

# --- FUNGSI PEMROSESAN (DENGAN BATASAN 10MB) ---
@st.cache_resource(show_spinner=False)
def get_vector_db(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        # Cek ukuran file (10MB = 10 * 1024 * 1024 bytes)
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error(f"❌ File '{uploaded_file.name}' terlalu besar! Maksimal 10MB agar tetap kencang.")
            continue
            
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        loader = PyPDFLoader(temp_path)
        all_docs.extend(loader.load())
        os.remove(temp_path)

    if not all_docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )
    
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_db.as_retriever(search_kwargs={"k": 3})

# --- SIDEBAR ---
with st.sidebar:
    st.title("📂 Dokumen PDF")
    st.write("Upload file untuk mulai chat.")
    
    uploaded_files = st.file_uploader(
        "Pilih file (Maks 10MB/file)", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    st.divider()
    if st.button("🗑️ Hapus Riwayat Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.info("⚠️ **Note:** File di atas 10MB akan otomatis dilewati demi kecepatan proses.")

# --- TAMPILAN UTAMA ---
st.title("💬 Chat AI dengan Sitasi")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    # Jalankan pemrosesan
    with st.spinner("Mempelajari dokumen..."):
        retriever = get_vector_db(uploaded_files)

    if retriever:
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile", 
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

        # Logic Chain
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rangkum riwayat chat jadi pertanyaan mandiri."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Jawab pertanyaan hanya berdasarkan konteks: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        # Render History
        for message in st.session_state.chat_history:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)

        # Input User
        if user_query := st.chat_input("Tanyakan sesuatu tentang PDF ini..."):
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.spinner("Berpikir..."):
                response = rag_chain.invoke({
                    "input": user_query, 
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
                sources = response.get("context", [])

                with st.chat_message("assistant"):
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📌 Referensi Sumber"):
                            for i, doc in enumerate(sources):
                                fname = os.path.basename(doc.metadata.get('source', 'PDF'))
                                p = doc.metadata.get('page', 0) + 1
                                st.write(f"**{fname} (Hal. {p})**")
                                st.caption(f"_{doc.page_content[:200]}..._")

                st.session_state.chat_history.extend([
                    HumanMessage(content=user_query), 
                    AIMessage(content=answer)
                ])
    else:
        st.error("Tidak ada dokumen yang bisa diproses (mungkin semua file di atas 10MB).")
else:
    st.warning("👈 Upload PDF di sidebar dulu ya!")