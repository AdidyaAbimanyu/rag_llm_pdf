import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Pro PDF AI + Google Embeddings", layout="wide")

# --- FUNGSI PEMROSESAN DOKUMEN (DENGAN CACHE) ---
@st.cache_resource
def get_vector_db(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        # Simpan file sementara
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Load PDF
        loader = PyPDFLoader(temp_path)
        all_docs.extend(loader.load())
        os.remove(temp_path)

    # Split teks menjadi potongan kecil
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    
    # MENGGUNAKAN GOOGLE EMBEDDINGS (GRATIS & CEPAT)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Simpan ke Vector Store (ChromaDB)
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_db.as_retriever(search_kwargs={"k": 3}) # Ambil 3 sumber teratas

# --- SIDEBAR ---
with st.sidebar:
    st.title("📂 Manajemen Dokumen")
    uploaded_files = st.file_uploader(
        "Upload PDF kamu di sini", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("🗑️ Hapus Riwayat Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.info("Brain: Llama 3.3 70B\nMemory: Google Embedding 001")

# --- TAMPILAN UTAMA ---
st.title("💬 Chat AI dengan Sitasi")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    # Proses dokumen hanya jika ada file baru
    retriever = get_vector_db(uploaded_files)

    # Inisialisasi LLM (Groq)
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    # Prompt untuk membuat pertanyaan mandiri dari history
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rangkum riwayat obrolan menjadi pertanyaan mandiri yang bisa dipahami tanpa konteks sebelumnya."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Prompt utama penjawab
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Kamu adalah asisten pintar. Jawab pertanyaan hanya berdasarkan konteks dokumen di bawah ini:\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    # Tampilkan History Chat
    for message in st.session_state.chat_history:
        with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
            st.markdown(message.content)

    # Input User
    if user_query := st.chat_input("Tanyakan sesuatu tentang dokumenmu..."):
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Berpikir..."):
            # Jalankan RAG
            response = rag_chain.invoke({
                "input": user_query, 
                "chat_history": st.session_state.chat_history
            })
            
            answer = response["answer"]
            sources = response.get("context", [])

            # Tampilkan Jawaban Assistant
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                # FITUR SITASI: Tampilkan sumber referensi
                if sources:
                    with st.expander("📌 Lihat Referensi Dokumen"):
                        for i, doc in enumerate(sources):
                            file_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                            page = doc.metadata.get('page', 0) + 1
                            st.write(f"**Sumber {i+1}:** {file_name} (Halaman {page})")
                            st.caption(f"_{doc.page_content[:250]}..._")
            
            # Update History
            st.session_state.chat_history.extend([
                HumanMessage(content=user_query), 
                AIMessage(content=answer)
            ])
else:
    st.warning("👈 Silakan upload minimal satu file PDF di sidebar untuk memulai.")