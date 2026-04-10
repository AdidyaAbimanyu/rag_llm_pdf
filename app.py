import streamlit as st
import os
from dotenv import load_dotenv

# --- IMPORT CORE & CHAINS ---
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

# 1. PERCANTIK UI: Set tema dan ikon
st.set_page_config(page_title="PDF Intelligence Pro", page_icon="🤖", layout="wide")

# Custom CSS untuk mempercantik chat
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stSidebar { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_vector_db(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_path)
        all_docs.extend(loader.load())
        os.remove(temp_path)

    # 2. OPTIMASI CHUNKING: Ukuran yang lebih seimbang untuk akurasi
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )
    
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_db.as_retriever(search_kwargs={"k": 3})

# --- SIDEBAR PRO ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
    st.title("Doc-Brain Settings")
    
    uploaded_files = st.file_uploader("📂 Upload Dokumen (PDF)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} File terdeteksi")
        with st.expander("Daftar File"):
            for f in uploaded_files:
                st.caption(f"• {f.name}")

    st.divider()
    if st.button("🗑️ Reset Obrolan", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.info("💡 **Tips:** Tanyakan hal spesifik seperti 'Apa kesimpulan dari bab 2?'")

# --- MAIN CHAT UI ---
st.title("🤖 AI PDF Assistant")
st.caption("Tanya jawab cerdas dengan referensi dokumen otomatis")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    # Tampilkan loading yang lebih keren
    with st.status("🧠 Memproses memori dokumen...", expanded=False) as status:
        retriever = get_vector_db(uploaded_files)
        status.update(label="✅ Dokumen siap dipelajari!", state="complete", expanded=False)

    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.2)

    # Logika RAG tetap sama (Sudah stabil)
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Ubah riwayat chat jadi pertanyaan mandiri."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Kamu adalah asisten ahli. Jawab dengan ramah hanya berdasarkan konteks: {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    # 1. PERCANTIK CHAT: Gunakan avatar
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message.content)

    if user_query := st.chat_input("Ketik pertanyaan Anda di sini..."):
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Menganalisa dokumen..."):
                response = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
                answer = response["answer"]
                sources = response.get("context", [])

                st.markdown(answer)
                
                # Fitur Sitasi yang lebih cantik
                if sources:
                    cols = st.columns(len(sources))
                    for i, doc in enumerate(sources):
                        with st.expander(f"📍 Sumber {i+1}"):
                            st.write(f"**File:** {os.path.basename(doc.metadata.get('source'))}")
                            st.write(f"**Halaman:** {doc.metadata.get('page')+1}")
                            st.caption(doc.page_content[:300] + "...")

            st.session_state.chat_history.extend([HumanMessage(content=user_query), AIMessage(content=answer)])
else:
    # Tampilan saat kosong
    st.info("👈 Silakan unggah dokumen PDF di sidebar untuk mulai bertanya.")
    st.image("https://p7.hiclipart.com/preview/264/343/1021/chatbot-customer-service-robot-user-experience-computer-icons-robot.jpg", width=300)