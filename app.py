import streamlit as st
import os
from dotenv import load_dotenv

# --- IMPORT CORE ---
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- IMPORT CHAINS ---
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

st.set_page_config(page_title="Multi-Doc AI Chat", page_icon="📑")

# --- FUNGSI PEMROSESAN MULTI-FORMAT ---
@st.cache_resource(show_spinner=False)
def get_vector_db(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        # Satpam 10MB
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error(f"❌ '{uploaded_file.name}' kegedean! Maks 10MB.")
            continue
            
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # LOGIKA PEMILIHAN LOADER BERDASARKAN EKSTENSI
        try:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            elif uploaded_file.name.endswith(".docx") or uploaded_file.name.endswith(".doc"):
                loader = Docx2txtLoader(temp_path)
            elif uploaded_file.name.endswith(".txt"):
                loader = TextLoader(temp_path)
            else:
                st.warning(f"Format {uploaded_file.name} tidak didukung.")
                os.remove(temp_path)
                continue
                
            all_docs.extend(loader.load())
        except Exception as e:
            st.error(f"Gagal baca {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(temp_path):
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
    st.title("📑 Multi-Doc Chat")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, atau TXT", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    st.divider()
    if st.button("🗑️ Reset Chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- MAIN UI ---
st.title("💬 Chat Multi-Document AI")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    with st.spinner("Membaca dokumen..."):
        retriever = get_vector_db(uploaded_files)

    if retriever:
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0)

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rangkum chat jadi pertanyaan mandiri."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Jawab berdasarkan dokumen: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        for message in st.session_state.chat_history:
            with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
                st.markdown(message.content)

        if user_query := st.chat_input("Tanya apa saja tentang dokumenmu..."):
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.spinner("Berpikir..."):
                response = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
                answer = response["answer"]
                sources = response.get("context", [])

                with st.chat_message("assistant"):
                    st.markdown(answer)
                    if sources:
                        with st.expander("📌 Referensi"):
                            for i, doc in enumerate(sources):
                                fname = os.path.basename(doc.metadata.get('source', 'Doc'))
                                page = doc.metadata.get('page', 0) + 1
                                # DOCX/TXT biasanya tidak punya metadata 'page' seakurat PDF
                                loc = f"Halaman {page}" if "pdf" in fname.lower() else "Bagian Teks"
                                st.write(f"**{fname} ({loc})**")
                                st.caption(f"_{doc.page_content[:150]}..._")

                st.session_state.chat_history.extend([HumanMessage(content=user_query), AIMessage(content=answer)])
    else:
        st.error("Gagal memproses file.")
else:
    st.info("👈 Upload file PDF, DOCX, atau TXT di sidebar.")