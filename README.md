# 📑 Multi-Doc AI Chat (RAG System)

A smart chatbot application that lets you chat with your personal documents using **Retrieval-Augmented Generation (RAG)**. Powered by Llama 3.3 (via Groq) and Hugging Face Embeddings.

## ✨ Key Features
- **Multi-Format Support:** Reads `.pdf`, `.docx`, and `.txt` files.
- **Powerful AI Model:** Uses Llama 3.3 70B from Groq for fast and accurate answers.
- **Efficient Processing:** Hugging Face Inference API for text embeddings with no local computing load.
- **Smart Citations:** Shows source references (file name & page) for each answer.
- **Session Memory:** AI remembers previous conversation context.
- **File Size Limit:** Maximum 10MB per file to maintain server performance.

## 🛠️ Tech Stack
- **Framework:** Streamlit
- **Orchestration:** LangChain
- **LLM:** Groq (Llama 3.3 70B)
- **Embeddings:** Hugging Face (`all-MiniLM-L6-v2`)
- **Vector Store:** ChromaDB

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install Dependencies
Make sure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

3. Set Up Environment Variables
Create a .env file in the root directory and add your API keys:

Code snippet
```python
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

4. Run the Application
```bash
streamlit run app.py
```

🌐 Deployment (Streamlit Cloud)

This app is ready for deployment on Streamlit Cloud. Add your GROQ_API_KEY and HF_TOKEN in the Advanced Settings > Secrets section of your Streamlit dashboard.

📝 Usage Notes

File uploads are limited to 10MB per file to ensure fast embedding processing.

Use the "Reset Chat" button in the sidebar to clear conversation history and start a new topic.

Built with ❤️ using LangChain & Streamlit.