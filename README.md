# 📑 Multi-Doc AI Chat (RAG System)

Aplikasi Chatbot cerdas yang memungkinkan pengguna untuk berinteraksi dengan dokumen pribadi menggunakan teknik **Retrieval-Augmented Generation (RAG)**. Didukung oleh model bahasa Llama 3.3 (via Groq) dan Hugging Face Embeddings.

## ✨ Fitur Utama
- **Multi-Format Support:** Membaca file `.pdf`, `.docx`, dan `.txt`.
- **Large Context Brain:** Menggunakan Llama 3.3 70B dari Groq untuk jawaban yang cepat dan akurat.
- **Efficient Memory:** Hugging Face Inference API untuk embedding teks tanpa beban komputasi lokal.
- **Smart Citations:** Menampilkan referensi sumber (nama file & halaman) untuk setiap jawaban.
- **Session Memory:** AI mengingat konteks percakapan sebelumnya.
- **Security Satpam:** Pembatasan ukuran file maksimal 10MB untuk menjaga performa server.

## 🛠️ Stack Teknologi
- **Framework:** Streamlit
- **Orchestration:** LangChain
- **LLM:** Groq (Llama 3.3 70B)
- **Embeddings:** Hugging Face (`all-MiniLM-L6-v2`)
- **Vector Store:** ChromaDB

## 🚀 Cara Menjalankan Lokal

### 1. Clone Repositori
```bash
git clone [https://github.com/username-kamu/nama-repo-kamu.git](https://github.com/username-kamu/nama-repo-kamu.git)
cd nama-repo-kamu
```

2. Install Dependensi
Pastikan kamu sudah menginstal Python 3.9+.
```bash
pip install -r requirements.txt
```

3. Konfigurasi Environment
Buat file .env di direktori utama dan isi dengan API Key kamu:

Code snippet
```python
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

4. Jalankan Aplikasi
```bash
streamlit run app.py
```

🌐 Deployment (Streamlit Cloud)
Aplikasi ini dioptimalkan untuk dideploy ke Streamlit Cloud. Pastikan kamu menambahkan GROQ_API_KEY dan HF_TOKEN di bagian Advanced Settings > Secrets pada dashboard Streamlit.

📝 Catatan Penggunaan
Aplikasi membatasi upload file maksimal 10MB per file untuk memastikan kecepatan proses embedding.

Gunakan tombol "Reset Chat" di sidebar untuk membersihkan riwayat percakapan dan memulai topik baru.

Dibuat dengan ❤️ menggunakan LangChain & Streamlit.
