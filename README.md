# 🚀 YouTube Summarizer (Groq + RAG)

Summarize any YouTube video using:

* 📥 Transcript extraction
* 🧠 Embeddings (Sentence Transformers)
* 🎯 Retrieval (RAG)
* ⚡ Fast LLM inference via Groq

---

## 🔧 Features

* Extracts YouTube transcripts automatically
* Uses embeddings to select the most relevant content
* Summarizes using Groq (LLaMA models)
* Lightweight and fast

---

## 🛠️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/yt-summarizer.git
cd yt-summarizer
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API key

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Usage

```bash
python main.py
```

Then paste a YouTube link:

```
https://www.youtube.com/watch?v=example
```

---

## 🧠 How it works

```
YouTube → Transcript → Chunking → Embeddings → Similarity Search → Groq → Summary
```

---

## ⚡ Models Used

* Sentence Transformers (`all-MiniLM-L6-v2`)
* Groq LLaMA models (`llama-3.3-70b-versatile`)

---

## 🔐 Security

* API keys are stored in `.env`
* `.env` is excluded via `.gitignore`

---

## 🚀 Future Improvements

* Chrome extension (summarize button on YouTube)
* FastAPI backend
* Batch summarization
* TexSpeech integration

---

## 👤 Author

Built by Maaz
