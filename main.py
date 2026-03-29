import os
import requests
import numpy as np
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer

# ----------------------------
# Load ENV
# ----------------------------
load_dotenv(dotenv_path=".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Check your .env file.")


# ----------------------------
# 1. Extract Video ID
# ----------------------------
def extract_video_id(url):
    parsed_url = urlparse(url)

    if "youtube" in url:
        return parse_qs(parsed_url.query)["v"][0]
    elif "youtu.be" in url:
        return parsed_url.path[1:]
    else:
        raise ValueError("Invalid YouTube URL")


# ----------------------------
# 2. Get Transcript
# ----------------------------
def get_transcript(video_id):
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id)
        text = " ".join([t.text for t in transcript_list])
        return text
    except Exception as e:
        print("Transcript Error:", str(e))
        raise Exception("Transcript not available for this video")


# ----------------------------
# 3. Generate Embeddings
# ----------------------------
def get_embeddings(text):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    embeddings = model.encode(chunks)

    return chunks, embeddings, model


# ----------------------------
# 4. Select Best Chunks (basic RAG)
# ----------------------------
def select_best_chunks(chunks, embeddings, model, top_k=5):
    query = "Summarize this video"
    query_embedding = model.encode([query])[0]

    similarities = []

    for emb in embeddings:
        sim = np.dot(query_embedding, emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(emb)
        )
        similarities.append(sim)

    top_indices = np.argsort(similarities)[-top_k:]
    selected_chunks = [chunks[i] for i in top_indices]

    return selected_chunks


# ----------------------------
# 5. Summarize with Groq
# ----------------------------
def summarize_with_groq(chunks):

    combined_text = " ".join(chunks)

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant"
    ]

    for model in MODELS:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Summarize clearly and concisely."},
                {"role": "user", "content": combined_text}
            ],
            "temperature": 0.4
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Model {model} failed:", response.text)

    raise Exception("All Groq models failed")


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    youtube_url = input("Enter YouTube URL: ").strip()

    print("\n📥 Fetching transcript...")
    video_id = extract_video_id(youtube_url)

    transcript = get_transcript(video_id)

    print("🧠 Generating embeddings...")
    chunks, embeddings, model = get_embeddings(transcript)

    print("🎯 Selecting relevant chunks...")
    selected_chunks = select_best_chunks(chunks, embeddings, model)

    print("⚡ Summarizing with Groq...")
    summary = summarize_with_groq(selected_chunks)

    print("\n=== ✅ SUMMARY ===\n")
    print(summary)