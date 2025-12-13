# 🎥 YouTube RAG Based Chatbot

A lightweight YouTube video question-answering system that transcribes a video's audio using OpenAI Whisper, indexes the transcript with a vector store (FAISS), and answers user questions with Google Gemini using Retrieval-Augmented Generation (RAG). This approach avoids YouTube caption restrictions by transcribing audio directly, so it works reliably for any video.

- 🔊 Audio → Text using Faster Whisper  
- 🧠 RAG over video content with Google Gemini  
- 🐳 Dockerized and deployable on any container host  
- ⚡ FastAPI backend with simple REST endpoints  
- 🌐 Optional minimal frontend (index.html + app.js + styles.css)

---

## Table of contents

- Features
- Architecture
- Tech stack
- Project structure
- Environment variables
- Run locally
- Run with Docker
- Simple frontend
- API endpoints
- Limitations
- Why Whisper instead of YouTube captions
- Future improvements
- License
- Acknowledgements

---

## 🚀 Features

- Transcribe video audio with Whisper (works even when captions are blocked)
- Text chunking + embeddings for retrieval
- Vector search using FAISS
- Answer generation with Google Gemini
- Simple FastAPI endpoints for fetching transcripts and chatting with videos
- Minimal web UI (optional) for quick interactions
- Dockerized for easy deployment

---

## 🧩 Architecture

```
YouTube Video
↓
Audio Download (FFmpeg)
↓
Whisper Transcription
↓
Text Chunking + Embeddings
↓
Vector Search (FAISS)
↓
Gemini LLM Answer
```

---

## 🛠️ Tech Stack

- Backend: FastAPI  
- Speech-to-Text: Faster Whisper  
- LLM (Answering): Google Gemini  
- Vector Store: FAISS  
- Embeddings: Sentence Transformers  
- Audio Processing: FFmpeg  
- Deployment: Docker or any container host  
- Frontend (optional): index.html, app.js, styles.css

---

## 📁 Project Structure

```
.
├── main.py              # FastAPI backend
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker build config
├── README.md            # Documentation
├── index.html           # Optional minimal frontend
├── app.js               # Frontend JS to call backend endpoints
├── styles.css           # Frontend styles
```

---

## 🔑 Environment Variables

Create a `.env` file or set the environment variable:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Warning: Do NOT commit API keys to GitHub.

---

## ▶️ Run Locally (Without Docker)

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Install FFmpeg

- Download from: https://ffmpeg.org/download.html  
- Add `ffmpeg/bin` to your system PATH

Verify:

```bash
ffmpeg -version
```

3. Run server

```bash
uvicorn main:app --reload --port 8000
```

Open the API docs:

```
http://127.0.0.1:8000/docs
```

---

## 🐳 Run with Docker

1. Build image

```bash
docker build -t youtube-rag-whisper .
```

2. Run container

```bash
docker run -p 8000:8000 -e GOOGLE_API_KEY=your_gemini_key youtube-rag-whisper
```

The FastAPI backend will be available on port 8000 by default.

---

## 🌐 Simple Frontend (Optional)

A minimal static frontend is included (index.html, app.js, styles.css) to let you interact with the API without building a separate client. It's intentionally simple so you can extend it for your use case.

To serve the frontend locally:

- Option A: Open index.html directly in a browser (CORS may prevent direct calls to localhost backend depending on browser security).
- Option B: Run a simple static server (recommended):

```bash
# from the project root
python -m http.server 8080
# then open http://localhost:8080/index.html
```

Make sure the backend (FastAPI) is running at http://localhost:8000 and that CORS is enabled in main.py if calling from a different origin.

---

## 📌 API Endpoints

- POST /fetch_transcript  
  Request body:
  ```json
  {
    "url": "https://www.youtube.com/watch?v=VIDEO_ID"
  }
  ```

- POST /chat  
  Request body:
  ```json
  {
    "question": "What is the main idea of the video?"
  }
  ```

- POST /ask (one-shot ask)  
  Request body:
  ```json
  {
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "question": "Summarize the video"
  }
  ```

---

## ⚠️ Limitations

- Whisper is CPU-heavy → slower on CPU-only environments. Expect long transcriptions for long videos.
- Very long videos may take a long time to transcribe and index.
- Gemini usage is subject to rate limits and API costs — plan accordingly.
- If you serve the frontend from a different origin than the backend, enable CORS in FastAPI.

---

## ✅ Why Whisper Instead of YouTube Captions?

| Method                 | Reliability         |
| ---------------------- | ------------------- |
| YouTube Transcript API | ❌ Often blocked     |
| yt-dlp captions        | ❌ Frequently breaks |
| Whisper (Audio)        | ✅ Always works      |

Whisper transcribes the audio itself, avoiding issues with missing or blocked captions and improving reliability for any video.

---

## 🧠 Future Improvements

- Timestamped answers (point to transcript positions)
- Multilingual transcription support
- Caching transcripts to avoid reprocessing
- Streaming responses for faster UX
- Split Whisper & RAG into microservices for scalability
- Add a richer frontend UI


---

## 📜 License

MIT License — free to use for learning, demos, and projects. See LICENSE for details.

---

## 🙌 Acknowledgements

- Faster Whisper
- Google Gemini
- LangChain
- FFmpeg

