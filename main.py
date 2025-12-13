import os
import re
import subprocess
import tempfile
import threading
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv
import google.generativeai as genai

from faster_whisper import WhisperModel

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

# --------------------------------------------------
# APP
# --------------------------------------------------
app = FastAPI(title="YouTube RAG Chat (Whisper)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# MODELS
# --------------------------------------------------
class FetchRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    question: str

# --------------------------------------------------
# GLOBAL STATE
# --------------------------------------------------
_lock = threading.Lock()
_vector_store = None
_retriever = None
_transcript_text = ""

# --------------------------------------------------
# UTIL
# --------------------------------------------------
def extract_video_id(url: str) -> Optional[str]:
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None

# --------------------------------------------------
# DOWNLOAD AUDIO
# --------------------------------------------------
def download_audio(video_id: str, out_dir: str) -> str:
    output_path = os.path.join(out_dir, "audio.%(ext)s")
    subprocess.run([
        "yt-dlp",
        "-f", "bestaudio",
        "-x", "--audio-format", "mp3",
        "-o", output_path,
        f"https://www.youtube.com/watch?v={video_id}"
    ], check=True)

    return os.path.join(out_dir, "audio.mp3")

# --------------------------------------------------
# WHISPER TRANSCRIBE
# --------------------------------------------------
def whisper_transcribe(audio_path: str) -> str:
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path)
    return " ".join(seg.text for seg in segments)

# --------------------------------------------------
# RAG
# --------------------------------------------------
def build_vector_store(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    store = FAISS.from_documents(docs, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 4})
    return store, retriever

PROMPT = PromptTemplate(
    template="""
Answer ONLY using the transcript context.

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# --------------------------------------------------
# API
# --------------------------------------------------
@app.post("/fetch_transcript")
async def fetch_transcript(req: FetchRequest):
    global _vector_store, _retriever, _transcript_text

    vid = extract_video_id(req.url)
    if not vid:
        return JSONResponse({"success": False, "message": "Invalid YouTube URL"}, 400)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = download_audio(vid, tmp)
            transcript = whisper_transcribe(audio_path)

        with _lock:
            _vector_store, _retriever = build_vector_store(transcript)
            _transcript_text = transcript

        return {
            "success": True,
            "message": "Transcript generated using Whisper",
            "preview": transcript[:800]
        }

    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, 500)

@app.post("/chat")
async def chat(req: ChatRequest):
    if not _retriever:
        return JSONResponse({"success": False, "message": "No transcript loaded"}, 400)

    docs = _retriever.get_relevant_documents(req.question)
    context = "".join(d.page_content for d in docs)

    llm = get_llm()
    result = llm.invoke(PROMPT.invoke({
        "context": context,
        "question": req.question
    }))

    return {"success": True, "answer": result.content}

@app.get("/health")
async def health():
    return {"ok": True}
