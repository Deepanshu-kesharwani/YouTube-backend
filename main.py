"""
FastAPI backend for YouTube RAG Chat using yt-dlp for transcripts.
This replaces youtube-transcript-api because YouTube now blocks scraping.
"""

import os
import re
import threading
from typing import Optional
from urllib.parse import urlparse, parse_qs
import traceback as _traceback

import yt_dlp
import requests
import xml.etree.ElementTree as ET

# FastAPI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# LangChain + FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

import google.generativeai as genai
from dotenv import load_dotenv

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY is missing")

genai.configure(api_key=GOOGLE_API_KEY)

# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------
app = FastAPI(title="YouTube RAG Chat API (yt-dlp)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
class FetchRequest(BaseModel):
    url: str
    lang: Optional[str] = "en"

class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4

class AskRequest(BaseModel):
    url: str
    lang: Optional[str] = "en"
    question: str
    top_k: Optional[int] = 4

# ---------------------------------------------------------
# Global State
# ---------------------------------------------------------
_state_lock = threading.Lock()
_vector_store = None
_retriever = None
_transcript_text = ""
_video_id = None

# ---------------------------------------------------------
# Utility: Extract YouTube Video ID
# ---------------------------------------------------------
def extract_youtube_id(url: str):
    """Returns an 11-char YouTube video ID."""
    if not url:
        return None

    s = url.strip()

    # Direct ID
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s

    try:
        parsed = urlparse(s)
        hostname = (parsed.hostname or "").lower()

        # watch?v=
        if "youtube.com" in hostname:
            qs = parse_qs(parsed.query)
            if "v" in qs:
                vid = qs["v"][0]
                if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
                    return vid

        # youtu.be/ID
        if "youtu.be" in hostname:
            vid = parsed.path.lstrip("/")
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
                return vid

    except Exception:
        pass

    m = re.search(r"([A-Za-z0-9_-]{11})", s)
    return m.group(1) if m else None

# ---------------------------------------------------------
# Transcript Fetcher Using yt-dlp
# ---------------------------------------------------------
def fetch_transcript(video_id: str, lang="en"):
    """
    Fetches YouTube transcript using yt-dlp (supports auto captions).
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        subtitles = info.get("subtitles") or info.get("automatic_captions")

        if not subtitles or lang not in subtitles:
            raise RuntimeError(f"No subtitles found for language: {lang}")

        sub_url = subtitles[lang][0]["url"]

        r = requests.get(sub_url)
        r.raise_for_status()
        xml_text = r.text

        root = ET.fromstring(xml_text)

        segments = []
        for child in root.findall(".//text"):
            text = child.text or ""
            start = float(child.attrib.get("start", 0))
            dur = float(child.attrib.get("dur", 0))
            segments.append({"text": text, "start": start, "duration": dur})

        return segments

# ---------------------------------------------------------
# Build RAG Vector Store
# ---------------------------------------------------------
def build_vector_store(transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = splitter.create_documents([transcript])

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(docs, emb)
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    return store, retriever

PROMPT = PromptTemplate(
    template="""
Answer ONLY from the video transcript.

CONTEXT:
{context}

QUESTION:
{question}
""",
    input_variables=["context", "question"]
)

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.post("/fetch_transcript")
async def fetch_transcript_api(req: FetchRequest):
    global _vector_store, _retriever, _transcript_text, _video_id

    vid = extract_youtube_id(req.url)
    if not vid:
        return JSONResponse({"success": False, "message": "Invalid YouTube URL"}, status_code=400)

    try:
        segments = fetch_transcript(vid, lang=req.lang)
    except Exception as e:
        print("[ERROR] Transcript fetch failed:", e)
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

    transcript = " ".join(s["text"] for s in segments)

    with _state_lock:
        _vector_store, _retriever = build_vector_store(transcript)
        _transcript_text = transcript
        _video_id = vid

    return {
        "success": True,
        "message": "Transcript fetched & indexed",
        "video_id": vid,
        "transcript_preview": transcript[:800]
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    if not _retriever:
        return JSONResponse({"success": False, "message": "No transcript loaded"}, status_code=400)

    docs = _retriever.get_relevant_documents(req.question)
    context = "".join(d.page_content for d in docs)

    llm = get_llm()
    result = llm.invoke(PROMPT.invoke({"context": context, "question": req.question}))

    return {"success": True, "answer": str(result)}

@app.post("/ask")
async def ask(req: AskRequest):
    """
    Fetch transcript + answer question in one call.
    """
    global _vector_store, _retriever, _transcript_text, _video_id

    vid = extract_youtube_id(req.url)
    if not vid:
        return JSONResponse({"success": False, "message": "Invalid YouTube URL"}, status_code=400)

    # Re-index if different video
    if _video_id != vid:
        try:
            segments = fetch_transcript(vid, lang=req.lang)
        except Exception as e:
            return JSONResponse({"success": False, "message": str(e)}, status_code=500)

        transcript = " ".join(s["text"] for s in segments)

        with _state_lock:
            _vector_store, _retriever = build_vector_store(transcript)
            _transcript_text = transcript
            _video_id = vid

    docs = _retriever.get_relevant_documents(req.question)
    context = "".join(d.page_content for d in docs)
    llm = get_llm()
    result = llm.invoke(PROMPT.invoke({"context": context, "question": req.question}))

    return {
        "success": True,
        "answer": str(result),
        "video_id": vid,
        "transcript_preview": _transcript_text[:800]
    }

@app.get("/health")
async def health():
    return {"ok": True, "indexed": _retriever is not None}
