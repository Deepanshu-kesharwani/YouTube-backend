"""
FastAPI backend for YouTube RAG Chat — yt-dlp + cookies support
"""

import os
import re
import traceback
from urllib.parse import urlparse, parse_qs
import threading
from typing import Optional
import yt_dlp
import requests
from xml.etree import ElementTree as ET

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 🔥 LangChain & FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# -------------------- ENV / CONFIG --------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("⚠️ WARNING: GOOGLE_API_KEY missing!")

genai.configure(api_key=GOOGLE_API_KEY)

COOKIE_FILE = "./cookies.txt"
if not os.path.exists(COOKIE_FILE):
    print("⚠️ WARNING: cookies.txt NOT FOUND — Some YouTube videos may fail!")


# -------------------- FASTAPI APP --------------------
app = FastAPI(title="YouTube RAG Chat API (yt-dlp + cookies)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- REQUEST MODELS --------------------
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


# -------------------- GLOBAL STATE --------------------
_state_lock = threading.Lock()
_vector_store = None
_retriever = None
_transcript_text = ""
_video_id = None


# -------------------- YOUTUBE ID EXTRACTION --------------------
def extract_youtube_id(url: str) -> Optional[str]:
    """Robust extraction of 11-character YouTube video ID"""

    if not url:
        return None

    s = url.strip()

    # Raw ID?
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s

    parsed = urlparse(s)
    qs = parse_qs(parsed.query)

    if "v" in qs:
        vid = qs["v"][0]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
            return vid

    # youtu.be/
    if "youtu.be" in (parsed.hostname or ""):
        vid = parsed.path.lstrip("/")
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
            return vid

    # /embed/VIDEO_ID
    m = re.search(r"(?:embed|shorts|v)/([A-Za-z0-9_-]{11})", s)
    if m:
        return m.group(1)

    return None


# -------------------- yt-dlp TRANSCRIPT FETCHER --------------------
def fetch_transcript_ytdlp(video_id: str, lang="en") -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"

    options = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "quiet": True,
        "cookiefile": COOKIE_FILE if os.path.exists(COOKIE_FILE) else None,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)

        subs = info.get("subtitles") or info.get("automatic_captions")
        if not subs:
            raise RuntimeError("No transcript/subtitle available.")

        tracks = subs.get(lang)
        if not tracks:
            raise RuntimeError(f"No transcript found for language={lang}")

        subtitle_url = tracks[0]["url"]
        res = requests.get(subtitle_url)
        if res.status_code != 200:
            raise RuntimeError(f"Could not download subtitles")

        return res.text


def xml_to_text(xml_data: str) -> str:
    root = ET.fromstring(xml_data)
    lines = []
    for child in root.findall(".//text"):
        txt = (child.text or "").replace("\n", " ").strip()
        lines.append(txt)
    return " ".join(lines)


# -------------------- RAG SETUP --------------------
def build_vector_store(transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(docs, embeddings)
    retriever = store.as_retriever()

    return store, retriever


PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer using ONLY the following transcript context:

{context}

Question:
{question}
""",
)


def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)


# -------------------- ROUTES --------------------
@app.post("/fetch_transcript")
async def fetch_transcript(req: FetchRequest):
    global _vector_store, _retriever, _transcript_text, _video_id

    vid = extract_youtube_id(req.url)
    if not vid:
        return JSONResponse({"success": False, "message": "Invalid YouTube URL"}, status_code=400)

    try:
        xml = fetch_transcript_ytdlp(vid, req.lang)
        transcript = xml_to_text(xml)
    except Exception as e:
        print("[TRANSCRIPT ERROR]", e)
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

    with _state_lock:
        _vector_store, _retriever = build_vector_store(transcript)
        _transcript_text = transcript
        _video_id = vid

    return {
        "success": True,
        "message": "Transcript successfully indexed",
        "video_id": vid,
        "transcript_preview": transcript[:800],
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    global _retriever

    if not _retriever:
        return JSONResponse({"success": False, "message": "No transcript indexed"}, status_code=400)

    docs = _retriever.get_relevant_documents(req.question)
    context = "".join(d.page_content for d in docs)

    llm = get_llm()
    final_prompt = PROMPT.invoke({"context": context, "question": req.question})

    result = llm.invoke(final_prompt)

    return {"success": True, "answer": str(result)}


@app.post("/ask")
async def ask(req: AskRequest):
    url = req.url
    lang = req.lang
    question = req.question

    vid = extract_youtube_id(url)
    if not vid:
        return JSONResponse({"success": False, "message": "Invalid YouTube URL"}, status_code=400)

    global _vector_store, _retriever, _transcript_text, _video_id

    # Re-index if switching videos
    if _video_id != vid or _retriever is None:
        try:
            xml = fetch_transcript_ytdlp(vid, lang)
            transcript = xml_to_text(xml)
        except Exception as e:
            return JSONResponse({"success": False, "message": str(e)}, status_code=500)

        with _state_lock:
            _vector_store, _retriever = build_vector_store(transcript)
            _transcript_text = transcript
            _video_id = vid

    docs = _retriever.get_relevant_documents(question)
    context = "".join(d.page_content for d in docs)

    llm = get_llm()
    final_prompt = PROMPT.invoke({"context": context, "question": question})
    result = llm.invoke(final_prompt)

    return {
        "success": True,
        "answer": str(result),
        "video_id": vid,
        "transcript_preview": _transcript_text[:800],
    }


@app.get("/health")
async def health():
    return {"ok": True, "indexed": _retriever is not None}
