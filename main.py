"""
FastAPI backend for YouTube RAG Chat.
"""
import os
import re
import traceback as _traceback
from urllib.parse import urlparse, parse_qs
import threading
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# LangChain + FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------- ENV / CONFIG --------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY missing")

genai.configure(api_key=GOOGLE_API_KEY)

# -------------------- FASTAPI APP --------------------
app = FastAPI(title="YouTube RAG Chat API")

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
def extract_youtube_id_from_url(url: str) -> Optional[str]:
    """ Robust extraction supporting all formats """
    if not url:
        return None

    s = url.strip()

    # Case: raw 11-char ID
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s

    try:
        parsed = urlparse(s)
        hostname = (parsed.hostname or "").lower()

        # watch?v=VIDEO_ID
        if "youtube.com" in hostname:
            qs = parse_qs(parsed.query)
            if "v" in qs:
                vid = qs["v"][0]
                if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
                    return vid

        # youtu.be/VIDEO_ID
        if "youtu.be" in hostname:
            vid = parsed.path.lstrip("/")
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
                return vid

        # shorts / embed / v
        m = re.search(r"/(?:shorts|embed|v)/([A-Za-z0-9_-]{11})", parsed.path or "")
        if m:
            return m.group(1)

    except Exception:
        pass

    # fallback: search anywhere
    m = re.search(r"([A-Za-z0-9_-]{11})", s)
    return m.group(1) if m else None

# -------------------- TRANSCRIPT NORMALIZER --------------------
def _to_seconds(val):
    try:
        if val is None: return None
        v = float(val)
        if v > 10000: return v/1000.0
        if v > 1000 and v % 1000 == 0: return v/1000.0
        return v
    except:
        return None

def _extract_time_field(item, keys):
    for k in keys:
        if isinstance(item, dict) and k in item:
            return _to_seconds(item[k])
        if hasattr(item, k):
            return _to_seconds(getattr(item, k))
    return None

def _normalize_to_text_with_timestamps(obj):
    """ Convert transcript segments to uniform dict format """
    def make(text, start=None, dur=None):
        return {"text": text or "", "start": start, "duration": dur}

    start_keys = ("start", "start_ms", "timestamp", "time", "offset")
    dur_keys   = ("duration", "duration_ms", "length", "end_time")

    if isinstance(obj, list):
        out = []
        for item in obj:
            if isinstance(item, dict):
                text = item.get("text") or ""
                start = _extract_time_field(item, start_keys)
                dur = _extract_time_field(item, dur_keys)
                out.append(make(text, start, dur))
            else:
                text = getattr(item, "text", None) or str(item)
                start = _extract_time_field(item, start_keys)
                dur = _extract_time_field(item, dur_keys)
                out.append(make(text, start, dur))
        return out

    if hasattr(obj, "lines"):
        out = []
        for seg in obj.lines:
            text = getattr(seg, "text", None) or ""
            start = _extract_time_field(seg, start_keys)
            dur = _extract_time_field(seg, dur_keys)
            out.append(make(text, start, dur))
        return out

    return [make(str(obj))]

# -------------------- DEBUG-ENABLED TRANSCRIPT FETCHER --------------------
def get_transcript_list_with_timestamps(video_id, languages=None, debug=True):
    api = YouTubeTranscriptApi

    # 1) instance.fetch()
    try:
        if debug: print("[DEBUG] Trying instance.fetch()")
        inst = api()
        fetched = inst.fetch(video_id, languages=languages)
        raw = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else fetched
        return _normalize_to_text_with_timestamps(raw)
    except Exception as e:
        print("[DEBUG] instance.fetch FAILED:", type(e).__name__, e)
        print(_traceback.format_exc())

    # 2) api.list(video)
    try:
        if debug: print("[DEBUG] Trying api.list(video).fetch()")
        lst = api.list(video_id)
        if hasattr(lst, "fetch"):
            fetched = lst.fetch(languages=languages)
            raw = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else fetched
            return _normalize_to_text_with_timestamps(raw)
        return _normalize_to_text_with_timestamps(lst)
    except Exception as e:
        print("[DEBUG] api.list FAILED:", type(e).__name__, e)
        print(_traceback.format_exc())

    # 3) get_transcript fallback (older versions)
    try:
        method = getattr(api, "get_transcript", None)
        if callable(method):
            if debug: print("[DEBUG] Trying get_transcript()")
            raw = method(video_id, languages=languages)
            return _normalize_to_text_with_timestamps(raw)
    except Exception as e:
        print("[DEBUG] get_transcript FAILED:", type(e).__name__, e)
        print(_traceback.format_exc())

    raise RuntimeError(
        "Failed to retrieve transcript: no usable interface worked. "
        "Check Render logs for debug tracebacks."
    )

# -------------------- RAG SETUP --------------------
def build_vector_store(transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([transcript])

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(docs, emb)
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return store, retriever

PROMPT = PromptTemplate(
    template="""
Answer only from the transcript.

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# -------------------- ROUTES --------------------
@app.post("/fetch_transcript")
async def fetch_transcript(req: FetchRequest):
    global _vector_store, _retriever, _transcript_text, _video_id

    url = req.url.strip()
    vid = extract_youtube_id_from_url(url)

    if not vid:
        return JSONResponse({"success": False, "message": "Could not extract video id"}, status_code=400)

    try:
        segments = get_transcript_list_with_timestamps(vid, languages=[req.lang])
    except TranscriptsDisabled:
        return JSONResponse({"success": False, "message": "Transcript disabled"}, status_code=404)
    except NoTranscriptFound:
        return JSONResponse({"success": False, "message": "No transcript found"}, status_code=404)
    except Exception as e:
        print("[DEBUG] FINAL TRANSCRIPT ERROR:", e)
        print(_traceback.format_exc())
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

    transcript = " ".join(s["text"] for s in segments)

    with _state_lock:
        _vector_store, _retriever = build_vector_store(transcript)
        _transcript_text = transcript
        _video_id = vid

    return {
        "success": True,
        "message": "Transcript indexed",
        "transcript_preview": transcript[:800],
        "video_id": vid
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    if not _retriever:
        return JSONResponse({"success": False, "message": "No transcript indexed"}, status_code=400)

    question = req.question.strip()
    docs = _retriever.get_relevant_documents(question)
    context = "".join(d.page_content for d in docs)

    llm = get_llm()
    final_prompt = PROMPT.invoke({"context": context, "question": question})
    result = llm.invoke(final_prompt)

    return {"success": True, "answer": str(result)}

@app.post("/ask")
async def ask(req: AskRequest):
    """Fetch + Index + Answer"""
    url = req.url.strip()
    vid = extract_youtube_id_from_url(url)

    if not vid:
        return JSONResponse({"success": False, "message": "Could not extract ID"}, status_code=400)

    reindex = False
    with _state_lock:
        if _video_id != vid or _retriever is None:
            reindex = True

    if reindex:
        try:
            segments = get_transcript_list_with_timestamps(vid, languages=[req.lang])
        except Exception as e:
            print("[DEBUG] ERROR DURING /ask TRANSCRIPT:", e)
            print(_traceback.format_exc())
            return JSONResponse({"success": False, "message": str(e)}, status_code=500)

        transcript = " ".join(s["text"] for s in segments)

        with _state_lock:
            _vector_store, _retriever = build_vector_store(transcript)
            _transcript_text = transcript
            _video_id = vid

    docs = _retriever.get_relevant_documents(req.question)
    context = "".join(d.page_content for d in docs)
    llm = get_llm()

    final_prompt = PROMPT.invoke({"context": context, "question": req.question})
    result = llm.invoke(final_prompt)

    return {
        "success": True,
        "answer": str(result),
        "video_id": vid,
        "transcript_preview": _transcript_text[:800]
    }

@app.get("/health")
async def health():
    return {"ok": True, "indexed": _retriever is not None}
