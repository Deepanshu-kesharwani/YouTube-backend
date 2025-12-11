"""
FastAPI backend for YouTube RAG Chat.
Provides /fetch_transcript, /chat and /ask endpoints.

Run locally:
  export GOOGLE_API_KEY="YOUR_KEY"
  uvicorn main:app --reload --port 8000
"""
import re
import traceback
import threading
from typing import Optional
import os
from urllib.parse import urlparse, parse_qs

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Transcript helper imports (youtube-transcript-api)
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# LangChain / embeddings / FAISS / Gemini
# NOTE: pin your langchain / related packages in requirements if you hit import errors.
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------- Config ----------
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not set in environment. Set it before production run.")

genai.configure(api_key=GOOGLE_API_KEY)

# ---------- App ----------
app = FastAPI(title="YouTube RAG Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic models ----------
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

# ---------- Global state (simple) ----------
_state_lock = threading.Lock()
_vector_store = None
_retriever = None
_transcript_text = ""   # keep last transcript (preview)
_video_id = None

# ---------- Utilities ----------
def extract_youtube_id_from_url(url: str) -> Optional[str]:
    """
    Extract a YouTube video id from:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - /embed/VIDEO_ID
      - /shorts/VIDEO_ID
      - plain 11-char VIDEO_ID
    Returns None if not found.
    """
    if not url:
        return None
    s = url.strip()

    # Accept raw 11-char IDs directly
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s

    try:
        parsed = urlparse(s)
        hostname = (parsed.hostname or "").lower()

        # youtube.com/watch?v=...
        if "youtube.com" in hostname:
            qs = parse_qs(parsed.query)
            v = qs.get("v")
            if v and len(v) > 0:
                candidate = v[0]
                if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                    return candidate

        # youtu.be/VIDEO_ID
        if "youtu.be" in hostname:
            path = (parsed.path or "").lstrip("/")
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", path):
                return path

        # embed/ or v/ or shorts/
        m = re.search(r"/(?:embed|v|shorts)/([A-Za-z0-9_-]{11})", parsed.path or "")
        if m:
            return m.group(1)
    except Exception:
        pass

    # fallback: search anywhere in the string for an 11-char candidate
    m = re.search(r"([A-Za-z0-9_-]{11})", s)
    if m:
        return m.group(1)

    return None

# ---------- Transcript normalization helper ----------
def _to_seconds(val):
    try:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            v = float(val)
        else:
            v = float(str(val).strip())
        if v > 10000:
            return v / 1000.0
        if v > 1000 and v % 1000 == 0:
            return v / 1000.0
        return v
    except Exception:
        return None

def _extract_time_field(item, keys):
    for k in keys:
        if isinstance(item, dict) and k in item:
            return _to_seconds(item[k])
        if hasattr(item, k):
            return _to_seconds(getattr(item, k))
    return None

def _normalize_to_text_with_timestamps(obj):
    """
    Normalize different transcript structures into list of:
      {"text": str, "start": float|None, "duration": float|None}
    """
    def make_entry(text, start=None, duration=None):
        return {"text": text if text is not None else "", "start": start, "duration": duration}

    start_keys = ("start", "time", "timestamp", "offset", "start_ms", "start_time")
    dur_keys   = ("duration", "dur", "length", "duration_ms", "end_time")

    if isinstance(obj, list):
        out = []
        for item in obj:
            if isinstance(item, dict):
                text = item.get("text") or item.get("line") or item.get("snippet") or ""
                start = _extract_time_field(item, start_keys)
                duration = _extract_time_field(item, dur_keys)
                out.append(make_entry(text, start, duration))
            elif hasattr(item, "text") or hasattr(item, "start") or hasattr(item, "duration"):
                text = getattr(item, "text", None) or getattr(item, "line", None) or str(item)
                start = _extract_time_field(item, start_keys)
                duration = _extract_time_field(item, dur_keys)
                out.append(make_entry(text, start, duration))
            else:
                out.append(make_entry(str(item), None, None))
        return out

    if hasattr(obj, "lines"):
        out = []
        for seg in obj.lines:
            text = getattr(seg, "text", None) or (seg.get("text") if isinstance(seg, dict) else None) or str(seg)
            start = _extract_time_field(seg, start_keys)
            duration = _extract_time_field(seg, dur_keys)
            out.append(make_entry(text, start, duration))
        return out

    try:
        as_list = list(obj)
        if as_list:
            return _normalize_to_text_with_timestamps(as_list)
    except Exception:
        pass

    return [make_entry(str(obj), None, None)]

# ---------- Robust transcript fetcher (supports library version differences) ----------
def get_transcript_list_with_timestamps(video_id, languages=None, debug=False):
    """
    Attempts multiple ways to fetch a transcript to support multiple versions
    of youtube-transcript-api:
      - YouTubeTranscriptApi.get_transcript(...)  (older)
      - YouTubeTranscriptApi().fetch(...)          (newer)
      - YouTubeTranscriptApi.list(...).fetch(...)  (alternate)
    Raises TranscriptsDisabled / NoTranscriptFound per the library when appropriate.
    """
    api = YouTubeTranscriptApi

    # 1) Try old classmethod first (if present)
    try:
        if hasattr(api, "get_transcript"):
            if debug: print("Using YouTubeTranscriptApi.get_transcript(...)")
            raw = api.get_transcript(video_id, languages=languages)
            return _normalize_to_text_with_timestamps(raw)
    except TranscriptsDisabled:
        raise
    except NoTranscriptFound:
        raise
    except Exception as e:
        if debug: print("get_transcript failed:", type(e).__name__, e)

    # 2) Try instance-based fetch()
    try:
        if debug: print("Trying instance.fetch()")
        instance = api()
        fetched = instance.fetch(video_id, languages=languages)
        raw = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else fetched
        return _normalize_to_text_with_timestamps(raw)
    except TranscriptsDisabled:
        raise
    except NoTranscriptFound:
        raise
    except Exception as e:
        if debug: print("instance.fetch failed:", type(e).__name__, e)

    # 3) Try api.list(video_id) path
    try:
        if hasattr(api, "list"):
            if debug: print("Trying YouTubeTranscriptApi.list(...)")
            lst = api.list(video_id)
            if hasattr(lst, "fetch"):
                fetched = lst.fetch(languages=languages)
                raw = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else fetched
                return _normalize_to_text_with_timestamps(raw)
            else:
                return _normalize_to_text_with_timestamps(lst)
    except TranscriptsDisabled:
        raise
    except NoTranscriptFound:
        raise
    except Exception as e:
        if debug: print("api.list path failed:", type(e).__name__, e)

    # If nothing worked, raise a clear error (endpoints catch this and return 500)
    raise RuntimeError(
        "Failed to retrieve transcript: youtube-transcript-api did not expose any known interface "
        "(checked get_transcript, instance.fetch, api.list)."
    )

# ---------- RAG helpers ----------
def build_vector_store_from_transcript(transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([transcript])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return vector_store, retriever

PROMPT = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say you don't know.

{context}

Question: {question}
""",
    input_variables=['context', 'question']
)

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# ---------- Endpoints ----------
@app.post("/fetch_transcript")
async def fetch_transcript(req: FetchRequest):
    global _vector_store, _retriever, _transcript_text, _video_id
    try:
        url = req.url.strip()
        lang = req.lang or "en"
        if not url:
            return JSONResponse({"success": False, "message": "Missing url"}, status_code=400)

        video_id = extract_youtube_id_from_url = None
        try:
            video_id = extract_youtube_id_from_url(url)
        except Exception:
            video_id = None

        if not video_id:
            return JSONResponse({"success": False, "message": "Could not extract video id from URL"}, status_code=400)

        try:
            transcript_list = get_transcript_list_with_timestamps(video_id, languages=[lang], debug=False)
        except TranscriptsDisabled:
            return JSONResponse({"success": False, "message": "Transcripts are disabled for this video."}, status_code=404)
        except NoTranscriptFound:
            return JSONResponse({"success": False, "message": "No transcript found for this video."}, status_code=404)

        transcript = " ".join(chunk["text"] for chunk in transcript_list if chunk.get("text"))
        if not transcript.strip():
            return JSONResponse({"success": False, "message": "Transcript is empty after parsing."}, status_code=500)

        with _state_lock:
            _vector_store, _retriever = build_vector_store_from_transcript(transcript)
            _transcript_text = transcript
            _video_id = video_id

        return JSONResponse({
            "success": True,
            "message": "Transcript fetched and indexed",
            "transcript_preview": transcript[:1000],
            "video_id": video_id,
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@app.post("/chat")
async def chat(req: ChatRequest):
    global _retriever, _transcript_text, _vector_store
    try:
        question = req.question.strip()
        top_k = req.top_k or 4
        if not question:
            return JSONResponse({"success": False, "message": "Question is empty"}, status_code=400)

        if _retriever is None:
            return JSONResponse({"success": False, "message": "No transcript indexed. Call /fetch_transcript first."}, status_code=400)

        retrieved_docs = None
        try:
            if hasattr(_retriever, "get_relevant_documents"):
                retrieved_docs = _retriever.get_relevant_documents(question)
            elif hasattr(_retriever, "get_relevant_texts"):
                texts = _retriever.get_relevant_texts(question)
                retrieved_docs = [type("D", (), {"page_content": t}) for t in texts]
            else:
                retrieved_docs = _retriever.invoke(question)
        except Exception:
            try:
                if _vector_store is not None and hasattr(_vector_store, "similarity_search"):
                    retrieved_docs = _vector_store.similarity_search(question, k=top_k)
            except Exception:
                pass

        if not retrieved_docs:
            context_text = (_transcript_text or "")[:3000]
        else:
            context_text = "".join(getattr(d, "page_content", str(d)) for d in retrieved_docs)

        llm = get_llm()
        final_prompt = PROMPT.invoke({"context": context_text, "question": question})
        answer_obj = llm.invoke(final_prompt)

        answer_text = None
        if isinstance(answer_obj, dict):
            answer_text = answer_obj.get("content") or answer_obj.get("answer") or str(answer_obj)
        else:
            answer_text = getattr(answer_obj, "content", None) or getattr(answer_obj, "text", None) or str(answer_obj)

        return JSONResponse({"success": True, "answer": answer_text, "source_snippet": context_text[:1000]})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@app.post("/ask")
async def ask(req: AskRequest):
    """
    Combined flow: fetch+index (if needed) then retrieve+answer in one call.
    """
    global _vector_store, _retriever, _transcript_text, _video_id
    try:
        url = req.url.strip()
        lang = req.lang or "en"
        question = req.question.strip()
        top_k = req.top_k or 4

        if not url:
            return JSONResponse({"success": False, "message": "Missing url"}, status_code=400)
        if not question:
            return JSONResponse({"success": False, "message": "Missing question"}, status_code=400)

        video_id = extract_youtube_id_from_url = None
        try:
            video_id = extract_youtube_id_from_url(url)
        except Exception:
            video_id = None

        if not video_id:
            return JSONResponse({"success": False, "message": "Could not extract video id from URL"}, status_code=400)

        need_index = False
        with _state_lock:
            if _retriever is None or _video_id != video_id:
                need_index = True

        if need_index:
            try:
                transcript_list = get_transcript_list_with_timestamps(video_id, languages=[lang], debug=False)
            except TranscriptsDisabled:
                return JSONResponse({"success": False, "message": "Transcripts disabled for this video."}, status_code=404)
            except NoTranscriptFound:
                return JSONResponse({"success": False, "message": "No transcript found for this video."}, status_code=404)

            transcript = " ".join(chunk["text"] for chunk in transcript_list if chunk.get("text"))
            if not transcript.strip():
                return JSONResponse({"success": False, "message": "Transcript is empty."}, status_code=500)

            with _state_lock:
                _vector_store, _retriever = build_vector_store_from_transcript(transcript)
                _transcript_text = transcript
                _video_id = video_id

        retrieved_docs = None
        try:
            if hasattr(_retriever, "get_relevant_documents"):
                retrieved_docs = _retriever.get_relevant_documents(question)
            elif hasattr(_retriever, "get_relevant_texts"):
                texts = _retriever.get_relevant_texts(question)
                retrieved_docs = [type("D", (), {"page_content": t}) for t in texts]
            elif hasattr(_retriever, "invoke"):
                retrieved_docs = _retriever.invoke(question)
        except Exception:
            pass

        if not retrieved_docs:
            context_text = (_transcript_text or "")[:3000]
        else:
            context_text = "".join(getattr(d, "page_content", str(d)) for d in retrieved_docs)

        llm = get_llm()
        final_prompt = PROMPT.invoke({"context": context_text, "question": question})
        answer_obj = llm.invoke(final_prompt)

        answer_text = None
        if isinstance(answer_obj, dict):
            answer_text = answer_obj.get("content") or answer_obj.get("answer") or str(answer_obj)
        else:
            answer_text = getattr(answer_obj, "content", None) or getattr(answer_obj, "text", None) or str(answer_obj)

        return JSONResponse({"success": True, "answer": answer_text, "transcript_preview": (_transcript_text or "")[:1000], "video_id": video_id})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@app.get("/health")
async def health():
    return {"ok": True, "indexed": _retriever is not None}
