"""
FastAPI backend for YouTube RAG Chat
Features:
- YouTube Data API captions
- Hindi + English fallback
- Gemini answers
- Timestamped answers
"""

import os
import re
import traceback
from typing import Optional, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai
from googleapiclient.discovery import build

# -------------------- ENV --------------------
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not YOUTUBE_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("Missing API keys")

genai.configure(api_key=GEMINI_API_KEY)
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# -------------------- APP --------------------
app = FastAPI(title="YouTube RAG Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- MODELS --------------------
class FetchRequest(BaseModel):
    url: str
    lang: Optional[str] = None  # auto if None

class ChatRequest(BaseModel):
    question: str

class AskRequest(BaseModel):
    url: str
    question: str
    lang: Optional[str] = None

# -------------------- STATE --------------------
TRANSCRIPT = []   # list of {start, text}
VIDEO_ID = None

# -------------------- HELPERS --------------------
def extract_video_id(url: str) -> Optional[str]:
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def srt_to_segments(srt: str):
    segments = []
    current_time = None

    for line in srt.splitlines():
        if "-->" in line:
            start = line.split("-->")[0].strip()
            h, m, s = start.replace(",", ":").split(":")
            current_time = int(h)*3600 + int(m)*60 + int(float(s))
        elif line.strip() and not line.strip().isdigit():
            segments.append({
                "start": current_time,
                "text": line.strip()
            })

    return segments


def fetch_captions(video_id: str, lang: Optional[str]):
    captions = youtube.captions().list(
        part="snippet",
        videoId=video_id
    ).execute()

    # Priority: Hindi → English → anything
    lang_priority = []
    if lang:
        lang_priority.append(lang)
    else:
        lang_priority.extend(["hi", "en"])

    caption_id = None
    for lp in lang_priority:
        for c in captions.get("items", []):
            if c["snippet"]["language"] == lp:
                caption_id = c["id"]
                break
        if caption_id:
            break

    if not caption_id:
        raise RuntimeError("No captions available")

    srt = youtube.captions().download(
        id=caption_id,
        tfmt="srt"
    ).execute()

    return srt.decode("utf-8")


def ask_gemini(question: str, transcript_text: str):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
Answer ONLY from the transcript.

Transcript:
{transcript_text}

Question:
{question}
"""
    return model.generate_content(prompt).text.strip()


def find_timestamps(answer: str):
    hits = []
    for seg in TRANSCRIPT:
        if seg["text"].lower() in answer.lower():
            hits.append({
                "time": f"{seg['start']//60:02d}:{seg['start']%60:02d}",
                "seconds": seg["start"],
                "text": seg["text"]
            })
    return hits[:5]

# -------------------- ROUTES --------------------
@app.post("/fetch_transcript")
async def fetch_transcript(req: FetchRequest):
    global TRANSCRIPT, VIDEO_ID

    try:
        vid = extract_video_id(req.url)
        if not vid:
            return JSONResponse({"success": False, "message": "Invalid URL"}, 400)

        srt = fetch_captions(vid, req.lang)
        TRANSCRIPT = srt_to_segments(srt)
        VIDEO_ID = vid

        preview = " ".join(s["text"] for s in TRANSCRIPT[:20])

        return {
            "success": True,
            "video_id": vid,
            "language_used": req.lang or "hi → en fallback",
            "transcript_preview": preview
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"success": False, "message": str(e)}, 500)


@app.post("/chat")
async def chat(req: ChatRequest):
    if not TRANSCRIPT:
        return JSONResponse({"success": False, "message": "No transcript"}, 400)

    transcript_text = " ".join(s["text"] for s in TRANSCRIPT)
    answer = ask_gemini(req.question, transcript_text)
    timestamps = find_timestamps(answer)

    return {
        "success": True,
        "answer": answer,
        "timestamps": timestamps,
        "video_id": VIDEO_ID
    }


@app.post("/ask")
async def ask(req: AskRequest):
    global TRANSCRIPT, VIDEO_ID

    vid = extract_video_id(req.url)
    if not vid:
        return JSONResponse({"success": False, "message": "Invalid URL"}, 400)

    if VIDEO_ID != vid or not TRANSCRIPT:
        srt = fetch_captions(vid, req.lang)
        TRANSCRIPT = srt_to_segments(srt)
        VIDEO_ID = vid

    transcript_text = " ".join(s["text"] for s in TRANSCRIPT)
    answer = ask_gemini(req.question, transcript_text)
    timestamps = find_timestamps(answer)

    return {
        "success": True,
        "answer": answer,
        "timestamps": timestamps,
        "video_id": vid
    }


@app.get("/health")
async def health():
    return {"ok": True, "segments": len(TRANSCRIPT)}
