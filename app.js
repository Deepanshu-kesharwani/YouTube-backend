// --------- Configuration ---------
const API_BASE = 'http://localhost:8000';
const FETCH_ENDPOINT = API_BASE + '/fetch_transcript';
const CHAT_ENDPOINT  = API_BASE + '/chat';

// --------- Helpers ---------
const el = id => document.getElementById(id);
const status = s => el('status').textContent = 'Status: ' + s;

const setLoading = on => {
  el('topLoader').innerHTML = on ? '<div class="loader"></div>' : '';
};

function extractYouTubeID(url){
  if(!url) return null;
  try{
    const u = new URL(url.trim());
    if(u.hostname.includes('youtube.com')) return new URLSearchParams(u.search).get('v');
    if(u.hostname.includes('youtu.be')) return u.pathname.slice(1);
  }catch{
    const m = url.match(/(?:v=|youtu\.be\/)([A-Za-z0-9_-]{6,})/);
    return m ? m[1] : null;
  }
  return null;
}

function addMessage(text, who='ai'){
  const msg = document.createElement('div');
  msg.className = 'bubble ' + (who === 'user' ? 'user' : 'ai');
  msg.textContent = text;
  el('messages').appendChild(msg);
  el('messages').scrollTop = el('messages').scrollHeight;
}

let _indexedVideoId = null;

// --------- Fetch Transcript ---------
async function fetchTranscript(url){
  setLoading(true);
  status('Fetching transcript...');

  const resp = await fetch(FETCH_ENDPOINT, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ url })
  });

  const data = await resp.json();
  setLoading(false);

  if(!resp.ok) throw new Error(data?.message || 'Transcript error');
  return data;
}

// --------- UI wiring ---------
el('fetchBtn').addEventListener('click', async ()=>{
  const url = el('yturl').value.trim();
  if(!url) return status('Please paste a YouTube URL.');

  try{
    const data = await fetchTranscript(url);
    el('ingestedIndicator').textContent = '(transcript indexed)';
    el('questionInput').disabled = false;
    el('sendBtn').disabled = false;
    _indexedVideoId = data.video_id;
    status('Ready');
  }catch(err){
    status(err.message);
  }
});

el('sendBtn').addEventListener('click', sendQuestion);
el('questionInput').addEventListener('keydown', e=>{
  if(e.key === 'Enter') sendQuestion();
});

el('clearChat').addEventListener('click', ()=>{
  el('messages').innerHTML = '';
  addMessage('Chat cleared. Ask another question!', 'ai');
});

// --------- Chat ---------
async function sendQuestion(){
  const q = el('questionInput').value.trim();
  if(!q) return;

  addMessage(q,'user');
  el('questionInput').value = '';
  setLoading(true);
  status('Thinking...');

  const resp = await fetch(CHAT_ENDPOINT, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ question: q })
  });

  const data = await resp.json();
  setLoading(false);

  addMessage(data.answer || '(no answer)');
  status('Idle');
}

// Initial message
addMessage('Hello! Paste a YouTube link and fetch transcript to begin.', 'ai');
