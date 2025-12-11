// --------- Configuration ---------
// Set this to your backend address.
const API_BASE = 'https://youtube-backend-chat.onrender.com';
const FETCH_ENDPOINT = API_BASE + '/fetch_transcript';
const CHAT_ENDPOINT  = API_BASE + '/chat';
const ASK_ENDPOINT   = API_BASE + '/ask';
const HEALTH_ENDPOINT = API_BASE + '/health';

// request timeout (ms)
const FETCH_TIMEOUT = 30_000; // 30s

// --------- Helpers ---------
const el = id => document.getElementById(id);
const status = (s)=> {
  const node = el('status');
  if(node) node.textContent = 'Status: '+s;
};
const setLoading = (on, where='top')=>{
  const node = where==='top'? el('topLoader') : el('status');
  if(!node) return;
  node.innerHTML = on ? '<div class="loader"></div>' : '';
}

function extractYouTubeID(url){
  if(!url) return null;
  try{
    const u = new URL(url.trim());
    const host = u.hostname.toLowerCase();
    if(host.includes('youtube.com')){
      const v = new URLSearchParams(u.search).get('v');
      if(v) return v;
    }
    if(host.includes('youtu.be')){
      const p = u.pathname.slice(1);
      if(p) return p;
    }
  }catch(e){
    // fall through to regex
  }
  // Prefer standard 11-char YouTube IDs, fallback to looser match
  const m11 = url.match(/(?:v=|youtu\.be\/|\/v\/|embed\/)([A-Za-z0-9_-]{11})/);
  if(m11) return m11[1];
  const m = url.match(/(?:v=|youtu\.be\/|\/v\/|embed\/)([A-Za-z0-9_-]{6,})/);
  return m? m[1] : null;
}

function addMessage(text, who='ai'){
  const messages = el('messages');
  if(!messages) {
    console.warn('messages element not found');
    return;
  }
  const msg = document.createElement('div');
  msg.className = 'bubble ' + (who==='user' ? 'user' : 'ai');
  msg.textContent = text;
  messages.appendChild(msg);
  messages.scrollTop = messages.scrollHeight;
}

// Track last indexed video id in the frontend so we can decide whether to call /ask
let _indexedVideoId = null;

// small helper to parse JSON body safely
async function parseJsonSafe(resp) {
  const contentType = resp.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    try { return await resp.json(); } catch(e){ return null; }
  }
  return null;
}

// health check utility
async function isServerHealthy() {
  try {
    const ctl = new AbortController();
    const id = setTimeout(()=>ctl.abort(), 5000);
    const r = await fetch(HEALTH_ENDPOINT, {signal: ctl.signal});
    clearTimeout(id);
    if(!r.ok) return false;
    const body = await parseJsonSafe(r);
    return body && body.ok === true;
  } catch(e) {
    return false;
  }
}

// general fetch with timeout
async function fetchWithTimeout(url, opts = {}, timeout = FETCH_TIMEOUT) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  opts.signal = controller.signal;
  try {
    const r = await fetch(url, opts);
    clearTimeout(id);
    return r;
  } catch (err) {
    clearTimeout(id);
    throw err;
  }
}

// Robust fetch for /fetch_transcript
async function robustFetchTranscript(url, lang) {
  setLoading(true, 'top'); status('Fetching transcript...');

  // quick health check
  const healthy = await isServerHealthy();
  if(!healthy){
    setLoading(false,'top');
    status('Error: backend health check failed. Is the server running?');
    throw new Error('Backend health check failed');
  }

  try {
    const resp = await fetchWithTimeout(FETCH_ENDPOINT, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({url, lang})
    });

    const data = await parseJsonSafe(resp);
    if (!resp.ok) {
      // try to extract helpful message
      const msg = data?.message || data?.error || (data ? JSON.stringify(data) : await resp.text().catch(()=>resp.statusText));
      throw new Error(`Server error ${resp.status}: ${msg}`);
    }

    setLoading(false,'top');
    return data;
  } catch (err) {
    setLoading(false,'top');
    const msg = err.name === 'AbortError' ? 'Request timed out' : err.message || String(err);
    status('Error fetching transcript: ' + msg);
    console.error('Fetch transcript error:', err);
    throw err;
  }
}

// --------- UI wiring ---------
if(el('clearBtn')) {
  el('clearBtn').addEventListener('click', ()=>{
    if(el('yturl')) el('yturl').value = '';
    if(el('transcriptPreview')) el('transcriptPreview').textContent = '';
    if(el('videoWrap')) el('videoWrap').style.display = 'none';
    status('Idle');
    _indexedVideoId = null;
  });
}

if(el('fetchBtn')) {
  el('fetchBtn').addEventListener('click', async ()=>{
    const url = (el('yturl') && el('yturl').value) ? el('yturl').value.trim() : '';
    const lang = (el('lang') && el('lang').value) ? el('lang').value : 'en';
    if(!url){ status('Please paste a YouTube URL.'); return; }
    const videoId = extractYouTubeID(url);
    try{
      const data = await robustFetchTranscript(url, lang);
      status(data?.message || 'Transcript fetched.');
      if(el('transcriptPreview')) el('transcriptPreview').textContent = data?.transcript_preview || (data?.transcript || '').slice(0,300) || '—';
      if(el('ingestedIndicator')) el('ingestedIndicator').textContent = '(transcript indexed)';
      if(el('questionInput')) el('questionInput').disabled = false;
      if(el('sendBtn')) el('sendBtn').disabled = false;
      const vid = data?.video_id || videoId;
      if(vid && el('videoPreview')) {
        el('videoPreview').innerHTML = `<iframe src="https://www.youtube.com/embed/${vid}" frameborder="0" allowfullscreen></iframe>`;
        if(el('videoWrap')) el('videoWrap').style.display = 'block';
        _indexedVideoId = vid;
      }
    }catch(err){
      // robustFetchTranscript already handled UI messaging
      console.error('robustFetchTranscript failed:', err);
    }
  });
}

if(el('sendBtn')) el('sendBtn').addEventListener('click', sendQuestion);
if(el('questionInput')) el('questionInput').addEventListener('keydown', (e)=>{ if(e.key==='Enter') sendQuestion(); });
if(el('clearChat')) el('clearChat').addEventListener('click', ()=>{ if(el('messages')) el('messages').innerHTML=''; addMessage('Hello! Ask anything about the video after fetching transcript.','ai'); });

// Updated sendQuestion: will call /ask when a video URL is provided but not indexed yet.
async function sendQuestion(){
  const q = el('questionInput') ? el('questionInput').value.trim() : '';
  if(!q) return;

  // Determine URL & lang from left pane (so we can include them)
  const url = el('yturl') ? el('yturl').value.trim() : '';
  const lang = el('lang') ? el('lang').value : 'en';

  // show user bubble
  addMessage(q,'user');
  if(el('questionInput')) el('questionInput').value='';

  const currentVid = extractYouTubeID(url);
  const needsAsk = !!url && (_indexedVideoId === null || _indexedVideoId !== currentVid);

  setLoading(true,'top'); status('Generating answer...');

  try{
    if(needsAsk){
      // Check health before ask
      const healthy = await isServerHealthy();
      if(!healthy){
        setLoading(false,'top');
        status('Error: backend health check failed before /ask.');
        addMessage('Error: backend unreachable.','ai');
        return;
      }

      // POST to /ask
      const resp = await fetchWithTimeout(ASK_ENDPOINT, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ url, lang, question: q, top_k: 4 })
      }, FETCH_TIMEOUT);

      const data = await parseJsonSafe(resp) || { rawText: await resp.text().catch(()=>null) };
      setLoading(false,'top');

      if(!resp.ok){
        const msg = data?.message || resp.statusText || JSON.stringify(data);
        addMessage('Error: ' + msg,'ai');
        status('Idle');
        return;
      }

      // update UI: set preview, video preview, and indexing marker
      if(el('transcriptPreview')) el('transcriptPreview').textContent = data?.transcript_preview || '';
      if(el('ingestedIndicator')) el('ingestedIndicator').textContent = '(transcript indexed)';
      _indexedVideoId = data?.video_id || currentVid;

      // show video preview if present
      if(_indexedVideoId && el('videoPreview')){
        el('videoPreview').innerHTML = `<iframe src="https://www.youtube.com/embed/${_indexedVideoId}" frameborder="0" allowfullscreen></iframe>`;
        if(el('videoWrap')) el('videoWrap').style.display = 'block';
      }

      // Display answer
      const ans = data?.answer || '(no answer returned)';
      addMessage(ans, 'ai');
      status('Idle');
      return;
    }

    // Otherwise fallback to simple /chat path (faster)
    const resp = await fetchWithTimeout(CHAT_ENDPOINT, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ question: q })
    }, FETCH_TIMEOUT);

    const data = await parseJsonSafe(resp) || { rawText: await resp.text().catch(()=>null) };
    setLoading(false,'top');

    if(!resp.ok){
      const msg = data?.message || resp.statusText || JSON.stringify(data);
      addMessage('Error: ' + msg,'ai');
      status('Idle');
      return;
    }

    const content = data?.answer || data?.response || data?.content || JSON.stringify(data);
    addMessage(content, 'ai');
    status('Idle');

  }catch(err){
    setLoading(false,'top');
    const msg = err.name === 'AbortError' ? 'Request timed out' : (err.message || String(err));
    status('Error: ' + msg);
    addMessage('Error: ' + msg,'ai');
    console.error('sendQuestion error:', err);
  }
}

// initial welcome
addMessage('Hello! Paste a YouTube link on the left and click "Fetch Transcript" to begin.','ai');
