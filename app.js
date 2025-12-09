// --------- Configuration ---------
// Set this to your backend address. 
const API_BASE = 'https://youtube-backend-chat.onrender.com';
const FETCH_ENDPOINT = API_BASE + '/fetch_transcript';
const CHAT_ENDPOINT  = API_BASE + '/chat';
const ASK_ENDPOINT   = API_BASE + '/ask';

// --------- Helpers ---------
const el = id => document.getElementById(id);
const status = (s)=> el('status').textContent = 'Status: '+s;
const setLoading = (on, where='top')=>{
  const node = where==='top'? el('topLoader') : el('status');
  node.innerHTML = on ? '<div class="loader"></div>' : '';
}

function extractYouTubeID(url){
  if(!url) return null;
  try{
    const u = new URL(url.trim());
    if(u.hostname.includes('youtube.com')) return new URLSearchParams(u.search).get('v');
    if(u.hostname.includes('youtu.be')) return u.pathname.slice(1);
  }catch(e){
    const m = url.match(/(?:v=|youtu\.be\/|\/v\/|embed\/)([A-Za-z0-9_-]{6,})/);
    return m? m[1] : null;
  }
  return null;
}

function addMessage(text, who='ai'){
  const msg = document.createElement('div');
  msg.className = 'bubble ' + (who==='user' ? 'user' : 'ai');
  msg.textContent = text;
  el('messages').appendChild(msg);
  el('messages').scrollTop = el('messages').scrollHeight;
}

// Track last indexed video id in the frontend so we can decide whether to call /ask
let _indexedVideoId = null;

// Robust fetch for /fetch_transcript
async function robustFetchTranscript(url, lang) {
  setLoading(true, 'top'); status('Fetching transcript...');
  try {
    const resp = await fetch(FETCH_ENDPOINT, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({url, lang})
    });

    const contentType = resp.headers.get('content-type') || '';
    let data;
    if (contentType.includes('application/json')) {
      data = await resp.json();
    } else {
      const text = await resp.text();
      throw new Error(`Server returned non-JSON response (content-type: ${contentType}). Response preview:

${text.slice(0,1000)}`);
    }

    if (!resp.ok) {
      const msg = data?.message || data?.error || JSON.stringify(data);
      throw new Error(`Server error ${resp.status}: ${msg}`);
    }

    setLoading(false,'top');
    return data;
  } catch (err) {
    setLoading(false,'top');
    status('Error fetching transcript: ' + err.message);
    console.error('Fetch transcript error:', err);
    throw err;
  }
}

// --------- UI wiring ---------
el('clearBtn').addEventListener('click', ()=>{
  el('yturl').value = '';
  el('transcriptPreview').textContent = '';
  el('videoWrap').style.display = 'none';
  status('Idle');
});

el('fetchBtn').addEventListener('click', async ()=>{
  const url = el('yturl').value.trim();
  const lang = el('lang').value;
  if(!url){ status('Please paste a YouTube URL.'); return; }
  const videoId = extractYouTubeID(url);
  try{
    const data = await robustFetchTranscript(url, lang);
    status(data?.message || 'Transcript fetched.');
    el('transcriptPreview').textContent = data?.transcript_preview || (data?.transcript || '').slice(0,300) || '—';
    el('ingestedIndicator').textContent = '(transcript indexed)';
    el('questionInput').disabled = false; el('sendBtn').disabled = false;
    const vid = data?.video_id || videoId;
    if(vid){
      el('videoPreview').innerHTML = `<iframe src="https://www.youtube.com/embed/${vid}" frameborder="0" allowfullscreen></iframe>`;
      el('videoWrap').style.display = 'block';
      _indexedVideoId = vid;
    }
  }catch(err){
    // robustFetchTranscript already handled UI messaging
  }
});

el('sendBtn').addEventListener('click', sendQuestion);
el('questionInput').addEventListener('keydown', (e)=>{ if(e.key==='Enter') sendQuestion(); });
el('clearChat').addEventListener('click', ()=>{ el('messages').innerHTML=''; addMessage('Hello! Ask anything about the video after fetching transcript.','ai'); });

// Updated sendQuestion: will call /ask when a video URL is provided but not indexed yet.
async function sendQuestion(){
  const q = el('questionInput').value.trim();
  if(!q) return;

  // Determine URL & lang from left pane (so we can include them)
  const url = el('yturl').value.trim();
  const lang = el('lang').value;

  // show user bubble
  addMessage(q,'user');
  el('questionInput').value='';

  const currentVid = extractYouTubeID(url);
  const needsAsk = !!url && (_indexedVideoId === null || _indexedVideoId !== currentVid);

  setLoading(true,'top'); status('Generating answer...');

  try{
    if(needsAsk){
      // POST to /ask
      const resp = await fetch(ASK_ENDPOINT, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ url, lang, question: q, top_k: 4 })
      });

      const contentType = resp.headers.get('content-type') || '';
      let data;
      if(contentType.includes('application/json')) data = await resp.json();
      else {
        const text = await resp.text();
        throw new Error('Server returned non-JSON: ' + text.slice(0,1000));
      }

      setLoading(false,'top');

      if(!resp.ok){
        const msg = data?.message || resp.statusText;
        addMessage('Error: ' + msg,'ai');
        status('Idle');
        return;
      }

      // update UI: set preview, video preview, and indexing marker
      el('transcriptPreview').textContent = data?.transcript_preview || '';
      el('ingestedIndicator').textContent = '(transcript indexed)';
      _indexedVideoId = data?.video_id || currentVid;

      // show video preview if present
      if(_indexedVideoId){
        el('videoPreview').innerHTML = `<iframe src="https://www.youtube.com/embed/${_indexedVideoId}" frameborder="0" allowfullscreen></iframe>`;
        el('videoWrap').style.display = 'block';
      }

      // Display answer
      const ans = data?.answer || '(no answer returned)';
      addMessage(ans, 'ai');
      status('Idle');
      return;
    }

    // Otherwise fallback to simple /chat path (faster)
    const resp = await fetch(CHAT_ENDPOINT, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ question: q })
    });
    const contentType = resp.headers.get('content-type') || '';
    let data;
    if(contentType.includes('application/json')) data = await resp.json();
    else data = { answer: await resp.text() };

    setLoading(false,'top');

    if(!resp.ok){
      const msg = data?.message || resp.statusText;
      addMessage('Error: ' + msg,'ai');
      status('Idle');
      return;
    }

    const content = data?.answer || data?.response || data?.content || JSON.stringify(data);
    addMessage(content, 'ai');
    status('Idle');

  }catch(err){
    setLoading(false,'top');
    status('Error: ' + err.message);
    addMessage('Error: ' + err.message,'ai');
  }
}

// initial welcome
addMessage('Hello! Paste a YouTube link on the left and click "Fetch Transcript" to begin.','ai');



