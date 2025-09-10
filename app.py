# app.py
# FastAPI + Single-page UI
# Continuous "real-time" conversation (auto listen -> STT -> LLM -> TTS -> repeat)
# Start = toggle start/stop; Restart button restarts
# Chat bubbles + Google TTS voice picker
# Uses ONLY the default knowledge_base.txt from disk (UI uploads are ignored)
# Run:  python app.py  ->  http://127.0.0.1:8400

import os, io, base64, wave
from dotenv import load_dotenv
load_dotenv()
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# ---------- KEYS (from .env only) ----------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434")

# Set GOOGLE_APPLICATION_CREDENTIALS env var for Google SDKs
if GOOGLE_APPLICATION_CREDENTIALS:
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

sdk_errors = []

# Gemini
try:
    import google.generativeai as genai
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    genai = None
    sdk_errors.append(f"Gemini error: {e}")

# Google Cloud STT/TTS
try:
    from google.cloud import speech, texttospeech
    speech_client = speech.SpeechClient()
    tts_client = texttospeech.TextToSpeechClient()
except Exception as e:
    speech = None
    texttospeech = None
    speech_client = None
    tts_client = None
    sdk_errors.append(f"Google STT/TTS error: {e}")

# OpenAI (optional)
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception as e:
    openai_client = None
    sdk_errors.append(f"OpenAI error: {e}")

# Ollama
try:
    import requests
except Exception as e:
    requests = None
    sdk_errors.append(f"requests error (Ollama): {e}")

# pyttsx3
try:
    import pyttsx3
except Exception as e:
    pyttsx3 = None
    sdk_errors.append(f"pyttsx3 error: {e}")

def pyttsx3_tts_bytes(text: str, voice_name: str = None) -> bytes:
    if not pyttsx3:
        return b""
    engine = pyttsx3.init()
    # Try to select Hindi voice if available
    if voice_name:
        for v in engine.getProperty('voices'):
            if voice_name.lower() in v.name.lower():
                engine.setProperty('voice', v.id)
                break
    engine.save_to_file(text, "pyttsx3_output.mp3")
    engine.runAndWait()
    try:
        with open("pyttsx3_output.mp3", "rb") as f:
            data = f.read()
        os.remove("pyttsx3_output.mp3")
        return data
    except Exception:
        return b""

def ensure_wav16k_mono(file_bytes: bytes) -> bytes:
    try:
        with wave.open(io.BytesIO(file_bytes), "rb") as wf:
            if wf.getnchannels()==1 and wf.getframerate()==16000 and wf.getsampwidth()==2:
                return file_bytes
    except:
        pass
    return file_bytes

def save_instr(text: str):
    with open(INSTR_FILE, "w", encoding="utf-8") as f:
        f.write(text)

# Always read KB from disk each time (ignore UI uploads)
def get_kb_from_disk() -> str:
    try:
        with open(KB_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def transcribe_audio_wav(wav_bytes: bytes) -> str:
    if not speech_client or not speech:
        return ""
    audio = speech.RecognitionAudio(content=wav_bytes)
    cfg = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-IN",
        alternative_language_codes=["en-IN","en-US"],
        enable_automatic_punctuation=True,
        use_enhanced=True,
        model="latest_short",  # <-- use short model for faster response
    )
    try:
        r = speech_client.recognize(config=cfg, audio=audio)
        t = r.results[0].alternatives[0].transcript.strip() if r.results else ""
        if t: return t
    except Exception:
        pass
    # fallback english primary
    try:
        cfg2 = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            alternative_language_codes=["hi-IN","en-IN"],
            enable_automatic_punctuation=True,
            use_enhanced=True,
            model="latest_short",  # <-- use short model for fallback too
        )
        r2 = speech_client.recognize(config=cfg2, audio=audio)
        t2 = r2.results[0].alternatives[0].transcript.strip() if r2.results else ""
        if t2: return t2
    except Exception:
        pass
    return ""

STOP_PHRASES = {"stop","bye","cut the call","band karo","bas","exit","quit","ruk jao"}

def llm_answer(provider: str, model: str, user_text: str) -> str:
    kb = get_kb_from_disk()  # always read the default file
    instr = STATE["instructions"]
    prompt = f"""You are young female helpful calling assistant for Metabull Universe.
Be concise, friendly, Hinglish ok. Use the KB if relevant.
Never use any kind of symbols in the answer to the user questions.
you can use the knowledge base to answer the user questions.
you can use some symbols like - " , "  " . " and numbers are allowed.

INSTRUCTIONS:
{instr or '(none)'}

KNOWLEDGE BASE:
{kb or '(empty)'}

USER:
{user_text}
"""
    if provider=="gemini":
        if not (genai and GOOGLE_API_KEY): return "(Gemini not configured)"
        try:
            m = genai.GenerativeModel(model or "gemini-1.5-flash")
            resp = m.generate_content(prompt)
            return (getattr(resp,"text","") or "").strip() or "(no content)"
        except Exception as e:
            return f"(Gemini error: {e})"
    if provider=="openai":
        if not openai_client: return "(OpenAI not configured)"
        try:
            try:
                r = openai_client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.6,
                )
                return r.choices[0].message.content.strip()
            except Exception:
                r = openai_client.responses.create(model=model or "gpt-4o-mini", input=prompt)
                return r.output_text.strip()
        except Exception as e:
            return f"(OpenAI error: {e})"
    if provider=="ollama":
        if not requests: return "(requests not available)"
        try:
            mm = model or "llama3"
            r = requests.post(
                f"{OLLAMA_ENDPOINT}/api/generate",
                json={"model":mm,"prompt":prompt,"stream":False},
                timeout=120
            )
            r.raise_for_status()
            return (r.json().get("response") or "").strip() or "(no content)"
        except Exception as e:
            return f"(Ollama error: {e})"
    return "(unknown provider)"

def tts_bytes(text: str, language_code: str, voice_name: str) -> bytes:
    # Use pyttsx3 if selected
    if voice_name.startswith("pyttsx3"):
        return pyttsx3_tts_bytes(text, voice_name.replace("pyttsx3-", ""))
    if not tts_client or not texttospeech: return b""
    try:
        synth = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name)
        cfg = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        resp = tts_client.synthesize_speech(input=synth, voice=voice, audio_config=cfg)
        return resp.audio_content
    except Exception:
        return b""

# ---------- APP ----------
app = FastAPI(title="Metabull Universe — Calling Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Default state (voice set to hi-IN-Chirp3-HD-Kore as requested)
STATE = {
    "provider": "gemini",
    "model": "gemini-1.5-flash",
    "phone": "",
    "instructions": "",
    "kb": "",  # not used at runtime anymore; KB is always read from disk
    "tts_language_code": "hi-IN",
    "tts_voice_name": "hi-IN-Chirp3-HD-Kore",
}

KB_FILE = "knowledge_base.txt"
INSTR_FILE = "llm_instructions.txt"

def _load_files():
    # Only instructions are loaded into STATE at startup; KB is always read fresh from disk per turn.
    try:
        if os.path.exists(INSTR_FILE):
            with open(INSTR_FILE, "r", encoding="utf-8") as f:
                STATE["instructions"] = f.read()
    except:
        pass

_load_files()

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

@app.get("/api/config")
def get_config():
    return {
        "provider": STATE["provider"],
        "model": STATE["model"],
        "phone": STATE["phone"],
        "instructions": STATE["instructions"],
        # reflect actual KB on disk:
        "has_kb": os.path.exists(KB_FILE) and os.path.getsize(KB_FILE) > 0,
        "tts_language_code": STATE["tts_language_code"],
        "tts_voice_name": STATE["tts_voice_name"],
        "sdk_errors": sdk_errors,
    }

@app.post("/api/config")
def set_config(
    provider: str = Form(...),
    model: str = Form(""),
    phone: str = Form(""),
    tts_language_code: str = Form("en-IN"),
    tts_voice_name: str = Form("en-IN-Wavenet-A"),
):
    STATE["provider"] = provider
    if model: STATE["model"] = model
    STATE["phone"] = phone
    STATE["tts_language_code"] = tts_language_code
    STATE["tts_voice_name"] = tts_voice_name
    return {"ok": True}

@app.post("/api/instructions")
def set_instructions(text: str = Form("")):
    save_instr(text)
    return {"ok": True}

# UI uploads are ignored; we always use KB_FILE on disk.
@app.post("/api/upload_kb")
async def upload_kb(file: UploadFile = File(...)):
    return {"ok": True, "ignored": True, "using": KB_FILE}

@app.get("/api/models")
def list_models(provider: str):
  if provider=="gemini":
    return {"models":["gemini-1.5-flash","gemini-1.5-pro"]}
  if provider=="openai":
    return {"models":[
      "gpt-4o-mini",
      "gpt-4o",
      "gpt-4.1-mini",
      "gpt-4.1",
      "gpt-4",
      "gpt-4-turbo",
      "gpt-3.5-turbo-0125",
      "gpt-3.5-turbo-1106",
      "gpt-3.5-turbo",
      "gpt-3.5-turbo-16k"
    ]}
  if provider=="ollama":
    return {"models":["llama3","llama3.1","qwen2.5","phi3","llama3.2:3b"]}  # <-- Added here
  return {"models":[]}

@app.get("/api/voices")
def list_voices(lang_hint: Optional[str]=Query(None)):
    voices=[]
    # Add pyttsx3 voices (Hindi only)
    if pyttsx3:
        try:
            engine = pyttsx3.init()
            for v in engine.getProperty('voices'):
                # Show all voices, mark Hindi if found in name
                lang_codes = ["pyttsx3"]
                if "hindi" in v.name.lower():
                    lang_codes.append("hi-IN")
                voices.append({"name": f"pyttsx3-{v.name}", "language_codes": lang_codes})
        except Exception:
            pass
    # ...existing Google TTS code...
    if not tts_client or not texttospeech: return {"voices":voices}
    try:
        resp = tts_client.list_voices()
        for v in resp.voices:
            langs = list(v.language_codes)
            if lang_hint:
                if lang_hint in langs:
                    voices.append({"name":v.name,"language_codes":langs})
            else:
                if any(code.startswith(("hi-","en-")) for code in langs):
                    voices.append({"name":v.name,"language_codes":langs})
    except Exception: pass
    uniq={v["name"]:v for v in voices}
    def rk(v):
        lc=v.get("language_codes",[])
        if any(x.startswith("hi-") for x in lc): return (0,v["name"])
        if any(x.startswith("en-") for x in lc): return (1,v["name"])
        if any(x=="pyttsx3" for x in lc): return (2,v["name"])
        return (3,v["name"])
    return {"voices": sorted(uniq.values(), key=rk)}

@app.post("/api/ask")
async def api_ask(audio: UploadFile = File(None), text: Optional[str]=Form(None)):
    user_text = (text or "").strip()
    stt_text = ""
    if not user_text and audio is not None:
        raw = await audio.read()
        wav = ensure_wav16k_mono(raw)
        stt_text = transcribe_audio_wav(wav)
        user_text = stt_text

    if not user_text:
        return JSONResponse({"error":"No text/audio provided"}, status_code=400)

    if any(phrase in user_text.lower() for phrase in STOP_PHRASES):
        return {"stt": user_text, "answer": "The call has been stopped. Thank you!", "audio_b64": "" , "should_stop": True}

    ans = llm_answer(STATE["provider"], STATE["model"], user_text)
    mp3 = tts_bytes(ans, STATE["tts_language_code"], STATE["tts_voice_name"])
    audio_b64 = base64.b64encode(mp3).decode("utf-8") if mp3 else ""
    return {"stt": stt_text, "answer": ans, "audio_b64": audio_b64, "should_stop": False}

# TTS-only endpoint for greeting or any custom speech
@app.post("/api/tts")
def api_tts(text: str = Form(...)):
    mp3 = tts_bytes(text, STATE["tts_language_code"], STATE["tts_voice_name"])
    audio_b64 = base64.b64encode(mp3).decode("utf-8") if mp3 else ""
    return {"audio_b64": audio_b64}

# ---------- UI ----------
HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Metabull — Real-time Assistant</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root{--bg:#0b0c10;--card:#14161b;--txt:#e8e8ea;--muted:#9aa0a6;--ac:#2bc0e4;--ok:#19c37d;--gray:#2a2f3b}
    *{box-sizing:border-box}
    body{margin:0;background:radial-gradient(80vh 80vh at 20% 10%, #10121a 0%, #0b0c10 60%);color:var(--txt);font-family:Inter,system-ui,Segoe UI,Roboto,Arial}
    .wrap{max-width:1000px;margin:24px auto;padding:16px}
    h1{margin:0 0 12px}
    .grid{display:grid;gap:16px}
    .card{background:var(--card);border:1px solid #1b1d24;border-radius:16px;padding:16px}
    label{font-size:12px;color:var(--muted)}
    input,select,textarea{width:100%;padding:10px;border-radius:10px;background:#0e1116;color:var(--txt);border:1px solid #222632}
    textarea{min-height:80px}
    button{border:none;border-radius:12px;padding:10px 14px;font-weight:700;color:#fff;cursor:pointer}
    .btn{background:linear-gradient(90deg,var(--ac),#8a60ff)}
    .btn.gray{background:var(--gray)}
    .btn.green{background:var(--ok)}
    .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
    .cols{display:grid;gap:12px;grid-template-columns:1fr 1fr}
    @media (max-width:900px){.cols{grid-template-columns:1fr}}
    .status{font-size:13px;color:var(--muted)}
    .chat{background:#0e1116;border:1px solid #222632;border-radius:14px;padding:10px;max-height:420px;overflow:auto}
    .msg{margin:8px 0;display:flex}
    .msg .bubble{padding:10px 12px;border-radius:12px;max-width:80%}
    .user .bubble{background:#16324a; margin-left:auto}
    .agent .bubble{background:#1f2430; margin-right:auto}
    .label{font-size:11px;color:#c7cbd1;margin-bottom:3px}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Metabull Universe — Real-time Calling Assistant</h1>

  <div class="grid">
    <div class="card">
      <div class="cols">
        <div>
          <label>Provider</label>
          <select id="provider">
            <option value="gemini">Gemini</option>
            <option value="openai">OpenAI</option>
            <option value="ollama">Ollama</option>
          </select>
        </div>
        <div>
          <label>Model</label>
          <select id="model"></select>
        </div>
        <div>
          <label>Voice (Google TTS)</label>
          <select id="voice"></select>
        </div>
        <div>
          <label>Phone (placeholder)</label>
          <input id="phone" placeholder="+91 98xxxxxx"/>
        </div>
      </div>

      <div class="row" style="margin-top:12px">
        <button id="btnStart" class="btn green">Start</button>
        <button id="btnRestart" class="btn gray">Restart</button>
        <span id="status" class="status"></span>
      </div>
    </div>

    <div class="card">
      <label>Conversation</label>
      <div id="chat" class="chat"></div>
      <div class="row" style="margin-top:8px">
        <textarea id="manualText" placeholder='Type to test (optional)… e.g. "Metabull kya provide karta hai?"'></textarea>
        <button id="btnSend" class="btn">Send as user</button>
        <audio id="player" controls style="width:220px"></audio>
      </div>
    </div>

    <div class="card">
      <label>LLM Instructions</label>
      <textarea id="instr"></textarea>
      <div class="row" style="margin-top:6px">
        <button id="saveInstr" class="btn gray">Save</button>
        <span id="sdk" class="status"></span>
      </div>
    </div>

    <div class="card">
      <label>Knowledge Base (fixed)</label>
      <div class="row">
        <input type="file" id="kbFile" accept=".txt"/>
        <button id="uploadKb" class="btn gray">Upload (ignored)</button>
        <span id="kbState" class="status">Using default: knowledge_base.txt</span>
      </div>
    </div>
  </div>
</div>

<script>
const $ = (id)=>document.getElementById(id);
let running = false;
let mediaStream, audioCtx, processor, inputNode, recorder;
let vadSilenceMs = 3000;        // silence to stop recording
let vadMinSpeakMs = 100;        // minimum speaking before allow silence stop
let energyThresh = 0.008;       // simple RMS threshold (tune as needed)
let lastAbove = 0, startedAt = 0;
let pcmChunks = [];

function setStatus(msg){ $("status").textContent = msg||""; }
function appendMsg(role, text){
  const chat = $("chat");
  const wrap = document.createElement("div");
  wrap.className = "msg " + (role==="user"?"user":"agent");
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const label = document.createElement("div");
  label.className = "label";
  label.textContent = (role==="user"?"user":"agent");
  const body = document.createElement("div");
  body.textContent = text;
  bubble.appendChild(label); bubble.appendChild(body);
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

async function loadConfig(){
  const cfg = await (await fetch("/api/config")).json();
  $("provider").value = cfg.provider;
  $("instr").value = cfg.instructions||"";
  $("phone").value = cfg.phone||"";
  $("sdk").textContent = (cfg.sdk_errors||[]).join(" | ")||"(OK)";
  await refreshModels(cfg.provider, cfg.model);
  await refreshVoices(cfg.tts_voice_name);
}

async function refreshModels(provider, selected){
  const r = await (await fetch("/api/models?provider="+encodeURIComponent(provider))).json();
  const sel = $("model"); sel.innerHTML = "";
  (r.models||[]).forEach(m=>{
    const o = document.createElement("option");
    o.value=m; o.textContent=m;
    if(selected===m) o.selected=true;
    sel.appendChild(o);
  });
}
$("provider").addEventListener("change", async e=>{
  await refreshModels(e.target.value);
  await saveConfig(); // persist provider+model
});

async function refreshVoices(selected){
  const r = await (await fetch("/api/voices")).json();
  const sel = $("voice"); sel.innerHTML="";
  (r.voices||[]).forEach(v=>{
    const o=document.createElement("option");
    o.value=v.name; o.textContent = v.name+" ("+(v.language_codes||[]).join(", ")+")";
    if(selected===v.name) o.selected = true;
    sel.appendChild(o);
  });
}

async function saveConfig(){
  const fd = new FormData();
  fd.append("provider", $("provider").value);
  fd.append("model", $("model").value);
  fd.append("phone", $("phone").value);
  const voiceName = $("voice").value;
  fd.append("tts_voice_name", voiceName);
  fd.append("tts_language_code", voiceName.includes("hi-")?"hi-IN":"en-IN");
  await fetch("/api/config",{method:"POST",body:fd});
}

$("saveInstr").addEventListener("click", async ()=>{
  const fd = new FormData(); fd.append("text", $("instr").value);
  await fetch("/api/instructions",{method:"POST",body:fd});
  setStatus("Instructions saved");
});

// Upload is ignored on backend; keep UI consistent.
$("uploadKb").addEventListener("click", async ()=>{
  const f = $("kbFile").files[0];
  if(!f){ setStatus("Upload ignored — using default knowledge_base.txt"); return; }
  const fd = new FormData(); fd.append("file", f);
  const r = await (await fetch("/api/upload_kb",{method:"POST",body:fd})).json();
  $("kbState").textContent = r.ok ? "Upload received (ignored). Using default knowledge_base.txt" : "Upload failed (and ignored).";
  setStatus("Using default knowledge_base.txt");
});

$("btnSend").addEventListener("click", async ()=>{
  const txt = $("manualText").value.trim();
  if(!txt) return;
  $("manualText").value = "";
  appendMsg("user", txt);
  await askLLMWithText(txt, /*loopAfter*/ running);
});

// Greeting on start then listen
async function playStartGreeting() {
  const fd = new FormData();
  fd.append("text", 'Hello i  am ekta from Meta bull univers, how can i help you today?');
  try {
    const r = await (await fetch("/api/tts", { method: "POST", body: fd })).json();
    appendMsg("agent", 'Hello i  am ekta from Meta bull univers, how can i help you today?');
    if (r.audio_b64) {
      const player = $("player");
      player.onended = () => { if (running) loopListen(); };
      player.src = "data:audio/mp3;base64," + r.audio_b64;
      player.play();
      setStatus("Greeting…");
      return;
    }
  } catch (e) {}
  if (running) loopListen();
}

$("btnStart").addEventListener("click", async ()=>{
  if(!running){
    running = true;
    $("btnStart").textContent = "Stop";
    $("btnStart").classList.remove("green");
    $("btnStart").classList.add("gray");
    await saveConfig();
    setStatus("Starting…");
    playStartGreeting(); // greeting then auto-listen
  }else{
    running = false;
    $("btnStart").textContent = "Start";
    $("btnStart").classList.remove("gray");
    $("btnStart").classList.add("green");
    stopCapture(); // stop if recording
    setStatus("Stopped.");
  }
});

$("btnRestart").addEventListener("click", async ()=>{
  running = false;
  stopCapture();
  $("btnStart").textContent = "Stop";
  $("btnStart").classList.remove("green");
  $("btnStart").classList.add("gray");
  running = true;
  setStatus("Restarting…");
  await saveConfig();
  playStartGreeting();
});

function startCapture(){
  return navigator.mediaDevices.getUserMedia({audio:true}).then(stream=>{
    mediaStream = stream;
    audioCtx = new (window.AudioContext||window.webkitAudioContext)({sampleRate:16000});
    processor = audioCtx.createScriptProcessor(4096,1,1);
    inputNode = audioCtx.createMediaStreamSource(stream);
    pcmChunks = [];
    lastAbove = 0; startedAt = Date.now();

    processor.onaudioprocess = e=>{
      const ch = e.inputBuffer.getChannelData(0);
      // compute RMS energy
      let sum=0; for(let i=0;i<ch.length;i++){ sum += ch[i]*ch[i]; }
      const rms = Math.sqrt(sum/ch.length);
      const buf = new Int16Array(ch.length);
      for(let i=0;i<ch.length;i++){
        let s = Math.max(-1, Math.min(1, ch[i]));
        buf[i] = s<0 ? s*0x8000 : s*0x7FFF;
      }
      pcmChunks.push(buf);
      const now = Date.now();
      if(rms>energyThresh){ lastAbove = now; }
      const spokenLongEnough = (now - startedAt) > 600;
      const silentLong = (now - lastAbove) > 2000;
      if(spokenLongEnough && silentLong){
        // stop on silence
        stopCapture(true);
      }
    };
    inputNode.connect(processor);
    processor.connect(audioCtx.destination);
  });
}

function stopCapture(triggerAsk=false){
  try{ processor && processor.disconnect(); }catch{}
  try{ inputNode && inputNode.disconnect(); }catch{}
  try{ audioCtx && audioCtx.close(); }catch{}
  try{ mediaStream && mediaStream.getTracks().forEach(t=>t.stop()); }catch{}
  processor=null; inputNode=null; audioCtx=null; mediaStream=null;

  if(triggerAsk){
    // pack WAV (16k mono PCM16)
    const pcmLen = pcmChunks.reduce((a,b)=>a+b.length,0);
    const wavBuffer = new ArrayBuffer(44 + pcmLen*2);
    const view = new DataView(wavBuffer);
    function wstr(off,str){ for(let i=0;i<str.length;i++) view.setUint8(off+i, str.charCodeAt(i)); }
    let off=0; wstr(off,"RIFF"); off+=4;
    view.setUint32(off, 36 + pcmLen*2, true); off+=4;
    wstr(off,"WAVE"); off+=4; wstr(off,"fmt "); off+=4;
    view.setUint32(off,16,true); off+=4;
    view.setUint16(off,1,true); off+=2;
    view.setUint16(off,1,true); off+=2;
    view.setUint32(off,16000,true); off+=4;
    view.setUint32(off,16000*2,true); off+=4;
    view.setUint16(off,2,true); off+=2;
    view.setUint16(off,16,true); off+=2;
    wstr(off,"data"); off+=4;
    view.setUint32(off, pcmLen*2, true); off+=4;
    let pos=44;
    for(const b of pcmChunks){ for(let i=0;i<b.length;i++,pos+=2) view.setInt16(pos,b[i],true); }
    const blob = new Blob([view], {type:"audio/wav"});
    askLLMWithAudio(blob, /*loopAfter*/ running);
  }
}

async function loopListen(){
  if(!running) return;
  setStatus("Listening… (speak)");
  await startCapture();
}

async function askLLMWithAudio(wavBlob, loopAfter){
  setStatus("Transcribing…");
  const fd = new FormData(); fd.append("audio", wavBlob, "input.wav");
  const r = await (await fetch("/api/ask",{method:"POST", body:fd})).json();
  handleAskResponse(r, loopAfter);
}

async function askLLMWithText(text, loopAfter){
  const fd = new FormData(); fd.append("text", text);
  const r = await (await fetch("/api/ask",{method:"POST", body:fd})).json();
  handleAskResponse(r, loopAfter);
}

function handleAskResponse(r, loopAfter){
  if(r.error){ setStatus(r.error); return; }
  const userSaid = r.stt || ""; if(userSaid){ appendMsg("user", userSaid); }
  appendMsg("agent", r.answer || "");

  if(r.should_stop){
    running=false;
    $("btnStart").textContent="Start";
    $("btnStart").classList.remove("gray"); $("btnStart").classList.add("green");
    setStatus("Stopped by voice command.");
    return;
  }

  if(r.audio_b64){
    const src="data:audio/mp3;base64,"+r.audio_b64;
    const player=$("player");
    player.onended = ()=>{
      if(loopAfter){ loopListen(); }
    };
    player.src=src; player.play();
    setStatus("Speaking…");
  }else{
    if(loopAfter){ loopListen(); }
  }
}

loadConfig();
</script>

</body>
</html>
"""

# ---------- MAIN ----------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8500)
