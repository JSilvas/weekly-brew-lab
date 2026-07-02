"""
Magentic Improv — Gradio app.

Style canvas (same physics as Personal DJ) → MusicCoCa embedding → RT2 style conditioning.
Mic input → real-time pitch detection → NoteStateTracker → RT2 notes conditioning.

Run:  uv run python app.py  →  http://localhost:7862
macOS: grant Microphone access to the terminal.
"""

from __future__ import annotations

import atexit
import json
import signal
import threading

import gradio as gr
import sounddevice as sd

from engine import ImprovEngine

# ── Shared state ──────────────────────────────────────────────────────────────

engine = ImprovEngine()
_mic_on = False
_mic_lock = threading.Lock()


def _shutdown():
    engine.pause()
    engine.stop_mic()


atexit.register(_shutdown)
signal.signal(signal.SIGTERM, lambda *_: (_shutdown(), exit(0)))


# ── Input device helpers ──────────────────────────────────────────────────────

# Names that identify virtual/loopback audio drivers (case-insensitive substring match).
_LOOPBACK_KEYWORDS = {"blackhole", "loopback", "soundflower", "stereo mix", "virtual", "cable", "jack"}


def _list_mic_devices() -> list[tuple[str, int | None]]:
    try:
        devices = sd.query_devices()
        result: list[tuple[str, int | None]] = [("System Default", None)]
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                result.append((d["name"], i))
        return result
    except Exception:
        return [("System Default", None)]


def _list_loopback_devices() -> list[tuple[str, int | None]]:
    """
    Return input devices that look like virtual/loopback drivers (BlackHole,
    SoundFlower, Loopback, etc.).  If none are found, fall back to all input
    devices so the user can still pick manually, with a hint at the top.
    """
    try:
        devices = sd.query_devices()
        loopback, all_inputs = [], []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                all_inputs.append((d["name"], i))
                if any(kw in d["name"].lower() for kw in _LOOPBACK_KEYWORDS):
                    loopback.append((d["name"], i))
        if loopback:
            return loopback
        # No known loopback driver — show hint + every input device
        hint: list[tuple[str, int | None]] = [("⚠ Install BlackHole for loopback", None)]
        return hint + all_inputs
    except Exception:
        return [("No devices found", None)]


def _list_output_devices(exclude_name: str | None = None) -> list[tuple[str, int | None]]:
    """
    List output-capable audio devices.  When exclude_name is provided (the
    name of the loopback input device), any device whose name contains that
    string is omitted so Magenta's output cannot feed back into the pipeline.
    """
    try:
        devices = sd.query_devices()
        result: list[tuple[str, int | None]] = [("System Default", None)]
        for i, d in enumerate(devices):
            if d["max_output_channels"] > 0:
                if exclude_name and exclude_name.lower() in d["name"].lower():
                    continue
                result.append((d["name"], i))
        return result
    except Exception:
        return [("System Default", None)]


# ── Mic status HTML ───────────────────────────────────────────────────────────

def _mic_html() -> str:
    if not engine.mic_enabled:
        return (
            '<div style="background:#09090f;border-radius:0 0 10px 10px;'
            "padding:14px 16px;border-top:1px solid rgba(255,255,255,0.06);"
            'display:flex;align-items:center;gap:16px;">'
            '<span style="font-size:36px;color:#333">🎙</span>'
            '<span style="color:#555;font-size:13px;">Mic inactive — press <b>Enable Mic</b> to start pitch detection</span>'
            "</div>"
        )

    level = engine.mic_level
    level_pct = min(int(level * 1200), 100)

    note_name = engine.detected_note_name
    freq = engine.detected_freq
    midi = engine.detected_midi
    conf = engine.detected_confidence

    if freq is not None:
        color = "#48dbfb"
        info = f"{freq:.1f} Hz · MIDI {midi} · conf {conf:.2f}"
        note_size = "42px"
    else:
        color = "#444"
        note_name = "—"
        info = "No pitch detected"
        note_size = "42px"

    return f"""
    <div style="background:#09090f;border-radius:0 0 10px 10px;
                padding:14px 16px;border-top:1px solid rgba(255,255,255,0.06);">
      <div style="display:flex;align-items:center;gap:20px;">
        <div style="min-width:110px;text-align:center;">
          <div style="font-size:{note_size};color:{color};font-weight:300;line-height:1.1;">{note_name}</div>
          <div style="font-size:11px;color:#666;margin-top:4px;">{info}</div>
        </div>
        <div style="flex:1;">
          <div style="font-size:10px;color:#555;margin-bottom:5px;letter-spacing:0.5px;">LEVEL</div>
          <div style="height:5px;background:#1a1a2e;border-radius:3px;overflow:hidden;">
            <div style="width:{level_pct}%;height:100%;background:{color};border-radius:3px;"></div>
          </div>
        </div>
        <div style="font-size:11px;color:#555;min-width:80px;text-align:right;">
          🎙 {engine.active_input_device or "live"}
        </div>
      </div>
    </div>
    """


# ── Canvas HTML ───────────────────────────────────────────────────────────────
# Style-node physics canvas — same as Personal DJ but without the MIDI keyboard.
# Nodes bounce around; the listener's proximity to each node sets its weight in
# the style embedding blend.  Bridge: pdj-data div → Gradio timer → set_style().

CANVAS_HTML = """
<div id="pdj-outer" style="width:100%;">

<div id="pdj-wrap" style="
  position:relative;width:100%;height:480px;
  background:#09090f;border-radius:10px;overflow:hidden;
  cursor:default;user-select:none;">

  <canvas id="pdj-canvas" style="position:absolute;inset:0;width:100%;height:100%;"></canvas>

  <input id="pdj-edit" type="text" autocomplete="off" spellcheck="false"
    placeholder="edit prompt…"
    style="
      position:absolute;display:none;z-index:20;
      background:#1a1a2e;color:#eee;border:1px solid #555;border-radius:6px;
      padding:4px 8px;font-size:13px;min-width:180px;outline:none;">

  <div id="pdj-data" style="display:none;"></div>

  <div style="
    position:absolute;bottom:0;left:0;right:0;height:44px;z-index:10;
    background:rgba(9,9,15,0.85);display:flex;align-items:center;
    padding:0 14px;gap:10px;">
    <span style="font-size:16px;line-height:1;">🐢</span>
    <input id="pdj-speed" type="range" min="0" max="1" step="0.005" value="0.35"
      style="flex:1;accent-color:#48dbfb;">
    <span style="font-size:16px;line-height:1;">🐇</span>
    <button id="pdj-add" title="Add prompt (or double-click canvas)"
      style="
        background:rgba(255,255,255,0.1);border:none;color:#eee;
        border-radius:6px;width:28px;height:28px;font-size:17px;
        cursor:pointer;line-height:1;display:flex;align-items:center;justify-content:center;">
      +
    </button>
  </div>
</div>

</div>
"""

# Canvas JS — physics, weight calc, bridge. Keyboard and focus-node logic removed.
CANVAS_JS = r"""
(function () {

// ── Suggestions ──────────────────────────────────────────────────────────────
const ALL_SUGGESTIONS = [
  "Dreamy Ambient Pads","Jazz Piano Trio","Electro Synthpop","Chiptune","Synthwave",
  "Jazz Guitar","Saturated Gamelan Choir","Flamenco Nylon Guitar Rasgueado",
  "Reggae Rhythm Guitar","Synthpop Groove Club Mix",
  "Fast Swing Jazz Clarinet and Guitar","Afrobeat Band with Horns and Complex Drums",
  "French House Disco Loops Filter Sweeps","Country Banjo Picking","R&B Smooth Keys",
  "Soft Rock","Cyberpunk Synthwave Mariachi Horns","Trap Beat with Sampled Funk",
  "UK Post-Dubstep String Quartet","African Kalimba","Lo-fi Hip Hop Beat",
  "Smooth Bossa Nova","West African Kora Polyrhythms",
  "Danceable Latin Jazz Salsa with Trombone","Celtic Fiddle Jig",
  "Acoustic Folk Guitar","Church Organ","Surf Rock Guitar",
  "Cavernous Endless Reverb Electric Guitar Swells","Japanese Koto",
  "Gritty Garage Rock","Indian Classical Sitar and Tabla Raga",
  "Middle Eastern Oud and Darbuka Groove","Ambient IDM Glitch Beats",
  "Euphoric Washed-Out Noise Pop Fuzz","Andean Pan Flute Mountain Melody",
  "Balinese Gamelan Metallic Percussion","Dark Cinematic Soundtrack",
  "Brazilian Samba Batucada Percussion Ensemble",
  "Heavily Digitally Distorted Harp Shimmer","Warm Vinyl Crackle Dusty Organ Chords",
  "Supersaw Complextro Chords","Melodramatic Tremolo Mandolin",
  "Trance Arpeggiated Synth","Baroque Cello Meets 90s Trance Euphoria",
  "Polka Accordion","Dubstep Wobble Bass Synth","Medieval Rain",
  "Granular Synthesis Frozen Vocal Textures","Slow Pad Sweeps Up",
  "Cinematic Orchestral Hits","Violin Chamber Ensemble","Fanfare French Horn",
  "Retro Synthwave Analog Lead","Ambient Pad Synthesizer",
  "Bowed Vibraphone Sustained Metallic Ringing","Classical Cello",
  "Delicate Vintage Music Box","Bluegrass Picked Banjo","Gentle Microtonal Flutes",
  "Latin Mallet Marimba","Fingerpicked Acoustic Guitar","Chinese Guzheng",
  "Orchestral Sustained Oboe","Nylon String Classical Guitar","Deep Future Garage",
  "80s Cinematic Synthpop","Ambient Drone with distant bells"
];

const SHUFFLED = [...ALL_SUGGESTIONS];
for (let i = SHUFFLED.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [SHUFFLED[i], SHUFFLED[j]] = [SHUFFLED[j], SHUFFLED[i]];
}

// ── Constants ─────────────────────────────────────────────────────────────────
const PROMPT_R   = 22;
const LISTENER_R = 26;
const FALLOFF    = 2.0;
const MAX_NODES  = 6;
const MIN_NODES  = 1;
const MAX_SPEED  = 700;
const DAMPING    = 0.35;
const COLORS     = ['#ff6b6b','#48dbfb','#ffd32a','#0be881','#f8b500','#ff5e57'];
const BRIDGE_HZ  = 10;

// ── State ─────────────────────────────────────────────────────────────────────
let W = 0, H = 0, PLAY_H = 0;
let nodes    = [];
let listener = {x:0, y:0, vx:0, vy:0};
let physicsSpeed = 0;
let drag        = null;
let dragMX      = 0, dragMY = 0;
let selectedId  = null;
let nextId      = 1;
let nextColor   = 1;
let deckIdx     = 1;
let dashOffsets = [];
let lastBridge  = 0;
let animId      = null;
let lastT       = null;
let recentPos   = [];

let canvas, ctx, wrap, labelEdit;

// ── Resize ────────────────────────────────────────────────────────────────────
function resize() {
  const r = wrap.getBoundingClientRect();
  W = r.width  || 600;
  H = r.height || 480;
  canvas.width  = W;
  canvas.height = H;
  PLAY_H = H - 44;
  clampAll();
}

function clampBall(b, R) {
  if (b.x < R)        { b.x = R;          b.vx =  Math.abs(b.vx); }
  if (b.x > W - R)    { b.x = W - R;      b.vx = -Math.abs(b.vx); }
  if (b.y < R)        { b.y = R;          b.vy =  Math.abs(b.vy); }
  if (b.y > PLAY_H-R) { b.y = PLAY_H - R; b.vy = -Math.abs(b.vy); }
}

function clampAll() {
  nodes.forEach(n => clampBall(n, PROMPT_R));
  clampBall(listener, LISTENER_R);
}

// ── Initial layout ────────────────────────────────────────────────────────────
function buildLayout() {
  const cx = W / 2;
  const cy = PLAY_H / 2;
  nodes = [{
    id: 0, colorIdx: 0,
    x: cx, y: cy,
    vx: 0, vy: 0,
    label: SHUFFLED[0] || 'Jazz Piano Trio',
  }];
  listener    = {x: cx, y: cy - 160, vx: 0, vy: 0};
  deckIdx     = 1;
  dashOffsets = [0];
}

// ── Weight calculation ────────────────────────────────────────────────────────
function calcWeights() {
  if (!nodes.length) return [];
  const dists = nodes.map(n => Math.hypot(n.x - listener.x, n.y - listener.y));
  const zi = dists.findIndex(d => d < 1);
  if (zi !== -1) return nodes.map((_, i) => i === zi ? 1 : 0);
  const raw = dists.map(d => 1 / Math.pow(d, FALLOFF));
  const sum = raw.reduce((a, b) => a + b, 0);
  return raw.map(w => w / sum);
}

// ── Physics ───────────────────────────────────────────────────────────────────
function advanceBall(b, dt) {
  if (physicsSpeed < 0.001) return;
  const damp = Math.exp(-DAMPING * dt);
  b.vx *= damp;
  b.vy *= damp;
  const s = Math.hypot(b.vx, b.vy);
  const cap = MAX_SPEED * physicsSpeed;
  if (s > cap) { b.vx *= cap/s; b.vy *= cap/s; }
  b.x += b.vx * dt;
  b.y += b.vy * dt;
}

// ── Trash zone ────────────────────────────────────────────────────────────────
const TRASH_CX = 44, TRASH_CY_FRAC = 0.5, TRASH_R = 28;
function trashCY() { return PLAY_H * TRASH_CY_FRAC; }
function overTrash(x, y) {
  const dx = x - TRASH_CX, dy = y - trashCY();
  return dx*dx + dy*dy <= TRASH_R*TRASH_R;
}

function drawTrash(alpha, hot) {
  const cx = TRASH_CX, cy = trashCY();
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.arc(cx, cy, TRASH_R, 0, Math.PI*2);
  ctx.fillStyle = hot ? 'rgba(220,50,50,0.85)' : 'rgba(60,60,70,0.75)';
  ctx.fill();
  ctx.strokeStyle = hot ? 'rgba(255,100,100,0.9)' : 'rgba(180,180,190,0.4)';
  ctx.lineWidth = 1.5;
  ctx.stroke();
  const s = 0.72;
  ctx.translate(cx, cy);
  ctx.scale(s, s);
  ctx.strokeStyle = hot ? '#fff' : 'rgba(220,220,230,0.9)';
  ctx.lineWidth = 1.8 / s;
  ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(-11, -9); ctx.lineTo(11, -9); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(-5, -9); ctx.lineTo(-5, -13); ctx.lineTo(5, -13); ctx.lineTo(5, -9); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(-9, -7); ctx.lineTo(-7, 12); ctx.lineTo(7, 12); ctx.lineTo(9, -7); ctx.stroke();
  [-4, 0, 4].forEach(x => { ctx.beginPath(); ctx.moveTo(x, -5); ctx.lineTo(x * 0.8, 10); ctx.stroke(); });
  ctx.restore();
}

// ── Draw ──────────────────────────────────────────────────────────────────────
function draw(weights) {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#09090f';
  ctx.fillRect(0, 0, W, H);

  if (drag && drag.type === 'node') {
    const n = nodes.find(n => n.id === drag.id);
    const hot = n && overTrash(n.x, n.y);
    drawTrash(hot ? 1.0 : 0.55, hot);
  }

  nodes.forEach((n, i) => {
    const w = weights[i] || 0;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(listener.x, listener.y);
    ctx.lineTo(n.x, n.y);
    ctx.strokeStyle = `rgba(255,255,255,${(0.06 + 0.45 * w).toFixed(2)})`;
    ctx.lineWidth   = 0.5 + 2.5 * w;
    ctx.setLineDash([6, 6]);
    ctx.lineDashOffset = -(dashOffsets[i] || 0);
    ctx.stroke();
    ctx.restore();
  });

  nodes.forEach((n, i) => {
    const color    = COLORS[n.colorIdx % COLORS.length];
    const w        = weights[i] || 0;
    const selected = n.id === selectedId;
    ctx.save();
    ctx.shadowBlur  = 6 + 22 * w;
    ctx.shadowColor = color;
    ctx.beginPath();
    ctx.arc(n.x, n.y, PROMPT_R, 0, Math.PI*2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.restore();
    if (selected) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, PROMPT_R + 5, 0, Math.PI*2);
      ctx.strokeStyle = 'rgba(255,255,255,0.75)';
      ctx.lineWidth   = 1.5;
      ctx.stroke();
    }
    ctx.fillStyle    = 'rgba(255,255,255,0.88)';
    ctx.font         = '11px system-ui,sans-serif';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'bottom';
    const words = n.label.split(' ');
    const maxLW = 130;
    let line = '', lines = [];
    words.forEach(word => {
      const test = line ? line + ' ' + word : word;
      if (ctx.measureText(test).width > maxLW && line) { lines.push(line); line = word; }
      else line = test;
    });
    if (line) lines.push(line);
    lines.forEach((l, li) => {
      ctx.fillText(l, n.x, n.y - PROMPT_R - 4 - (lines.length - 1 - li) * 13);
    });
  });

  ctx.save();
  ctx.shadowBlur  = 18;
  ctx.shadowColor = 'rgba(255,255,255,0.55)';
  ctx.beginPath();
  ctx.arc(listener.x, listener.y, LISTENER_R, 0, Math.PI*2);
  ctx.fillStyle = 'rgba(255,255,255,0.90)';
  ctx.fill();
  ctx.restore();
  ctx.beginPath();
  ctx.arc(listener.x, listener.y, 4, 0, Math.PI*2);
  ctx.fillStyle = '#09090f';
  ctx.fill();
}

// ── Animation loop ────────────────────────────────────────────────────────────
function loop(ts) {
  animId = requestAnimationFrame(loop);
  const dt = lastT ? Math.min((ts - lastT) / 1000, 0.05) : 0;
  lastT = ts;

  if (physicsSpeed > 0) {
    if (!drag || drag.type !== 'listener') { advanceBall(listener, dt); clampBall(listener, LISTENER_R); }
    nodes.forEach(n => {
      if (!drag || drag.type !== 'node' || drag.id !== n.id) { advanceBall(n, dt); clampBall(n, PROMPT_R); }
    });
  }

  const weights = calcWeights();
  nodes.forEach((_, i) => { dashOffsets[i] = (dashOffsets[i] || 0) + (weights[i] || 0) * 55 * dt; });

  draw(weights);

  const now = performance.now();
  if (now - lastBridge >= 1000 / BRIDGE_HZ) {
    lastBridge = now;
    pushBridge(weights);
  }
}

// ── Bridge ────────────────────────────────────────────────────────────────────
function pushBridge(weights) {
  const el = document.getElementById('pdj-data');
  if (el) el.textContent = JSON.stringify({
    weights,
    prompts: nodes.map(n => n.label),
  });
}

// ── Hit testing ───────────────────────────────────────────────────────────────
function hitTest(mx, my) {
  if (Math.hypot(mx - listener.x, my - listener.y) < LISTENER_R + 10)
    return { type: 'listener' };
  for (let i = nodes.length - 1; i >= 0; i--) {
    if (Math.hypot(mx - nodes[i].x, my - nodes[i].y) < PROMPT_R + 10)
      return { type: 'node', id: nodes[i].id };
  }
  return null;
}

function canvasXY(e) {
  const r  = canvas.getBoundingClientRect();
  const sx = W / r.width;
  const sy = H / r.height;
  return [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy];
}

// ── Node management ───────────────────────────────────────────────────────────
function addNode(x, y) {
  if (nodes.length >= MAX_NODES) return;
  const label = SHUFFLED[deckIdx % SHUFFLED.length] || 'New Prompt';
  deckIdx++;
  nodes.push({ id: nextId++, colorIdx: nextColor++, x, y, vx: 0, vy: 0, label });
  dashOffsets.push(0);
}

function startEdit(nodeId, clientX, clientY) {
  const n = nodes.find(n => n.id === nodeId);
  if (!n) return;
  labelEdit.value          = n.label;
  labelEdit.dataset.nodeId = String(nodeId);
  const r = wrap.getBoundingClientRect();
  labelEdit.style.left    = Math.max(4, clientX - r.left - 90) + 'px';
  labelEdit.style.top     = Math.max(4, clientY - r.top  - 34) + 'px';
  labelEdit.style.display = 'block';
  labelEdit.focus();
  labelEdit.select();
}

function commitEdit() {
  const id = parseInt(labelEdit.dataset.nodeId);
  const n  = nodes.find(n => n.id === id);
  if (n && labelEdit.value.trim()) n.label = labelEdit.value.trim();
  labelEdit.style.display = 'none';
}

// ── Boot ──────────────────────────────────────────────────────────────────────
function initWhenReady() {
  canvas = document.getElementById('pdj-canvas');
  if (!canvas) { setTimeout(initWhenReady, 150); return; }

  ctx       = canvas.getContext('2d');
  wrap      = document.getElementById('pdj-wrap');
  labelEdit = document.getElementById('pdj-edit');

  canvas.addEventListener('mousedown', e => {
    if (labelEdit && labelEdit.style.display !== 'none') commitEdit();
    const [mx, my] = canvasXY(e);
    const hit = hitTest(mx, my);
    if (hit) {
      drag = { type: hit.type, id: hit.id };
      selectedId = hit.type === 'node' ? hit.id : null;
      recentPos  = [{ x: mx, y: my, t: Date.now() }];
      e.preventDefault();
    } else {
      selectedId = null;
    }
  });

  window.addEventListener('mousemove', e => {
    if (!drag) return;
    const [mx, my] = canvasXY(e);
    dragMX = mx; dragMY = my;
    if (drag.type === 'listener') {
      listener.x = Math.max(LISTENER_R, Math.min(W - LISTENER_R, mx));
      listener.y = Math.max(LISTENER_R, Math.min(PLAY_H - LISTENER_R, my));
    } else {
      const n = nodes.find(n => n.id === drag.id);
      if (n) {
        n.x = Math.max(PROMPT_R, Math.min(W - PROMPT_R, mx));
        n.y = Math.max(PROMPT_R, Math.min(PLAY_H - PROMPT_R, my));
      }
    }
    recentPos.push({ x: mx, y: my, t: Date.now() });
    if (recentPos.length > 8) recentPos.shift();
  });

  window.addEventListener('mouseup', () => {
    if (!drag) return;
    if (drag.type === 'node' && overTrash(dragMX, dragMY)) {
      nodes = nodes.filter(nd => nd.id !== drag.id);
      dashOffsets = nodes.map(() => 0);
      if (selectedId === drag.id) selectedId = null;
      drag = null;
      return;
    }
    if (physicsSpeed > 0 && recentPos.length >= 2) {
      const a  = recentPos[Math.max(0, recentPos.length - 4)];
      const b  = recentPos[recentPos.length - 1];
      const dt = (b.t - a.t) / 1000;
      if (dt > 0 && dt < 0.25) {
        const vx = (b.x - a.x) / dt;
        const vy = (b.y - a.y) / dt;
        if (drag.type === 'listener') { listener.vx = vx; listener.vy = vy; }
        else { const n = nodes.find(n => n.id === drag.id); if (n) { n.vx = vx; n.vy = vy; } }
      }
    }
    drag = null;
  });

  canvas.addEventListener('dblclick', e => {
    const [mx, my] = canvasXY(e);
    const hit = hitTest(mx, my);
    if (hit?.type === 'node') startEdit(hit.id, e.clientX, e.clientY);
    else if (!hit) addNode(mx, my);
  });

  canvas.addEventListener('contextmenu', e => {
    e.preventDefault();
    const [mx, my] = canvasXY(e);
    const hit = hitTest(mx, my);
    if (hit?.type === 'node' && nodes.length > MIN_NODES) {
      const idx = nodes.findIndex(n => n.id === hit.id);
      nodes.splice(idx, 1);
      dashOffsets.splice(idx, 1);
      if (selectedId === hit.id) selectedId = null;
    }
  });

  labelEdit.addEventListener('keydown', e => {
    if (e.key === 'Enter')  { commitEdit(); e.preventDefault(); }
    if (e.key === 'Escape') { labelEdit.style.display = 'none'; }
  });
  labelEdit.addEventListener('blur', () => setTimeout(commitEdit, 80));

  const speedEl = document.getElementById('pdj-speed');
  if (speedEl) speedEl.addEventListener('input', e => {
    physicsSpeed = Math.pow(parseFloat(e.target.value), 2);
  });

  const addEl = document.getElementById('pdj-add');
  if (addEl) addEl.addEventListener('click', () => {
    const pad = 70;
    addNode(pad + Math.random() * (W - 2*pad), pad + Math.random() * (PLAY_H - 2*pad));
  });

  resize();
  buildLayout();
  const ro = new ResizeObserver(() => resize());
  ro.observe(wrap);
  animId = requestAnimationFrame(loop);
}

setTimeout(initWhenReady, 200);

})();
"""


# ── Gradio handlers ───────────────────────────────────────────────────────────

def handle_weights(bridge_json: str, transition_s: float):
    """Style timer (2Hz): read canvas weights → update RT2 style embedding."""
    if not bridge_json or not engine.is_loaded:
        return gr.skip()
    try:
        data    = json.loads(bridge_json)
        prompts = data.get("prompts", [])
        weights = data.get("weights", [])
        engine.set_style(prompts, weights, "", 0.0, transition_s)
    except Exception:
        pass
    return f"buffer {engine.buffer_s:.1f}s"


def toggle_play(playing: bool, out_device_val):
    if playing:
        engine.pause()
        engine.preferred_out_device = None
        return False, gr.Button("▶ Play", variant="primary")
    else:
        engine.preferred_out_device = out_device_val  # None = system default
        engine.play()
        return True, gr.Button("⏸ Pause", variant="secondary")


def _enable_label(source: str) -> str:
    return "🎙 Enable Mic" if "Microphone" in source else "🖥 Enable System Audio"


def toggle_mic(mic_on: bool, device_val, source: str):
    if mic_on:
        engine.stop_mic()
        return False, gr.Button(_enable_label(source), variant="primary")
    else:
        engine.start_mic(device=device_val)
        return True, gr.Button("⏹ Disable Input", variant="secondary")


def on_input_source_change(source: str, mic_on: bool):
    """Switch input device list and refresh output list to exclude the loopback device."""
    if mic_on:
        engine.stop_mic()
    engine.preferred_out_device = None

    if "Microphone" in source:
        in_choices  = _list_mic_devices()
        in_default  = None                            # system default mic
        out_choices = _list_output_devices()
        out_default = None
    else:
        in_choices = _list_loopback_devices()
        # Auto-select the first real loopback device (skip the warning hint which
        # has value=None) so the user doesn't accidentally leave it on the built-in mic.
        in_default = next((v for _, v in in_choices if v is not None), None)
        # Exclude that same device from the output list to prevent feedback.
        loopback_name = next((n for n, v in in_choices if v == in_default), None)
        out_choices = _list_output_devices(exclude_name=loopback_name)
        out_default = None

    return (
        False,
        gr.Dropdown(choices=in_choices,  value=in_default),
        gr.Dropdown(choices=out_choices, value=out_default),
        gr.Button(_enable_label(source), variant="primary"),
    )


def mic_status_tick():
    return _mic_html()


def load_model(model_size: str):
    engine.pause()
    return engine.load(model_size)


def on_page_load(model_size: str):
    status = load_model(model_size)
    return status, False, gr.Button("▶ Play", variant="primary")


# ── Build UI ──────────────────────────────────────────────────────────────────

mic_devices = _list_mic_devices()

with gr.Blocks(title="Magentic Improv") as demo:

    gr.Markdown("# 🎸 Magentic Improv")
    gr.Markdown(
        "Play or sing into your mic — pitches are detected in real time and routed as MIDI "
        "conditioning to Magenta RT2. Use the style canvas to set the musical context."
    )

    playing_state = gr.State(False)
    mic_state     = gr.State(False)

    with gr.Row():

        # ── Canvas + mic panel ────────────────────────────────────────────────
        with gr.Column(scale=4):
            canvas_html = gr.HTML(CANVAS_HTML)
            mic_display = gr.HTML(value=_mic_html(), elem_id="mic-display")
            bridge_tb   = gr.Textbox(value="", visible=False, label="bridge")

        # ── Controls ──────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=210):

            model_dd = gr.Dropdown(
                ["mrt2_small", "mrt2_base"],
                value="mrt2_small",
                label="Model",
            )
            load_status = gr.Textbox(
                value="Not loaded", label="Model status", interactive=False,
            )

            gr.Markdown("---")

            input_source = gr.Radio(
                choices=["🎙 Microphone", "🖥 System Audio"],
                value="🎙 Microphone",
                label="Input source",
                info="System Audio requires a loopback driver (e.g. BlackHole)",
            )
            mic_device_dd = gr.Dropdown(
                choices=mic_devices,
                value=None,
                label="Input device",
            )
            output_device_dd = gr.Dropdown(
                choices=_list_output_devices(),
                value=None,
                label="Output device",
                info="In System Audio mode, pick a non-loopback device to avoid feedback",
            )
            confidence_sl = gr.Slider(
                0.1, 0.9, value=0.4, step=0.05,
                label="Pitch confidence",
                info="Higher = fewer false detections",
            )

            mic_btn  = gr.Button("🎙 Enable Mic", variant="primary")
            play_btn = gr.Button("▶ Play", variant="secondary")

            gr.Markdown("---")

            volume_sl     = gr.Slider(0, 1, value=0.7, step=0.05, label="Volume")
            transition_sl = gr.Slider(
                0, 30, value=0, step=1, label="Style transition (s)",
            )
            buffer_display = gr.Textbox(
                label="Engine", value="idle", interactive=False,
            )

    # ── Timers ────────────────────────────────────────────────────────────────
    style_timer = gr.Timer(value=0.5, active=True)   # 2 Hz style update
    mic_timer   = gr.Timer(value=0.2, active=True)   # 5 Hz mic display update

    # ── Event wiring ──────────────────────────────────────────────────────────

    model_dd.change(fn=load_model, inputs=[model_dd], outputs=[load_status])

    demo.load(
        fn=on_page_load,
        inputs=[model_dd],
        outputs=[load_status, playing_state, play_btn],
    )

    play_btn.click(
        fn=toggle_play,
        inputs=[playing_state, output_device_dd],
        outputs=[playing_state, play_btn],
    )

    input_source.change(
        fn=on_input_source_change,
        inputs=[input_source, mic_state],
        outputs=[mic_state, mic_device_dd, output_device_dd, mic_btn],
    )

    mic_btn.click(
        fn=toggle_mic,
        inputs=[mic_state, mic_device_dd, input_source],
        outputs=[mic_state, mic_btn],
    )

    confidence_sl.change(
        fn=lambda v: setattr(engine, "confidence_threshold", v),
        inputs=[confidence_sl],
    )

    volume_sl.change(
        fn=lambda v: engine.set_volume(v),
        inputs=[volume_sl],
    )

    style_timer.tick(
        fn=handle_weights,
        js="(_bj, trans) => { const d = document.getElementById('pdj-data'); return [d ? d.textContent : '', trans]; }",
        inputs=[bridge_tb, transition_sl],
        outputs=[buffer_display],
    )

    mic_timer.tick(fn=mic_status_tick, outputs=[mic_display])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        head=f"<script>{CANVAS_JS}</script>",
    )
