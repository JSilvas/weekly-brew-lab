"""
Personal DJ — Gradio app.

Tab 1: Collider-style prompt node canvas → Magenta RT2 → sounddevice audio.
Tab 2: Screenshot context capture → LLM focus suffix → biases the canvas mix.

Run:  uv run python app.py  →  http://localhost:7860
macOS: grant Screen Recording and Accessibility permissions to the terminal.
"""

from __future__ import annotations

import atexit
import json
import signal
import threading
import time
from datetime import datetime

import gradio as gr

from engine import DJEngine
from context import capture_screen, get_window_title, get_lm_client, evolve_focus

# ── Shared state ──────────────────────────────────────────────────────────────

engine           = DJEngine()
DEFAULT_CTX_MODEL = "google/gemma-4-12b-qat"

def _shutdown():
    """Stop audio and generation cleanly on any exit path."""
    engine.pause()

atexit.register(_shutdown)
signal.signal(signal.SIGTERM, lambda *_: (_shutdown(), exit(0)))
_ctx_running = False
_ctx_lock    = threading.Lock()
_focus_suffix     = ""
_focus_history: list[str] = []
_capture_log: list[str]   = []

LOG_MAX = 40

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _append_log(log: str, msg: str) -> str:
    lines = log.splitlines()
    lines.append(f"[{_ts()}] {msg}")
    return "\n".join(lines[-LOG_MAX:])

# ── Canvas HTML ───────────────────────────────────────────────────────────────
# Ported from dev/magenta-realtime/examples/collider/
# Physics: throw-on-release, wall bounce, exponential speed curve (turtle→rabbit).
# Weight calc: inverse distance squared (FALLOFF=2), matching collider exactly.
# Bridge: throttled JSON → hidden Gradio textbox → Python .input() handler.

# Gradio 6: <script> tags in gr.HTML don't execute (injected via innerHTML).
# Solution: HTML structure goes in gr.HTML; JS runs via demo.load(fn=None, js=...).

CANVAS_HTML = """
<div id="pdj-outer" style="width:100%;">

<div id="pdj-wrap" style="
  position:relative;width:100%;height:480px;
  background:#09090f;border-radius:10px 10px 0 0;overflow:hidden;
  cursor:default;user-select:none;">

  <canvas id="pdj-canvas" style="position:absolute;inset:0;width:100%;height:100%;"></canvas>

  <input id="pdj-edit" type="text" autocomplete="off" spellcheck="false"
    placeholder="edit prompt…"
    style="
      position:absolute;display:none;z-index:20;
      background:#1a1a2e;color:#eee;border:1px solid #555;border-radius:6px;
      padding:4px 8px;font-size:13px;min-width:180px;outline:none;">

  <!-- weight+notes store: JS writes here; Gradio timer js= reads it -->
  <div id="pdj-data" style="display:none;"></div>
  <!-- focus bridge: Python writes here; JS polls to update focus node label -->
  <div id="pdj-focus-bridge" style="display:none;"></div>

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
    <button id="pdj-kbd-toggle" title="Toggle MIDI keyboard"
      style="
        background:rgba(255,255,255,0.1);border:none;color:#eee;
        border-radius:6px;padding:0 8px;height:28px;font-size:13px;
        cursor:pointer;line-height:1;">
      🎹
    </button>
  </div>
</div>

<!-- Collapsible keyboard section -->
<div id="pdj-keyboard-section" style="
  display:none;width:100%;height:90px;
  background:#06060e;border-radius:0 0 10px 10px;
  border-top:1px solid rgba(255,255,255,0.06);
  overflow:hidden;position:relative;">
  <div id="pdj-keyboard" style="
    position:absolute;inset:8px 10px;"></div>
</div>

</div>
"""

# JS injected via demo.load(fn=None, js=CANVAS_JS) after Gradio renders the DOM.
CANVAS_JS = r"""
(function () {

// ── Suggestions (from collider's promptSuggestion.ts) ────────────────────────
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
  "Orchestral Sustained Oboe","Nylon String Classical Guitar",
  "Deep Future Garage",
];

// Fisher-Yates shuffle
const SHUFFLED = [...ALL_SUGGESTIONS];
for (let i = SHUFFLED.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [SHUFFLED[i], SHUFFLED[j]] = [SHUFFLED[j], SHUFFLED[i]];
}

// ── Constants (matching collider) ─────────────────────────────────────────────
const PROMPT_R    = 22;
const LISTENER_R  = 26;
const FALLOFF     = 2.0;
const MAX_NODES   = 6;
const MIN_NODES   = 1;
const MAX_SPEED   = 700;     // px/s ceiling
const DAMPING     = 0.35;    // exponential decay rate per second (~5s coast)
const COLORS      = ['#ff6b6b','#48dbfb','#ffd32a','#0be881','#f8b500','#ff5e57'];
const BRIDGE_HZ   = 10;
const FOCUS_COLOR = '#a29bfe';  // lavender — visually distinct from prompt nodes
const FOCUS_ID    = 'focus';

// ── State ─────────────────────────────────────────────────────────────────────
let W = 0, H = 0, PLAY_H = 0; // PLAY_H = H minus controls bar
let nodes    = [];
let listener = {x:0, y:0, vx:0, vy:0};
let physicsSpeed = 0;      // 0..1 mapped exponentially to velocity multiplier
let drag        = null;    // {type:'node'|'listener', id?, ox, oy}
let dragMX      = 0, dragMY = 0;  // raw mouse position during drag
let selectedId  = null;
let nextId      = 1;
let nextColor   = 1;
let deckIdx     = 1;
let focusDeletedByUser = false;  // true after user explicitly trashes focus node
let dashOffsets = [];
let lastBridge  = 0;
let animId      = null;
let lastT       = null;
let recentPos   = [];      // [{x,y,t}] for throw velocity

// DOM refs — declared here so closures see them, assigned in initWhenReady
let canvas, ctx, wrap, labelEdit;

// ── Resize ────────────────────────────────────────────────────────────────────
function resize() {
  const r = wrap.getBoundingClientRect();
  W = r.width  || 600;
  H = r.height || 480;
  canvas.width  = W;
  canvas.height = H;
  PLAY_H = H - 44; // subtract controls bar
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

// ── Initial layout — single centered node ─────────────────────────────────────
function buildLayout() {
  const cx = W / 2;
  const cy = PLAY_H / 2;
  nodes = [{
    id: 0, colorIdx: 0,
    x: cx, y: cy,
    vx: 0, vy: 0,
    label: SHUFFLED[0] || 'Dreamy Ambient Pads',
  }];
  listener    = {x: cx, y: cy - 160, vx: 0, vy: 0};
  deckIdx     = 1;
  dashOffsets = [0];
}

// ── Weight calculation (identical to collider's calculateWeights) ─────────────
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

// ── Draw ──────────────────────────────────────────────────────────────────────
// ── Trash zone constants ──────────────────────────────────────────────────────
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

  // Circle background
  ctx.beginPath();
  ctx.arc(cx, cy, TRASH_R, 0, Math.PI*2);
  ctx.fillStyle = hot ? 'rgba(220,50,50,0.85)' : 'rgba(60,60,70,0.75)';
  ctx.fill();
  ctx.strokeStyle = hot ? 'rgba(255,100,100,0.9)' : 'rgba(180,180,190,0.4)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Trash icon (lid + body + lines) scaled to fit
  const s = 0.72;
  ctx.translate(cx, cy);
  ctx.scale(s, s);
  ctx.strokeStyle = hot ? '#fff' : 'rgba(220,220,230,0.9)';
  ctx.lineWidth = 1.8 / s;
  ctx.lineCap = 'round';

  // Lid
  ctx.beginPath(); ctx.moveTo(-11, -9); ctx.lineTo(11, -9); ctx.stroke();
  // Handle on lid
  ctx.beginPath(); ctx.moveTo(-5, -9); ctx.lineTo(-5, -13); ctx.lineTo(5, -13); ctx.lineTo(5, -9); ctx.stroke();
  // Body
  ctx.beginPath();
  ctx.moveTo(-9, -7); ctx.lineTo(-7, 12); ctx.lineTo(7, 12); ctx.lineTo(9, -7);
  ctx.stroke();
  // Inner lines
  [-4, 0, 4].forEach(x => {
    ctx.beginPath(); ctx.moveTo(x, -5); ctx.lineTo(x * 0.8, 10); ctx.stroke();
  });

  ctx.restore();
}

function draw(weights) {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#09090f';
  ctx.fillRect(0, 0, W, H);

  // Trash zone — visible only while dragging a node
  if (drag && drag.type === 'node') {
    const n = nodes.find(n => n.id === drag.id);
    const hot = n && overTrash(n.x, n.y);
    drawTrash(hot ? 1.0 : 0.55, hot);
  }

  // Lines: listener → each node
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

  // Prompt nodes
  nodes.forEach((n, i) => {
    const color    = n.isFocus ? FOCUS_COLOR : COLORS[n.colorIdx % COLORS.length];
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

    // Focus node: dashed outer ring to distinguish it
    if (n.isFocus) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(n.x, n.y, PROMPT_R + 7, 0, Math.PI*2);
      ctx.strokeStyle = `rgba(162,155,254,${0.35 + 0.5 * w})`;
      ctx.lineWidth   = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.restore();
    }

    if (selected) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, PROMPT_R + 5, 0, Math.PI*2);
      ctx.strokeStyle = 'rgba(255,255,255,0.75)';
      ctx.lineWidth   = 1.5;
      ctx.stroke();
    }

    // Label (above node, word-wrapped at 130px)
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

  // Listener
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

// ── Bridge: write canvas state to plain div, read by Gradio timer js= ─────────
// Gradio 6 Svelte doesn't fire .input() for programmatic textbox events, so we
// write to a plain <div> and let the gr.Timer js= parameter read it each tick.
function pushBridge(weights) {
  const el = document.getElementById('pdj-data');
  if (el) el.textContent = JSON.stringify({
    weights,
    prompts: nodes.map(n => n.label),
    notes:   [...activeNotes],
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

// ── Node add/edit (pure logic, no direct DOM — safe to define before init) ────
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

// ── Expose focus update for Python → canvas ───────────────────────────────────
window.pdjSetFocus = function(text) {
  const pill = document.getElementById('pdj-focus-pill');
  const span = document.getElementById('pdj-focus-text');
  if (!span || !pill) return;
  if (text) { span.textContent = text; pill.style.display = 'block'; }
  else       { pill.style.display = 'none'; }
};

// ── Boot — poll for canvas; head= fires before Svelte renders the components ──
// All DOM access (getElementById, addEventListener) happens here after elements exist.
function initWhenReady() {
  canvas = document.getElementById('pdj-canvas');
  if (!canvas) { setTimeout(initWhenReady, 150); return; }

  ctx      = canvas.getContext('2d');
  wrap     = document.getElementById('pdj-wrap');
  labelEdit = document.getElementById('pdj-edit');

  // ── Mouse events ────────────────────────────────────────────────────────────
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

    // Drop on trash zone — no minimum enforced, zero nodes = full context-driven mode
    if (drag.type === 'node' && overTrash(dragMX, dragMY)) {
      const trashed = nodes.find(nd => nd.id === drag.id);
      nodes = nodes.filter(nd => nd.id !== drag.id);
      dashOffsets = nodes.map(() => 0);
      if (selectedId === drag.id) selectedId = null;
      if (trashed?.isFocus) focusDeletedByUser = true;
      drag = null;
      return;
    }

    // Normal release — apply throw velocity
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
    if (hit?.type === 'node') {
      const n = nodes.find(n => n.id === hit.id);
      if (n?.isFocus || nodes.length > MIN_NODES) {
        const idx = nodes.findIndex(n => n.id === hit.id);
        nodes.splice(idx, 1);
        dashOffsets.splice(idx, 1);
        if (selectedId === hit.id) selectedId = null;
        if (n?.isFocus) focusDeletedByUser = true;
      }
    }
  });

  // ── Label edit listeners ─────────────────────────────────────────────────────
  labelEdit.addEventListener('keydown', e => {
    if (e.key === 'Enter')  { commitEdit(); e.preventDefault(); }
    if (e.key === 'Escape') { labelEdit.style.display = 'none'; }
  });
  labelEdit.addEventListener('blur', () => setTimeout(commitEdit, 80));

  // ── Controls ─────────────────────────────────────────────────────────────────
  const speedEl = document.getElementById('pdj-speed');
  if (speedEl) speedEl.addEventListener('input', e => {
    physicsSpeed = Math.pow(parseFloat(e.target.value), 2);
  });

  const addEl = document.getElementById('pdj-add');
  if (addEl) addEl.addEventListener('click', () => {
    const pad = 70;
    addNode(pad + Math.random() * (W - 2*pad), pad + Math.random() * (PLAY_H - 2*pad));
  });

  const kbdToggle = document.getElementById('pdj-kbd-toggle');
  if (kbdToggle) kbdToggle.addEventListener('click', toggleKeyboard);

  // ── Start ─────────────────────────────────────────────────────────────────────
  resize();
  buildLayout();
  const ro = new ResizeObserver(() => resize());
  ro.observe(wrap);
  animId = requestAnimationFrame(loop);
}

// ── Focus node ────────────────────────────────────────────────────────────────

function getFocusText() {
  return (document.getElementById('pdj-focus-bridge')?.textContent || '').trim();
}

function hasFocusNode() {
  return nodes.some(n => n.isFocus);
}

function addFocusNode() {
  if (hasFocusNode()) return;
  focusDeletedByUser = false;
  const label = getFocusText() || 'Context Flavor';
  nodes.push({
    id: FOCUS_ID, colorIdx: -1, isFocus: true,
    x: W / 2, y: PLAY_H / 2,
    vx: (Math.random() - 0.5) * 80,
    vy: (Math.random() - 0.5) * 80,
    label,
  });
  dashOffsets.push(0);
}

// Expose for Gradio button + debug
window.pdjAddFocusNode = addFocusNode;
window.pdjKick = (speed = 0.6) => {
  const speedEl = document.getElementById('pdj-speed');
  if (speedEl) { speedEl.value = speed; speedEl.dispatchEvent(new Event('input')); }
  const v = 250 + Math.random() * 150;
  nodes.forEach(n => { n.vx = (Math.random()-0.5)*2*v; n.vy = (Math.random()-0.5)*2*v; });
  listener.vx = (Math.random()-0.5)*2*v; listener.vy = (Math.random()-0.5)*2*v;
};

// Poll focus bridge: update focus node label when context capture produces new text,
// and auto-add the focus node on first arrival (unless user explicitly deleted it).
setInterval(() => {
  const text = getFocusText();
  if (!text) return;
  const fn = nodes.find(n => n.isFocus);
  if (fn) {
    if (fn.label !== text) fn.label = text;
  } else if (!focusDeletedByUser) {
    addFocusNode();
  }
}, 1500);

// ── MIDI keyboard ─────────────────────────────────────────────────────────────

const KB_START   = 60;   // C4
const KB_END     = 76;   // E5 (matches JAM app default)
const KB_BLACK   = new Set([1,3,6,8,10]);
const KB_GAP     = 2;    // px between white keys
const KB_ACCENT  = '#48dbfb';

// Semitone offset → QWERTY key (same mapping as JAM PianoKeyboard.tsx)
const KB_S2K = {0:'a',1:'w',2:'s',3:'e',4:'d',5:'f',6:'t',7:'g',8:'y',9:'h',10:'u',11:'j',12:'k',13:'o',14:'l',15:'p',16:';'};
const KB_K2S = Object.fromEntries(Object.entries(KB_S2K).map(([s,k])=>[k,+s]));

let activeNotes = new Set();
let kbBuilt     = false;
let kbOpen      = false;

function noteOn(note) {
  if (activeNotes.has(note)) return;
  activeNotes.add(note);
  const el = document.querySelector(`#pdj-keyboard [data-note="${note}"]`);
  if (el) el.dataset.active = '1';
  renderKeyColors();
}

function noteOff(note) {
  if (!activeNotes.has(note)) return;
  activeNotes.delete(note);
  const el = document.querySelector(`#pdj-keyboard [data-note="${note}"]`);
  if (el) delete el.dataset.active;
  renderKeyColors();
}

function renderKeyColors() {
  document.querySelectorAll('#pdj-keyboard [data-note]').forEach(el => {
    const isBlack = el.dataset.black === '1';
    const isActive = el.dataset.active === '1';
    el.style.backgroundColor = isActive ? KB_ACCENT : (isBlack ? '#111' : '#e8e8e8');
  });
}

function buildKeyboard() {
  if (kbBuilt) return;
  kbBuilt = true;
  const kbd = document.getElementById('pdj-keyboard');
  if (!kbd) return;

  // Collect white keys in order
  const whites = [];
  for (let n = KB_START; n <= KB_END; n++) {
    if (!KB_BLACK.has(n % 12)) whites.push(n);
  }
  const wCount = whites.length;

  // Style kbd as flex row for white keys
  kbd.style.display    = 'flex';
  kbd.style.gap        = KB_GAP + 'px';
  kbd.style.userSelect = 'none';
  kbd.style.touchAction= 'none';
  kbd.style.width      = '100%';
  kbd.style.height     = '100%';

  // White keys
  whites.forEach((note, wi) => {
    const offset = note - KB_START;
    const qKey   = KB_S2K[offset];
    const div    = document.createElement('div');
    div.dataset.note  = note;
    div.dataset.black = '0';
    div.style.cssText = `flex:1;height:100%;background:#e8e8e8;border-radius:0 0 5px 5px;position:relative;cursor:pointer;`;
    // QWERTY label
    if (qKey) {
      const lbl = document.createElement('span');
      lbl.textContent = qKey.toUpperCase();
      lbl.style.cssText = `position:absolute;bottom:7px;left:50%;transform:translateX(-50%);font-size:10px;color:rgba(0,0,0,0.35);pointer-events:none;`;
      div.appendChild(lbl);
    }
    kbd.appendChild(div);
  });

  // Black keys — positioned using same calc() formula as PianoKeyboard.tsx
  const wExpr = `((100% - ${(wCount-1)*KB_GAP}px) / ${wCount})`;
  for (let n = KB_START; n <= KB_END; n++) {
    if (!KB_BLACK.has(n % 12)) continue;
    const prevWi = whites.indexOf(n - 1);
    if (prevWi < 0) continue;
    const offset = n - KB_START;
    const qKey   = KB_S2K[offset];
    const leftExpr = `calc(${prevWi+1} * ${wExpr} + ${prevWi} * ${KB_GAP}px + ${KB_GAP/2}px - (${wExpr} * 0.65 / 2))`;
    const div = document.createElement('div');
    div.dataset.note  = n;
    div.dataset.black = '1';
    div.style.cssText = `position:absolute;top:0;left:${leftExpr};width:calc(${wExpr} * 0.65);height:60%;background:#111;border-radius:0 0 4px 4px;box-shadow:0 4px 8px rgba(0,0,0,0.5);z-index:2;cursor:pointer;`;
    if (qKey) {
      const lbl = document.createElement('span');
      lbl.textContent = qKey.toUpperCase();
      lbl.style.cssText = `position:absolute;bottom:6px;left:50%;transform:translateX(-50%);font-size:9px;color:rgba(255,255,255,0.45);pointer-events:none;`;
      div.appendChild(lbl);
    }
    kbd.appendChild(div);
  }

  // Pointer events — glissando support
  let heldNote = null;
  const getNote = (x, y) => {
    for (const el of document.elementsFromPoint(x, y)) {
      if (el.dataset && el.dataset.note !== undefined && el.closest('#pdj-keyboard')) return +el.dataset.note;
    }
    return null;
  };
  kbd.addEventListener('pointerdown', e => {
    e.preventDefault();
    kbd.setPointerCapture(e.pointerId);
    const note = getNote(e.clientX, e.clientY);
    if (note !== null) { heldNote = note; noteOn(note); }
  });
  kbd.addEventListener('pointermove', e => {
    if (heldNote === null) return;
    const note = getNote(e.clientX, e.clientY);
    if (note !== null && note !== heldNote) { noteOff(heldNote); heldNote = note; noteOn(note); }
  });
  kbd.addEventListener('pointerup',     () => { if (heldNote !== null) { noteOff(heldNote); heldNote = null; } });
  kbd.addEventListener('pointercancel', () => { if (heldNote !== null) { noteOff(heldNote); heldNote = null; } });
}

function toggleKeyboard() {
  kbOpen = !kbOpen;
  const section = document.getElementById('pdj-keyboard-section');
  const wrap    = document.getElementById('pdj-wrap');
  if (section) section.style.display = kbOpen ? 'block' : 'none';
  if (wrap)    wrap.style.borderRadius = kbOpen ? '10px 10px 0 0' : '10px';
  if (kbOpen) buildKeyboard();
  // QWERTY events only active while keyboard is open
  if (kbOpen) {
    window.addEventListener('keydown', onKbdDown);
    window.addEventListener('keyup',   onKbdUp);
  } else {
    window.removeEventListener('keydown', onKbdDown);
    window.removeEventListener('keyup',   onKbdUp);
    // Release all held notes on close
    [...activeNotes].forEach(noteOff);
  }
}

function onKbdDown(e) {
  if (e.repeat || e.metaKey || e.ctrlKey) return;
  const semi = KB_K2S[e.key.toLowerCase()];
  if (semi !== undefined) { e.preventDefault(); noteOn(KB_START + semi); }
}

function onKbdUp(e) {
  const semi = KB_K2S[e.key.toLowerCase()];
  if (semi !== undefined) noteOff(KB_START + semi);
}

setTimeout(initWhenReady, 200);

})(); // end IIFE
"""


# ── Gradio app ────────────────────────────────────────────────────────────────

def handle_weights(bridge_json: str, alpha: float, transition_s: float):
    """Called by the 2Hz style timer; bridge_json comes from the pdj-data div via js=."""
    if not bridge_json or not engine.is_loaded:
        return gr.skip()
    try:
        data    = json.loads(bridge_json)
        prompts = data.get("prompts", [])
        weights = data.get("weights", [])
        notes   = data.get("notes", [])
        # Focus node is now a canvas node — no separate α blending needed
        engine.set_style(prompts, weights, '', 0, transition_s)
        engine.set_notes(set(int(n) for n in notes))
    except Exception:
        pass
    return f"buffer {engine.buffer_s:.1f}s"


def toggle_play(playing_state: bool, bridge_json: str, alpha: float, transition_s: float):
    if playing_state:
        engine.pause()
        return False, gr.Button("▶ Play", variant="primary")
    else:
        if bridge_json:
            try:
                data    = json.loads(bridge_json)
                prompts = data.get("prompts", [])
                weights = data.get("weights", [])
                engine.set_style(prompts, weights, '', 0, 0)
            except Exception:
                pass
        engine.play()
        return True, gr.Button("⏸ Pause", variant="secondary")


def _lmstudio_base(lmstudio_url: str) -> str:
    return (lmstudio_url or "http://localhost:1234/v1").rstrip("/").removesuffix("/v1")


def get_lmstudio_models(lmstudio_url: str) -> list[str]:
    """Return loaded models from LM Studio using the v1 REST API."""
    import httpx
    base = _lmstudio_base(lmstudio_url)
    # LM Studio 0.4+ native API
    try:
        r = httpx.get(f"{base}/api/v1/models", timeout=3.0)
        r.raise_for_status()
        data = r.json()
        items = data if isinstance(data, list) else data.get("data", [])
        if items:
            return [m.get("id") or m.get("path", "") for m in items]
    except Exception:
        pass
    # OpenAI-compat fallback
    try:
        r = httpx.get(f"{base}/v1/models", timeout=3.0)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        return []


def load_lmstudio_model(lmstudio_url: str, model_id: str) -> str:
    """Ask LM Studio to load a model via the v1 REST API."""
    import httpx
    if not model_id:
        return "No model selected"
    base = _lmstudio_base(lmstudio_url)
    try:
        r = httpx.post(
            f"{base}/api/v1/models/load",
            json={"path": model_id},
            timeout=60.0,
        )
        if r.status_code == 200:
            return f"✓ {model_id} loaded"
        return f"⚠ {r.status_code}: {r.text[:120]}"
    except Exception as e:
        return f"✗ {e}"


def test_lm_connection(lmstudio_url: str, model_name: str, log: str) -> str:
    """Fire a minimal test prompt and log the response to validate the connection."""
    client, name = get_lm_client(lmstudio_url)
    if client is None:
        return _append_log(log, f"✗ {name}")

    # Use the selected model, fall back to DEFAULT_CTX_MODEL
    model = model_name or DEFAULT_CTX_MODEL
    log = _append_log(log, f"🔌 Testing {name} ({model})…")

    def _attempt():
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with three words that describe ambient music."}],
            max_tokens=500,
            temperature=0.7,
        )
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        if not content:
            # Surface diagnostics so we can see what the model actually returned
            nonlocal log
            log = _append_log(log, f"  finish_reason={choice.finish_reason} usage={getattr(resp, 'usage', '?')}")
        return content

    try:
        reply = _attempt()
        if not reply:
            # Model accepted request but returned empty — likely still loading
            log = _append_log(log, "⏳ Model warming up, retrying in 3s…")
            time.sleep(3)
            reply = _attempt()
        if reply:
            return _append_log(log, f"✓ Connected — \"{reply}\"")
        return _append_log(log, "⚠ Model connected but returned empty response")
    except Exception as e:
        err = str(e)
        if "No models loaded" in err:
            log = _append_log(log, f"⏳ No model loaded — triggering load of {model}…")
            load_lmstudio_model(lmstudio_url, model)
            log = _append_log(log, "Waiting 5s for model to load…")
            time.sleep(5)
            try:
                reply = _attempt()
                if reply:
                    return _append_log(log, f"✓ Connected — \"{reply}\"")
                return _append_log(log, "⚠ Model loaded but returned empty — may still be warming up")
            except Exception as e2:
                return _append_log(log, f"✗ {e2}")
        return _append_log(log, f"✗ {err}")


def load_model(model_size: str):
    engine.pause()   # stop any running audio before (re)loading
    return engine.load(model_size)


def context_tick(
    lmstudio_url: str,
    model_name: str,
    alpha: float,
    transition_s: float,
    bridge_json: str,
    focus_state: str,
    log: str,
):
    global _focus_suffix

    if not _ctx_running:
        return focus_state, log, gr.skip(), gr.skip(), gr.skip(), gr.skip()

    log = _append_log(log, "📸 Capturing screen…")

    try:
        b64, img = capture_screen()
        title    = get_window_title()
        log      = _append_log(log, f"🪟 {title}")

        client, backend = get_lm_client(lmstudio_url)
        if client is None:
            log = _append_log(log, f"⚠️  {backend}")
            return focus_state, log, img, backend, gr.skip(), gr.skip()

        prompts = []
        if bridge_json:
            try:
                prompts = json.loads(bridge_json).get("prompts", [])
            except Exception:
                pass

        # Exclude focus node's own label from node_prompts so it doesn't bias itself
        node_prompts = [p for p in prompts if p != focus_state]

        log       = _append_log(log, f"🤖 {backend} — generating focus prompt…")
        new_focus = evolve_focus(
            client, model_name, node_prompts, focus_state, title, b64, alpha
        )
        _focus_suffix = new_focus
        log = _append_log(log, f"🎵 → \"{new_focus}\"")

        # Focus node label is updated via pdj-focus-bridge; engine blends it by position
        bridge_html = f'<span id="pdj-focus-bridge" style="display:none">{new_focus}</span>'
        return new_focus, log, img, f"✓ {backend}", new_focus, bridge_html

    except Exception as e:
        log = _append_log(log, f"✗ Error: {e}")
        return focus_state, log, gr.skip(), gr.skip(), gr.skip(), gr.skip()


def toggle_context(running: bool):
    global _ctx_running
    _ctx_running = not running
    return _ctx_running, ("⏸ Stop Capture" if _ctx_running else "▶ Start Capture")


# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Personal DJ") as demo:

    gr.Markdown("# 🎧 Personal DJ")

    # Shared state
    playing_state = gr.State(False)
    ctx_running   = gr.State(False)
    focus_state   = gr.State("")

    with gr.Tabs():

        # ── Tab 1: Music Studio ───────────────────────────────────────────────
        with gr.Tab("🎵 Music Studio"):
            with gr.Row():
                # Canvas column
                with gr.Column(scale=4):
                    canvas_html = gr.HTML(CANVAS_HTML)
                    with gr.Row():
                        focus_display = gr.Textbox(
                            label="🎯 Context Flavor",
                            interactive=False,
                            placeholder="Contextual flavor will appear here once capture is running…",
                            scale=4,
                        )
                        add_focus_btn = gr.Button("+ Add to canvas", size="sm", scale=1, min_width=120)
                    # Hidden bridges
                    bridge_tb    = gr.Textbox(value="", visible=False, elem_id="dj_weight_bridge", label="bridge")
                    focus_bridge = gr.HTML(value="", visible=False, elem_id="pdj-focus-bridge-outer")

                # Controls column
                with gr.Column(scale=1, min_width=200):
                    model_dd    = gr.Dropdown(
                        ["mrt2_small", "mrt2_base"],
                        value="mrt2_small",
                        label="Model",
                    )
                    load_status = gr.Textbox(
                        value="Not loaded", label="Model status", interactive=False,
                    )
                    play_btn    = gr.Button("▶ Play", variant="primary")
                    volume_sl   = gr.Slider(0, 1, value=0.7, step=0.05, label="Volume")
                    transition_sl = gr.Slider(
                        0, 60, value=20, step=1, label="Transition Duration (s)",
                        info="Ramp time when focus node embedding updates",
                    )
                    buffer_display = gr.Textbox(
                        label="Engine", value="idle", interactive=False,
                    )
                    alpha_sl = gr.Slider(0, 1, value=0, visible=False)  # kept for wiring only

        # ── Tab 2: Context Capture ────────────────────────────────────────────
        with gr.Tab("🔍 Context Capture"):

            # ── Configuration — full width, collapses on Start ────────────────
            with gr.Accordion("⚙️ Configuration", open=True) as config_accordion:
                gr.Markdown(
                    "_[LM Link](https://lmstudio.ai/docs/lmlink) routes `localhost:1234` "
                    "to your remote device automatically when configured._"
                )
                lmstudio_tb = gr.Textbox(
                    label="LMStudio / LM Link",
                    value="http://localhost:1234/v1",
                )
                with gr.Row():
                    model_dd_ctx = gr.Dropdown(
                        label="Model",
                        value=None,
                        allow_custom_value=True,
                        scale=3,
                        info="Use 🔄 to detect from LM Studio",
                    )
                    lmstudio_refresh_btn = gr.Button("🔄", scale=0, min_width=48)

            ctx_btn      = gr.Button("▶ Start Capture", variant="primary")
            ctx_stop_btn = gr.Button("⏸ Stop Capture", variant="secondary", visible=False)

            # ── Two-column live section ───────────────────────────────────────
            with gr.Row():

                # Left — Context
                with gr.Column(scale=1):
                    capture_log = gr.Textbox(
                        label="Activity Log", lines=12, max_lines=12,
                        interactive=False, value="",
                    )
                    focus_hist  = gr.Textbox(
                        label="Music Flavor Prompt",
                        interactive=False,
                        placeholder="(awaiting first capture)",
                        lines=3,
                    )

                # Right — Capture
                with gr.Column(scale=1):
                    cadence_sl = gr.Slider(
                        5, 120, value=30, step=5, label="Capture Cadence (s)",
                    )
                    screen_img = gr.Image(
                        label="Last Capture", interactive=False,
                    )

            model_tb   = model_dd_ctx
            ctx_status = gr.Textbox(visible=False)  # wiring only
            capture_timer = gr.Timer(value=30, active=False)

        # Style update timer — 2Hz, always active.
        # Replaces bridge_tb.input() which doesn't reliably fire for programmatic
        # DOM events in Gradio 6's Svelte runtime.
        style_timer = gr.Timer(value=0.5, active=True)

    # ── Event wiring ──────────────────────────────────────────────────────────

    model_dd.change(
        fn=load_model,
        inputs=[model_dd],
        outputs=[load_status],
    )

    def on_page_load(model_size: str, lmstudio_url: str):
        """Stop audio, reload Magenta model, ping LM Studio to load the vision
        model, and update the model dropdown if LM Studio has something loaded."""
        status = load_model(model_size)

        # Check what's actually loaded in LM Studio right now
        loaded_models = get_lmstudio_models(lmstudio_url)
        if loaded_models:
            # LM Studio is up and has a model — show what's actually there
            ctx_model = gr.Dropdown(choices=loaded_models, value=loaded_models[0])
        else:
            # Not reachable or nothing loaded — ping in background, keep default
            def _ping():
                load_lmstudio_model(lmstudio_url, DEFAULT_CTX_MODEL)
            threading.Thread(target=_ping, daemon=True).start()
            ctx_model = gr.skip()

        return status, False, gr.Button("▶ Play", variant="primary"), ctx_model

    demo.load(
        fn=on_page_load,
        inputs=[model_dd, lmstudio_tb],
        outputs=[load_status, playing_state, play_btn, model_dd_ctx],
    )

    play_btn.click(
        fn=toggle_play,
        js="(playing, _bj, alpha, trans) => { const d = document.getElementById('pdj-data'); return [playing, d ? d.textContent : '', alpha, trans]; }",
        inputs=[playing_state, bridge_tb, alpha_sl, transition_sl],
        outputs=[playing_state, play_btn],
    )

    volume_sl.change(
        fn=lambda v: engine.set_volume(v),
        inputs=[volume_sl],
    )

    # Style timer (2Hz): js= reads pdj-data div client-side, passes to Python.
    # This bypasses the textbox bridge entirely — Gradio 6 Svelte doesn't update
    # component state from programmatic DOM events on form elements.
    style_timer.tick(
        fn=handle_weights,
        js="(_bj, alpha, trans) => { const d = document.getElementById('pdj-data'); return [d ? d.textContent : '', alpha, trans]; }",
        inputs=[bridge_tb, alpha_sl, transition_sl],
        outputs=[buffer_display],
    )

    # Context capture toggle
    def _start_ctx(running, cadence):
        new_run, _ = toggle_context(running)
        return (
            new_run,
            gr.Timer(value=cadence, active=new_run),
            gr.Accordion(open=False),   # collapse config on start
            gr.Button(visible=False),   # hide start btn
            gr.Button(visible=True),    # show stop btn
        )

    def _stop_ctx(running, cadence):
        new_run, _ = toggle_context(running)
        return (
            new_run,
            gr.Timer(value=cadence, active=new_run),
            gr.Accordion(open=True),    # re-expand config on stop
            gr.Button(visible=True),    # show start btn
            gr.Button(visible=False),   # hide stop btn
        )

    ctx_btn.click(
        fn=_start_ctx,
        inputs=[ctx_running, cadence_sl],
        outputs=[ctx_running, capture_timer, config_accordion, ctx_btn, ctx_stop_btn],
    ).then(
        fn=context_tick,
        inputs=[lmstudio_tb, model_tb, alpha_sl, transition_sl, bridge_tb, focus_state, capture_log],
        outputs=[focus_state, capture_log, screen_img, ctx_status, focus_display, focus_bridge],
    )
    ctx_stop_btn.click(
        fn=_stop_ctx,
        inputs=[ctx_running, cadence_sl],
        outputs=[ctx_running, capture_timer, config_accordion, ctx_btn, ctx_stop_btn],
    )

    def refresh_lmstudio_models(url: str):
        models = get_lmstudio_models(url)
        return gr.Dropdown(choices=models, value=models[0] if models else None)

    lmstudio_refresh_btn.click(
        fn=refresh_lmstudio_models,
        inputs=[lmstudio_tb],
        outputs=[model_dd_ctx],
    ).then(
        fn=test_lm_connection,
        inputs=[lmstudio_tb, model_dd_ctx, capture_log],
        outputs=[capture_log],
    )

    model_dd_ctx.change(
        fn=load_lmstudio_model,
        inputs=[lmstudio_tb, model_dd_ctx],
        outputs=[ctx_status],
    )

    # Capture timer tick
    capture_timer.tick(
        fn=context_tick,
        inputs=[
            lmstudio_tb, model_tb,
            alpha_sl, transition_sl,
            bridge_tb, focus_state, capture_log,
        ],
        outputs=[focus_state, capture_log, screen_img, ctx_status, focus_display, focus_bridge],
    )

    # Mirror focus_state → Tab 2 history display
    focus_state.change(
        fn=lambda f: f,
        inputs=[focus_state],
        outputs=[focus_hist],
    )

    add_focus_btn.click(
        fn=None,
        js="() => { window.pdjAddFocusNode && window.pdjAddFocusNode(); }",
    )

    # Gradio 6: inject canvas JS via head= (executes on page load; IIFE polls for canvas element)
    # demo.load(fn=None, js=...) stalls startup-events in Gradio 6, so we use head= instead.

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        head=f"<script>{CANVAS_JS}</script>",
    )
