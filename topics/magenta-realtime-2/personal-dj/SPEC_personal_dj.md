# Personal DJ — Prototype Spec

**Tagline:** Realtime generative music that scores your day based on what you're actually doing.

---

## Vision

Build a Magenta RealTime 2 inference-powered music generation app that captures the focal screen context of what you're doing at any point in time (macOS window focus), and integrates that context into a continually evolving composite live music prompt powering your personal soundtrack — your Magenta DJ.

Spotify has a personalized AI DJ selecting real tracks based on algorithmic recommendation. LaurieWired built a version of contextual radio a couple years back. But realtime music gen has come pretty far since. This is a fresh crack at generating realtime background music and ambiance based on your initially prompted musical flavors, evolving from there based on regular context captures of what you're focused on at any point in time.

---

## Reference Sources

| What | Where | What to Borrow |
|------|-------|----------------|
| Screen capture | `dev/Noema/Prototype/app.py` | `capture_screen()` with `mss` (lines 230–252), `gr.Timer` tick pattern |
| Music gen engine | `dev/magenta-realtime-2/test_generate.py` | `MagentaRT2Mlxfn` init, `embed_style()`, stateful `generate()` |
| Prompt node canvas | `dev/magenta-realtime/examples/collider/` | Full collider UX: draggable nodes + listener, physics, inverse-distance weight blending |

---

## Known Performance Constraint

`mrt2_small` via the Python `MagentaRT2Mlxfn` path runs at approximately **0.9x realtime** on the current machine — slightly below the threshold needed for gapless playback. Buffer underruns occur periodically. The C++ `RealtimeRunner` used by the Collider app runs faster (native, no Python per-frame overhead) but requires more integration work. This is an open issue for the next iteration; the prototype is functional with occasional gaps.

---

## Architecture

### Two UI Tabs

**Tab 1 — Music Studio** (default focus): prompt node canvas drives continuous Magenta RT2 generation; LLM focus context rides as a background bias.

**Tab 2 — Context Capture**: periodic screenshot → LMStudio/LM Link multimodal LLM → focus suffix that biases the canvas mix.

### Embedding Model

The canvas has N prompt nodes (3 default, up to 6). A listener node sits on the canvas; its distance to each prompt determines blend weights via inverse-distance weighting (FALLOFF = 2, matching collider exactly):

```
w_i = (1 / d_i²) / Σ(1 / d_j²)       # listener→node distances

canvas_embed = Σ w_i · embed_style(prompt_i)   # cached per node text
focus_embed  = embed_style(focus_suffix)         # from LLM, Tab 2

final_embed  = normalize(canvas_embed + α · focus_embed)
```

`α` is the **Context Influence** slider (0–1). At 0 the focus context has no effect; at 1 it contributes equally to the full canvas mix. Canvas node prompts are never mutated by the LLM.

**Known gap:** the normalization in `engine.py:compute_embed` currently adds `alpha * focus_embed` directly without re-normalizing as a separate weighted term. To fix: compute `normalize(canvas_embed + α · focus_embed)` explicitly after adding the focus component.

**Embedding cache:** `embed_style()` uses TFLite (CPU) and is safe to call from any thread. Results cached by prompt text string in a plain dict; recomputed only when node text changes.

**Transition smoothing:** when `focus_suffix` updates, lerp `final_embed` from old to new over `transition_duration` seconds in a dedicated ramp thread at 20Hz. The canvas weights update continuously as the listener moves; only the focus component triggers a ramp.

### Audio Playback

`MagentaRT2Mlxfn` generates chunks into a ring buffer. `sounddevice` drains it via a callback. Gradio owns controls only — no Gradio audio streaming component.

**Critical:** MLX GPU streams are thread-local. Both `load()` and `generate()` must run on the same thread. Implemented via `ThreadPoolExecutor(max_workers=1)` — all MLX work is submitted to this single worker thread.

### Threads

```
ThreadPoolExecutor (1 worker — the MLX thread)
├── load()      — MagentaRT2Mlxfn init
└── _gen_loop() — generate(style=final_embed, frames=50) → ring buffer

Main (Gradio)
├── style_timer (2Hz)  — reads pdj-data div via js=, calls engine.set_style()
├── ramp_thread        — lerps focus_embed old→new over transition_duration
├── sounddevice stream — callback drains ring buffer → speakers
└── capture_timer      — every N seconds: screenshot → LLM → focus_suffix
```

---

## Tab 1: Music Studio

### Prompt Node Canvas

Implemented as a `gr.HTML` block containing a Canvas2D element with vanilla JS. Script injected via `launch(head=...)` since Gradio 6 does not execute `<script>` tags inserted via `innerHTML`. The IIFE polls for `#pdj-canvas` until Svelte renders the components.

**Canvas behavior:**
- 1 default prompt node, centered on the canvas — user builds up from there
- Each node is a colored circle; double-click to edit label inline
- Right-click a node to remove it (minimum 2 nodes enforced)
- Double-click empty canvas to add a node (maximum 6)
- Listener is a white circle; drag to blend between prompts
- Dashed lines from listener to each node, thickness and dash speed proportional to weight
- **Full collider physics shipped in v0:** throw-on-release, wall bounce, exponential speed curve (🐢/🐇 slider), per-ball velocity and damping

**Weight → Python bridge:**

Gradio 6 Svelte does not fire `.input()` events for programmatic DOM writes on form elements. Instead, canvas JS writes weights to a plain `<div id="pdj-data">`:

```js
function pushBridge(weights) {
  const el = document.getElementById('pdj-data');
  if (el) el.textContent = JSON.stringify({ weights, prompts: nodes.map(n => n.label) });
}
```

A `gr.Timer(value=0.5)` ticks every 500ms; its `js=` parameter reads `pdj-data.textContent` client-side before calling Python `handle_weights()`. This gives ~500ms style update latency when dragging the listener — acceptable for musical blending.

**Default prompt nodes (shuffled from collider's suggestion list):**  
Picked randomly from `ALL_SUGGESTIONS` in `promptSuggestion.ts` on each page load.

### Controls

- **Context influence α** (slider 0–1) — how much the LLM focus bias colors the mix. Default 0.3.
- **Transition duration** (slider 0–60s) — ramp time when focus suffix updates. Default 20s.
- **Model size** (dropdown) — `mrt2_small` / `mrt2_base`. Default `mrt2_small`.
- **Play / Pause** (button) — toggles label between ▶ Play and ⏸ Pause on the button itself.
- **Volume** (slider)

### Displays

- **Canvas** — the prompt node / listener visual
- **Focus suffix** (read-only) — current LLM-generated bias; updated by Tab 2
- **Engine** — live buffer level (`buffer X.Xs`) or `idle`
- **Model status** — load result

### Open display gaps

- **Active style readout** (combined prompt string) — not yet shown; useful for debugging what the engine is actually receiving
- **Transition progress** ("Transitioning Xs / Ys") — not yet shown

---

## Tab 2: Context Capture

### Config

- **LMStudio / LM Link** (text) — default `http://localhost:1234/v1`. LM Link routes this transparently to a remote device when configured in LM Studio — no separate URL needed. See [LM Link docs](https://lmstudio.ai/docs/lmlink).
- **Ollama local** (text) — default `http://localhost:11434/v1`. Fallback if LMStudio is unreachable.
- **Model** (dropdown + 🔄 Ollama refresh) — type any model name or click refresh to populate from Ollama's `/api/tags`. Default `llama-3.2-vision-instruct`.
- **Capture cadence** (slider, 15–120s) — default 30s.
- **Start / Stop** toggle.

### LLM Backend

LM Link is transparent through `localhost:1234` — no distinct URL is needed. The fallback chain is:

```
LMStudio (localhost:1234)  ← picks up LM Link automatically if configured
        ↓ (if unreachable)
Ollama  (localhost:11434)
```

Both are OpenAI-compatible (`/v1/chat/completions`). The active backend is probed on each tick and shown in the status display; falls through automatically if a backend goes down.

### Screen Capture

`capture_screen()` from Noema — `mss` grabs primary monitor, resizes to 768px max side, encodes as base64 JPEG. Window title via `osascript` (Accessibility permission, no Screen Recording required) is included as supplementary text alongside the image.

### Prompt Evolution

```
System: You are a music-context assistant. Given a screenshot, write a short
        music flavor suffix reflecting the mood/energy of the current activity.
        Output only the suffix — no quotes, no explanation.

User:   Music canvas prompts: {node_prompts}
        Current focus suffix: "{current_focus}"
        Active window: {window_title}
        Context influence: {alpha} (0=subtle, 1=strong)

        Write a music flavor suffix ({length_hint}) that complements the canvas
        prompts and reflects the screenshot activity.
```

`length_hint` = "2–3 evocative words" when α < 0.4, "evocative phrase up to 80 chars" otherwise.

### Displays

- **Active backend** — `✓ LMStudio / LM Link` or error
- **Last capture** — screenshot thumbnail (200px)
- **Latest music flavor prompt** — the focus suffix the LLM most recently generated
- **Capture log** — rolling timestamped steps: 📸 capture → 🪟 window → 🤖 LLM call → 🎵 result

**Open display gap:** "Current combined style" (mirroring Tab 1's active style) not yet shown on Tab 2.

---

## Data / Prompt Flow

```
[Canvas] N prompt nodes + listener position
              │
    inverse-distance weights w_i (FALLOFF=2)
              │
    canvas_embed = Σ w_i · embed_style(prompt_i)   ← TFLite, cached
              │
    final_embed = normalize(canvas_embed + α · focus_embed)
              │
    [MLX thread] generate(style=final_embed, frames=50)
              │
          ring buffer (AudioRingBuffer, max 16s)
              │
      sounddevice callback → speakers

[Tab 2 loop, every 30s]
  mss screenshot + osascript window title
              │
  LMStudio/LM Link → Ollama (first reachable)
              │
  focus_suffix (≤80 chars)
              │
  embed_style(focus_suffix) → ramp thread lerps → final_embed
```

---

## File Structure

```
personal-DJ/
├── SPEC_personal_dj.md
├── app.py          # Gradio UI: canvas HTML/JS, tab wiring, timer ticks, bridge
├── engine.py       # DJEngine: ThreadPoolExecutor MLX thread, ring buffer, sounddevice, ramp
├── context.py      # capture_screen(), get_window_title(), get_lm_client(), evolve_focus()
└── pyproject.toml
```

**Key deps:**

```toml
[project]
dependencies = [
    "gradio>=6.0",
    "magenta-rt[mlx]",
    "openai>=1.0",
    "httpx",
    "sounddevice",
    "mss",
    "pillow",
    "numpy",
]

[tool.uv.sources]
magenta-rt = { path = "../../../../magenta-realtime", editable = true }
```

---

## Gradio 6 Wiring Notes

- Canvas JS in `launch(head=...)` — `gr.HTML` `<script>` tags don't execute in Gradio 6
- IIFE polls for `#pdj-canvas` via `setTimeout(initWhenReady, 200)` — `head=` fires before Svelte renders components
- Weight bridge: `<div id="pdj-data">` written by JS, read by `style_timer.tick(js=...)` — programmatic `.input()` events on Svelte textboxes don't trigger Python handlers
- `gr.Timer(value=0.5, active=True)` for style updates; `gr.Timer(value=N, active=False)` for capture cadence (activated on Start)
- `play_btn.click(js=...)` reads `pdj-data.textContent` at click time to seed initial embedding immediately

---

## Scope Boundaries

- No MIDI input in v0
- No persistent storage (no RAG, no compaction from Noema)
- LM Link transparent through `localhost:1234`; no separate LM-Link URL field
- Canvas physics **shipped in v0** (throw, bounce, speed slider) — full collider UX
- C++ RealtimeRunner not integrated — all inference via Python `MagentaRT2Mlxfn`

## Known Issues / Next Iteration

- **Buffer underruns** (~0.9x realtime on current hardware via Python MLX path). Options: integrate C++ RealtimeRunner, tune chunk size, or pre-buffer more aggressively before starting playback.
- **Embedding normalization gap** — `normalize(canvas_embed + α·focus_embed)` should be applied after the full expression, not just to `canvas_embed`. Minor at low α values.
- **Tab 2 combined style display** — mirror of active combined prompt not yet shown.
- **Tab 1 active style / transition progress** — readouts specced but not yet implemented.
