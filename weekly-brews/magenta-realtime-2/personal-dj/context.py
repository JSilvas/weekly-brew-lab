"""
Context capture for Personal DJ.

capture_screen() is adapted from dev/Noema/Prototype/app.py.
LLM client falls through LMStudio (localhost:1234, transparent LM Link support) → Ollama.

LM Link docs: requests to localhost:1234 are automatically routed to a remote device
when LM Link is configured in LM Studio — no separate URL needed.
"""

from __future__ import annotations

import base64
import io
import subprocess
import time

import httpx
import mss
from openai import OpenAI
from PIL import Image

MAX_SIDE       = 768
JPEG_QUALITY   = 80
PROBE_TIMEOUT  = 2.0   # seconds per backend probe

_sct = None   # lazy-initialised on first capture to avoid blocking at import time

def _get_sct():
    global _sct
    if _sct is None:
        _sct = mss.mss()
    return _sct

# ── Screen capture ─────────────────────────────────────────────────────────

def capture_screen() -> tuple[str, Image.Image]:
    """Grab primary monitor → base64 JPEG + PIL image."""
    sct = _get_sct()
    monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
    raw = sct.grab(monitor)
    img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64, img


def get_window_title() -> str:
    """Return frontmost app + window title via osascript (no Screen Recording needed)."""
    script = (
        'tell application "System Events"\n'
        '  set frontApp to name of first application process whose frontmost is true\n'
        '  set frontWindow to ""\n'
        '  try\n'
        '    set frontWindow to name of front window of (first process whose frontmost is true)\n'
        '  end try\n'
        '  return frontApp & " — " & frontWindow\n'
        'end tell'
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip() or "Unknown"
    except Exception:
        return "Unknown"


# ── LLM client fallback chain ──────────────────────────────────────────────

def _probe(url: str) -> bool:
    try:
        httpx.get(url, timeout=PROBE_TIMEOUT)
        return True
    except Exception:
        return False


def get_lm_client(
    lmstudio_url: str = "http://localhost:1234/v1",
    ollama_url: str = "http://localhost:11434/v1",
) -> tuple[OpenAI, str] | tuple[None, str]:
    """Try LMStudio (localhost:1234, picks up LM Link automatically) → Ollama.
    Returns (client, backend_name) or (None, error_message).
    """
    backends = [
        ("LMStudio / LM Link", lmstudio_url),
        ("Ollama local",        ollama_url),
    ]
    for name, url in backends:
        if not url or not url.strip():
            continue
        base = url.rstrip("/").removesuffix("/v1")
        if _probe(base) or _probe(url):
            return OpenAI(base_url=url, api_key="lm-studio"), name

    return None, "No LLM backend reachable — start LM Studio or Ollama"


# ── Prompt evolution ────────────────────────────────────────────────────────

EVOLVE_SYSTEM = (
    "You are a music-context assistant. "
    "Given a screenshot of what the user is working on, write a short music flavor suffix "
    "that reflects the mood/energy of their current activity. "
    "Output only the suffix — no quotes, no explanation."
)

EVOLVE_USER = (
    "Music canvas prompts: {node_prompts}\n"
    "Current focus suffix: \"{current_focus}\"\n"
    "Active window: {window_title}\n"
    "Context influence: {alpha:.2f} (0=subtle, 1=strong)\n\n"
    "Write a music flavor suffix ({length_hint}) "
    "that complements the canvas prompts and reflects the screenshot activity."
)


def evolve_focus(
    client: OpenAI,
    model: str,
    node_prompts: list[str],
    current_focus: str,
    window_title: str,
    b64_image: str,
    alpha: float,
) -> str:
    """Call the LLM with the screenshot and return an updated focus suffix."""
    length_hint = "2–3 evocative words" if alpha < 0.4 else "evocative phrase up to 80 chars"

    user_text = EVOLVE_USER.format(
        node_prompts=", ".join(f'"{p}"' for p in node_prompts),
        current_focus=current_focus or "none",
        window_title=window_title,
        alpha=alpha,
        length_hint=length_hint,
    )

    messages = [
        {"role": "system", "content": EVOLVE_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text",       "text": user_text},
                {"type": "image_url",  "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
            ],
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=60,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip().strip('"').strip("'")
