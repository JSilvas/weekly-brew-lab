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

def get_lm_client(
    lmstudio_url: str = "http://localhost:1234/v1",
) -> tuple[OpenAI, str] | tuple[None, str]:
    """Return an OpenAI-compatible client pointed at LMStudio / LM Link.

    No probe — just hand back the client and let the actual API call surface
    any connection error in the Activity Log where it's visible.
    """
    url = lmstudio_url.strip()
    if not url:
        return None, "No LMStudio URL configured"
    return OpenAI(base_url=url, api_key="lm-studio"), "LMStudio / LM Link"


# ── Prompt evolution ────────────────────────────────────────────────────────

# ── Suffix mode (canvas nodes present) ────────────────────────────────────────
# LLM writes a short flavour addition that blends with the existing node prompts.

SUFFIX_SYSTEM = (
    "You are a background music DJ. "
    "Given a screenshot of what the user is working on, write a short thematically appropriate music prompt suffix"
    "that would soundtrack their current activity. "
    "Output only the suffix — no quotes, no explanation."
)

SUFFIX_USER = (
    "Music canvas prompts: {node_prompts}\n"
    "Current focus suffix: \"{current_focus}\"\n"
    "Active window: {window_title}\n"
    "Context influence: {alpha:.2f} (0=subtle, 1=strong)\n\n"
    "Write a music flavor suffix ({length_hint}) "
    "that complements the canvas prompts and reflects the screenshot activity."
)

# ── Full-drive mode (no canvas nodes) ─────────────────────────────────────────
# No user prompts to anchor to — LLM authors the entire music style description.

FULL_SYSTEM = (
    "You are a music director scoring a live scene. "
    "Given a screenshot of what the user is working on, write a short thematically appropriate music prompt"
    "that would soundtrack their current activity. "
    "Be specific: include genre, tempo feel, instrumentation, and emotional tone. "
    "Output only the style description — no quotes, no explanation."
)

FULL_USER = (
    "Active window: {window_title}\n\n"
    "Looking at this screenshot, write a complete music style description (up to 120 chars) "
    "that would score what this person is focused on right now. "
    "Consider their activity's energy, concentration level, and emotional context."
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
    """Call the LLM with the screenshot and return a music style string.

    When node_prompts is empty the canvas has been cleared — the LLM takes full
    authorship and returns a complete style description rather than a suffix.
    """
    if node_prompts:
        length_hint = "2–3 evocative words" if alpha < 0.4 else "evocative phrase up to 80 chars"
        system    = SUFFIX_SYSTEM
        user_text = SUFFIX_USER.format(
            node_prompts=", ".join(f'"{p}"' for p in node_prompts),
            current_focus=current_focus or "none",
            window_title=window_title,
            alpha=alpha,
            length_hint=length_hint,
        )
        max_tokens = 500
    else:
        system    = FULL_SYSTEM
        user_text = FULL_USER.format(window_title=window_title)
        max_tokens = 500

    _model = model or "local-model"
    _kwargs = dict(model=_model, max_tokens=max_tokens, temperature=0.7)

    def _call(with_image: bool) -> str:
        user_content = (
            [
                {"type": "text",      "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
            ]
            if with_image
            else user_text
        )
        resp = client.chat.completions.create(
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": user_content},
            ],
            **_kwargs,
        )
        return (resp.choices[0].message.content or "").strip().strip('"').strip("'")

    result = _call(with_image=True)
    if not result:
        # Model didn't return content with image — rebuild user_text with more
        # explicit context for text-only fallback so the model doesn't ask for a screenshot
        if node_prompts:
            user_text = SUFFIX_USER.format(
                node_prompts=", ".join(f'"{p}"' for p in node_prompts),
                current_focus=current_focus or "none",
                window_title=window_title,
                alpha=alpha,
                length_hint=length_hint,
            ) + f"\n(No screenshot available — use window title '{window_title}' as context.)"
        else:
            user_text = FULL_USER.format(window_title=window_title) + \
                f"\n(No screenshot available — base your response on the window title '{window_title}' alone.)"
        result = _call(with_image=False)
    return result
