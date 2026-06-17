# Weekly Focus — Testing Magenta RealTime 2 for Local Generative Music

This week's focus is Magenta RealTime 2, testing the practicality of using it on local consumer hardware to generate music in real time.

---

## What Is It?

Magenta RealTime 2 is Google DeepMind's open-weights model for continuous, low-latency music generation steered in real time by text prompts, audio examples, or MIDI. It's the successor to Magenta RealTime and the Lyria RealTime API, now running fully on-device.

The key differentiator from every other music gen model: it generates music frame-by-frame at 25Hz (one 40ms chunk at a time), so it can be steered *while it's playing* with ~200ms latency.

- [Release Page](https://magenta.withgoogle.com/magenta-realtime-2)
- [GitHub Repo](https://github.com/magenta/magenta-realtime)

---

### Architecture

Three components under the hood:

- **SpectroStream** — a discrete audio codec (RVQ, 25Hz, 48kHz stereo) that's the input/output format for everything
- **MusicCoCa** — a contrastive model (like CLIP but for music) that embeds both text descriptions and audio clips into the same 768-dim space — this is what lets you say "heavy metal" and have it mean something to the model
- **Decoder-only Transformer LLM** — the actual generator; takes codec context tokens, style tokens from MusicCoCa, and optional MIDI, and autoregressively produces the next audio frame

The model carries previous state forward as new frames are generated. Runs on GPU via MLX on macOS Apple Silicon.

An interesting design detail: despite generating one frame at a time, both model sizes maintain a **20-second effective receptive field** through windowed attention across their layers — which is what gives it musical coherence over longer stretches.

---

### Model Sizes

| Model | Parameters | Real-time hardware |
| --- | --- | --- |
| `mrt2_small` | 230M | Any Apple Silicon Mac (including Air) |
| `mrt2_base` | 2.4B | M Pro Max or better |

---

## Training & License

- **Training data** — ~71k hours of stock instrumental music, trained on Google TPUs with JAX
- **License** — code is Apache 2.0, weights are CC-BY 4.0. Google claims no rights over outputs; you're responsible for them.
- **Vocals** — no lexical vocals by design (trained on instrumental data), though non-lexical vocal textures can appear with certain prompts

---

## Further Reading

[Live Music Models](https://arxiv.org/html/2508.04651v3) — DeepMind's 2025 research paper covering the Magenta RealTime model series (NeurIPS Creative AI 2025).
