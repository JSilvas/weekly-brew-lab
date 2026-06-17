# Weekly Focus - Testing Magenta Realtime 2 for local generative music
This week's focused on the Magenta Realtime 2 model, and testing out the practicality of using it on local consumer hardware to generate music in realtime.

### Tell Me More?
Magenta RealTime 2 is Google DeepMind's open-weights model for
continuous, low-latency music generation steered in real time by text
prompts, audio examples, or MIDI. It's the successor to Magenta RealTime
and the Lyria RealTime API, now running fully on-device. 
The core difference from every other music gen model: it generates music
frame-by-frame at 25Hz (one 40ms chunk at a time), so it can be steered
while it's playing with ~200ms latency.


Release Page: https://magenta.withgoogle.com/magenta-realtime-2
Github Repo: https://github.com/magenta/magenta-realtime


### Architectural Notes
- streaming model that generates 25 frames of audio per second

  - SpectroStream — a discrete audio codec (RVQ, 25Hz, 48kHz stereo) that's
   the input/output format for everything
  - MusicCoCa — a contrastive model (like CLIP but for music) that embeds
  both text descriptions and audio clips into the same 768-dim space — this
   is what lets you say "heavy metal" and have it mean something to the
  model
  - Decoder-only Transformer LLM — the actual generator; takes codec
  context tokens, style tokens from MusicCoCa, and optional MIDI, and
  autoregressively produces the next audio frame


- carries the previous state forward as new frames are generated
- Runs on GPU (MLX on MacOS M series hardware)

### Further Reading
Intersting 2025 Research Paper released by DeepMind with the first Magenta Realtime model series: https://arxiv.org/html/2508.04651v3



  Three components under the hood
  - SpectroStream — a discrete audio codec (RVQ, 25Hz, 48kHz stereo) that's
   the input/output format for everything
  - MusicCoCa — a contrastive model (like CLIP but for music) that embeds
  both text descriptions and audio clips into the same 768-dim space — this
   is what lets you say "heavy metal" and have it mean something to the
  model
  - Decoder-only Transformer LLM — the actual generator; takes codec
  context tokens, style tokens from MusicCoCa, and optional MIDI, and
  autoregressively produces the next audio frame
  
  Two sizes
  - mrt2_small — 230M params, runs in real time on any Apple Silicon Mac
  (including Air)
  - mrt2_base — 2.4B params, higher quality, needs M Pro Max or better for
  real time
  
  Training data — ~71k hours of stock instrumental music, trained on Google
   TPUs with JAX

  License — code is Apache 2.0, weights are CC-BY 4.0. Google claims no
  rights over your outputs, but you're responsible for them. No lexical
  vocals by design (it trained on instrumental data), though non-lexical
  vocal textures can appear with certain prompts.

  The interesting design choice for your audience: the 20-second effective
  receptive field (even though it generates frame by frame) — both model
  sizes achieve this through windowed attention across their layers, which
  is what gives it musical coherence over longer stretches despite
  streaming one frame at a time.