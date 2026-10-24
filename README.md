# Breeze TTS 2 — Getting Started

A short version of the official [README](README.md), plus how to launch the local Web UI.

## Requirements

- Linux and Python 3.10 or newer
- A CUDA-capable NVIDIA GPU
- GPU memory: about 7.7 GiB for eager inference, or 14.4 GiB with `--fast-all`
- A 12 GB GPU for eager mode, or a 24 GB GPU for the fast path
- The Breeze TTS 2 checkpoint

> Use Python 3.10–3.13. The pinned PyTorch 2.9.1 build fails on Python 3.14.

## Installation

Download the inference code:

```bash
git clone https://github.com/breezeblue-ai/breeze-tts.git
cd breeze-tts
```

Create and activate a virtual environment so the dependencies stay isolated:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

All required model components are included in the Breeze TTS 2 checkpoint.

## Checkpoint

Use a local checkpoint directory, or pass the Hugging Face repo id and let the API download it into the standard cache.

```text
../breeze-tts-2
BreezeBlue/breeze-tts-2
```

## CLI

Voice design, no reference audio:

```bash
python infer.py ../breeze-tts-2 \
  --text "(sigh) Welcome aboard. Your journey begins now." \
  --instruction "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery." \
  --cfg-scale 4 \
  --output outputs/voice_design.wav
```

Voice clone, reference audio plus its exact transcript:

```bash
python infer.py ../breeze-tts-2 \
  --ref-audio reference.wav \
  --ref-text "This is the exact transcript of the reference audio." \
  --text "(sigh) It is good to hear your voice again." \
  --output outputs/voice_clone.wav
```

Voice direction, clone plus delivery instruction:

```bash
python infer.py ../breeze-tts-2 \
  --ref-audio reference.wav \
  --ref-text "This is the exact transcript of the reference audio." \
  --text "(clears throat) We need to discuss what happened last night." \
  --instruction "Speak slowly with a restrained, serious tone." \
  --cfg-scale 4 \
  --output outputs/voice_direction.wav
```

## Streaming API

```bash
python -m breeze_infer.api ../breeze-tts-2 --host 0.0.0.0 --port 7860
```

```bash
curl -X POST http://127.0.0.1:7860/v1/audio/speech \
  -F "cfg_scale=4" \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=This is the exact transcript of the reference audio." \
  -F "text=(clears throat) We need to discuss what happened last night." \
  -F "instruction=Speak slowly with a restrained, serious tone." \
  -F "seed=42" \
  --output voice_direction.pcm
```

The response is streaming mono 24 kHz signed 16-bit little-endian PCM. Start the API with `--fast-all` to enable the fast path.

To interrupt a request, `POST /v1/cancel`. The running request stops at its next chunk boundary and releases the model, so the next request can start immediately.

## Web UI

Start the API first, then run the UI in another terminal:

```bash
python webui.py --api-url http://127.0.0.1:7860
```

Open **http://127.0.0.1:7861**.

The Generate tab asks two questions: what the voice should sound like, and whether you have a sample clip.

- **Voice** — choose **Describe a voice** or **Clone from a clip**.
  - *Describe* shows one **Voice & delivery** field. Vocal events such as `(laugh)`, `(sigh)`, `(cough)`, and `(clears throat)` can be typed anywhere in the script; the Tips panel lists them.
  - *Clone* takes an uploaded or recorded clip plus its exact transcript, with an optional delivery direction.
- **Presets** — **Save preset** stores the voice source, description, delivery, reference clip, and every sampling setting in one JSON file. The script is deliberately excluded. Pick a preset from the dropdown to load it, use **Download JSON** to save a copy to your computer in one click, or use **Import JSON** to bring in a preset from somewhere else. If you have unsaved changes, switching presets asks whether to save them first.
- **Clear** — resets the script, generated audio, takes, and last-seed display without changing the current voice or sampling settings.
- **Settings** — CFG, temperature, top-p/top-k, repetition penalty, max duration, takes, seed mode, timeout, and autoplay. In random seed mode, the **Last seed** field shows the seed used by the most recent take; saving a preset records that seed.

Presets are stored as self-contained JSON files (the reference clip is embedded) in `~/.breeze2/presets/` in your home directory, outside the repository. The app creates the folder automatically and reloads it from disk, so you can drop preset JSON files in directly. Existing presets from `~/.local/share/breeze-tts/` (or from an older checkout's `.breeze2/`) are copied over automatically the first time the new folder is used. Override the root with `--data-dir` or `BREEZE_DATA_DIR`.

## Batch API

```bash
curl -X POST http://127.0.0.1:7860/v1/audio/speech/batch \
  -F 'texts=["The first sentence.", "The second sentence."]' \
  -F "instruction=A calm, clear narrator." \
  -F "cfg_scale=4" \
  -F "seed=42" \
  -F "max_new_tokens=600" \
  --output batch.pcm
```

`texts` is a JSON array of up to 128 non-empty strings. The response is the concatenated PCM of every segment; `X-Segment-Bytes` gives each segment's length.

## Notes

- The API handles one request at a time; a concurrent request returns `409`.
- `--fast-all` enables every optimized stage. Eager execution is the default and needs no extra build tools.
- Model weights and self-hosted outputs are for research and non-commercial use only. See the [README](README.md#license-and-responsible-use) for the full license.
