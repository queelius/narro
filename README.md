<!-- Version 0.1.0 -->
<div align="center">
  
  # Soprano: Instant, Ultra‑Realistic Text‑to‑Speech

  [![Alt Text](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/ekwek/Soprano-1.1-80M)
  [![Alt Text](https://img.shields.io/badge/HuggingFace-Demo-yellow?logo=huggingface)](https://huggingface.co/spaces/ekwek/Soprano-TTS)
  
  <img width="640" height="320" alt="soprano-github" src="https://github.com/user-attachments/assets/4d612eac-23b8-44e6-8c59-d7ac14ebafd1" />
</div>

### 📰 News
**2026.01.14 - [Soprano-1.1-80M](https://huggingface.co/ekwek/Soprano-1.1-80M) released! 95% fewer hallucinations and a 63% preference rate over Soprano-80M.**  
2026.01.13 - [Soprano-Factory](https://github.com/ekwek1/soprano-factory) released! You can now train/fine-tune your own Soprano models.  
2025.12.22 - Soprano-80M released! [Model](https://huggingface.co/ekwek/Soprano-80M) | [Demo](https://huggingface.co/spaces/ekwek/Soprano-TTS)

---

## Overview

**Soprano** is an ultra‑lightweight, on-device text‑to‑speech (TTS) model designed for expressive, high‑fidelity speech synthesis at unprecedented speed. Soprano was designed with the following features:
- Up to **20x** real-time generation on CPU
- **<1 GB** memory usage with a compact 80M parameter architecture
- **Infinite generation length** with automatic text splitting
- Highly expressive, crystal clear audio generation at **32kHz**
- Minimal dependencies — lightweight core inference library

https://github.com/user-attachments/assets/525cf529-e79e-4368-809f-6be620852826

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [CLI](#cli)
  - [Python script](#python-script)
- [Usage tips](#usage-tips)
- [Roadmap](#roadmap)

## Installation

```bash
pip install soprano-tts
```

Or install from source:

```bash
git clone https://github.com/ekwek1/soprano.git
cd soprano
pip install -e .
```

---

## Usage

### CLI

```
soprano "Soprano is an extremely lightweight text to speech model."

optional arguments:
  --output, -o                  Output audio file path. Defaults to 'output.wav'
  --model-path, -m              Path to local model directory (optional)
  --no-compile                  Disable torch.compile optimization
  --no-quantize                 Disable INT8 quantization
  --decoder-batch-size, -bs     Decoder batch size. Defaults to 4
  --num-threads, -t             Number of CPU threads for inference
```

> **Note:** The CLI reloads the model every time it is called. For repeated inference, use the Python API directly.

### Python script

```python
from soprano import SopranoTTS

model = SopranoTTS(decoder_batch_size=4)

# Basic inference
out = model.infer("Soprano is an extremely lightweight text to speech model.")

# Save output to a file
out = model.infer("Soprano is an extremely lightweight text to speech model.", out_path="out.wav")

# Custom sampling parameters
out = model.infer(
    "Soprano is an extremely lightweight text to speech model.",
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.2,
)

# Batched inference
out = model.infer_batch(["Soprano is an extremely lightweight text to speech model."] * 10)

# Save batch outputs to a directory
out = model.infer_batch(["Soprano is an extremely lightweight text to speech model."] * 10, out_dir="/dir")

# Streaming inference (yields audio tensors)
for chunk in model.infer_stream("Soprano is an extremely lightweight text to speech model.", chunk_size=1):
    # chunk is a float32 tensor in [-1, 1] range
    process_audio(chunk)
```

### 3rd-party tools

#### ComfyUI Nodes

https://github.com/jo-nike/ComfyUI-SopranoTTS

https://github.com/SanDiegoDude/ComfyUI-Soprano-TTS

## Usage tips:

* When quoting, use double quotes instead of single quotes.
* Soprano works best when each sentence is between 2 and 30 seconds long.
* Although Soprano recognizes numbers and some special characters, it occasionally mispronounces them. Best results can be achieved by converting these into their phonetic form. (1+1 -> one plus one, etc)
* If Soprano produces unsatisfactory results, you can easily regenerate it for a new, potentially better generation. You may also change the sampling settings for more varied results.

---

## Roadmap

* [x] Add model and inference code
* [x] Seamless streaming
* [x] Batched inference
* [x] Command-line interface (CLI)
* [x] CPU support
* [ ] Additional LLM backends
* [ ] Voice cloning
* [ ] Multilingual support

---

## Limitations

Soprano is currently English-only and does not support voice cloning. In addition, Soprano was trained on only 1,000 hours of audio (~100x less than other TTS models), so mispronunciation of uncommon words may occur. This is expected to diminish as Soprano is trained on more data.

---

## Acknowledgements

Soprano uses and/or is inspired by the following projects:

* [Vocos](https://github.com/gemelo-ai/vocos)
* [XTTS](https://github.com/coqui-ai/TTS)

---

## License

This project is licensed under the **Apache-2.0** license. See `LICENSE` for details.
