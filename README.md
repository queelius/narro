# Narro

Lightweight CPU text-to-speech. Wraps the [Soprano-1.1-80M](https://huggingface.co/ekwek/Soprano-1.1-80M) model for fast, expressive speech synthesis.

- Up to **20x** real-time on CPU
- **<1 GB** memory, 80M parameters
- **32kHz** audio output
- Infinite length via automatic text splitting

## Installation

```bash
pip install narro
```

Or from source:

```bash
git clone https://github.com/ekwek1/soprano.git
cd soprano
pip install -e .
```

## Usage

### CLI

```bash
narro "Hello world" -o output.wav
```

Options:

```
--output, -o              Output audio file path (default: output.wav)
--model-path, -m          Path to local model directory (optional)
--no-compile              Disable torch.compile optimization
--no-quantize             Disable INT8 quantization
--decoder-batch-size, -bs Decoder batch size (default: 4)
--num-threads, -t         Number of CPU threads for inference
```

> **Note:** The CLI reloads the model on each invocation. For repeated inference, use the Python API.

### Python API

```python
from narro import Narro

model = Narro(decoder_batch_size=4)

# Basic inference
out = model.infer("Hello world.")

# Save to file
out = model.infer("Hello world.", out_path="out.wav")

# Custom sampling parameters
out = model.infer("Hello world.", temperature=0.3, top_p=0.95, repetition_penalty=1.2)

# Batched inference
out = model.infer_batch(["Hello world."] * 10)

# Streaming inference (yields float32 tensors in [-1, 1])
for chunk in model.infer_stream("Hello world.", chunk_size=1):
    process_audio(chunk)
```

## Usage Tips

- Use double quotes instead of single quotes when quoting.
- Best results with sentences between 2 and 30 seconds long.
- Spell out numbers and special characters phonetically for best pronunciation (e.g., `1+1` -> `one plus one`).
- Unsatisfactory results? Regenerate or adjust sampling settings.

## License

Apache-2.0. See `LICENSE` for details.
