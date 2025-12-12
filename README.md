# Soprano: High Efficiency, High Fidelity Text-to-Speech

## Overview

Soprano is an ultra-lightweight TTS model capable of generating highly realistic speech in real-time. It addresses significant gaps in the field of open-source text-to-speech, particularly in terms of **scalability**, **speed**, and **naturalness**. 

At just **80 million parameters**, Soprano can achieve an RTF of **2000x**, generating **10 hours** of audio in **under 20 seconds**. Using a novel seamless streaming technique, Soprano also supports **real-time streaming**, producing speech in **<15 ms**, multiple orders of magnitude lower than existing TTS models.

## Installation

```bash
git clone https://github.com/ekwek1/soprano
cd soprano
pip install -r requirements.txt
```

## Usage

```python
from tts import TTS

model = TTS()



# basic inference
out = model.infer("Hello world!")

# saving output to a file
out = model.infer("Hello world!", "out.wav")

# custom sampling parameters
out = model.infer("Hello world!", temperature=0.3, top_p=0.95, repetition_penalty=1.2)



# batched inference
out = model.infer_batch(["Hello world!"]*10)

# saving outputs to a directory
out = model.infer_batch(["Hello world!"]*10, "/dir") 



# streamed inference w/ seamless streaming
import torch
stream = model.infer_stream("Hello world!", chunk_size=1)
# audio chunks can be accessed via an iterator
out = []
for chunk in stream: 
    out.append(chunk)
out = torch.cat(out)
```
