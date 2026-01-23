import io

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from scipy.io.wavfile import write
from torch import Tensor

from soprano.tts import SopranoTTS

# Load model at startup
tts = SopranoTTS()

app = FastAPI(title="Soprano TTS API")

def _tensor_to_wav_bytes(tensor: Tensor) -> bytes:
    """
    Convert a 1D fp32 torch tensor to a WAV byte stream.
    """
    audio_int16 = (np.clip(tensor.numpy(), -1.0, 1.0) * 32767).astype(np.int16)

    wav_io = io.BytesIO()
    write(wav_io, 32000, audio_int16)
    wav_io.seek(0)
    return wav_io.read()


@app.post("/v1/audio/speech")
async def create_speech(payload: dict):
    """
    Minimal implementation of OpenAI's Speech endpoint.
    Fields:
      - input: string - text to synthesize
      - model, voice, etc. are accepted but ignored.
      - response_format: str - ignored, only support wav.
    """
    text = payload.get("input")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="`input` field must be a non-empty string.")

    audio_tensor = tts.infer(text)
    wav_bytes = _tensor_to_wav_bytes(audio_tensor)
    return Response(content=wav_bytes, media_type="audio/wav", headers={"Content-Disposition": 'attachment; filename="speech.wav"'})
