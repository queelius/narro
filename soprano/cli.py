#!/usr/bin/env python3
"""
Soprano TTS Command Line Interface
"""
import argparse
import logging

from soprano import SopranoTTS
from soprano.utils.streaming import play_stream

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Soprano Text-to-Speech CLI')
    parser.add_argument('text', help='Text to synthesize')
    parser.add_argument('--output', '-o', default='output.wav',
                        help='Output audio file path (non-streaming only)')
    parser.add_argument('--model-path', '-m',
                        help='Path to local model directory (optional)')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile optimization')
    parser.add_argument('--quantize', '-q', action='store_true',
                        help='Enable INT8 quantization for faster CPU inference')
    parser.add_argument('--decoder-batch-size', '-bs', type=int, default=1,
                        help='Batch size when decoding audio')
    parser.add_argument('--streaming', '-s', action='store_true',
                        help='Enable streaming playback to speakers')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    tts = SopranoTTS(
        model_path=args.model_path,
        compile=not args.no_compile,
        quantize=args.quantize,
        decoder_batch_size=args.decoder_batch_size
    )

    logger.info("Generating speech for: '%s'", args.text)
    if args.streaming:
        stream = tts.infer_stream(args.text, chunk_size=1)
        play_stream(stream)
    else:
        tts.infer(args.text, out_path=args.output)
        logger.info("Audio saved to: %s", args.output)


if __name__ == "__main__":
    main()
