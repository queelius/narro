#!/usr/bin/env python3
"""Soprano TTS Command Line Interface"""
import argparse
import logging

logger = logging.getLogger(__name__)


def _add_common_args(parser):
    """Add model/compile/quantize args shared across subcommands."""
    parser.add_argument('--model-path', '-m',
                        help='Path to local model directory (optional)')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile optimization')
    parser.add_argument('--quantize', action='store_true',
                        help='Enable INT8 quantization (faster but lower quality)')
    parser.add_argument('--num-threads', '-t', type=int,
                        help='Number of CPU threads for inference')


def cmd_speak(args):
    """Default command: encode + decode text to WAV."""
    from soprano import SopranoTTS
    tts = SopranoTTS(
        model_path=args.model_path,
        compile=not args.no_compile,
        quantize=args.quantize,
        decoder_batch_size=args.decoder_batch_size,
        num_threads=args.num_threads,
    )
    logger.info("Generating speech for: '%s'", args.text)
    tts.infer(args.text, out_path=args.output)
    logger.info("Audio saved to: %s", args.output)


def cmd_encode(args):
    """Encode text to .soprano file."""
    from soprano import SopranoTTS
    from soprano.encoded import save
    tts = SopranoTTS(
        model_path=args.model_path,
        compile=not args.no_compile,
        quantize=args.quantize,
        num_threads=args.num_threads,
    )
    logger.info("Encoding: '%s'", args.text)
    encoded = tts.encode(args.text, include_attention=args.include_attention)
    save(encoded, args.output)
    logger.info("Encoded %d tokens (est. %.1fs audio) -> %s",
                encoded.total_tokens, encoded.estimated_duration, args.output)


def cmd_decode(args):
    """Decode .soprano file to WAV."""
    from soprano.encoded import load
    from soprano.decode_only import decode_to_wav, load_decoder
    logger.info("Loading encoded speech from: %s", args.input)
    encoded = load(args.input)
    decoder = load_decoder(model_path=args.model_path, compile=not args.no_compile)
    decode_to_wav(encoded, args.output, decoder=decoder,
                  decoder_batch_size=args.decoder_batch_size)
    logger.info("Audio saved to: %s", args.output)


def _add_speak_args(parser):
    """Add speak-specific args (text, output, batch size)."""
    parser.add_argument('text', help='Text to synthesize')
    parser.add_argument('--output', '-o', default='output.wav',
                        help='Output audio file path')
    parser.add_argument('--decoder-batch-size', '-bs', type=int, default=4,
                        help='Batch size when decoding audio')


def main():
    import sys

    # Default to 'speak' when first arg isn't a known subcommand.
    # This lets `soprano "Hello world"` work as shorthand for `soprano speak "Hello world"`.
    _subcommands = {'speak', 'encode', 'decode'}
    if len(sys.argv) > 1 and sys.argv[1] not in _subcommands and sys.argv[1] not in ('-h', '--help'):
        sys.argv.insert(1, 'speak')

    parser = argparse.ArgumentParser(
        description='Soprano Text-to-Speech CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  soprano "Hello world" -o output.wav
  soprano encode "Hello world" -o encoded.soprano
  soprano decode encoded.soprano -o output.wav
""")

    subparsers = parser.add_subparsers(dest='command')

    # --- speak ---
    speak_parser = subparsers.add_parser('speak', help='Synthesize speech (default)')
    _add_speak_args(speak_parser)
    _add_common_args(speak_parser)
    speak_parser.set_defaults(func=cmd_speak)

    # --- encode ---
    encode_parser = subparsers.add_parser('encode', help='Encode text to .soprano file')
    encode_parser.add_argument('text', help='Text to encode')
    encode_parser.add_argument('--output', '-o', default='output.soprano',
                               help='Output .soprano file path')
    encode_parser.add_argument('--include-attention', action='store_true',
                               help='Include attention weights (larger file)')
    _add_common_args(encode_parser)
    encode_parser.set_defaults(func=cmd_encode)

    # --- decode ---
    decode_parser = subparsers.add_parser('decode', help='Decode .soprano file to WAV')
    decode_parser.add_argument('input', help='Input .soprano file path')
    decode_parser.add_argument('--output', '-o', default='output.wav',
                               help='Output WAV file path')
    decode_parser.add_argument('--decoder-batch-size', '-bs', type=int, default=4,
                               help='Batch size when decoding audio')
    decode_parser.add_argument('--model-path', '-m',
                               help='Path to local model directory (optional)')
    decode_parser.add_argument('--no-compile', action='store_true',
                               help='Disable torch.compile optimization')
    decode_parser.set_defaults(func=cmd_decode)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command is not None:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
