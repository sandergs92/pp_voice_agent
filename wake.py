#!/usr/bin/env python3
"""
wake.py — Wake word listener for PinePhone Pro.

Streams microphone audio through sherpa-onnx KeywordSpotter.
On detection, prints the keyword to stdout and (later) triggers
the ASR → LLM → TTS pipeline.

Target: RK3399S aarch64, 4GB RAM.
CPU: 1 ONNX thread on A53 — KWS is tiny (~3.3M params).
Memory: ~15MB resident.

Usage:
  python3 wake.py [--device hw:0,0]
"""

import os
import sys
import signal
import time

# --- Configuration via environment ---
KWS_MODEL_DIR = os.environ.get(
    "KWS_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.path.expanduser("~/models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01")),
)
KEYWORDS_FILE = os.environ.get(
    "KEYWORDS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "keywords.txt"),
)
NUM_THREADS = int(os.environ.get("KWS_NUM_THREADS", "1"))
SAMPLE_RATE = 16000

# Tuning: lower threshold = more sensitive, higher = fewer false positives
# Start conservative; tune down if it doesn't trigger reliably
KEYWORDS_THRESHOLD = float(os.environ.get("KWS_THRESHOLD", "0.2"))
KEYWORDS_SCORE = float(os.environ.get("KWS_SCORE", "1.0"))


def load_kws():
    import sherpa_onnx

    encoder = os.path.join(KWS_MODEL_DIR,
                           "encoder-epoch-12-avg-2-chunk-16-left-64.onnx")
    decoder = os.path.join(KWS_MODEL_DIR,
                           "decoder-epoch-12-avg-2-chunk-16-left-64.onnx")
    joiner = os.path.join(KWS_MODEL_DIR,
                          "joiner-epoch-12-avg-2-chunk-16-left-64.onnx")
    tokens = os.path.join(KWS_MODEL_DIR, "tokens.txt")

    for p in (encoder, decoder, joiner, tokens, KEYWORDS_FILE):
        if not os.path.isfile(p):
            print(f"FATAL: missing {p}", file=sys.stderr)
            sys.exit(1)

    kws = sherpa_onnx.KeywordSpotter(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        num_threads=NUM_THREADS,
        keywords_file=KEYWORDS_FILE,
        keywords_score=KEYWORDS_SCORE,
        keywords_threshold=KEYWORDS_THRESHOLD,
        provider="cpu",
    )
    print(f"wake: KWS model loaded ({NUM_THREADS} thread, "
          f"threshold={KEYWORDS_THRESHOLD})", file=sys.stderr)
    return kws


def main():
    try:
        import sounddevice as sd
    except ImportError:
        print("FATAL: pip install sounddevice", file=sys.stderr)
        sys.exit(1)

    kws = load_kws()
    stream = kws.create_stream()

    running = True

    def _shutdown(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Determine audio device
    device = os.environ.get("AUDIO_DEVICE", None)
    if device and device.isdigit():
        device = int(device)

    # Audio callback — called from a separate thread by sounddevice.
    # We just feed samples into the KWS stream; decoding happens in main loop.
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"wake: audio status: {status}", file=sys.stderr)
        # indata is (frames, channels) float32 — take channel 0
        stream.accept_waveform(SAMPLE_RATE, indata[:, 0].tolist())

    print(f"wake: listening on device={device or 'default'} "
          f"@ {SAMPLE_RATE}Hz", file=sys.stderr)
    print("wake: say your wake word...", file=sys.stderr)

    try:
        with sd.InputStream(
            device=device,
            channels=1,
            dtype="float32",
            samplerate=SAMPLE_RATE,
            # 100ms chunks — good balance of latency vs overhead on ARM
            blocksize=SAMPLE_RATE // 10,
            callback=audio_callback,
        ):
            while running:
                # Process available frames
                while kws.is_ready(stream):
                    kws.decode_stream(stream)

                result = kws.get_result(stream)
                if result:
                    keyword = result.strip()
                    if keyword:
                        now = time.strftime("%H:%M:%S")
                        print(f"wake: [{now}] DETECTED: {keyword}",
                              file=sys.stderr)
                        # Flush stdout for downstream piping
                        print(keyword, flush=True)

                # Sleep briefly to avoid busy-spinning — 50ms is fine
                # for wake word latency (detection is not time-critical
                # to the millisecond)
                time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    except sd.PortAudioError as e:
        print(f"FATAL: audio device error: {e}", file=sys.stderr)
        print("  Try: AUDIO_DEVICE=0 python3 wake.py", file=sys.stderr)
        sys.exit(1)

    print("wake: shutdown", file=sys.stderr)


if __name__ == "__main__":
    main()
