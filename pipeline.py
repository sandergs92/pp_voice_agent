#!/usr/bin/env python3
"""
pipeline.py — Voice assistant pipeline for PinePhone Pro.

Flow:
  1. Wake word detection (KWS) — always listening
  2. On wake: close wake mic, start recording with VAD + ASR
  3. On silence: Smart Turn decides if user is done
  4. If done: get final ASR text, send to LLM
  5. Stream LLM response, buffer to sentences, send to TTS daemon
  6. After TTS finishes, reopen wake mic

Audio architecture:
  - Only one InputStream open at a time (avoids ALSA/PipeWire conflicts)
  - Audio callback only queues samples (fast, no blocking)
  - Main thread feeds ASR + runs VAD + runs Smart Turn (thread-safe)

Target: RK3399S aarch64, 4GB RAM.
"""

import collections
import json
import os
import re
import signal
import socket
import sys
import threading
import time

import numpy as np

# --- Paths (override via environment) ---
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/models"))

KWS_MODEL_DIR = os.path.join(MODELS_DIR,
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01")
ASR_MODEL_DIR = os.path.join(MODELS_DIR,
    "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17")
SMART_TURN_MODEL = os.path.join(MODELS_DIR, "smart-turn-v3.2-cpu.onnx")
SILERO_VAD_MODEL = os.path.join(MODELS_DIR, "silero_vad.onnx")

KEYWORDS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "keywords.txt")
TTS_SOCKET = os.environ.get("TTS_SOCKET",
    os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "pp-tts.sock"))
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8080/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5-0.8b")

SAMPLE_RATE = 16000

# VAD parameters
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
SILENCE_TIMEOUT_S = float(os.environ.get("SILENCE_TIMEOUT", "0.6"))
MAX_RECORD_S = float(os.environ.get("MAX_RECORD_S", "30"))

# Smart Turn
SMART_TURN_THRESHOLD = float(os.environ.get("SMART_TURN_THRESHOLD", "0.5"))


# ─── Component loaders ───────────────────────────────────────────────

def load_kws():
    import sherpa_onnx
    kws = sherpa_onnx.KeywordSpotter(
        tokens=os.path.join(KWS_MODEL_DIR, "tokens.txt"),
        encoder=os.path.join(KWS_MODEL_DIR,
            "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        decoder=os.path.join(KWS_MODEL_DIR,
            "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        joiner=os.path.join(KWS_MODEL_DIR,
            "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
        num_threads=1,
        keywords_file=KEYWORDS_FILE,
        keywords_threshold=0.2,
        provider="cpu",
    )
    print("pipeline: KWS loaded", file=sys.stderr)
    return kws


def load_asr():
    import sherpa_onnx
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=os.path.join(ASR_MODEL_DIR,
            "encoder-epoch-99-avg-1.onnx"),
        decoder=os.path.join(ASR_MODEL_DIR,
            "decoder-epoch-99-avg-1.onnx"),
        joiner=os.path.join(ASR_MODEL_DIR,
            "joiner-epoch-99-avg-1.onnx"),
        tokens=os.path.join(ASR_MODEL_DIR, "tokens.txt"),
        num_threads=2,
        provider="cpu",
    )
    print("pipeline: ASR loaded", file=sys.stderr)
    return recognizer


def load_vad():
    import sherpa_onnx
    config = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=SILERO_VAD_MODEL,
            threshold=VAD_THRESHOLD,
            min_silence_duration=SILENCE_TIMEOUT_S,
            min_speech_duration=0.25,
        ),
        sample_rate=SAMPLE_RATE,
        num_threads=1,
        provider="cpu",
    )
    vad = sherpa_onnx.VadModel.create(config)
    print(f"pipeline: VAD loaded (window={vad.window_size()})", file=sys.stderr)
    return vad


def load_smart_turn():
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(SMART_TURN_MODEL,
        providers=["CPUExecutionProvider"], sess_options=opts)
    fe = WhisperFeatureExtractor(chunk_length=8)
    print("pipeline: Smart Turn loaded", file=sys.stderr)
    return sess, fe


def smart_turn_predict(sess, fe, audio_samples):
    """Returns probability that the turn is complete."""
    target = 8 * SAMPLE_RATE
    if len(audio_samples) > target:
        audio_samples = audio_samples[-target:]
    elif len(audio_samples) < target:
        audio_samples = np.concatenate([
            np.zeros(target - len(audio_samples)), audio_samples])

    inputs = fe(audio_samples, sampling_rate=SAMPLE_RATE,
                return_tensors="np", padding="max_length",
                max_length=target, truncation=True, do_normalize=True)
    feat = inputs.input_features.squeeze(0).astype(np.float32)
    feat = np.expand_dims(feat, 0)
    result = sess.run(None, {"input_features": feat})
    return result[0][0].item()


# ─── TTS client ──────────────────────────────────────────────────────

def tts_speak(text):
    """Send text to TTS daemon. Non-blocking (daemon queues it)."""
    text = text.strip()
    if not text:
        return
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(TTS_SOCKET)
        sock.sendall((text + "\n").encode())
        resp = sock.recv(64)
        sock.close()
        status = resp.decode().strip()
        if status != "OK":
            print(f"pipeline: TTS {status}", file=sys.stderr)
    except Exception as e:
        print(f"pipeline: TTS error: {e}", file=sys.stderr)


# ─── LLM streaming ──────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think>")


def stream_llm(user_text, on_sentence):
    """Stream LLM response, call on_sentence(str) for each sentence."""
    import urllib.request

    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content":
             "/no_think You are a helpful voice assistant on a PinePhone. "
             "Keep responses concise — 1-3 sentences. "
             "No markdown, no lists, no special characters."},
            {"role": "user", "content": user_text},
        ],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 256,
    }).encode()

    req = urllib.request.Request(LLM_URL,
        data=payload,
        headers={"Content-Type": "application/json"})

    buffer = ""
    sent_re = re.compile(r'([.!?])\s')

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    token = delta.get("content", "")
                    if token:
                        buffer += token
                        buffer = _THINK_RE.sub("", buffer)
                        buffer = _THINK_TAG_RE.sub("", buffer)
                        while True:
                            m = sent_re.search(buffer)
                            if not m:
                                break
                            end = m.end()
                            sentence = buffer[:end].strip()
                            buffer = buffer[end:]
                            if sentence:
                                on_sentence(sentence)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        remaining = buffer.strip()
        if remaining:
            on_sentence(remaining)

    except Exception as e:
        print(f"pipeline: LLM error: {e}", file=sys.stderr)


# ─── Recording + VAD + ASR + Smart Turn ─────────────────────────────

def record_and_transcribe(asr, vad, smart_turn_sess, smart_turn_fe):
    """Record from mic with queue-based architecture.

    Audio callback only appends to a deque (fast, never blocks).
    Main thread pulls from deque, feeds VAD + ASR, runs Smart Turn.
    This keeps everything thread-safe — sherpa-onnx streams are only
    touched from the main thread.
    """
    import sounddevice as sd

    asr_stream = asr.create_stream()
    vad.reset()

    # Prime ASR with silence so first word isn't clipped
    asr_stream.accept_waveform(SAMPLE_RATE, [0.0] * 8000)
    while asr.is_ready(asr_stream):
        asr.decode_stream(asr_stream)

    window_size = vad.window_size()  # 512 for 16kHz

    # Shared state
    audio_q = collections.deque()
    q_lock = threading.Lock()
    done = threading.Event()

    # VAD state (main thread only)
    all_samples = []
    speech_started = False
    silence_start = None

    def audio_callback(indata, frames, time_info, status):
        """Minimal callback — just queue the audio."""
        if done.is_set():
            return
        with q_lock:
            audio_q.append(indata[:, 0].copy())

    print("pipeline: listening...", file=sys.stderr)

    device = os.environ.get("AUDIO_DEVICE", None)
    if device and device.isdigit():
        device = int(device)

    max_samples = int(MAX_RECORD_S * SAMPLE_RATE)

    with sd.InputStream(
        device=device,
        channels=1,
        dtype="float32",
        samplerate=SAMPLE_RATE,
        blocksize=window_size * 3,  # ~96ms
        callback=audio_callback,
    ):
        while not done.is_set():
            # Pull all queued audio
            chunks = []
            with q_lock:
                while audio_q:
                    chunks.append(audio_q.popleft())

            for chunk in chunks:
                samples = chunk
                all_samples.extend(samples.tolist())

                # Feed ASR
                asr_stream.accept_waveform(SAMPLE_RATE, samples.tolist())

                # Run VAD on window_size sub-chunks
                for j in range(0, len(samples), window_size):
                    sub = samples[j:j + window_size]
                    if len(sub) < window_size:
                        break

                    is_speech = vad.is_speech(sub.tolist())

                    if is_speech:
                        if not speech_started:
                            print("pipeline: speech detected",
                                  file=sys.stderr)
                        speech_started = True
                        silence_start = None
                    elif speech_started:
                        if silence_start is None:
                            silence_start = time.monotonic()

            # Decode ASR
            while asr.is_ready(asr_stream):
                asr.decode_stream(asr_stream)

            # Check silence timeout → Smart Turn
            if (speech_started
                    and silence_start is not None
                    and time.monotonic() - silence_start >= SILENCE_TIMEOUT_S
                    and not done.is_set()):

                audio_arr = np.array(all_samples, dtype=np.float32)
                t0 = time.monotonic()
                prob_complete = smart_turn_predict(
                    smart_turn_sess, smart_turn_fe, audio_arr)
                st_ms = (time.monotonic() - t0) * 1000
                print(f"pipeline: Smart Turn {prob_complete:.2f} "
                      f"({st_ms:.0f}ms)", file=sys.stderr)

                if prob_complete >= SMART_TURN_THRESHOLD:
                    done.set()
                else:
                    silence_start = None

            # Safety: max recording length
            if len(all_samples) >= max_samples:
                done.set()

            time.sleep(0.01)

        # Final drain: pull remaining queued audio and decode
        with q_lock:
            while audio_q:
                chunk = audio_q.popleft()
                asr_stream.accept_waveform(SAMPLE_RATE, chunk.tolist())

        while asr.is_ready(asr_stream):
            asr.decode_stream(asr_stream)

    # Let PipeWire fully release before TTS plays
    time.sleep(0.15)

    result_text = asr.get_result(asr_stream).strip()
    return result_text


# ─── Main loop ───────────────────────────────────────────────────────

def get_audio_device():
    device = os.environ.get("AUDIO_DEVICE", None)
    if device and device.isdigit():
        device = int(device)
    return device


def main():
    import sounddevice as sd

    print("pipeline: loading models...", file=sys.stderr)
    kws = load_kws()
    asr = load_asr()
    vad = load_vad()
    smart_turn_sess, smart_turn_fe = load_smart_turn()

    running = True

    def _shutdown(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    device = get_audio_device()
    kws_blocksize = SAMPLE_RATE // 10  # 1600 samples, 100ms

    print("pipeline: ready — say your wake word!", file=sys.stderr)

    while running:
        # --- Wake word phase ---
        # Open mic for KWS, close it before recording/playback
        kws_stream = kws.create_stream()
        wake_detected = False

        def wake_callback(indata, frames, time_info, status):
            kws_stream.accept_waveform(SAMPLE_RATE, indata[:, 0].tolist())

        try:
            with sd.InputStream(
                device=device,
                channels=1,
                dtype="float32",
                samplerate=SAMPLE_RATE,
                blocksize=kws_blocksize,
                callback=wake_callback,
            ):
                while running and not wake_detected:
                    while kws.is_ready(kws_stream):
                        kws.decode_stream(kws_stream)

                    result = kws.get_result(kws_stream)
                    if result and result.strip():
                        keyword = result.strip()
                        print(f"pipeline: wake [{keyword}]",
                              file=sys.stderr)
                        wake_detected = True

                    time.sleep(0.05)
        except Exception as e:
            print(f"pipeline: wake error: {e}", file=sys.stderr)
            time.sleep(1)
            continue

        if not wake_detected:
            continue

        # Wake InputStream is now closed (exited with block)
        # Small gap to let PipeWire release the device
        time.sleep(0.1)

        # --- Record + transcribe phase ---
        text = record_and_transcribe(asr, vad, smart_turn_sess, smart_turn_fe)

        if text:
            print(f"pipeline: ASR → \"{text}\"", file=sys.stderr)

            # --- LLM + TTS phase (no mic open) ---
            print("pipeline: LLM streaming...", file=sys.stderr)
            stream_llm(text, lambda s: (
                print(f"pipeline: TTS ← \"{s}\"", file=sys.stderr),
                tts_speak(s),
            ))
            print("pipeline: response complete", file=sys.stderr)

            # Wait for TTS to finish playing before reopening mic
            # TTS daemon queues sentences, so we wait a bit
            time.sleep(1.0)
        else:
            print("pipeline: no speech detected", file=sys.stderr)

    print("pipeline: shutdown", file=sys.stderr)


if __name__ == "__main__":
    main()
