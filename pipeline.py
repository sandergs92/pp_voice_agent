#!/usr/bin/env python3
"""
pipeline.py — Voice assistant pipeline for PinePhone Pro.

Flow:
  1. Wake word detection (KWS) — always listening
  2. On wake: say "Listening.", enter conversation mode
  3. Conversation mode loop:
     a. Record with VAD + ASR + Smart Turn
     b. If LLM responds with ACTION: → execute tool, speak template
     c. If LLM responds with text → stream to TTS as phrases
     d. Wait for TTS to finish, listen again
     e. If no speech for CONVO_TIMEOUT_S, exit to wake mode
  4. Back to step 1

Target: RK3399S aarch64, 4GB RAM.
"""

import collections
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np

# --- Paths (override via environment) ---
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/models"))

KWS_MODEL_DIR = os.path.join(MODELS_DIR,
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01")
ASR_MODEL_DIR = os.path.join(MODELS_DIR,
    "sherpa-onnx-streaming-zipformer-en-2023-06-26")
SMART_TURN_MODEL = os.path.join(MODELS_DIR, "smart-turn-v3.2-cpu.onnx")
SILERO_VAD_MODEL = os.path.join(MODELS_DIR, "silero_vad.onnx")

KEYWORDS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kws", "keywords.txt")
TTS_SOCKET = os.environ.get("TTS_SOCKET",
    os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "pp-tts.sock"))
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8080/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5-0.8b")

SAMPLE_RATE = 16000

VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
SILENCE_TIMEOUT_S = float(os.environ.get("SILENCE_TIMEOUT", "0.6"))
MAX_RECORD_S = float(os.environ.get("MAX_RECORD_S", "30"))

SMART_TURN_THRESHOLD = float(os.environ.get("SMART_TURN_THRESHOLD", "0.5"))

TTS_WORD_FLUSH = int(os.environ.get("TTS_WORD_FLUSH", "5"))

CONVO_TIMEOUT_S = float(os.environ.get("CONVO_TIMEOUT", "4.0"))

PRE_SPEECH_BUFFER_S = float(os.environ.get("PRE_SPEECH_BUFFER", "0.5"))

DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")


# ─── Tool definitions ────────────────────────────────────────────────

TOOLS = {
    "toggle_flashlight": {
        "command": ["sxmo_flashtoggle.sh"],
        "response": "I've toggled the flashlight.",
    },
}

SYSTEM_PROMPT = (
    "/no_think You are a helpful voice assistant on a PinePhone. "
    "For normal conversation, just respond naturally and concisely. "
    "Only use ACTION when the user explicitly asks to control the phone. "
    "Available actions (use ONLY when asked):\n"
    "ACTION: toggle_flashlight\n"
    "Respond with exactly one ACTION line and nothing else when acting. "
    "Keep responses to 1-2 sentences. No markdown or special characters."
)

_ACTION_RE = re.compile(
    r'ACTION:\s*(\w+)(?::(.+))?', re.IGNORECASE)


def execute_tool(action_str):
    m = _ACTION_RE.match(action_str.strip())
    if not m:
        return False, None

    name = m.group(1).lower().strip()
    args_str = m.group(2) or ""

    tool = TOOLS.get(name)
    if not tool:
        return False, f"I don't know how to do {name}."

    print(f"pipeline: executing {name}", file=sys.stderr)

    try:
        cmd = tool["command"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, tool["response"]
        else:
            err = result.stderr.strip() or f"exit code {result.returncode}"
            print(f"pipeline: tool error: {err}", file=sys.stderr)
            return False, f"Sorry, {name} failed."

    except subprocess.TimeoutExpired:
        return False, f"Sorry, {name} timed out."
    except Exception as e:
        print(f"pipeline: tool exception: {e}", file=sys.stderr)
        return False, "Sorry, something went wrong."


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
            "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
        decoder=os.path.join(ASR_MODEL_DIR,
            "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
        joiner=os.path.join(ASR_MODEL_DIR,
            "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
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


def tts_sync():
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(120.0)
        sock.connect(TTS_SOCKET)
        sock.sendall(b"__SYNC__\n")
        resp = sock.recv(64)
        sock.close()
    except Exception as e:
        print(f"pipeline: TTS sync error: {e}", file=sys.stderr)


# ─── LLM streaming with ACTION: detection ────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think>")
_PHRASE_BREAK = re.compile(r'([.!?,;:—\-])\s')


def stream_llm(user_text, on_chunk):
    import urllib.request

    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 256,
    }).encode()

    req = urllib.request.Request(LLM_URL,
        data=payload,
        headers={"Content-Type": "application/json"})

    full_response = ""
    buffer = ""

    def try_flush():
        nonlocal buffer
        m = _PHRASE_BREAK.search(buffer)
        if m:
            end = m.end()
            chunk = buffer[:end].strip()
            buffer = buffer[end:]
            if chunk:
                on_chunk(chunk)
            return True
        words = buffer.split()
        if len(words) >= TTS_WORD_FLUSH:
            chunk = buffer.strip()
            buffer = ""
            if chunk:
                on_chunk(chunk)
            return True
        return False

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
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
                        full_response += token
                        buffer += token
                        buffer = _THINK_RE.sub("", buffer)
                        buffer = _THINK_TAG_RE.sub("", buffer)

                        clean = full_response.strip()
                        clean = _THINK_RE.sub("", clean)
                        clean = _THINK_TAG_RE.sub("", clean)

                        if clean.upper().startswith("ACTION"):
                            continue

                        while try_flush():
                            pass
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        clean = full_response.strip()
        clean = _THINK_RE.sub("", clean)
        clean = _THINK_TAG_RE.sub("", clean).strip()

        if _ACTION_RE.match(clean):
            return clean

        remaining = buffer.strip()
        if remaining:
            on_chunk(remaining)
        return None

    except Exception as e:
        print(f"pipeline: LLM error: {e}", file=sys.stderr)
        return None


# ─── Recording + VAD + ASR + Smart Turn ─────────────────────────────

def record_and_transcribe(asr, vad, smart_turn_sess, smart_turn_fe,
                          initial_timeout=None):
    import sounddevice as sd

    asr_stream = asr.create_stream()
    vad.reset()

    window_size = vad.window_size()

    pre_speech_max = int(PRE_SPEECH_BUFFER_S * SAMPLE_RATE)
    pre_speech_buf = collections.deque()
    pre_speech_len = 0

    audio_q = collections.deque()
    q_lock = threading.Lock()
    done = threading.Event()

    all_samples = []
    asr_samples = []
    speech_started = False
    silence_start = None
    listen_start = time.monotonic()

    def audio_callback(indata, frames, time_info, status):
        if done.is_set():
            return
        with q_lock:
            audio_q.append(indata[:, 0].copy())

    device = get_audio_device()
    max_samples = int(MAX_RECORD_S * SAMPLE_RATE)

    print("pipeline: listening...", file=sys.stderr)

    with sd.InputStream(
        device=device,
        channels=1,
        dtype="float32",
        samplerate=SAMPLE_RATE,
        blocksize=window_size * 3,
        callback=audio_callback,
    ):
        while not done.is_set():
            chunks = []
            with q_lock:
                while audio_q:
                    chunks.append(audio_q.popleft())

            for chunk in chunks:
                samples = chunk
                samples_list = samples.tolist()
                all_samples.extend(samples_list)

                for j in range(0, len(samples), window_size):
                    sub = samples[j:j + window_size]
                    if len(sub) < window_size:
                        break
                    is_speech = vad.is_speech(sub.tolist())

                    if is_speech and not speech_started:
                        speech_started = True
                        print("pipeline: speech detected", file=sys.stderr)

                        while pre_speech_buf:
                            old = pre_speech_buf.popleft()
                            asr_stream.accept_waveform(SAMPLE_RATE, old)
                            if DEBUG:
                                asr_samples.extend(old)
                        pre_speech_len = 0

                        asr_stream.accept_waveform(SAMPLE_RATE, samples_list)
                        if DEBUG:
                            asr_samples.extend(samples_list)
                        break

                    elif is_speech:
                        silence_start = None
                    elif speech_started:
                        if silence_start is None:
                            silence_start = time.monotonic()
                else:
                    if speech_started:
                        asr_stream.accept_waveform(SAMPLE_RATE, samples_list)
                        if DEBUG:
                            asr_samples.extend(samples_list)
                    else:
                        pre_speech_buf.append(samples_list)
                        pre_speech_len += len(samples_list)
                        while (pre_speech_len > pre_speech_max
                               and pre_speech_buf):
                            removed = pre_speech_buf.popleft()
                            pre_speech_len -= len(removed)

            while asr.is_ready(asr_stream):
                asr.decode_stream(asr_stream)

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

            if (initial_timeout is not None
                    and not speech_started
                    and time.monotonic() - listen_start >= initial_timeout):
                print("pipeline: conversation timeout, "
                      "returning to wake mode", file=sys.stderr)
                done.set()

            if len(all_samples) >= max_samples:
                done.set()

            time.sleep(0.01)

        with q_lock:
            while audio_q:
                chunk = audio_q.popleft()
                chunk_list = chunk.tolist()
                asr_stream.accept_waveform(SAMPLE_RATE, chunk_list)
                if DEBUG:
                    asr_samples.extend(chunk_list)
        while asr.is_ready(asr_stream):
            asr.decode_stream(asr_stream)

    time.sleep(0.15)

    if not speech_started:
        return None

    if DEBUG:
        import wave as _wave
        import array as _array
        _dbg_path = "/tmp/pipeline_asr_input.wav"
        _int = _array.array("h",
            (int(max(-1.0, min(1.0, s)) * 32767) for s in asr_samples))
        with _wave.open(_dbg_path, "wb") as _wf:
            _wf.setnchannels(1)
            _wf.setsampwidth(2)
            _wf.setframerate(SAMPLE_RATE)
            _wf.writeframes(_int.tobytes())
        print(f"pipeline: DEBUG saved {len(asr_samples)/SAMPLE_RATE:.1f}s "
              f"to {_dbg_path}", file=sys.stderr)

    asr_stream.input_finished()
    while asr.is_ready(asr_stream):
        asr.decode_stream(asr_stream)

    return asr.get_result(asr_stream).strip()


# ─── Main loop ───────────────────────────────────────────────────────

def get_audio_device():
    device = os.environ.get("AUDIO_DEVICE", None)
    if device and device.isdigit():
        device = int(device)
    return device


def handle_llm_response(text):
    print("pipeline: LLM streaming...", file=sys.stderr)

    action = stream_llm(text, lambda s: (
        print(f"pipeline: TTS ← \"{s}\"", file=sys.stderr),
        tts_speak(s),
    ))

    if action:
        print(f"pipeline: action → {action}", file=sys.stderr)
        success, response = execute_tool(action)
        if response:
            print(f"pipeline: TTS ← \"{response}\"", file=sys.stderr)
            tts_speak(response)

    print("pipeline: waiting for TTS...", file=sys.stderr)
    tts_sync()
    print("pipeline: response complete", file=sys.stderr)


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
    kws_blocksize = SAMPLE_RATE // 10

    print("pipeline: ready — say your wake word!", file=sys.stderr)
    if DEBUG:
        print("pipeline: DEBUG mode enabled", file=sys.stderr)

    while running:
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
                        print(f"pipeline: wake [{result.strip()}]",
                              file=sys.stderr)
                        wake_detected = True
                    time.sleep(0.05)
        except Exception as e:
            print(f"pipeline: wake error: {e}", file=sys.stderr)
            time.sleep(1)
            continue

        if not wake_detected:
            continue

        # Wake mic closed — TTS feedback then record
        time.sleep(0.3)
        tts_speak("Listening.")
        tts_sync()
        time.sleep(0.15)

        # ── Conversation mode ──
        is_first_turn = True

        while running:
            if is_first_turn:
                text = record_and_transcribe(
                    asr, vad, smart_turn_sess, smart_turn_fe,
                    initial_timeout=None)
            else:
                text = record_and_transcribe(
                    asr, vad, smart_turn_sess, smart_turn_fe,
                    initial_timeout=CONVO_TIMEOUT_S)

            if text is None:
                break

            if text:
                print(f"pipeline: ASR → \"{text}\"", file=sys.stderr)
                handle_llm_response(text)
            else:
                print("pipeline: no speech detected", file=sys.stderr)

            is_first_turn = False

        print("pipeline: back to wake mode", file=sys.stderr)

    print("pipeline: shutdown", file=sys.stderr)


if __name__ == "__main__":
    main()
