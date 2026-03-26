#!/usr/bin/env python3
"""
pipeline.py — Voice assistant pipeline for PinePhone Pro.

Flow:
  1. Wake word detection (KWS) — always listening
  2. On wake: enter conversation mode
  3. Conversation mode loop:
     a. Record with VAD + ASR + Smart Turn
     b. Send to LLM (lightweight tool prompt)
     c. If LLM responds with ACTION: → execute tool, speak template
     d. If LLM responds with text → stream to TTS as phrases
     e. Wait for TTS to finish, listen again
     f. If no speech for CONVO_TIMEOUT_S, exit to wake mode
  4. Back to step 1

Tool calling uses lightweight ACTION: keyword detection instead of
OpenAI function calling, cutting prompt from ~360 to ~70 tokens
(8s vs 52s on RK3399S A72 cores).

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

VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
SILENCE_TIMEOUT_S = float(os.environ.get("SILENCE_TIMEOUT", "0.6"))
MAX_RECORD_S = float(os.environ.get("MAX_RECORD_S", "30"))

SMART_TURN_THRESHOLD = float(os.environ.get("SMART_TURN_THRESHOLD", "0.5"))

TTS_WORD_FLUSH = int(os.environ.get("TTS_WORD_FLUSH", "5"))

CONVO_TIMEOUT_S = float(os.environ.get("CONVO_TIMEOUT", "4.0"))


# ─── Tool definitions ────────────────────────────────────────────────

TOOLS = {
    "toggle_flashlight": {
        "command": ["sxmo_flashtoggle.sh"],
        "response": "I've toggled the flashlight.",
    },
    "set_brightness": {
        "command": None,  # built dynamically
        "response": "I've set the brightness to {level} percent.",
    },
    "toggle_wifi": {
        "command": ["doas", "sxmo_wifitoggle.sh"],
        "response": "I've toggled the WiFi.",
    },
    "send_sms": {
        "command": None,  # built dynamically
        "response": "I've sent the message.",
    },
    "make_call": {
        "command": None,  # built dynamically
        "response": "I'm calling {number}.",
    },
}

# System prompt with tool descriptions — kept short for fast prompts
SYSTEM_PROMPT = (
    "/no_think You are a voice assistant on a PinePhone. "
    "You can perform actions by responding ONLY with ACTION: followed by "
    "the action. Available actions:\n"
    "ACTION: toggle_flashlight\n"
    "ACTION: set_brightness:LEVEL (0-100)\n"
    "ACTION: toggle_wifi\n"
    "ACTION: send_sms:NUMBER:MESSAGE\n"
    "ACTION: make_call:NUMBER\n"
    "For normal conversation, respond normally. "
    "Keep responses concise — 1-3 sentences. "
    "No markdown, no lists, no special characters."
)

# Pattern to detect ACTION: in LLM output
_ACTION_RE = re.compile(
    r'ACTION:\s*(\w+)(?::(.+))?', re.IGNORECASE)


def execute_tool(action_str):
    """Parse and execute an ACTION: string.

    Format: ACTION: name or ACTION: name:arg1:arg2
    Returns (success, tts_response).
    """
    m = _ACTION_RE.match(action_str.strip())
    if not m:
        return False, None

    name = m.group(1).lower().strip()
    args_str = m.group(2) or ""

    tool = TOOLS.get(name)
    if not tool:
        return False, f"I don't know how to do {name}."

    print(f"pipeline: executing {name} (args: {args_str})", file=sys.stderr)

    try:
        if name == "toggle_flashlight":
            cmd = tool["command"]
        elif name == "set_brightness":
            level = args_str.strip() or "50"
            cmd = ["brightnessctl", "-q", "set", f"{level}%"]
            tool["response"] = f"I've set the brightness to {level} percent."
        elif name == "toggle_wifi":
            cmd = tool["command"]
        elif name == "send_sms":
            parts = args_str.split(":", 1)
            number = parts[0].strip() if parts else ""
            message = parts[1].strip() if len(parts) > 1 else ""
            if not number:
                return False, "I need a phone number to send a message."
            cmd = ["sxmo_modemsendsms.sh", number, message]
        elif name == "make_call":
            number = args_str.strip()
            if not number:
                return False, "I need a phone number to make a call."
            cmd = ["sxmo_modemdial.sh", number]
            tool["response"] = f"I'm calling {number}."
        else:
            return False, f"I don't know how to do {name}."

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
    """Stream LLM response. Detects ACTION: lines for tool execution.

    Returns:
        None if regular text response (already streamed to on_chunk).
        String like "toggle_flashlight" or "set_brightness:75" if action.
    """
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

                        # Check if this is an ACTION response
                        # Don't stream to TTS yet — accumulate first
                        # to detect ACTION: pattern
                        clean = full_response.strip()
                        clean = _THINK_RE.sub("", clean)
                        clean = _THINK_TAG_RE.sub("", clean)

                        if clean.upper().startswith("ACTION"):
                            # Keep accumulating, don't flush to TTS
                            continue

                        # Regular text — flush phrases to TTS
                        while try_flush():
                            pass
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        # Final check: is the full response an ACTION?
        clean = full_response.strip()
        clean = _THINK_RE.sub("", clean)
        clean = _THINK_TAG_RE.sub("", clean).strip()

        if _ACTION_RE.match(clean):
            return clean

        # Regular text — flush remainder
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

    asr_stream.accept_waveform(SAMPLE_RATE, [0.0] * 8000)
    while asr.is_ready(asr_stream):
        asr.decode_stream(asr_stream)

    window_size = vad.window_size()

    audio_q = collections.deque()
    q_lock = threading.Lock()
    done = threading.Event()

    all_samples = []
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
                all_samples.extend(samples.tolist())
                asr_stream.accept_waveform(SAMPLE_RATE, samples.tolist())

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
                asr_stream.accept_waveform(SAMPLE_RATE, chunk.tolist())
        while asr.is_ready(asr_stream):
            asr.decode_stream(asr_stream)

    time.sleep(0.15)

    if not speech_started:
        return None

    result_text = asr.get_result(asr_stream).strip()
    return result_text


# ─── Main loop ───────────────────────────────────────────────────────

def get_audio_device():
    device = os.environ.get("AUDIO_DEVICE", None)
    if device and device.isdigit():
        device = int(device)
    return device


def handle_llm_response(text):
    """Send text to LLM, handle ACTION: or stream text to TTS."""
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
    print(f"pipeline: tools: {', '.join(TOOLS.keys())}", file=sys.stderr)

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

        time.sleep(0.1)

        is_first_turn = True

        while running:
            if is_first_turn:
                print("pipeline: listening...", file=sys.stderr)
                text = record_and_transcribe(
                    asr, vad, smart_turn_sess, smart_turn_fe,
                    initial_timeout=None)
            else:
                print("pipeline: listening for follow-up...",
                      file=sys.stderr)
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
