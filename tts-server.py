#!/usr/bin/env python3
"""
tts-server.py — Persistent TTS daemon for PinePhone Pro.

Loads the sherpa-onnx VITS model once into memory (~40MB resident),
then listens on a UNIX domain socket for text lines.

Architecture — three-stage pipeline:
  accept loop → synth thread → playback thread

Protocol (per connection):
  Client sends: <text>\n
  Server sends: OK\n  (accepted into queue)
               | FULL\n (queue at capacity, try again)
               | ERR <reason>\n

  Special command:
  Client sends: __SYNC__\n
  Server blocks until both synth and playback queues are drained,
  then sends: OK\n

Socket: $XDG_RUNTIME_DIR/pp-tts.sock (or /tmp/pp-tts.sock)
"""

import os
import sys
import signal
import socket
import subprocess
import tempfile
import threading
import time
import wave
import array
from collections import deque

# --- Configuration via environment ---
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.expanduser("~/sherpa-onnx/models/vits-piper-en_US-amy-low"),
)
SOCKET_PATH = os.environ.get(
    "SOCKET_PATH",
    os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "pp-tts.sock"),
)
NUM_THREADS = int(os.environ.get("NUM_THREADS", "2"))
SID = int(os.environ.get("SID", "0"))
SPEED = float(os.environ.get("SPEED", "1.0"))
MAX_TEXT_LEN = 2000
QUEUE_DEPTH = int(os.environ.get("QUEUE_DEPTH", "4"))

# Paths derived from model dir
MODEL_FILE = os.path.join(MODEL_DIR, "en_US-amy-low.onnx")
TOKENS_FILE = os.path.join(MODEL_DIR, "tokens.txt")
DATA_DIR = os.path.join(MODEL_DIR, "espeak-ng-data")

WAV_DIR = os.environ.get("XDG_RUNTIME_DIR", "/tmp")

_POISON = None


def load_tts():
    """Load sherpa-onnx TTS model. Dies hard if anything is missing."""
    import sherpa_onnx

    for p in (MODEL_FILE, TOKENS_FILE):
        if not os.path.isfile(p):
            print(f"FATAL: missing {p}", file=sys.stderr)
            sys.exit(1)

    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=MODEL_FILE,
                tokens=TOKENS_FILE,
                data_dir=DATA_DIR,
            ),
            num_threads=NUM_THREADS,
            provider="cpu",
        ),
    )
    tts = sherpa_onnx.OfflineTts(config)
    print(f"pp-tts: model loaded ({NUM_THREADS} threads)", file=sys.stderr)
    return tts


def samples_to_wav(samples, sample_rate):
    """Convert float32 samples to WAV file on tmpfs. Returns path."""
    int_samples = array.array("h",
        (int(max(-1.0, min(1.0, s)) * 32767) for s in samples)
    )
    fd, wav_path = tempfile.mkstemp(suffix=".wav", dir=WAV_DIR, prefix="pp-tts-")
    try:
        with os.fdopen(fd, "wb") as f:
            with wave.open(f, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(int_samples.tobytes())
        return wav_path
    except Exception:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        raise


def play_wav(wav_path):
    """Play a WAV file via the best available audio backend."""
    try:
        for cmd in (
            ["pw-play", wav_path],
            ["paplay", wav_path],
            ["aplay", "-q", wav_path],
        ):
            try:
                subprocess.run(cmd, check=True, timeout=30,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except (FileNotFoundError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired):
                continue
        print("pp-tts: WARN no audio backend played", file=sys.stderr)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


class TTSPipeline:
    """Three-stage pipeline: accept → synthesise → play.

    Synth and playback run on separate threads so synthesis of the
    next utterance overlaps playback of the current one. Both queues
    are bounded to cap memory on a 4GB device.
    """

    def __init__(self, tts):
        self.tts = tts
        self._running = True

        self._text_q_lock = threading.Lock()
        self._text_q_not_empty = threading.Condition(self._text_q_lock)
        self._text_q_not_full = threading.Condition(self._text_q_lock)
        self._text_q = deque()

        self._audio_q_lock = threading.Lock()
        self._audio_q_not_empty = threading.Condition(self._audio_q_lock)
        self._audio_q_not_full = threading.Condition(self._audio_q_lock)
        self._audio_q = deque()

        # Idle tracking: set when both queues are empty AND no
        # synth/playback is in progress
        self._synth_busy = False
        self._play_busy = False
        self._idle_event = threading.Event()
        self._idle_event.set()  # starts idle

        self._synth_thread = threading.Thread(
            target=self._synth_loop, name="tts-synth", daemon=True)
        self._play_thread = threading.Thread(
            target=self._play_loop, name="tts-play", daemon=True)
        self._synth_thread.start()
        self._play_thread.start()

    def enqueue(self, text):
        """Try to enqueue text for synthesis. Returns True if accepted."""
        with self._text_q_not_full:
            if len(self._text_q) >= QUEUE_DEPTH:
                return False
            self._text_q.append(text)
            self._idle_event.clear()
            self._text_q_not_empty.notify()
            return True

    def wait_idle(self, timeout=60.0):
        """Block until both queues are drained and nothing is playing."""
        return self._idle_event.wait(timeout=timeout)

    def _update_idle(self):
        """Check if pipeline is fully idle and signal if so."""
        with self._text_q_lock:
            text_empty = len(self._text_q) == 0
        with self._audio_q_lock:
            audio_empty = len(self._audio_q) == 0
        if text_empty and audio_empty and not self._synth_busy and not self._play_busy:
            self._idle_event.set()

    def shutdown(self):
        self._running = False
        with self._text_q_not_empty:
            self._text_q.append(_POISON)
            self._text_q_not_empty.notify()
        with self._audio_q_not_empty:
            self._audio_q.append(_POISON)
            self._audio_q_not_empty.notify()
        self._synth_thread.join(timeout=5)
        self._play_thread.join(timeout=5)

    def _synth_loop(self):
        while self._running:
            with self._text_q_not_empty:
                while not self._text_q and self._running:
                    self._text_q_not_empty.wait(timeout=1.0)
                if not self._text_q:
                    continue
                text = self._text_q.popleft()
                self._text_q_not_full.notify()

            if text is _POISON:
                break

            self._synth_busy = True
            try:
                t0 = time.monotonic()
                audio = self.tts.generate(text, sid=SID, speed=SPEED)
                if audio.samples is None or len(audio.samples) == 0:
                    print("pp-tts: synth produced no audio", file=sys.stderr)
                    continue

                t_synth = time.monotonic() - t0
                dur = len(audio.samples) / audio.sample_rate
                print(f"pp-tts: synth {dur:.1f}s audio in {t_synth:.2f}s "
                      f"(RTF {t_synth/dur:.3f})", file=sys.stderr)

                wav_path = samples_to_wav(audio.samples, audio.sample_rate)

                with self._audio_q_not_full:
                    while len(self._audio_q) >= QUEUE_DEPTH and self._running:
                        self._audio_q_not_full.wait(timeout=1.0)
                    self._audio_q.append(wav_path)
                    self._audio_q_not_empty.notify()

            except Exception as e:
                print(f"pp-tts: synth error: {e}", file=sys.stderr)
            finally:
                self._synth_busy = False
                self._update_idle()

    def _play_loop(self):
        while self._running:
            with self._audio_q_not_empty:
                while not self._audio_q and self._running:
                    self._audio_q_not_empty.wait(timeout=1.0)
                if not self._audio_q:
                    self._update_idle()
                    continue
                wav_path = self._audio_q.popleft()
                self._audio_q_not_full.notify()

            if wav_path is _POISON:
                break

            self._play_busy = True
            try:
                play_wav(wav_path)
            except Exception as e:
                print(f"pp-tts: play error: {e}", file=sys.stderr)
            finally:
                self._play_busy = False
                self._update_idle()


def handle_connection(conn, pipeline):
    """Read one text line, enqueue for synthesis, respond immediately.

    Special command __SYNC__: block until pipeline is fully idle.
    """
    try:
        conn.settimeout(10.0)
        data = b""
        while b"\n" not in data and len(data) < MAX_TEXT_LEN + 64:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk

        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            conn.sendall(b"ERR empty\n")
            return

        # Handle sync command
        if text == "__SYNC__":
            conn.settimeout(120.0)
            pipeline.wait_idle(timeout=60.0)
            conn.sendall(b"OK\n")
            return

        text = text[:MAX_TEXT_LEN]

        if pipeline.enqueue(text):
            conn.sendall(b"OK\n")
        else:
            conn.sendall(b"FULL\n")

    except Exception as e:
        print(f"pp-tts: connection error: {e}", file=sys.stderr)
        try:
            conn.sendall(f"ERR {e}\n".encode())
        except Exception:
            pass
    finally:
        conn.close()


def main():
    tts = load_tts()
    pipeline = TTSPipeline(tts)

    try:
        os.unlink(SOCKET_PATH)
    except FileNotFoundError:
        pass

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o660)
    srv.listen(8)
    srv.settimeout(None)

    running = True

    def _shutdown(signum, frame):
        nonlocal running
        running = False
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(SOCKET_PATH)
            s.close()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print(f"pp-tts: listening on {SOCKET_PATH} (queue depth {QUEUE_DEPTH})",
          file=sys.stderr)

    while running:
        try:
            conn, _ = srv.accept()
        except OSError:
            break
        if not running:
            conn.close()
            break
        handle_connection(conn, pipeline)

    pipeline.shutdown()
    srv.close()
    try:
        os.unlink(SOCKET_PATH)
    except FileNotFoundError:
        pass
    print("pp-tts: shutdown", file=sys.stderr)


if __name__ == "__main__":
    main()
