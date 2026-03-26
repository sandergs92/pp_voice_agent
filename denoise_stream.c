/*
 * denoise_stream.c — streaming speech enhancement for pipe-based audio
 *
 * Reads raw float32 mono 16kHz audio from stdin, runs sherpa-onnx
 * streaming denoiser (gtcrn), writes clean audio to stdout.
 *
 * Usage:
 *   pw-cat --record --target <mic> --rate 16000 --channels 1 --format f32 - \
 *     | ./denoise_stream ~/models/gtcrn_simple.onnx \
 *     | pw-cat --playback --target <virtual_sink> --rate 16000 --channels 1 --format f32 -
 *
 * Build:  make -f Makefile
 *
 * Copyright 2025, PinePhone Voice Agent project. MIT licence.
 */

#include <sherpa-onnx/c-api/c-api.h>

#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SAMPLE_RATE 16000

static volatile sig_atomic_t g_running = 1;

static void on_signal(int sig)
{
    (void)sig;
    g_running = 0;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.onnx>\n", argv[0]);
        return 1;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);  /* don't die if downstream closes */

    /* ── init denoiser ─────────────────────────────────────────────── */
    SherpaOnnxOnlineSpeechDenoiserConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.model.gtcrn.model = argv[1];
    cfg.model.num_threads = 1;
    cfg.model.debug = 0;
    cfg.model.provider = "cpu";

    const SherpaOnnxOnlineSpeechDenoiser *denoiser =
        SherpaOnnxCreateOnlineSpeechDenoiser(&cfg);
    if (!denoiser) {
        fprintf(stderr, "denoise_stream: failed to load model %s\n", argv[1]);
        return 1;
    }

    int32_t frame_shift =
        SherpaOnnxOnlineSpeechDenoiserGetFrameShiftInSamples(denoiser);
    fprintf(stderr, "denoise_stream: ready  frame_shift=%d (%.1fms)\n",
            frame_shift, frame_shift * 1000.0 / SAMPLE_RATE);

    /* ── read/process/write loop ───────────────────────────────────── */
    float *buf = malloc(frame_shift * sizeof(float));
    if (!buf) {
        fprintf(stderr, "denoise_stream: malloc failed\n");
        SherpaOnnxDestroyOnlineSpeechDenoiser(denoiser);
        return 1;
    }

    while (g_running) {
        /* read exactly frame_shift samples from stdin */
        size_t total = 0;
        size_t need = frame_shift * sizeof(float);
        uint8_t *p = (uint8_t *)buf;

        while (total < need) {
            ssize_t n = read(STDIN_FILENO, p + total, need - total);
            if (n <= 0) {
                if (n == 0)
                    goto done;  /* EOF */
                if (errno == EINTR)
                    continue;
                goto done;
            }
            total += (size_t)n;
        }

        /* denoise */
        const SherpaOnnxDenoisedAudio *out =
            SherpaOnnxOnlineSpeechDenoiserRun(
                denoiser, buf, frame_shift, SAMPLE_RATE);

        if (out) {
            /* write denoised samples to stdout */
            size_t to_write = out->n * sizeof(float);
            size_t written = 0;
            const uint8_t *op = (const uint8_t *)out->samples;

            while (written < to_write) {
                ssize_t n = write(STDOUT_FILENO, op + written,
                                  to_write - written);
                if (n <= 0) {
                    if (errno == EINTR)
                        continue;
                    SherpaOnnxDestroyDenoisedAudio(out);
                    goto done;
                }
                written += (size_t)n;
            }
            SherpaOnnxDestroyDenoisedAudio(out);
        }
    }

done:
    /* flush any remaining buffered audio */
    {
        const SherpaOnnxDenoisedAudio *tail =
            SherpaOnnxOnlineSpeechDenoiserFlush(denoiser);
        if (tail) {
            size_t to_write = tail->n * sizeof(float);
            size_t written = 0;
            const uint8_t *op = (const uint8_t *)tail->samples;

            while (written < to_write) {
                ssize_t n = write(STDOUT_FILENO, op + written,
                                  to_write - written);
                if (n <= 0)
                    break;
                written += (size_t)n;
            }
            SherpaOnnxDestroyDenoisedAudio(tail);
        }
    }

    free(buf);
    SherpaOnnxDestroyOnlineSpeechDenoiser(denoiser);
    fprintf(stderr, "denoise_stream: stopped\n");
    return 0;
}
