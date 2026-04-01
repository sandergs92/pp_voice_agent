/*
 * denoise_source.c — PipeWire source node with real-time speech enhancement
 *
 * Reads raw audio from ALSA DMIC (hw:0,0), runs sherpa-onnx streaming
 * denoiser (gtcrn), and presents clean audio as a PipeWire source node.
 * Apps see "Clean Mic" as an audio source — no filter routing, no null-sink.
 *
 * Build:  make -f Makefile
 * Run:    ./denoise_source ~/models/gtcrn_simple.onnx
 *
 * Copyright 2026, PinePhone Voice Agent project. MIT licence.
 */

#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>

#include <sherpa-onnx/c-api/c-api.h>

#include <alsa/asoundlib.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── defaults ────────────────────────────────────────────────────────── */
#define SAMPLE_RATE     16000
#define ALSA_CHANNELS   2       /* DMIC is stereo-only at hw level */
#define ALSA_DEVICE     "hw:0,0"
#define ALSA_PERIOD     256     /* samples per period */
#define ALSA_PERIODS    4

/* ring buffer: power-of-2, holds ~500ms of denoised mono audio */
#define RING_BITS       13      /* 8192 samples */
#define RING_SIZE       (1 << RING_BITS)
#define RING_MASK       (RING_SIZE - 1)

/* ── state ───────────────────────────────────────────────────────────── */
struct source_data {
    /* PipeWire */
    struct pw_main_loop *loop;
    struct pw_stream    *stream;
    struct spa_io_position *position;

    /* denoiser */
    const SherpaOnnxOnlineSpeechDenoiser *denoiser;
    int32_t frame_shift;

    /* ALSA capture thread */
    pthread_t capture_thread;
    volatile int running;

    /* lock-free SPSC ring: capture thread writes, PW thread reads */
    float    ring[RING_SIZE];
    volatile int32_t ring_wr;   /* written by capture thread only */
    volatile int32_t ring_rd;   /* written by PW thread only */
};

/* ── ring helpers (SPSC: single producer, single consumer) ────────── */
static inline int32_t ring_avail(const struct source_data *d)
{
    return d->ring_wr - d->ring_rd;
}

static inline void ring_write(struct source_data *d,
                               const float *src, int32_t n)
{
    for (int32_t i = 0; i < n; i++)
        d->ring[(d->ring_wr + i) & RING_MASK] = src[i];
    __atomic_store_n(&d->ring_wr, d->ring_wr + n, __ATOMIC_RELEASE);
}

static inline int32_t ring_read(struct source_data *d,
                                 float *dst, int32_t n)
{
    int32_t avail = ring_avail(d);
    if (avail < n) n = avail;
    for (int32_t i = 0; i < n; i++)
        dst[i] = d->ring[(d->ring_rd + i) & RING_MASK];
    __atomic_store_n(&d->ring_rd, d->ring_rd + n, __ATOMIC_RELEASE);
    return n;
}

/* ── ALSA capture + denoise thread ───────────────────────────────────── */
static void *capture_thread_fn(void *arg)
{
    struct source_data *d = arg;
    snd_pcm_t *pcm = NULL;
    int err;

    err = snd_pcm_open(&pcm, ALSA_DEVICE, SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
        fprintf(stderr, "ALSA open failed: %s\n", snd_strerror(err));
        return NULL;
    }

    /* configure ALSA: S16_LE stereo 16kHz */
    snd_pcm_hw_params_t *hw;
    snd_pcm_hw_params_alloca(&hw);
    snd_pcm_hw_params_any(pcm, hw);
    snd_pcm_hw_params_set_access(pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(pcm, hw, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(pcm, hw, ALSA_CHANNELS);
    unsigned int rate = SAMPLE_RATE;
    snd_pcm_hw_params_set_rate_near(pcm, hw, &rate, NULL);
    snd_pcm_uframes_t period = ALSA_PERIOD;
    snd_pcm_hw_params_set_period_size_near(pcm, hw, &period, NULL);
    unsigned int periods = ALSA_PERIODS;
    snd_pcm_hw_params_set_periods_near(pcm, hw, &periods, NULL);

    err = snd_pcm_hw_params(pcm, hw);
    if (err < 0) {
        fprintf(stderr, "ALSA hw_params failed: %s\n", snd_strerror(err));
        snd_pcm_close(pcm);
        return NULL;
    }

    fprintf(stderr, "ALSA: %s %uHz period=%lu periods=%u\n",
            ALSA_DEVICE, rate, period, periods);

    /* buffers: interleaved S16 stereo → mono float */
    int16_t *ibuf = malloc(period * ALSA_CHANNELS * sizeof(int16_t));
    float   *mono = malloc(period * sizeof(float));
    if (!ibuf || !mono) {
        fprintf(stderr, "capture: malloc failed\n");
        free(ibuf); free(mono);
        snd_pcm_close(pcm);
        return NULL;
    }

    while (d->running) {
        snd_pcm_sframes_t frames = snd_pcm_readi(pcm, ibuf, period);
        if (frames < 0) {
            frames = snd_pcm_recover(pcm, (int)frames, 1);
            if (frames < 0) {
                fprintf(stderr, "ALSA read error: %s\n",
                        snd_strerror((int)frames));
                break;
            }
            continue;
        }

        /* stereo S16 → mono float [-1,1] */
        for (snd_pcm_sframes_t i = 0; i < frames; i++) {
            int32_t sum = (int32_t)ibuf[i * ALSA_CHANNELS]
                        + (int32_t)ibuf[i * ALSA_CHANNELS + 1];
            mono[i] = (float)sum / 65536.0f;
        }

        /* feed denoiser in frame_shift-sized chunks */
        int32_t pos = 0;
        while (pos < (int32_t)frames) {
            int32_t chunk = (int32_t)frames - pos;
            if (chunk > d->frame_shift)
                chunk = d->frame_shift;

            const SherpaOnnxDenoisedAudio *out =
                SherpaOnnxOnlineSpeechDenoiserRun(
                    d->denoiser, mono + pos, chunk, SAMPLE_RATE);

            if (out) {
                /* drop samples if ring is nearly full to avoid stale data */
                if (ring_avail(d) + out->n < RING_SIZE)
                    ring_write(d, out->samples, out->n);
                SherpaOnnxDestroyDenoisedAudio(out);
            }
            pos += chunk;
        }
    }

    free(ibuf);
    free(mono);
    snd_pcm_close(pcm);
    return NULL;
}

/* ── PipeWire stream callback ────────────────────────────────────────── */
static void on_process(void *userdata)
{
    struct source_data *d = userdata;

    struct pw_buffer *buf = pw_stream_dequeue_buffer(d->stream);
    if (!buf)
        return;

    struct spa_buffer *sbuf = buf->buffer;
    float *dst = sbuf->datas[0].data;
    if (!dst) {
        pw_stream_queue_buffer(d->stream, buf);
        return;
    }

    uint32_t n_frames = sbuf->datas[0].maxsize / sizeof(float);
    if (buf->requested && buf->requested < n_frames)
        n_frames = buf->requested;

    /* fill from ring, zero-pad if not enough data yet */
    int32_t got = ring_read(d, dst, (int32_t)n_frames);
    if (got < (int32_t)n_frames)
        memset(dst + got, 0, ((int32_t)n_frames - got) * sizeof(float));

    sbuf->datas[0].chunk->offset = 0;
    sbuf->datas[0].chunk->size = n_frames * sizeof(float);
    sbuf->datas[0].chunk->stride = sizeof(float);

    pw_stream_queue_buffer(d->stream, buf);
}

static const struct pw_stream_events stream_events = {
    PW_VERSION_STREAM_EVENTS,
    .process = on_process,
};

/* ── signal handling ─────────────────────────────────────────────────── */
static struct source_data *g_data = NULL;

static void on_signal(int sig)
{
    (void)sig;
    if (g_data && g_data->loop)
        pw_main_loop_quit(g_data->loop);
}

/* ── main ────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.onnx>\n", argv[0]);
        return 1;
    }

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
        fprintf(stderr, "Failed to create denoiser from %s\n", argv[1]);
        return 1;
    }

    int32_t frame_shift =
        SherpaOnnxOnlineSpeechDenoiserGetFrameShiftInSamples(denoiser);
    fprintf(stderr, "Denoiser: frame_shift=%d (%.1fms)\n",
            frame_shift, frame_shift * 1000.0 / SAMPLE_RATE);

    /* ── init PipeWire ─────────────────────────────────────────────── */
    pw_init(&argc, &argv);

    struct source_data data;
    memset(&data, 0, sizeof(data));
    data.denoiser = denoiser;
    data.frame_shift = frame_shift;
    data.running = 1;
    g_data = &data;

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    data.loop = pw_main_loop_new(NULL);
    if (!data.loop) {
        fprintf(stderr, "Failed to create PipeWire main loop\n");
        SherpaOnnxDestroyOnlineSpeechDenoiser(denoiser);
        return 1;
    }

    struct pw_properties *props = pw_properties_new(
        PW_KEY_MEDIA_TYPE,       "Audio",
        PW_KEY_MEDIA_CATEGORY,   "Capture",
        PW_KEY_MEDIA_ROLE,       "Production",
        PW_KEY_NODE_NAME,        "clean_mic",
        PW_KEY_NODE_DESCRIPTION, "Clean Mic (Denoised)",
        PW_KEY_MEDIA_CLASS,      "Audio/Source",
        NULL);

    data.stream = pw_stream_new_simple(
        pw_main_loop_get_loop(data.loop),
        "clean_mic",
        props,
        &stream_events,
        &data);

    if (!data.stream) {
        fprintf(stderr, "Failed to create PipeWire stream\n");
        pw_main_loop_destroy(data.loop);
        SherpaOnnxDestroyOnlineSpeechDenoiser(denoiser);
        return 1;
    }

    /* define output format: mono float32 16kHz */
    uint8_t pod_buf[1024];
    struct spa_pod_builder b = SPA_POD_BUILDER_INIT(pod_buf, sizeof(pod_buf));
    const struct spa_pod *params[1];

    params[0] = spa_format_audio_raw_build(
        &b, SPA_PARAM_EnumFormat,
        &SPA_AUDIO_INFO_RAW_INIT(
            .format = SPA_AUDIO_FORMAT_F32,
            .channels = 1,
            .rate = SAMPLE_RATE));

    if (pw_stream_connect(data.stream,
                          PW_DIRECTION_OUTPUT,
                          PW_ID_ANY,
                          PW_STREAM_FLAG_MAP_BUFFERS |
                          PW_STREAM_FLAG_RT_PROCESS,
                          params, 1) < 0) {
        fprintf(stderr, "Failed to connect PipeWire stream\n");
        pw_stream_destroy(data.stream);
        pw_main_loop_destroy(data.loop);
        SherpaOnnxDestroyOnlineSpeechDenoiser(denoiser);
        return 1;
    }

    /* ── start ALSA capture thread ─────────────────────────────────── */
    if (pthread_create(&data.capture_thread, NULL,
                       capture_thread_fn, &data) != 0) {
        fprintf(stderr, "Failed to start capture thread\n");
        pw_stream_destroy(data.stream);
        pw_main_loop_destroy(data.loop);
        SherpaOnnxDestroyOnlineSpeechDenoiser(denoiser);
        return 1;
    }

    fprintf(stderr, "Clean Mic source running — Ctrl+C to stop\n");
    pw_main_loop_run(data.loop);

    /* ── cleanup ───────────────────────────────────────────────────── */
    data.running = 0;
    pthread_join(data.capture_thread, NULL);

    pw_stream_destroy(data.stream);
    pw_main_loop_destroy(data.loop);
    pw_deinit();
    SherpaOnnxDestroyOnlineSpeechDenoiser(denoiser);

    fprintf(stderr, "Clean Mic source stopped\n");
    return 0;
}
