// Transcribe a WAV file with a Shruti model through the Cactus engine's C library.
// The same calls work on every platform the engine ships a library for; only the file you link changes.
//
//   transcribe model.cact clip.wav            whole clip at once
//   transcribe model.cact clip.wav --live     fed one second at a time, printing words as they are committed
//
// Input: 16-bit PCM WAV, 16 kHz mono. Output: one line of JSON per result.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cactus.h"

#define OUT_SIZE (1 << 20)

static unsigned char *read_file(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    rewind(f);
    unsigned char *data = malloc((size_t)*size);
    if (fread(data, 1, (size_t)*size, f) != (size_t)*size) { perror(path); exit(1); }
    fclose(f);
    return data;
}

// Finds the WAV's "data" chunk and returns the samples as floats in [-1, 1]. Anything but 16-bit PCM, 16 kHz, mono is refused.
static float *read_wav(const char *path, int *samples) {
    long size;
    unsigned char *w = read_file(path, &size);
    long at = 12;
    while (at + 8 <= size && memcmp(w + at, "data", 4) != 0) {
        unsigned n = w[at + 4] | w[at + 5] << 8 | w[at + 6] << 16 | (unsigned)w[at + 7] << 24;
        if (memcmp(w + at, "fmt ", 4) == 0 && at + 24 <= size) {
            unsigned format = w[at + 8] | w[at + 9] << 8, channels = w[at + 10] | w[at + 11] << 8;
            unsigned rate = w[at + 12] | w[at + 13] << 8 | w[at + 14] << 16 | (unsigned)w[at + 15] << 24, bits = w[at + 22] | w[at + 23] << 8;
            if (format != 1 || channels != 1 || rate != 16000 || bits != 16) {
                fprintf(stderr, "%s: need 16-bit PCM, 16 kHz, mono; this is format %u, %u channels, %u Hz, %u bits\n", path, format, channels, rate, bits);
                exit(1);
            }
        }
        at += 8 + n + (n & 1);
    }
    if (at + 8 > size) { fprintf(stderr, "%s: no data chunk\n", path); exit(1); }
    unsigned n = w[at + 4] | w[at + 5] << 8 | w[at + 6] << 16 | (unsigned)w[at + 7] << 24;
    if (at + 8 + n > (unsigned long)size) n = (unsigned)(size - at - 8);
    *samples = (int)(n / 2);
    float *pcm = malloc(sizeof(float) * (size_t)*samples);
    for (int i = 0; i < *samples; i++) {
        short s = (short)(w[at + 8 + 2 * i] | w[at + 9 + 2 * i] << 8);
        pcm[i] = (float)s / 32768.0f;
    }
    free(w);
    return pcm;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.cact clip.wav [--live]\n", argv[0]); return 2; }
    long model_size;
    unsigned char *model = read_file(argv[1], &model_size);
    if (cactus_load(model, (unsigned long long)model_size) < 0) { fprintf(stderr, "load: %s\n", cactus_last_error()); return 1; }

    int samples;
    float *pcm = read_wav(argv[2], &samples);
    char *out = malloc(OUT_SIZE);

    if (argc > 3 && strcmp(argv[3], "--live") == 0) {
        for (int at = 0; at < samples; at += 16000) {
            int n = samples - at < 16000 ? samples - at : 16000;
            if (cactus_stream_transcribe_process(pcm + at, n, NULL, NULL, out, OUT_SIZE) < 0) { fprintf(stderr, "stream: %s\n", cactus_last_error()); return 1; }
            puts(out);
        }
        if (cactus_stream_transcribe_stop(out, OUT_SIZE) < 0) { fprintf(stderr, "stop: %s\n", cactus_last_error()); return 1; }
        puts(out);
    } else {
        // NULL language detects it; pass "en" for speech that is entirely English.
        if (cactus_transcribe(pcm, samples, NULL, NULL, 0, out, OUT_SIZE) < 0) { fprintf(stderr, "transcribe: %s\n", cactus_last_error()); return 1; }
        puts(out);
    }
    return 0;
}
