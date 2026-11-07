package com.udgamlabs.shruti.sample

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Reads a WAV file (PCM 8, 16, 24 or 32-bit, or float32) into 16 kHz mono samples. Other formats: use MediaCodec. */
object Wav {
    fun read(bytes: ByteArray): FloatArray {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(bytes.size > 12 && String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF" &&
            String(bytes, 8, 4, Charsets.US_ASCII) == "WAVE") { "This sample reads WAV files only" }
        var tag = 0; var channels = 0; var rate = 0; var bits = 0
        var start = -1; var length = 0
        var p = 12
        while (p + 8 <= bytes.size) {
            val id = String(bytes, p, 4, Charsets.US_ASCII)
            val len = b.getInt(p + 4)
            if (len < 0) break
            if (id == "fmt ") {
                tag = b.getShort(p + 8).toInt() and 0xffff
                channels = b.getShort(p + 10).toInt()
                rate = b.getInt(p + 12)
                bits = b.getShort(p + 22).toInt()
            } else if (id == "data") {
                start = p + 8
                length = minOf(len, bytes.size - start)
            }
            p += 8 + len + (len and 1)
        }
        require(start >= 0 && channels > 0 && bits in setOf(8, 16, 24, 32)) { "Unsupported WAV file" }
        val width = bits / 8
        val frames = length / width / channels
        val out = FloatArray(frames)
        for (i in 0 until frames) {
            var sum = 0f
            for (c in 0 until channels) {
                val o = start + (i * channels + c) * width
                sum += when {
                    tag == 3 -> b.getFloat(o)
                    width == 1 -> ((bytes[o].toInt() and 0xff) - 128) / 128f
                    width == 2 -> b.getShort(o) / 32768f
                    width == 3 -> ((bytes[o].toInt() and 0xff) or ((bytes[o + 1].toInt() and 0xff) shl 8) or
                        (bytes[o + 2].toInt() shl 16)) / 8388608f
                    else -> b.getInt(o) / 2147483648f
                }
            }
            out[i] = sum / channels
        }
        return resample(out, rate)
    }

    /** Linear resampling to 16 kHz, the same as the JavaScript package. */
    private fun resample(x: FloatArray, rate: Int): FloatArray {
        val target = ShrutiEngine.SAMPLE_RATE
        if (rate == target || x.isEmpty()) return x
        val n = Math.round(x.size.toDouble() * target / rate).toInt()
        return FloatArray(n) { i ->
            val t = i.toDouble() * rate / target
            val j = t.toInt()
            val f = (t - j).toFloat()
            if (j + 1 < x.size) x[j] * (1 - f) + x[j + 1] * f else x[x.size - 1]
        }
    }
}
