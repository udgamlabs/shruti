package com.udgamlabs.shruti.sample

import android.content.res.AssetManager
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineMoonshineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.SileroVadModelConfig
import com.k2fsa.sherpa.onnx.Vad
import com.k2fsa.sherpa.onnx.VadModelConfig
import kotlin.math.abs
import kotlin.math.min

/**
 * The part to copy into your own app: one language's Shruti pack, plus a voice-activity detector (VAD) to split long
 * audio and skip quiet. Create it off the main thread (it takes a few seconds), and call it from one thread at a time.
 *
 * This is everything needed to use a pack. There is no language option: sherpa-onnx's Moonshine API always detects
 * between the pack's language and English.
 *
 * @param assets pass the AssetManager if the pack is under assets/, or null if `dir` is a folder on disk.
 */
class ShrutiEngine(private val assets: AssetManager, dir: String, packInAssets: Boolean, threads: Int = 2) {

    private val recognizer = OfflineRecognizer(
        assetManager = if (packInAssets) assets else null,
        config = OfflineRecognizerConfig(
            modelConfig = OfflineModelConfig(
                moonshine = OfflineMoonshineModelConfig(
                    preprocessor = "$dir/preprocess.onnx",
                    encoder = "$dir/encode.onnx",
                    uncachedDecoder = "$dir/uncached_decode.onnx",
                    cachedDecoder = "$dir/cached_decode.onnx",
                ),
                tokens = "$dir/tokens.txt",
                numThreads = threads, // 2 was the fastest on the phone we tested
            ),
        ),
    )

    /** Text for one piece of speech of up to 30 s, 16 kHz mono samples in [-1, 1]. */
    fun decode(samples: FloatArray): String {
        if (isDigitalSilence(samples)) return ""
        val stream = recognizer.createStream()
        try {
            stream.acceptWaveform(samples, SAMPLE_RATE)
            recognizer.decode(stream)
            return recognizer.getResult(stream).text.trim()
        } finally {
            stream.release()
        }
    }

    /** Any length of audio. Up to 30 s is decoded whole; longer audio is split at pauses by the VAD first. */
    fun transcribe(samples: FloatArray, onText: (String) -> Unit) {
        if (samples.size <= ONE_PASS_SECONDS * SAMPLE_RATE) {
            onText(decode(samples))
            return
        }
        val splitter = SpeechSplitter(assets)
        try {
            var i = 0
            while (i < samples.size) {
                val end = min(i + SpeechSplitter.WINDOW, samples.size)
                splitter.accept(samples.copyOfRange(i, end)) { onText(decode(it)) }
                i = end
            }
            splitter.finish { onText(decode(it)) }
        } finally {
            splitter.release()
        }
    }

    fun release() = recognizer.release()

    companion object {
        const val SAMPLE_RATE = 16000
        const val ONE_PASS_SECONDS = 30

        // The same test as the Python and JavaScript packages. Real-world quiet is not silent: use the VAD for that.
        fun isDigitalSilence(x: FloatArray) = x.isEmpty() || x.all { abs(it) < 1e-4f }
    }
}

/**
 * Cuts a stream of audio into pieces of speech with sherpa-onnx's VAD. Pieces never run past 28 s, because the pack starts
 * repeating itself on long unsplit audio. Single-threaded: feed it from one thread.
 */
class SpeechSplitter(assets: AssetManager) {
    private val vad = Vad(
        assetManager = assets,
        config = VadModelConfig(
            sileroVadModelConfig = SileroVadModelConfig(
                model = "silero_vad.onnx",
                threshold = 0.5f,
                minSilenceDuration = 0.25f,
                minSpeechDuration = 0.25f,
                windowSize = WINDOW,
                maxSpeechDuration = 28f,
            ),
            sampleRate = ShrutiEngine.SAMPLE_RATE,
            numThreads = 1,
        ),
    )
    private val pending = FloatArray(WINDOW)
    private var filled = 0

    /** Takes any number of samples; calls [onSpeech] for each piece of speech that has ended. */
    fun accept(samples: FloatArray, onSpeech: (FloatArray) -> Unit) {
        var i = 0
        while (i < samples.size) {
            val n = min(WINDOW - filled, samples.size - i)
            samples.copyInto(pending, filled, i, i + n)
            filled += n
            i += n
            if (filled == WINDOW) { // the detector takes exactly one window at a time
                vad.acceptWaveform(pending.copyOf())
                filled = 0
                drain(onSpeech)
            }
        }
    }

    /** End of audio: hands over whatever speech is still open. */
    fun finish(onSpeech: (FloatArray) -> Unit) {
        if (filled > 0) {
            pending.fill(0f, filled, WINDOW)
            vad.acceptWaveform(pending.copyOf())
            filled = 0
        }
        vad.flush()
        drain(onSpeech)
    }

    private fun drain(onSpeech: (FloatArray) -> Unit) {
        while (!vad.empty()) {
            onSpeech(vad.front().samples)
            vad.pop()
        }
    }

    fun release() = vad.release()

    companion object {
        const val WINDOW = 512
    }
}
