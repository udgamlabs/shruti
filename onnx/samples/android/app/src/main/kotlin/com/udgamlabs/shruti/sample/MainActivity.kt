package com.udgamlabs.shruti.sample

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.Spinner
import android.widget.TextView
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import kotlin.math.abs

/**
 * A small working example: pick a language, then open a WAV file or record from the microphone. Everything runs on the
 * phone. The app has no INTERNET permission. ShrutiEngine.kt is the part to copy into your own app.
 *
 * A language's pack is one of:
 *   app/src/main/assets/models/<code>/        bundled in the APK (see scripts/get_pack.sh)
 *   <external files dir>/models/<code>/       pushed with adb, or downloaded by your own app
 */
class MainActivity : Activity() {

    private lateinit var languageView: Spinner
    private lateinit var openFile: Button
    private lateinit var record: Button
    private lateinit var status: TextView
    private lateinit var transcript: TextView

    private val worker = Executors.newSingleThreadExecutor() // loading and decoding, one at a time
    private var engine: ShrutiEngine? = null
    private var packs: List<Pack> = emptyList()
    @Volatile private var recording = false

    private data class Pack(val code: String, val dir: String, val inAssets: Boolean) {
        override fun toString() = "${LANGUAGES[code] ?: code} ($code)"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        languageView = findViewById(R.id.language)
        openFile = findViewById(R.id.open_file)
        record = findViewById(R.id.record)
        status = findViewById(R.id.status)
        transcript = findViewById(R.id.transcript)

        // Android slows the CPU a lot once the screen is off, so keep it on while the app is open.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        openFile.setOnClickListener {
            startActivityForResult(
                Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("audio/*"), PICK_WAV,
            )
        }
        record.setOnClickListener { if (recording) recording = false else startRecording() }

        packs = findPacks()
        if (packs.isEmpty()) {
            status.text = "No language pack found. See samples/android/README.md."
            return
        }
        languageView.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, packs)
        val prefs = getPreferences(MODE_PRIVATE)
        val wanted = (if (BuildConfig.DEBUG) intent.getStringExtra("language") else null)
            ?: prefs.getString("language", null)
        languageView.setSelection(packs.indexOfFirst { it.code == wanted }.coerceAtLeast(0))
        languageView.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                prefs.edit().putString("language", packs[position].code).apply()
                load(packs[position])
            }
            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
    }

    private fun findPacks(): List<Pack> {
        val fromAssets = assets.list("models").orEmpty().map { Pack(it, "models/$it", true) }
        val folder = File(getExternalFilesDir(null), "models")
        val fromDisk = folder.listFiles().orEmpty()
            .filter { File(it, "tokens.txt").exists() }
            .map { Pack(it.name, it.path, false) }
        return (fromAssets + fromDisk).distinctBy { it.code }.sortedBy { it.toString() }
    }

    private fun load(pack: Pack) {
        setBusy("Loading $pack…")
        worker.execute {
            val t0 = SystemClock.elapsedRealtime()
            engine?.release()
            engine = null
            val next = try {
                ShrutiEngine(assets, pack.dir, pack.inAssets)
            } catch (e: Throwable) {
                runOnUiThread { status.text = "Could not load $pack: ${e.message}" }
                return@execute
            }
            engine = next
            val ms = SystemClock.elapsedRealtime() - t0
            Log.i(TAG, "LOADED ${pack.code} ms=$ms peakKb=${peakMemoryKb()}")
            runOnUiThread {
                status.text = "$pack loaded in ${"%.1f".format(ms / 1000f)} s. Language is detected: ${LANGUAGES[pack.code]} or English."
                setReady(true)
            }
            if (BuildConfig.DEBUG) intent.getStringExtra("bench_dir")?.let { dir ->
                intent.removeExtra("bench_dir")
                bench(pack, next, File(dir))
            }
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val uri = data?.data
        if (requestCode == PICK_WAV && resultCode == RESULT_OK && uri != null) transcribeFile(uri)
    }

    private fun transcribeFile(uri: Uri) {
        setBusy("Transcribing…")
        transcript.text = ""
        worker.execute {
            try {
                val bytes = contentResolver.openInputStream(uri)!!.use { it.readBytes() }
                val samples = Wav.read(bytes)
                val t0 = SystemClock.elapsedRealtime()
                engine!!.transcribe(samples) { appendText(it) }
                val ms = SystemClock.elapsedRealtime() - t0
                val seconds = samples.size / ShrutiEngine.SAMPLE_RATE.toFloat()
                showDone("%.1f s of audio in %.1f s. Peak memory %d MB.".format(seconds, ms / 1000f, peakMemoryKb() / 1024))
            } catch (e: Exception) {
                showDone("Could not transcribe: ${e.message}")
            }
        }
    }

    /**
     * Records at 16 kHz and cuts the audio at pauses as it goes, so text appears phrase by phrase. The microphone is read on
     * its own thread so that decoding one phrase never drops the next one.
     */
    @SuppressLint("MissingPermission") // checked just below
    private fun startRecording() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), ASK_MIC)
            return
        }
        val rate = ShrutiEngine.SAMPLE_RATE
        val minBuffer = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val mic = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION, rate, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuffer, rate),
        )
        if (mic.state != AudioRecord.STATE_INITIALIZED) {
            mic.release()
            status.text = "Could not open the microphone."
            return
        }
        recording = true
        record.text = "Stop"
        openFile.isEnabled = false
        languageView.isEnabled = false
        status.text = "Listening…"
        transcript.text = ""
        Thread {
            val splitter = SpeechSplitter(assets)
            val pcm = ShortArray(rate / 10)
            var samplesRead = 0L
            var pieces = 0
            var loudest = 0f
            // Debug builds only, for testing with no one speaking: plays a clip through the phone's speaker while recording.
            val testClip = if (BuildConfig.DEBUG) intent.getStringExtra("mic_test_wav")?.let {
                MediaPlayer().apply { setDataSource(it); prepare(); start() }
            } else null
            val onSpeech = { speech: FloatArray ->
                pieces++
                onWorker { engine?.let { appendText(it.decode(speech)) } }
            }
            mic.startRecording()
            while (recording) {
                val n = mic.read(pcm, 0, pcm.size)
                if (n <= 0) continue
                samplesRead += n
                val samples = FloatArray(n) { pcm[it] / 32768f }
                samples.forEach { loudest = maxOf(loudest, abs(it)) }
                splitter.accept(samples, onSpeech)
            }
            mic.stop()
            mic.release()
            testClip?.release()
            splitter.finish(onSpeech)
            splitter.release()
            Log.i(TAG, "MIC seconds=${samplesRead / rate} speechPieces=$pieces loudest=$loudest")
            onWorker { showDone("Recorded ${samplesRead / rate} s, $pieces piece(s) of speech.") }
        }.start()
    }

    /** The recording thread can outlive the screen; after that, its remaining work is dropped. */
    private fun onWorker(block: () -> Unit) {
        try {
            worker.execute(block)
        } catch (_: RejectedExecutionException) {
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, results: IntArray) {
        if (requestCode == ASK_MIC && results.firstOrNull() == PackageManager.PERMISSION_GRANTED) startRecording()
    }

    /** Debug builds only, for measuring: decodes every WAV in a folder and writes timings to bench.jsonl. */
    private fun bench(pack: Pack, asr: ShrutiEngine, dir: File) {
        val out = File(getExternalFilesDir(null), "bench.jsonl").bufferedWriter()
        dir.listFiles().orEmpty().filter { it.name.endsWith(".wav") }.sortedBy { it.name }.forEach { f ->
            val samples = Wav.read(f.readBytes())
            val t0 = SystemClock.elapsedRealtime()
            val parts = mutableListOf<String>()
            asr.transcribe(samples) { parts += it }
            val ms = SystemClock.elapsedRealtime() - t0
            val text = parts.filter { it.isNotEmpty() }.joinToString(" ").replace("\\", "\\\\").replace("\"", "\\\"")
            out.write("""{"clip":"${f.name}","lang":"${pack.code}","audio_s":${samples.size / 16000.0},"ms":$ms,"peak_kb":${peakMemoryKb()},"text":"$text"}""")
            out.newLine()
            out.flush()
            Log.i(TAG, "BENCH ${f.name} ms=$ms")
        }
        out.close()
        Log.i(TAG, "BENCHDONE")
    }

    private fun peakMemoryKb(): Long = File("/proc/self/status").readLines()
        .firstOrNull { it.startsWith("VmHWM") }?.filter { it.isDigit() }?.toLong() ?: 0

    private fun appendText(text: String) {
        if (text.isEmpty()) return
        Log.i(TAG, "TEXT $text")
        runOnUiThread { transcript.append(if (transcript.text.isEmpty()) text else " $text") }
    }

    private fun setBusy(message: String) {
        status.text = message
        setReady(false)
    }

    private fun setReady(ready: Boolean) {
        openFile.isEnabled = ready
        record.isEnabled = ready
        languageView.isEnabled = ready
    }

    private fun showDone(message: String) = runOnUiThread {
        record.text = "Record"
        status.text = message
        setReady(true)
    }

    override fun onDestroy() {
        recording = false
        worker.execute { engine?.release(); engine = null }
        worker.shutdown()
        super.onDestroy()
    }

    override fun onStart() {
        super.onStart()
        if (BuildConfig.DEBUG && Build.VERSION.SDK_INT >= 27) { // lets adb-driven tests run on a locked test phone
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        }
    }

    companion object {
        private const val TAG = "ShrutiSample"
        private const val PICK_WAV = 1
        private const val ASK_MIC = 2

        val LANGUAGES = mapOf(
            "as" to "Assamese", "bn" to "Bengali", "brx" to "Bodo", "doi" to "Dogri", "gu" to "Gujarati", "hi" to "Hindi",
            "kn" to "Kannada", "ks" to "Kashmiri", "kok" to "Konkani", "mai" to "Maithili", "ml" to "Malayalam",
            "mni" to "Manipuri", "mr" to "Marathi", "ne" to "Nepali", "or" to "Odia", "pa" to "Punjabi", "sa" to "Sanskrit",
            "sat" to "Santali", "sd" to "Sindhi", "ta" to "Tamil", "te" to "Telugu", "ur" to "Urdu",
        )
    }
}
