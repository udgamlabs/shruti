// Runs ShrutiTranscriber.swift as a Mac command: main <pack folder> <silero_vad.onnx> clip.wav
let args = CommandLine.arguments
guard args.count == 4 else { fatalError("usage: main <pack folder> <silero_vad.onnx> clip.wav") }
let wave = SherpaOnnxWaveWrapper.readWave(filename: args[3])
let asr = ShrutiTranscriber(pack: args[1], vadModel: args[2])
print(asr.transcribe(wave.samples).joined(separator: " "))
