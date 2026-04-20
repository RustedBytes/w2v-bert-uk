import io.github.rustedbytes.w2vbertuk.W2vBertUk

import java.nio.file.Paths

object Transcribe {
  def main(args: Array[String]): Unit = {
    val root = Paths.get("").toAbsolutePath

    System.loadLibrary("w2v_bert_uk")

    val options = W2vBertUk.Options.defaults()
    options.model = root.resolve("model_optimized.onnx").toString
    options.tokenizer = root.resolve("tokenizer.model").toString
    options.lm = root.resolve("news-titles.arpa").toString
    options.beamWidth = 32
    options.lmWeight = 0.45f
    options.wordBonus = 0.2f
    options.logLanguageModel = false
    options.ortOptimization = "disable"
    options.logAccelerator = true
    options.fallbackSampleRate = 16000
    options.skipDecodeErrors = true
    options.blankId = 0
    options.nBest = 32
    options.normalizeSpaces = true
    options.dropEmptyCandidates = true
    options.lmBos = true
    options.lmEos = true

    val transcriber = W2vBertUk.createTranscriber(options)
    try {
      println(transcriber.transcribeFile(root.resolve("example_1.wav")))
      println(transcriber.transcribeFile(root.resolve("example_2.wav")))
    } finally {
      transcriber.close()
    }
  }
}
