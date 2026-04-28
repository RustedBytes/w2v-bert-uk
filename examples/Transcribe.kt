import io.github.rustedbytes.w2vbertuk.KotlinTranscriber
import io.github.rustedbytes.w2vbertuk.defaultOptions

fun main(args: Array<String>) {
    require(args.isNotEmpty()) { "usage: TranscribeKt <audio-file>" }

    val options = defaultOptions().copy(
        model = "model_optimized.onnx",
        tokenizer = "tokenizer.model",
        lm = "news-titles.arpa",
        beamWidth = 32,
        hotWords = listOf("Kyiv"),
        hotWordBonus = 2.0f,
    )

    KotlinTranscriber(options).use { transcriber ->
        println(transcriber.transcribeFile(args[0]))
    }
}
