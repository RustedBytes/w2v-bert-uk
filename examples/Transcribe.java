import io.github.rustedbytes.rustasr.RustAsr;

import java.nio.file.Path;

public final class Transcribe {
    private Transcribe() {
    }

    public static void main(String[] args) {
        Path root = Path.of("").toAbsolutePath();

        System.loadLibrary("rust_asr");

        RustAsr.Options options = RustAsr.Options.defaults();
        options.model = root.resolve("model_optimized.onnx").toString();
        options.tokenizer = root.resolve("tokenizer.model").toString();
        options.lm = root.resolve("news-titles.arpa").toString();
        options.beamWidth = 32;
        options.lmWeight = 0.45f;
        options.wordBonus = 0.2f;
        options.hotWords = new String[] { "Kyiv" };
        options.hotWordBonus = 2.0f;
        options.logLanguageModel = false;
        options.ortOptimization = "disable";
        options.logAccelerator = true;
        options.fallbackSampleRate = 16_000;
        options.skipDecodeErrors = true;
        options.blankId = 0;
        options.nBest = 32;
        options.normalizeSpaces = true;
        options.dropEmptyCandidates = true;
        options.lmBos = true;
        options.lmEos = true;

        try (RustAsr.Transcriber transcriber = RustAsr.createTranscriber(options)) {
            System.out.println(transcriber.transcribeFile(root.resolve("example_1.wav")));
            System.out.println(transcriber.transcribeFile(root.resolve("example_2.wav")));
        }
    }
}
