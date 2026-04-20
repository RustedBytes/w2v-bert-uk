import io.github.rustedbytes.w2vbertuk.W2vBertUk;

import java.nio.file.Path;

public final class Transcribe {
    private Transcribe() {
    }

    public static void main(String[] args) {
        Path root = Path.of("").toAbsolutePath();

        System.loadLibrary("w2v_bert_uk");

        W2vBertUk.Options options = W2vBertUk.Options.defaults();
        options.model = root.resolve("model_optimized.onnx").toString();
        options.tokenizer = root.resolve("tokenizer.model").toString();
        options.lm = root.resolve("news-titles.arpa").toString();
        options.beamWidth = 32;
        options.lmWeight = 0.45f;
        options.wordBonus = 0.2f;
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

        try (W2vBertUk.Transcriber transcriber = W2vBertUk.createTranscriber(options)) {
            System.out.println(transcriber.transcribeFile(root.resolve("example_1.wav")));
            System.out.println(transcriber.transcribeFile(root.resolve("example_2.wav")));
        }
    }
}
