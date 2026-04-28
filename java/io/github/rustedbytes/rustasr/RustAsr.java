package io.github.rustedbytes.rustasr;

import java.io.Closeable;
import java.nio.file.Path;
import java.util.Objects;

public final class RustAsr {
    private RustAsr() {
    }

    public static boolean initializeOrt(String ortDylibPath) {
        return nativeInitializeOrt(ortDylibPath);
    }

    public static void preloadCudaDylibs(String cudaLibDir, String cudnnLibDir) {
        nativePreloadCudaDylibs(cudaLibDir, cudnnLibDir);
    }

    public static String transcribeFile(Path audioFile, Options options) {
        Objects.requireNonNull(audioFile, "audioFile");
        Options resolved = options == null ? Options.defaults() : options;
        return nativeTranscribeFile(audioFile.toString(), resolved.model, resolved.tokenizer, resolved.lm,
                resolved.ortDylibPath, resolved.ortOptimization, resolved.w2vModelSource,
                resolved.beamWidth, resolved.lmWeight, resolved.wordBonus, resolved.hotWords, resolved.hotWordBonus,
                resolved.fallbackSampleRate, resolved.w2vSampleRate, resolved.w2vFeatureSize, resolved.w2vStride, resolved.w2vFeatureDim,
                resolved.w2vPaddingValue, resolved.blankId, resolved.nBest, resolved.logLanguageModel,
                resolved.logAccelerator, resolved.skipDecodeErrors, resolved.normalizeSpaces,
                resolved.dropEmptyCandidates, resolved.lmBos, resolved.lmEos);
    }

    public static Transcriber createTranscriber(Options options) {
        Options resolved = options == null ? Options.defaults() : options;
        long handle = nativeCreateTranscriber(resolved.model, resolved.tokenizer, resolved.lm,
                resolved.ortDylibPath, resolved.ortOptimization, resolved.w2vModelSource,
                resolved.beamWidth, resolved.lmWeight, resolved.wordBonus, resolved.hotWords, resolved.hotWordBonus,
                resolved.fallbackSampleRate, resolved.w2vSampleRate, resolved.w2vFeatureSize, resolved.w2vStride, resolved.w2vFeatureDim,
                resolved.w2vPaddingValue, resolved.blankId, resolved.nBest, resolved.logLanguageModel,
                resolved.logAccelerator, resolved.skipDecodeErrors, resolved.normalizeSpaces,
                resolved.dropEmptyCandidates, resolved.lmBos, resolved.lmEos);
        return new Transcriber(handle);
    }

    public static final class Transcriber implements Closeable {
        private long handle;

        private Transcriber(long handle) {
            if (handle == 0) {
                throw new IllegalStateException("native transcriber creation returned a null handle");
            }
            this.handle = handle;
        }

        public String transcribeFile(Path audioFile) {
            Objects.requireNonNull(audioFile, "audioFile");
            return nativeTranscriberTranscribeFile(requireOpen(), audioFile.toString());
        }

        public String transcribeBytes(byte[] audioBytes, String formatHint) {
            Objects.requireNonNull(audioBytes, "audioBytes");
            return nativeTranscriberTranscribeBytes(requireOpen(), audioBytes, formatHint);
        }

        @Override
        public void close() {
            long current = handle;
            handle = 0;
            if (current != 0) {
                nativeFreeTranscriber(current);
            }
        }

        private long requireOpen() {
            if (handle == 0) {
                throw new IllegalStateException("transcriber is closed");
            }
            return handle;
        }
    }

    public static final class Options {
        public String model = "model_optimized.onnx";
        public String tokenizer = "tokenizer.model";
        public String lm = "lm.binary";
        public String ortDylibPath = null;
        public String ortOptimization = "disable";
        public String w2vModelSource = null;
        public int beamWidth = 32;
        public float lmWeight = 0.45f;
        public float wordBonus = 0.2f;
        public String[] hotWords = new String[0];
        public float hotWordBonus = 0.0f;
        public int fallbackSampleRate = 16_000;
        public int w2vSampleRate = 0;
        public int w2vFeatureSize = 0;
        public int w2vStride = 0;
        public int w2vFeatureDim = 0;
        public float w2vPaddingValue = Float.NaN;
        public int blankId = 0;
        public int nBest = 32;
        public boolean logLanguageModel = true;
        public boolean logAccelerator = true;
        public boolean skipDecodeErrors = true;
        public boolean normalizeSpaces = true;
        public boolean dropEmptyCandidates = true;
        public boolean lmBos = true;
        public boolean lmEos = true;

        public static Options defaults() {
            return new Options();
        }
    }

    private static native boolean nativeInitializeOrt(String ortDylibPath);
    private static native void nativePreloadCudaDylibs(String cudaLibDir, String cudnnLibDir);
    private static native String nativeTranscribeFile(String audioFile, String model, String tokenizer, String lm,
            String ortDylibPath, String ortOptimization, String w2vModelSource, int beamWidth, float lmWeight,
            float wordBonus, String[] hotWords, float hotWordBonus, int fallbackSampleRate, int w2vSampleRate, int w2vFeatureSize, int w2vStride,
            int w2vFeatureDim, float w2vPaddingValue, int blankId, int nBest, boolean logLanguageModel,
            boolean logAccelerator, boolean skipDecodeErrors, boolean normalizeSpaces, boolean dropEmptyCandidates,
            boolean lmBos, boolean lmEos);
    private static native long nativeCreateTranscriber(String model, String tokenizer, String lm, String ortDylibPath,
            String ortOptimization, String w2vModelSource, int beamWidth, float lmWeight, float wordBonus,
            String[] hotWords, float hotWordBonus, int fallbackSampleRate, int w2vSampleRate, int w2vFeatureSize, int w2vStride, int w2vFeatureDim,
            float w2vPaddingValue, int blankId, int nBest, boolean logLanguageModel, boolean logAccelerator,
            boolean skipDecodeErrors, boolean normalizeSpaces, boolean dropEmptyCandidates, boolean lmBos,
            boolean lmEos);
    private static native void nativeFreeTranscriber(long handle);
    private static native String nativeTranscriberTranscribeFile(long handle, String audioFile);
    private static native String nativeTranscriberTranscribeBytes(long handle, byte[] audioBytes, String formatHint);
}
