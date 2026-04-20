package io.github.rustedbytes.w2vbertuk

import java.nio.file.Path
import java.util.Objects

object W2vBertUk {
  def initializeOrt(ortDylibPath: String = null): Boolean =
    W2vBertUkNative.nativeInitializeOrt(ortDylibPath)

  def preloadCudaDylibs(cudaLibDir: String = null, cudnnLibDir: String = null): Unit =
    W2vBertUkNative.nativePreloadCudaDylibs(cudaLibDir, cudnnLibDir)

  def transcribeFile(audioFile: Path, options: Options = Options.defaults()): String = {
    Objects.requireNonNull(audioFile, "audioFile")
    val resolved = if (options == null) Options.defaults() else options
    W2vBertUkNative.nativeTranscribeFile(
      audioFile.toString,
      resolved.model,
      resolved.tokenizer,
      resolved.lm,
      resolved.ortDylibPath,
      resolved.ortOptimization,
      resolved.w2vModelSource,
      resolved.beamWidth,
      resolved.lmWeight,
      resolved.wordBonus,
      resolved.fallbackSampleRate,
      resolved.w2vSampleRate,
      resolved.w2vFeatureSize,
      resolved.w2vStride,
      resolved.w2vFeatureDim,
      resolved.w2vPaddingValue,
      resolved.blankId,
      resolved.nBest,
      resolved.logLanguageModel,
      resolved.logAccelerator,
      resolved.skipDecodeErrors,
      resolved.normalizeSpaces,
      resolved.dropEmptyCandidates,
      resolved.lmBos,
      resolved.lmEos
    )
  }

  def createTranscriber(options: Options = Options.defaults()): Transcriber = {
    val resolved = if (options == null) Options.defaults() else options
    val handle = W2vBertUkNative.nativeCreateTranscriber(
      resolved.model,
      resolved.tokenizer,
      resolved.lm,
      resolved.ortDylibPath,
      resolved.ortOptimization,
      resolved.w2vModelSource,
      resolved.beamWidth,
      resolved.lmWeight,
      resolved.wordBonus,
      resolved.fallbackSampleRate,
      resolved.w2vSampleRate,
      resolved.w2vFeatureSize,
      resolved.w2vStride,
      resolved.w2vFeatureDim,
      resolved.w2vPaddingValue,
      resolved.blankId,
      resolved.nBest,
      resolved.logLanguageModel,
      resolved.logAccelerator,
      resolved.skipDecodeErrors,
      resolved.normalizeSpaces,
      resolved.dropEmptyCandidates,
      resolved.lmBos,
      resolved.lmEos
    )
    if (handle == 0L) {
      throw new IllegalStateException("native transcriber creation returned a null handle")
    }
    new Transcriber(handle)
  }

  final class Transcriber private[w2vbertuk] (private var handle: Long) extends AutoCloseable {
    def transcribeFile(audioFile: Path): String = {
      Objects.requireNonNull(audioFile, "audioFile")
      W2vBertUkNative.nativeTranscriberTranscribeFile(requireOpen(), audioFile.toString)
    }

    def transcribeBytes(audioBytes: Array[Byte], formatHint: String = null): String = {
      Objects.requireNonNull(audioBytes, "audioBytes")
      W2vBertUkNative.nativeTranscriberTranscribeBytes(requireOpen(), audioBytes, formatHint)
    }

    override def close(): Unit = {
      val current = handle
      handle = 0L
      if (current != 0L) {
        W2vBertUkNative.nativeFreeTranscriber(current)
      }
    }

    private def requireOpen(): Long = {
      if (handle == 0L) {
        throw new IllegalStateException("transcriber is closed")
      }
      handle
    }
  }

  final class Options {
    var model: String = "model_optimized.onnx"
    var tokenizer: String = "tokenizer.model"
    var lm: String = "lm.binary"
    var ortDylibPath: String = null
    var ortOptimization: String = "disable"
    var w2vModelSource: String = null
    var beamWidth: Int = 32
    var lmWeight: Float = 0.45f
    var wordBonus: Float = 0.2f
    var fallbackSampleRate: Int = 16000
    var w2vSampleRate: Int = 0
    var w2vFeatureSize: Int = 0
    var w2vStride: Int = 0
    var w2vFeatureDim: Int = 0
    var w2vPaddingValue: Float = Float.NaN
    var blankId: Int = 0
    var nBest: Int = 32
    var logLanguageModel: Boolean = true
    var logAccelerator: Boolean = true
    var skipDecodeErrors: Boolean = true
    var normalizeSpaces: Boolean = true
    var dropEmptyCandidates: Boolean = true
    var lmBos: Boolean = true
    var lmEos: Boolean = true
  }

  object Options {
    def defaults(): Options = new Options()
  }
}

private[w2vbertuk] object W2vBertUkNative {
  private val instance = new W2vBertUkNative()

  def nativeInitializeOrt(ortDylibPath: String): Boolean =
    instance.nativeInitializeOrt(ortDylibPath)

  def nativePreloadCudaDylibs(cudaLibDir: String, cudnnLibDir: String): Unit =
    instance.nativePreloadCudaDylibs(cudaLibDir, cudnnLibDir)

  def nativeTranscribeFile(
      audioFile: String,
      model: String,
      tokenizer: String,
      lm: String,
      ortDylibPath: String,
      ortOptimization: String,
      w2vModelSource: String,
      beamWidth: Int,
      lmWeight: Float,
      wordBonus: Float,
      fallbackSampleRate: Int,
      w2vSampleRate: Int,
      w2vFeatureSize: Int,
      w2vStride: Int,
      w2vFeatureDim: Int,
      w2vPaddingValue: Float,
      blankId: Int,
      nBest: Int,
      logLanguageModel: Boolean,
      logAccelerator: Boolean,
      skipDecodeErrors: Boolean,
      normalizeSpaces: Boolean,
      dropEmptyCandidates: Boolean,
      lmBos: Boolean,
      lmEos: Boolean
  ): String =
    instance.nativeTranscribeFile(
      audioFile,
      model,
      tokenizer,
      lm,
      ortDylibPath,
      ortOptimization,
      w2vModelSource,
      beamWidth,
      lmWeight,
      wordBonus,
      fallbackSampleRate,
      w2vSampleRate,
      w2vFeatureSize,
      w2vStride,
      w2vFeatureDim,
      w2vPaddingValue,
      blankId,
      nBest,
      logLanguageModel,
      logAccelerator,
      skipDecodeErrors,
      normalizeSpaces,
      dropEmptyCandidates,
      lmBos,
      lmEos
    )

  def nativeCreateTranscriber(
      model: String,
      tokenizer: String,
      lm: String,
      ortDylibPath: String,
      ortOptimization: String,
      w2vModelSource: String,
      beamWidth: Int,
      lmWeight: Float,
      wordBonus: Float,
      fallbackSampleRate: Int,
      w2vSampleRate: Int,
      w2vFeatureSize: Int,
      w2vStride: Int,
      w2vFeatureDim: Int,
      w2vPaddingValue: Float,
      blankId: Int,
      nBest: Int,
      logLanguageModel: Boolean,
      logAccelerator: Boolean,
      skipDecodeErrors: Boolean,
      normalizeSpaces: Boolean,
      dropEmptyCandidates: Boolean,
      lmBos: Boolean,
      lmEos: Boolean
  ): Long =
    instance.nativeCreateTranscriber(
      model,
      tokenizer,
      lm,
      ortDylibPath,
      ortOptimization,
      w2vModelSource,
      beamWidth,
      lmWeight,
      wordBonus,
      fallbackSampleRate,
      w2vSampleRate,
      w2vFeatureSize,
      w2vStride,
      w2vFeatureDim,
      w2vPaddingValue,
      blankId,
      nBest,
      logLanguageModel,
      logAccelerator,
      skipDecodeErrors,
      normalizeSpaces,
      dropEmptyCandidates,
      lmBos,
      lmEos
    )

  def nativeFreeTranscriber(handle: Long): Unit =
    instance.nativeFreeTranscriber(handle)

  def nativeTranscriberTranscribeFile(handle: Long, audioFile: String): String =
    instance.nativeTranscriberTranscribeFile(handle, audioFile)

  def nativeTranscriberTranscribeBytes(handle: Long, audioBytes: Array[Byte], formatHint: String): String =
    instance.nativeTranscriberTranscribeBytes(handle, audioBytes, formatHint)
}

private[w2vbertuk] final class W2vBertUkNative {
  @native def nativeInitializeOrt(ortDylibPath: String): Boolean
  @native def nativePreloadCudaDylibs(cudaLibDir: String, cudnnLibDir: String): Unit
  @native def nativeTranscribeFile(
      audioFile: String,
      model: String,
      tokenizer: String,
      lm: String,
      ortDylibPath: String,
      ortOptimization: String,
      w2vModelSource: String,
      beamWidth: Int,
      lmWeight: Float,
      wordBonus: Float,
      fallbackSampleRate: Int,
      w2vSampleRate: Int,
      w2vFeatureSize: Int,
      w2vStride: Int,
      w2vFeatureDim: Int,
      w2vPaddingValue: Float,
      blankId: Int,
      nBest: Int,
      logLanguageModel: Boolean,
      logAccelerator: Boolean,
      skipDecodeErrors: Boolean,
      normalizeSpaces: Boolean,
      dropEmptyCandidates: Boolean,
      lmBos: Boolean,
      lmEos: Boolean
  ): String
  @native def nativeCreateTranscriber(
      model: String,
      tokenizer: String,
      lm: String,
      ortDylibPath: String,
      ortOptimization: String,
      w2vModelSource: String,
      beamWidth: Int,
      lmWeight: Float,
      wordBonus: Float,
      fallbackSampleRate: Int,
      w2vSampleRate: Int,
      w2vFeatureSize: Int,
      w2vStride: Int,
      w2vFeatureDim: Int,
      w2vPaddingValue: Float,
      blankId: Int,
      nBest: Int,
      logLanguageModel: Boolean,
      logAccelerator: Boolean,
      skipDecodeErrors: Boolean,
      normalizeSpaces: Boolean,
      dropEmptyCandidates: Boolean,
      lmBos: Boolean,
      lmEos: Boolean
  ): Long
  @native def nativeFreeTranscriber(handle: Long): Unit
  @native def nativeTranscriberTranscribeFile(handle: Long, audioFile: String): String
  @native def nativeTranscriberTranscribeBytes(handle: Long, audioBytes: Array[Byte], formatHint: String): String
}
