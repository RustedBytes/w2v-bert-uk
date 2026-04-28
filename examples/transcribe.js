const path = require("node:path");

const w2vBertUk = require("../artifacts");

const root = path.resolve(__dirname, "..");
const fromRoot = (...parts) => path.join(root, ...parts);

const transcriber = new w2vBertUk.Transcriber({
  model: fromRoot("model_optimized.onnx"),
  tokenizer: fromRoot("tokenizer.model"),
  beamWidth: 32,
  lm: fromRoot("news-titles.arpa"),
  lmWeight: 0.45,
  wordBonus: 0.2,
  hotWords: ["Kyiv"],
  hotWordBonus: 2.0,
  logLanguageModel: false,
  ortDylibPath: null,
  ortOptimization: "disable",
  logAccelerator: true,
  fallbackSampleRate: 16000,
  skipDecodeErrors: true,
  w2VModelSource: null,
  w2VSampleRate: null,
  w2VFeatureSize: null,
  w2VStride: null,
  w2VFeatureDim: null,
  w2VPaddingValue: null,
  blankId: 0,
  nBest: 32,
  normalizeSpaces: true,
  dropEmptyCandidates: true,
  lmBos: true,
  lmEos: true,
});

console.log(transcriber.transcribeFile(fromRoot("example_1.wav")));
console.log(transcriber.transcribeFile(fromRoot("example_2.wav")));
