using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using W2vBertUk.Native;

unsafe class Transcribe
{
    static readonly string Root = GetRoot();

    static int Main()
    {
        try
        {
            using var model = new Utf8String(FromRoot("model_optimized.onnx"));
            using var tokenizer = new Utf8String(FromRoot("tokenizer.model"));
            using var languageModel = new Utf8String(FromRoot("news-titles.arpa"));
            using var ortOptimization = new Utf8String("disable");
            using var hotWord = new Utf8String("Kyiv");
            byte** hotWords = stackalloc byte*[1];
            hotWords[0] = hotWord.Pointer;

            var options = NativeMethods.w2v_bert_uk_options_default();
            options.model = model.Pointer;
            options.tokenizer = tokenizer.Pointer;
            options.beam_width = 32;
            options.lm = languageModel.Pointer;
            options.lm_weight = 0.45f;
            options.word_bonus = 0.2f;
            options.hot_words = hotWords;
            options.hot_words_len = 1;
            options.hot_word_bonus = 2.0f;
            options.log_language_model = 0;
            options.ort_dylib_path = null;
            options.ort_optimization = ortOptimization.Pointer;
            options.log_accelerator = 1;
            options.fallback_sample_rate = 16_000;
            options.skip_decode_errors = 1;
            options.w2v_model_source = null;
            options.w2v_sample_rate = 0;
            options.w2v_feature_size = 0;
            options.w2v_stride = 0;
            options.w2v_feature_dim = 0;
            options.w2v_padding_value = float.NaN;
            options.blank_id = 0;
            options.n_best = 32;
            options.normalize_spaces = 1;
            options.drop_empty_candidates = 1;
            options.lm_bos = 1;
            options.lm_eos = 1;

            W2vBertUkTranscriber* transcriber = null;
            Check(NativeMethods.w2v_bert_uk_transcriber_new(&options, &transcriber));

            try
            {
                Console.WriteLine(TranscribeFile(transcriber, FromRoot("example_1.wav")));
                Console.WriteLine(TranscribeFile(transcriber, FromRoot("example_2.wav")));
            }
            finally
            {
                NativeMethods.w2v_bert_uk_transcriber_free(transcriber);
            }

            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error.Message);
            return 1;
        }
    }

    static string FromRoot(string path)
    {
        return Path.Combine(Root, path);
    }

    static string GetRoot([CallerFilePath] string sourceFile = "")
    {
        string sourceDirectory = Path.GetDirectoryName(sourceFile)
            ?? throw new InvalidOperationException("Could not resolve the example source directory.");
        DirectoryInfo root = Directory.GetParent(sourceDirectory)
            ?? throw new InvalidOperationException("Could not resolve the repository root.");
        return root.FullName;
    }

    static string TranscribeFile(W2vBertUkTranscriber* transcriber, string audioFile)
    {
        using var audio = new Utf8String(audioFile);
        byte* transcript = null;

        Check(NativeMethods.w2v_bert_uk_transcriber_transcribe_file(
            transcriber,
            audio.Pointer,
            &transcript));

        try
        {
            return Marshal.PtrToStringUTF8((IntPtr)transcript) ?? string.Empty;
        }
        finally
        {
            NativeMethods.w2v_bert_uk_string_free(transcript);
        }
    }

    static void Check(int status)
    {
        if (status == 0)
        {
            return;
        }

        byte* message = NativeMethods.w2v_bert_uk_last_error_message();
        try
        {
            string text = message == null
                ? "w2v-bert-uk native call failed"
                : Marshal.PtrToStringUTF8((IntPtr)message) ?? "w2v-bert-uk native call failed";
            throw new InvalidOperationException(text);
        }
        finally
        {
            NativeMethods.w2v_bert_uk_string_free(message);
        }
    }

    sealed class Utf8String : IDisposable
    {
        public byte* Pointer { get; }

        public Utf8String(string value)
        {
            Pointer = (byte*)Marshal.StringToCoTaskMemUTF8(value);
        }

        public void Dispose()
        {
            Marshal.FreeCoTaskMem((IntPtr)Pointer);
        }
    }
}
