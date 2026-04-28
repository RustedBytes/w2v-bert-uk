package main

import (
	"fmt"
	"log"
	"os"

	rustasr "github.com/RustedBytes/rust-asr/go"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatalf("usage: %s <audio-file>", os.Args[0])
	}

	transcriber, err := rustasr.NewTranscriber(rustasr.Options{
		Model:              "model_optimized.onnx",
		Tokenizer:          "tokenizer.model",
		LM:                 "news-titles.arpa",
		BeamWidth:          32,
		LMWeight:           0.45,
		WordBonus:          0.2,
		HotWords:           []string{"Kyiv"},
		HotWordBonus:       2.0,
		LogLanguageModel:   rustasr.BoolFalse,
		LogAccelerator:     rustasr.BoolTrue,
		FallbackSampleRate: 16000,
		SkipDecodeErrors:   rustasr.BoolTrue,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer transcriber.Close()

	transcript, err := transcriber.TranscribeFile(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(transcript)
}
