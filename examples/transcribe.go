package main

import (
	"fmt"
	"log"
	"os"

	w2vbertuk "github.com/RustedBytes/w2v-bert-uk/go"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatalf("usage: %s <audio-file>", os.Args[0])
	}

	transcriber, err := w2vbertuk.NewTranscriber(w2vbertuk.Options{
		Model:              "model_optimized.onnx",
		Tokenizer:          "tokenizer.model",
		LM:                 "news-titles.arpa",
		BeamWidth:          32,
		LMWeight:           0.45,
		WordBonus:          0.2,
		LogLanguageModel:   w2vbertuk.BoolFalse,
		LogAccelerator:     w2vbertuk.BoolTrue,
		FallbackSampleRate: 16000,
		SkipDecodeErrors:   w2vbertuk.BoolTrue,
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
