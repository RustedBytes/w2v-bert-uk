package w2vbertuk

/*
#cgo CFLAGS: -I${SRCDIR}/../c
#cgo linux LDFLAGS: -L${SRCDIR}/../native -lw2v_bert_uk -Wl,-rpath,${SRCDIR}/../native
#cgo darwin LDFLAGS: -L${SRCDIR}/../native -lw2v_bert_uk -Wl,-rpath,${SRCDIR}/../native
#cgo windows LDFLAGS: -L${SRCDIR}/../native -lw2v_bert_uk
#include "w2v_bert_uk.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"runtime"
	"sync"
	"unsafe"
)

type BoolOption int32

const (
	BoolDefault BoolOption = iota
	BoolFalse
	BoolTrue
)

type Options struct {
	Model               string
	Tokenizer           string
	LM                  string
	OrtDylibPath        string
	OrtOptimization     string
	W2vModelSource      string
	BeamWidth           uint32
	LMWeight            float32
	WordBonus           float32
	HotWords            []string
	HotWordBonus        float32
	FallbackSampleRate  uint32
	W2vSampleRate       uint32
	W2vFeatureSize      uint32
	W2vStride           uint32
	W2vFeatureDim       uint32
	W2vPaddingValue     float32
	BlankID             uint32
	NBest               uint32
	LogLanguageModel    BoolOption
	LogAccelerator      BoolOption
	SkipDecodeErrors    BoolOption
	NormalizeSpaces     BoolOption
	DropEmptyCandidates BoolOption
	LMBOS               BoolOption
	LMEOS               BoolOption
}

type Transcriber struct {
	mu     sync.Mutex
	handle *C.W2vBertUkTranscriber
}

func InitializeOrt(ortDylibPath string) (bool, error) {
	path := cStringOrNil(ortDylibPath)
	if path != nil {
		defer C.free(unsafe.Pointer(path))
	}

	var initialized C.bool
	if err := checkStatus(C.w2v_bert_uk_initialize_ort(path, &initialized)); err != nil {
		return false, err
	}
	return bool(initialized), nil
}

func PreloadCudaDylibs(cudaLibDir, cudnnLibDir string) error {
	cuda := cStringOrNil(cudaLibDir)
	if cuda != nil {
		defer C.free(unsafe.Pointer(cuda))
	}
	cudnn := cStringOrNil(cudnnLibDir)
	if cudnn != nil {
		defer C.free(unsafe.Pointer(cudnn))
	}

	return checkStatus(C.w2v_bert_uk_preload_cuda_dylibs(cuda, cudnn))
}

func TranscribeFile(audioFile string, options Options) (string, error) {
	audio := C.CString(audioFile)
	defer C.free(unsafe.Pointer(audio))

	var transcript *C.char
	err := withCOptions(options, func(cOptions *C.W2vBertUkOptions) error {
		return checkStatus(C.w2v_bert_uk_transcribe_file(audio, cOptions, &transcript))
	})
	if err != nil {
		return "", err
	}
	defer C.w2v_bert_uk_string_free(transcript)
	return C.GoString(transcript), nil
}

func TranscribeBytes(audioBytes []byte, formatHint string, options Options) (string, error) {
	var data unsafe.Pointer
	if len(audioBytes) > 0 {
		data = C.CBytes(audioBytes)
		defer C.free(data)
	}

	hint := cStringOrNil(formatHint)
	if hint != nil {
		defer C.free(unsafe.Pointer(hint))
	}

	var transcript *C.char
	err := withCOptions(options, func(cOptions *C.W2vBertUkOptions) error {
		return checkStatus(C.w2v_bert_uk_transcribe_bytes(
			(*C.uchar)(data),
			C.uintptr_t(len(audioBytes)),
			hint,
			cOptions,
			&transcript,
		))
	})
	if err != nil {
		return "", err
	}
	defer C.w2v_bert_uk_string_free(transcript)
	return C.GoString(transcript), nil
}

func NewTranscriber(options Options) (*Transcriber, error) {
	var handle *C.W2vBertUkTranscriber
	err := withCOptions(options, func(cOptions *C.W2vBertUkOptions) error {
		return checkStatus(C.w2v_bert_uk_transcriber_new(cOptions, &handle))
	})
	if err != nil {
		return nil, err
	}

	transcriber := &Transcriber{handle: handle}
	runtime.SetFinalizer(transcriber, (*Transcriber).Close)
	return transcriber, nil
}

func (t *Transcriber) Close() {
	if t == nil {
		return
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	if t.handle != nil {
		C.w2v_bert_uk_transcriber_free(t.handle)
		t.handle = nil
		runtime.SetFinalizer(t, nil)
	}
}

func (t *Transcriber) TranscribeFile(audioFile string) (string, error) {
	if t == nil {
		return "", errors.New("w2v-bert-uk: transcriber is nil")
	}

	audio := C.CString(audioFile)
	defer C.free(unsafe.Pointer(audio))

	t.mu.Lock()
	defer t.mu.Unlock()
	if t.handle == nil {
		return "", errors.New("w2v-bert-uk: transcriber is closed")
	}

	var transcript *C.char
	if err := checkStatus(C.w2v_bert_uk_transcriber_transcribe_file(t.handle, audio, &transcript)); err != nil {
		return "", err
	}
	defer C.w2v_bert_uk_string_free(transcript)
	return C.GoString(transcript), nil
}

func (t *Transcriber) TranscribeBytes(audioBytes []byte, formatHint string) (string, error) {
	if t == nil {
		return "", errors.New("w2v-bert-uk: transcriber is nil")
	}

	var data unsafe.Pointer
	if len(audioBytes) > 0 {
		data = C.CBytes(audioBytes)
		defer C.free(data)
	}

	hint := cStringOrNil(formatHint)
	if hint != nil {
		defer C.free(unsafe.Pointer(hint))
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	if t.handle == nil {
		return "", errors.New("w2v-bert-uk: transcriber is closed")
	}

	var transcript *C.char
	if err := checkStatus(C.w2v_bert_uk_transcriber_transcribe_bytes(
		t.handle,
		(*C.uchar)(data),
		C.uintptr_t(len(audioBytes)),
		hint,
		&transcript,
	)); err != nil {
		return "", err
	}
	defer C.w2v_bert_uk_string_free(transcript)
	return C.GoString(transcript), nil
}

func withCOptions(options Options, call func(*C.W2vBertUkOptions) error) error {
	cOptions := C.w2v_bert_uk_options_default()
	strings := []*C.char{}
	hotWords := []*C.char{}
	defer func() {
		for _, value := range strings {
			C.free(unsafe.Pointer(value))
		}
	}()

	assignString := func(value string, target **C.char) {
		if value == "" {
			return
		}
		cValue := C.CString(value)
		strings = append(strings, cValue)
		*target = cValue
	}

	assignString(options.Model, &cOptions.model)
	assignString(options.Tokenizer, &cOptions.tokenizer)
	assignString(options.LM, &cOptions.lm)
	assignString(options.OrtDylibPath, &cOptions.ort_dylib_path)
	assignString(options.OrtOptimization, &cOptions.ort_optimization)
	assignString(options.W2vModelSource, &cOptions.w2v_model_source)

	if options.BeamWidth != 0 {
		cOptions.beam_width = C.uint32_t(options.BeamWidth)
	}
	if options.LMWeight != 0 {
		cOptions.lm_weight = C.float(options.LMWeight)
	}
	if options.WordBonus != 0 {
		cOptions.word_bonus = C.float(options.WordBonus)
	}
	if len(options.HotWords) != 0 {
		hotWords = make([]*C.char, 0, len(options.HotWords))
		for _, value := range options.HotWords {
			if value == "" {
				continue
			}
			cValue := C.CString(value)
			strings = append(strings, cValue)
			hotWords = append(hotWords, cValue)
		}
		if len(hotWords) != 0 {
			cOptions.hot_words = &hotWords[0]
			cOptions.hot_words_len = C.uintptr_t(len(hotWords))
		}
	}
	if options.HotWordBonus != 0 {
		cOptions.hot_word_bonus = C.float(options.HotWordBonus)
	}
	if options.FallbackSampleRate != 0 {
		cOptions.fallback_sample_rate = C.uint32_t(options.FallbackSampleRate)
	}
	if options.W2vSampleRate != 0 {
		cOptions.w2v_sample_rate = C.uint32_t(options.W2vSampleRate)
	}
	if options.W2vFeatureSize != 0 {
		cOptions.w2v_feature_size = C.uint32_t(options.W2vFeatureSize)
	}
	if options.W2vStride != 0 {
		cOptions.w2v_stride = C.uint32_t(options.W2vStride)
	}
	if options.W2vFeatureDim != 0 {
		cOptions.w2v_feature_dim = C.uint32_t(options.W2vFeatureDim)
	}
	if options.W2vPaddingValue != 0 {
		cOptions.w2v_padding_value = C.float(options.W2vPaddingValue)
	}
	cOptions.blank_id = C.uint32_t(options.BlankID)
	if options.NBest != 0 {
		cOptions.n_best = C.uint32_t(options.NBest)
	}
	assignBoolOption(options.LogLanguageModel, &cOptions.log_language_model)
	assignBoolOption(options.LogAccelerator, &cOptions.log_accelerator)
	assignBoolOption(options.SkipDecodeErrors, &cOptions.skip_decode_errors)
	assignBoolOption(options.NormalizeSpaces, &cOptions.normalize_spaces)
	assignBoolOption(options.DropEmptyCandidates, &cOptions.drop_empty_candidates)
	assignBoolOption(options.LMBOS, &cOptions.lm_bos)
	assignBoolOption(options.LMEOS, &cOptions.lm_eos)

	err := call(&cOptions)
	runtime.KeepAlive(hotWords)
	return err
}

func assignBoolOption(value BoolOption, target *C.int32_t) {
	switch value {
	case BoolFalse:
		*target = 0
	case BoolTrue:
		*target = 1
	}
}

func cStringOrNil(value string) *C.char {
	if value == "" {
		return nil
	}
	return C.CString(value)
}

func checkStatus(status C.int32_t) error {
	if status == C.W2V_BERT_UK_OK {
		return nil
	}

	message := C.w2v_bert_uk_last_error_message()
	if message == nil {
		return errors.New("w2v-bert-uk: native call failed")
	}
	defer C.w2v_bert_uk_string_free(message)
	return errors.New(C.GoString(message))
}
