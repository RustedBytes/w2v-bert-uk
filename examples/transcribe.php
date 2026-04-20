<?php

$audioFile = $argv[1] ?? null;
if ($audioFile === null) {
    fwrite(STDERR, "usage: php -d extension=./target/release/libw2v_bert_uk.so examples/transcribe.php <audio> [model] [tokenizer] [lm]\n");
    exit(2);
}

$options = [
    "model" => $argv[2] ?? "model_optimized.onnx",
    "tokenizer" => $argv[3] ?? "tokenizer.model",
    "beam_width" => 32,
];

if (isset($argv[4])) {
    $options["lm"] = $argv[4];
}

echo w2v_bert_uk_transcribe_file($audioFile, $options), PHP_EOL;
