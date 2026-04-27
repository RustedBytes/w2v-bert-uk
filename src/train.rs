use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use burn::module::AutodiffModule;
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use burn_nn::loss::{CTCLossConfig, Reduction};
use burn_optim::{AdamWConfig, GradientsParams, Optimizer};
use serde_json::{Value, json};

use crate::paraformer::{ParaformerV2, ParaformerV2Config};
use crate::squeezeformer::{SqueezeformerCtc, SqueezeformerCtcConfig, SqueezeformerEncoderConfig};
use crate::wav2vec::{Wav2VecBertConfig, Wav2VecBertCtc, Wav2VecBertCtcConfig};
use crate::zipformer::{ZipformerConfig, ZipformerCtc, ZipformerCtcConfig};

#[derive(Clone, Debug)]
pub enum TrainArchitecture {
    Squeezeformer,
    Zipformer,
    Paraformer,
    Wav2VecBert,
}

impl std::str::FromStr for TrainArchitecture {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "squeezeformer" => Ok(Self::Squeezeformer),
            "zipformer" => Ok(Self::Zipformer),
            "paraformer" => Ok(Self::Paraformer),
            "w2v-bert" | "w2v_bert" | "wav2vec" | "wav2vec-bert" => Ok(Self::Wav2VecBert),
            other => bail!("unknown architecture '{other}'"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BurnTrainConfig {
    pub architecture: TrainArchitecture,
    pub train_manifest: PathBuf,
    pub val_manifest: Option<PathBuf>,
    pub output_dir: PathBuf,
    pub variant: Option<String>,
    pub input_dim: usize,
    pub vocab_size: usize,
    pub blank_id: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub log_every: usize,
    pub validate_every_steps: Option<usize>,
    pub max_train_samples: Option<usize>,
    pub max_val_samples: Option<usize>,
    pub dry_run: bool,
}

impl Default for BurnTrainConfig {
    fn default() -> Self {
        Self {
            architecture: TrainArchitecture::Squeezeformer,
            train_manifest: PathBuf::from("train.jsonl"),
            val_manifest: None,
            output_dir: PathBuf::from("runs/burn"),
            variant: Some("sm".to_string()),
            input_dim: 80,
            vocab_size: 256,
            blank_id: 0,
            d_model: 256,
            num_layers: 16,
            num_heads: 4,
            batch_size: 8,
            epochs: 10,
            learning_rate: 1.0e-3,
            weight_decay: 1.0e-2,
            log_every: 10,
            validate_every_steps: None,
            max_train_samples: None,
            max_val_samples: None,
            dry_run: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FeatureRecord {
    pub id: String,
    pub rows: usize,
    pub cols: usize,
    pub features: Vec<f32>,
    pub tokens: Vec<i64>,
}

#[derive(Clone, Debug)]
pub struct TrainBatch {
    pub features: Vec<f32>,
    pub batch_size: usize,
    pub max_frames: usize,
    pub feature_dim: usize,
    pub feature_lengths: Vec<usize>,
    pub targets: Vec<i64>,
    pub max_target_len: usize,
    pub target_lengths: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct TrainSummary {
    pub epochs: usize,
    pub steps: usize,
    pub last_train_loss: Option<f32>,
    pub last_val_loss: Option<f32>,
}

pub fn run_burn_training(config: BurnTrainConfig) -> Result<TrainSummary> {
    type InnerBackend = burn_ndarray::NdArray<f32>;
    type TrainBackend = burn_autodiff::Autodiff<InnerBackend>;

    let device = Default::default();
    fs::create_dir_all(&config.output_dir).with_context(|| {
        format!(
            "failed to create output directory {}",
            config.output_dir.display()
        )
    })?;
    validate_config(&config)?;
    write_run_config(&config)?;

    match config.architecture {
        TrainArchitecture::Squeezeformer => {
            let encoder = config
                .variant
                .as_deref()
                .and_then(SqueezeformerEncoderConfig::variant)
                .unwrap_or_else(|| {
                    SqueezeformerEncoderConfig::new(
                        config.input_dim,
                        config.d_model,
                        config.num_layers,
                        config.num_heads,
                    )
                });
            let model = SqueezeformerCtcConfig {
                encoder,
                vocab_size: config.vocab_size,
            }
            .init::<TrainBackend>(&device);
            train_ctc_model(model, &config, &device)
        }
        TrainArchitecture::Zipformer => {
            let mut encoder = config
                .variant
                .as_deref()
                .and_then(ZipformerConfig::variant)
                .unwrap_or_else(|| ZipformerConfig::new(config.input_dim));
            encoder.input_dim = config.input_dim;
            let model = ZipformerCtcConfig {
                encoder,
                vocab_size: config.vocab_size,
            }
            .init::<TrainBackend>(&device);
            train_ctc_model(model, &config, &device)
        }
        TrainArchitecture::Paraformer => {
            let model_config = config
                .variant
                .as_deref()
                .and_then(|variant| {
                    ParaformerV2Config::variant(
                        variant,
                        config.input_dim,
                        config.vocab_size,
                        config.blank_id,
                    )
                })
                .unwrap_or_else(|| {
                    let mut value = ParaformerV2Config::new(config.input_dim, config.vocab_size)
                        .with_blank_id(config.blank_id);
                    value.encoder_dim = config.d_model;
                    value.decoder_dim = config.d_model;
                    value.encoder_layers = config.num_layers;
                    value.attention_heads = config.num_heads;
                    value
                });
            let model = model_config.init::<TrainBackend>(&device);
            train_ctc_model(model, &config, &device)
        }
        TrainArchitecture::Wav2VecBert => {
            let encoder = Wav2VecBertConfig::new(config.input_dim, config.d_model)
                .with_layers(config.num_layers);
            let model = Wav2VecBertCtcConfig {
                encoder,
                vocab_size: config.vocab_size,
            }
            .init::<TrainBackend>(&device);
            train_ctc_model(model, &config, &device)
        }
    }
}

trait TrainableCtc<B: AutodiffBackend>: AutodiffModule<B> {
    fn ctc_logits(&self, features: Tensor<B, 3>, lengths: Vec<usize>)
    -> (Tensor<B, 3>, Vec<usize>);
}

impl<B: AutodiffBackend> TrainableCtc<B> for SqueezeformerCtc<B> {
    fn ctc_logits(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        self.forward_with_lengths(features, Some(lengths))
    }
}

impl<B: AutodiffBackend> TrainableCtc<B> for ZipformerCtc<B> {
    fn ctc_logits(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        self.forward_with_lengths(features, lengths)
    }
}

impl<B: AutodiffBackend> TrainableCtc<B> for Wav2VecBertCtc<B> {
    fn ctc_logits(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        self.forward_with_lengths(features, lengths)
    }
}

impl<B: AutodiffBackend> TrainableCtc<B> for ParaformerV2<B> {
    fn ctc_logits(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        let output = self.forward(features, lengths);
        (output.ctc_log_probs, output.encoder_lengths)
    }
}

fn train_ctc_model<B, M>(
    mut model: M,
    config: &BurnTrainConfig,
    device: &B::Device,
) -> Result<TrainSummary>
where
    B: AutodiffBackend,
    M: TrainableCtc<B>,
{
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .init();
    let mut global_step = 0usize;
    let mut last_train_loss = None;
    let mut last_val_loss = None;
    let started = Instant::now();

    for epoch in 1..=config.epochs {
        let mut train_batches = StreamingBatchLoader::new(
            config.train_manifest.clone(),
            config.batch_size,
            config.input_dim,
            config.max_train_samples,
        )?;
        while let Some(batch) = train_batches.next_batch()? {
            let loss = ctc_loss_for_batch::<B, M>(&model, &batch, config.blank_id, device);
            let loss_value = scalar_value(loss.clone())?;
            last_train_loss = Some(loss_value);

            if !config.dry_run {
                let grads = GradientsParams::from_grads(loss.backward(), &model);
                model = optimizer.step(config.learning_rate, model, grads);
            }

            global_step += 1;
            if global_step == 1 || global_step % config.log_every == 0 {
                println!(
                    "epoch={epoch} step={global_step} train_ctc_loss={loss_value:.6} elapsed_sec={:.1}",
                    started.elapsed().as_secs_f64()
                );
            }

            if let (Some(_), Some(every)) = (&config.val_manifest, config.validate_every_steps) {
                if every > 0 && global_step % every == 0 {
                    let val_loss = evaluate_ctc_model::<B, M>(&model, config, device)?;
                    println!("epoch={epoch} step={global_step} val_ctc_loss={val_loss:.6}");
                    last_val_loss = Some(val_loss);
                    write_checkpoint_metadata(
                        config,
                        epoch,
                        global_step,
                        last_train_loss,
                        last_val_loss,
                    )?;
                }
            }
        }

        if config.val_manifest.is_some() {
            let val_loss = evaluate_ctc_model::<B, M>(&model, config, device)?;
            println!("epoch={epoch} val_ctc_loss={val_loss:.6}");
            last_val_loss = Some(val_loss);
        }
        write_checkpoint_metadata(config, epoch, global_step, last_train_loss, last_val_loss)?;
    }

    Ok(TrainSummary {
        epochs: config.epochs,
        steps: global_step,
        last_train_loss,
        last_val_loss,
    })
}

fn evaluate_ctc_model<B, M>(model: &M, config: &BurnTrainConfig, device: &B::Device) -> Result<f32>
where
    B: AutodiffBackend,
    M: TrainableCtc<B>,
{
    let mut total = 0.0f64;
    let mut count = 0usize;
    let val_manifest = config
        .val_manifest
        .as_ref()
        .ok_or_else(|| anyhow!("validation requested without val_manifest"))?;
    let mut batches = StreamingBatchLoader::new(
        val_manifest.clone(),
        config.batch_size,
        config.input_dim,
        config.max_val_samples,
    )?;
    while let Some(batch) = batches.next_batch()? {
        let loss = ctc_loss_for_batch::<B, M>(model, &batch, config.blank_id, device);
        total += f64::from(scalar_value(loss)?);
        count += 1;
    }
    if count == 0 {
        bail!("validation manifest is empty");
    }
    Ok((total / count as f64) as f32)
}

fn ctc_loss_for_batch<B, M>(
    model: &M,
    batch: &TrainBatch,
    blank_id: usize,
    device: &B::Device,
) -> Tensor<B, 1>
where
    B: AutodiffBackend,
    M: TrainableCtc<B>,
{
    let features = Tensor::<B, 3>::from_data(
        TensorData::new(
            batch.features.clone(),
            [batch.batch_size, batch.max_frames, batch.feature_dim],
        ),
        device,
    );
    let (logits_or_log_probs, output_lengths) =
        model.ctc_logits(features, batch.feature_lengths.clone());
    let log_probs = log_softmax(logits_or_log_probs, 2).swap_dims(0, 1);
    let targets = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            batch.targets.clone(),
            [batch.batch_size, batch.max_target_len],
        ),
        device,
    );
    let input_lengths = Tensor::<B, 1, Int>::from_data(
        TensorData::new(to_i64(output_lengths), [batch.batch_size]),
        device,
    );
    let target_lengths = Tensor::<B, 1, Int>::from_data(
        TensorData::new(to_i64(batch.target_lengths.clone()), [batch.batch_size]),
        device,
    );
    CTCLossConfig::new()
        .with_blank(blank_id)
        .with_zero_infinity(true)
        .init()
        .forward_with_reduction(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            Reduction::Mean,
        )
}

fn make_batch(records: &[FeatureRecord], expected_dim: usize) -> Result<TrainBatch> {
    if records.is_empty() {
        bail!("cannot build an empty batch");
    }
    let batch_size = records.len();
    let max_frames = records.iter().map(|record| record.rows).max().unwrap_or(0);
    let max_target_len = records
        .iter()
        .map(|record| record.tokens.len())
        .max()
        .unwrap_or(0);
    if max_frames == 0 || max_target_len == 0 {
        bail!("batch contains empty features or targets");
    }

    let mut features = vec![0.0; batch_size * max_frames * expected_dim];
    let mut targets = vec![0i64; batch_size * max_target_len];
    let mut feature_lengths = Vec::with_capacity(batch_size);
    let mut target_lengths = Vec::with_capacity(batch_size);

    for (batch_index, record) in records.iter().enumerate() {
        if record.cols != expected_dim {
            bail!(
                "record '{}' has feature dim {}, expected {}",
                record.id,
                record.cols,
                expected_dim
            );
        }
        feature_lengths.push(record.rows);
        target_lengths.push(record.tokens.len());
        for row in 0..record.rows {
            let src = row * expected_dim;
            let dst = (batch_index * max_frames + row) * expected_dim;
            features[dst..dst + expected_dim]
                .copy_from_slice(&record.features[src..src + expected_dim]);
        }
        let dst = batch_index * max_target_len;
        targets[dst..dst + record.tokens.len()].copy_from_slice(&record.tokens);
    }

    Ok(TrainBatch {
        features,
        batch_size,
        max_frames,
        feature_dim: expected_dim,
        feature_lengths,
        targets,
        max_target_len,
        target_lengths,
    })
}

pub fn load_manifest(path: &Path, limit: Option<usize>) -> Result<Vec<FeatureRecord>> {
    let files = manifest_files(path)?;
    let mut records = Vec::new();
    for file in files {
        let remaining = limit.map(|limit| limit.saturating_sub(records.len()));
        if remaining == Some(0) {
            break;
        }
        records.extend(load_manifest_file(&file, remaining)?);
    }
    if records.is_empty() {
        bail!("manifest {} contains no records", path.display());
    }
    Ok(records)
}

pub struct StreamingBatchLoader {
    files: Vec<PathBuf>,
    file_index: usize,
    current: Option<CurrentManifestFile>,
    batch_size: usize,
    expected_dim: usize,
    yielded: usize,
    limit: Option<usize>,
}

struct CurrentManifestFile {
    path: PathBuf,
    base_dir: PathBuf,
    reader: BufReader<fs::File>,
    line_number: usize,
}

impl StreamingBatchLoader {
    pub fn new(
        manifest: PathBuf,
        batch_size: usize,
        expected_dim: usize,
        limit: Option<usize>,
    ) -> Result<Self> {
        if batch_size == 0 {
            bail!("batch_size must be > 0");
        }
        Ok(Self {
            files: manifest_files(&manifest)?,
            file_index: 0,
            current: None,
            batch_size,
            expected_dim,
            yielded: 0,
            limit,
        })
    }

    pub fn next_batch(&mut self) -> Result<Option<TrainBatch>> {
        let mut records = Vec::with_capacity(self.batch_size);
        while records.len() < self.batch_size {
            if self.limit.is_some_and(|limit| self.yielded >= limit) {
                break;
            }
            match self.next_record()? {
                Some(record) => {
                    self.yielded += 1;
                    records.push(record);
                }
                None => break,
            }
        }

        if records.is_empty() {
            Ok(None)
        } else {
            make_batch(&records, self.expected_dim).map(Some)
        }
    }

    fn next_record(&mut self) -> Result<Option<FeatureRecord>> {
        loop {
            if self.current.is_none() {
                if self.file_index >= self.files.len() {
                    return Ok(None);
                }
                let path = self.files[self.file_index].clone();
                self.file_index += 1;
                self.current = Some(CurrentManifestFile::open(path)?);
            }

            let current = self.current.as_mut().expect("manifest file must be open");
            let mut line = String::new();
            let bytes = current
                .reader
                .read_line(&mut line)
                .with_context(|| format!("failed reading {}", current.path.display()))?;
            if bytes == 0 {
                self.current = None;
                continue;
            }
            current.line_number += 1;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let record = if line.starts_with('{') {
                parse_json_record(line, &current.base_dir, current.line_number)?
            } else {
                parse_tsv_record(line, &current.base_dir, current.line_number)?
            };
            return Ok(Some(record));
        }
    }
}

impl CurrentManifestFile {
    fn open(path: PathBuf) -> Result<Self> {
        let file = fs::File::open(&path)
            .with_context(|| format!("failed to open manifest {}", path.display()))?;
        let base_dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        Ok(Self {
            path,
            base_dir,
            reader: BufReader::new(file),
            line_number: 0,
        })
    }
}

fn manifest_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_dir() {
        let mut files = fs::read_dir(path)
            .with_context(|| format!("failed to read manifest directory {}", path.display()))?
            .map(|entry| {
                entry
                    .map(|entry| entry.path())
                    .with_context(|| format!("failed to read entry in {}", path.display()))
            })
            .collect::<Result<Vec<_>>>()?;
        files.retain(|path| is_manifest_file(path));
        files.sort();
        if files.is_empty() {
            bail!(
                "manifest directory {} contains no .jsonl/.json files",
                path.display()
            );
        }
        return Ok(files);
    }
    Ok(vec![path.to_path_buf()])
}

fn is_manifest_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| matches!(extension, "jsonl" | "json"))
}

fn load_manifest_file(path: &Path, limit: Option<usize>) -> Result<Vec<FeatureRecord>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read manifest {}", path.display()))?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut records = Vec::new();

    for (line_index, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let record = if line.starts_with('{') {
            parse_json_record(line, base_dir, line_index + 1)?
        } else {
            parse_tsv_record(line, base_dir, line_index + 1)?
        };
        records.push(record);
        if limit.is_some_and(|limit| records.len() >= limit) {
            break;
        }
    }

    if records.is_empty() {
        bail!("manifest {} contains no records", path.display());
    }
    Ok(records)
}

fn parse_json_record(line: &str, base_dir: &Path, line_number: usize) -> Result<FeatureRecord> {
    let value: Value = serde_json::from_str(line)
        .with_context(|| format!("invalid JSON manifest line {line_number}"))?;
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("line-{line_number}"));
    let tokens = parse_tokens_value(
        value
            .get("tokens")
            .or_else(|| value.get("target"))
            .or_else(|| value.get("targets"))
            .ok_or_else(|| anyhow!("record '{id}' is missing tokens"))?,
    )?;

    if let Some(features) = value.get("features") {
        let rows = value
            .get("rows")
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        let cols = value
            .get("cols")
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        let (features, rows, cols) = parse_inline_features(features, rows, cols, &id)?;
        return Ok(FeatureRecord {
            id,
            rows,
            cols,
            features,
            tokens,
        });
    }

    let feature_path = value
        .get("features_path")
        .or_else(|| value.get("feature_path"))
        .or_else(|| value.get("path"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("record '{id}' is missing features or features_path"))?;
    let rows = value
        .get("rows")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("record '{id}' with feature file must include rows"))?
        as usize;
    let cols = value
        .get("cols")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("record '{id}' with feature file must include cols"))?
        as usize;
    let features = read_feature_file(&resolve_path(base_dir, feature_path), rows, cols)?;
    Ok(FeatureRecord {
        id,
        rows,
        cols,
        features,
        tokens,
    })
}

fn parse_tsv_record(line: &str, base_dir: &Path, line_number: usize) -> Result<FeatureRecord> {
    let parts = line.split('\t').collect::<Vec<_>>();
    if parts.len() < 4 {
        bail!(
            "TSV manifest line {line_number} must have: features_path<TAB>rows<TAB>cols<TAB>tokens"
        );
    }
    let rows = parts[1]
        .parse::<usize>()
        .with_context(|| format!("invalid rows on line {line_number}"))?;
    let cols = parts[2]
        .parse::<usize>()
        .with_context(|| format!("invalid cols on line {line_number}"))?;
    let tokens = parse_token_string(parts[3])?;
    let features = read_feature_file(&resolve_path(base_dir, parts[0]), rows, cols)?;
    Ok(FeatureRecord {
        id: format!("line-{line_number}"),
        rows,
        cols,
        features,
        tokens,
    })
}

fn parse_inline_features(
    value: &Value,
    rows: Option<usize>,
    cols: Option<usize>,
    id: &str,
) -> Result<(Vec<f32>, usize, usize)> {
    if let Some(outer) = value.as_array().filter(|items| {
        items
            .first()
            .is_some_and(|first| matches!(first, Value::Array(_)))
    }) {
        let rows = outer.len();
        let cols = outer
            .first()
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        let mut features = Vec::with_capacity(rows * cols);
        for row in outer {
            let row = row
                .as_array()
                .ok_or_else(|| anyhow!("record '{id}' has a non-array feature row"))?;
            if row.len() != cols {
                bail!("record '{id}' has ragged inline features");
            }
            for value in row {
                features.push(
                    value
                        .as_f64()
                        .ok_or_else(|| anyhow!("record '{id}' has a non-number feature"))?
                        as f32,
                );
            }
        }
        return Ok((features, rows, cols));
    }

    let rows = rows.ok_or_else(|| anyhow!("record '{id}' inline flat features require rows"))?;
    let cols = cols.ok_or_else(|| anyhow!("record '{id}' inline flat features require cols"))?;
    let features = value
        .as_array()
        .ok_or_else(|| anyhow!("record '{id}' features must be an array"))?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .ok_or_else(|| anyhow!("record '{id}' has a non-number feature"))
        })
        .collect::<Result<Vec<_>>>()?;
    if features.len() != rows * cols {
        bail!(
            "record '{id}' feature shape {rows}x{cols} implies {} values, got {}",
            rows * cols,
            features.len()
        );
    }
    Ok((features, rows, cols))
}

fn parse_tokens_value(value: &Value) -> Result<Vec<i64>> {
    if let Some(text) = value.as_str() {
        return parse_token_string(text);
    }
    value
        .as_array()
        .ok_or_else(|| anyhow!("tokens must be an array or string"))?
        .iter()
        .map(|value| {
            value
                .as_i64()
                .ok_or_else(|| anyhow!("token array contains a non-integer"))
        })
        .collect()
}

fn parse_token_string(value: &str) -> Result<Vec<i64>> {
    let tokens = value
        .split(|ch: char| ch == ',' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<i64>()
                .with_context(|| format!("invalid token id '{part}'"))
        })
        .collect::<Result<Vec<_>>>()?;
    if tokens.is_empty() {
        bail!("target token sequence is empty");
    }
    Ok(tokens)
}

fn read_feature_file(path: &Path, rows: usize, cols: usize) -> Result<Vec<f32>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read feature file {}", path.display()))?;
    let values = text
        .split(|ch: char| ch == ',' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<f32>()
                .with_context(|| format!("invalid feature value '{part}' in {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    if values.len() != rows * cols {
        bail!(
            "feature file {} shape {rows}x{cols} implies {} values, got {}",
            path.display(),
            rows * cols,
            values.len()
        );
    }
    Ok(values)
}

fn resolve_path(base_dir: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}

fn scalar_value<B: Backend>(tensor: Tensor<B, 1>) -> Result<f32> {
    let values = tensor
        .into_data()
        .to_vec::<f32>()
        .context("failed to read scalar loss")?;
    values
        .first()
        .copied()
        .ok_or_else(|| anyhow!("loss tensor was empty"))
}

fn to_i64(values: Vec<usize>) -> Vec<i64> {
    values.into_iter().map(|value| value as i64).collect()
}

fn validate_config(config: &BurnTrainConfig) -> Result<()> {
    if config.batch_size == 0 {
        bail!("batch_size must be > 0");
    }
    if config.epochs == 0 {
        bail!("epochs must be > 0");
    }
    if config.input_dim == 0 || config.vocab_size == 0 {
        bail!("input_dim and vocab_size must be > 0");
    }
    if config.blank_id >= config.vocab_size {
        bail!("blank_id must be smaller than vocab_size");
    }
    Ok(())
}

fn write_run_config(config: &BurnTrainConfig) -> Result<()> {
    let path = config.output_dir.join("training_config.json");
    let architecture = match config.architecture {
        TrainArchitecture::Squeezeformer => "squeezeformer",
        TrainArchitecture::Zipformer => "zipformer",
        TrainArchitecture::Paraformer => "paraformer",
        TrainArchitecture::Wav2VecBert => "w2v_bert",
    };
    fs::write(
        &path,
        serde_json::to_string_pretty(&json!({
            "architecture": architecture,
            "train_manifest": config.train_manifest,
            "val_manifest": config.val_manifest,
            "variant": config.variant,
            "input_dim": config.input_dim,
            "vocab_size": config.vocab_size,
            "blank_id": config.blank_id,
            "d_model": config.d_model,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
        }))?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

fn write_checkpoint_metadata(
    config: &BurnTrainConfig,
    epoch: usize,
    global_step: usize,
    train_loss: Option<f32>,
    val_loss: Option<f32>,
) -> Result<()> {
    let path = config.output_dir.join("checkpoint_latest.json");
    fs::write(
        &path,
        serde_json::to_string_pretty(&json!({
            "epoch": epoch,
            "global_step": global_step,
            "train_ctc_loss": train_loss,
            "val_ctc_loss": val_loss,
        }))?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_loads_inline_jsonl_records() {
        let dir =
            std::env::temp_dir().join(format!("w2v_bert_uk_train_test_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let manifest = dir.join("train.jsonl");
        fs::write(
            &manifest,
            r#"{"id":"a","features":[[0.1,0.2],[0.3,0.4]],"tokens":[1,2]}"#,
        )
        .unwrap();

        let records = load_manifest(&manifest, None).unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].rows, 2);
        assert_eq!(records[0].cols, 2);
        assert_eq!(records[0].tokens, vec![1, 2]);
    }

    #[test]
    fn batch_pads_features_and_targets() {
        let records = vec![
            FeatureRecord {
                id: "a".to_string(),
                rows: 2,
                cols: 2,
                features: vec![1.0, 2.0, 3.0, 4.0],
                tokens: vec![1, 2],
            },
            FeatureRecord {
                id: "b".to_string(),
                rows: 1,
                cols: 2,
                features: vec![5.0, 6.0],
                tokens: vec![3],
            },
        ];

        let batch = make_batch(&records, 2).unwrap();

        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.max_frames, 2);
        assert_eq!(batch.feature_lengths, vec![2, 1]);
        assert_eq!(batch.target_lengths, vec![2, 1]);
    }

    #[test]
    fn manifest_directory_loads_jsonl_shards_in_order() {
        let dir = std::env::temp_dir().join(format!(
            "w2v_bert_uk_manifest_dir_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("b.jsonl"),
            r#"{"id":"b","features":[[0.2,0.3]],"tokens":[2]}"#,
        )
        .unwrap();
        fs::write(
            dir.join("a.jsonl"),
            r#"{"id":"a","features":[[0.1,0.2]],"tokens":[1]}"#,
        )
        .unwrap();

        let records = load_manifest(&dir, Some(1)).unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, "a");
    }
}
