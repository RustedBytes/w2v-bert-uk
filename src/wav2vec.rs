use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::module::Module;
use burn::module::Param;
use burn::tensor::activation::{gelu, log_softmax};
use burn::tensor::ops::PadMode;
use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};
use burn_nn::conv::{Conv1d, Conv1dConfig};
use burn_nn::loss::{CTCLossConfig, Reduction};
use burn_nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};
use burn_nn::{
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig1d,
};
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};
use serde_json::{Map, Value, json};

pub const DEFAULT_W2V_BERT_MODEL: &str = "facebook/w2v-bert-2.0";

#[derive(Clone, Debug)]
pub struct Wav2VecBertConfig {
    pub model_name: String,
    pub hidden_size: usize,
    pub feature_dim: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub conv_pos_kernel_size: usize,
    pub conv_pos_groups: usize,
    pub dropout: f64,
    pub sample_rate: usize,
    pub model_config: Map<String, Value>,
    pub add_adapter: bool,
    pub adapter_stride: usize,
    pub num_adapter_layers: usize,
    pub activation_checkpointing: bool,
}

impl Default for Wav2VecBertConfig {
    fn default() -> Self {
        Self {
            model_name: DEFAULT_W2V_BERT_MODEL.to_string(),
            hidden_size: 1024,
            feature_dim: 160,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            conv_pos_kernel_size: 128,
            conv_pos_groups: 16,
            dropout: 0.1,
            sample_rate: 16_000,
            model_config: Map::new(),
            add_adapter: false,
            adapter_stride: 2,
            num_adapter_layers: 0,
            activation_checkpointing: false,
        }
    }
}

impl Wav2VecBertConfig {
    pub fn new(feature_dim: usize, hidden_size: usize) -> Self {
        Self {
            feature_dim,
            hidden_size,
            intermediate_size: hidden_size * 4,
            num_attention_heads: (hidden_size / 64).max(1),
            ..Self::default()
        }
    }

    pub fn with_layers(mut self, num_hidden_layers: usize) -> Self {
        self.num_hidden_layers = num_hidden_layers;
        self
    }

    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn with_model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = model_name.into();
        self
    }

    pub fn with_sample_rate(mut self, sample_rate: usize) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn with_model_config(mut self, model_config: Map<String, Value>) -> Self {
        self.model_config = model_config;
        if let Some(hidden_size) = self
            .model_config
            .get("hidden_size")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            self.hidden_size = hidden_size;
        }
        if let Some(feature_dim) = self
            .model_config
            .get("feature_projection_input_dim")
            .or_else(|| self.model_config.get("feature_dim"))
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            self.feature_dim = feature_dim;
        }
        if let Some(intermediate_size) = self
            .model_config
            .get("intermediate_size")
            .or_else(|| self.model_config.get("encoder_ffn_dim"))
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            self.intermediate_size = intermediate_size;
        }
        if let Some(num_hidden_layers) = self
            .model_config
            .get("num_hidden_layers")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            self.num_hidden_layers = num_hidden_layers;
        }
        if let Some(num_attention_heads) = self
            .model_config
            .get("num_attention_heads")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            self.num_attention_heads = num_attention_heads;
        }
        if let Some(kernel_size) = self
            .model_config
            .get("conv_pos_kernel_size")
            .or_else(|| self.model_config.get("num_conv_pos_embeddings"))
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            self.conv_pos_kernel_size = kernel_size;
        }
        if let Some(groups) = self
            .model_config
            .get("conv_pos_groups")
            .or_else(|| self.model_config.get("num_conv_pos_embedding_groups"))
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            self.conv_pos_groups = groups;
        }
        if let Some(dropout) = self
            .model_config
            .get("hidden_dropout")
            .or_else(|| self.model_config.get("hidden_dropout_prob"))
            .and_then(Value::as_f64)
        {
            self.dropout = dropout;
        }
        if let Some(add_adapter) = self
            .model_config
            .get("add_adapter")
            .and_then(Value::as_bool)
        {
            self.add_adapter = add_adapter;
        }
        if let Some(stride) = self
            .model_config
            .get("adapter_stride")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            self.adapter_stride = stride.max(1);
        }
        if let Some(layers) = self
            .model_config
            .get("num_adapter_layers")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            self.num_adapter_layers = layers;
        }
        if let Some(activation_checkpointing) = self
            .model_config
            .get("activation_checkpointing")
            .or_else(|| self.model_config.get("gradient_checkpointing"))
            .and_then(Value::as_bool)
        {
            self.activation_checkpointing = activation_checkpointing;
        }
        self
    }

    pub fn from_mapping(mapping: Map<String, Value>) -> Self {
        let model_name = mapping
            .get("model_name")
            .and_then(Value::as_str)
            .unwrap_or(DEFAULT_W2V_BERT_MODEL)
            .to_string();
        let model_config = mapping
            .get("model_config")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let mut config = Self::default()
            .with_model_name(model_name)
            .with_model_config(model_config);
        if let Some(hidden_size) = mapping
            .get("hidden_size")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            config.hidden_size = hidden_size;
        }
        if let Some(feature_dim) = mapping
            .get("feature_dim")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            config.feature_dim = feature_dim;
        }
        if let Some(sample_rate) = mapping
            .get("sample_rate")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            config.sample_rate = sample_rate;
        }
        if let Some(activation_checkpointing) = mapping
            .get("activation_checkpointing")
            .and_then(Value::as_bool)
        {
            config.activation_checkpointing = activation_checkpointing;
        }
        config
    }

    pub fn from_huggingface_dir(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let config_path = if path.is_dir() {
            path.join("config.json")
        } else {
            path.to_path_buf()
        };
        let contents = fs::read_to_string(&config_path).with_context(|| {
            format!(
                "failed to read Hugging Face W2V-BERT config {}",
                config_path.display()
            )
        })?;
        let mut model_config = match serde_json::from_str::<Value>(&contents)
            .with_context(|| format!("failed to parse {}", config_path.display()))?
        {
            Value::Object(values) => values,
            _ => bail!("{} must contain a JSON object", config_path.display()),
        };
        model_config.insert("apply_spec_augment".to_string(), json!(false));
        let model_name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or(DEFAULT_W2V_BERT_MODEL)
            .to_string();
        Ok(Self::default()
            .with_model_name(model_name)
            .with_model_config(model_config))
    }

    pub fn with_adapter(mut self, adapter_stride: usize, num_adapter_layers: usize) -> Self {
        self.add_adapter = num_adapter_layers > 0;
        self.adapter_stride = adapter_stride.max(1);
        self.num_adapter_layers = num_adapter_layers;
        self
    }

    pub fn with_activation_checkpointing(mut self, activation_checkpointing: bool) -> Self {
        self.activation_checkpointing = activation_checkpointing;
        self
    }

    pub fn model_dim(&self) -> usize {
        self.hidden_size
    }

    pub fn to_config_dict(&self) -> Value {
        json!({
            "architecture": "w2v_bert",
            "model_name": self.model_name,
            "hidden_size": self.hidden_size,
            "feature_dim": self.feature_dim,
            "sample_rate": self.sample_rate,
            "model_config": self.model_config,
            "activation_checkpointing": self.activation_checkpointing,
        })
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Wav2VecBertModel<B> {
        Wav2VecBertModel {
            feature_projection: Wav2VecFeatureProjection {
                layer_norm: LayerNormConfig::new(self.feature_dim).init(device),
                projection: LinearConfig::new(self.feature_dim, self.hidden_size).init(device),
                dropout: DropoutConfig::new(self.dropout).init(),
                feature_dim: self.feature_dim,
            },
            positional_conv: Conv1dConfig::new(
                self.hidden_size,
                self.hidden_size,
                self.conv_pos_kernel_size,
            )
            .with_groups(self.conv_pos_groups.min(self.hidden_size).max(1))
            .with_padding(PaddingConfig1d::Explicit(
                self.conv_pos_kernel_size / 2,
                self.conv_pos_kernel_size / 2,
            ))
            .init(device),
            encoder: TransformerEncoderConfig::new(
                self.hidden_size,
                self.intermediate_size,
                self.num_attention_heads,
                self.num_hidden_layers,
            )
            .with_dropout(self.dropout)
            .with_norm_first(true)
            .init(device),
            final_norm: LayerNormConfig::new(self.hidden_size).init(device),
            add_adapter: self.add_adapter,
            adapter_stride: self.adapter_stride.max(1),
            num_adapter_layers: self.num_adapter_layers,
        }
    }
}

#[derive(Module, Debug)]
struct Wav2VecFeatureProjection<B: Backend> {
    layer_norm: LayerNorm<B>,
    projection: Linear<B>,
    dropout: Dropout,
    feature_dim: usize,
}

impl<B: Backend> Wav2VecFeatureProjection<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let output = self.layer_norm.forward(input);
        self.dropout.forward(self.projection.forward(output))
    }
}

#[derive(Module, Debug)]
pub struct Wav2VecBertModel<B: Backend> {
    feature_projection: Wav2VecFeatureProjection<B>,
    positional_conv: Conv1d<B>,
    encoder: TransformerEncoder<B>,
    final_norm: LayerNorm<B>,
    add_adapter: bool,
    adapter_stride: usize,
    num_adapter_layers: usize,
}

impl<B: Backend> Wav2VecBertModel<B> {
    pub fn forward(
        &self,
        input_features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        let device = input_features.device();
        assert_eq!(
            input_features.dims()[2],
            self.feature_projection.feature_dim,
            "W2V-BERT feature dim must match the encoder configuration"
        );
        let lengths = clamp_lengths(&lengths, input_features.dims()[1]);
        let mut hidden = self.feature_projection.forward(input_features);
        let [batch_size, seq_len, hidden_size] = hidden.dims();
        let pos = self
            .positional_conv
            .forward(hidden.clone().swap_dims(1, 2))
            .slice_dim(2, 0..seq_len)
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, hidden_size]);
        hidden = hidden + gelu(pos);
        hidden = mask_time(hidden, &lengths);
        let mask = padding_mask::<B>(&lengths, seq_len, &device);
        let encoded = self
            .encoder
            .forward(TransformerEncoderInput::new(hidden).mask_pad(mask));
        let mut encoded = mask_time(encoded, &lengths);
        let mut output_lengths = lengths;
        if self.add_adapter {
            for _ in 0..self.num_adapter_layers {
                (encoded, output_lengths) =
                    downsample_time(encoded, &output_lengths, self.adapter_stride);
            }
        }
        output_lengths = clamp_lengths(&output_lengths, encoded.dims()[1]);
        let encoded = self.final_norm.forward(mask_time(encoded, &output_lengths));
        (encoded, output_lengths)
    }
}

#[derive(Clone, Debug)]
pub struct Wav2VecBertCtcConfig {
    pub encoder: Wav2VecBertConfig,
    pub vocab_size: usize,
}

impl Wav2VecBertCtcConfig {
    pub fn new(feature_dim: usize, hidden_size: usize, vocab_size: usize) -> Self {
        Self {
            encoder: Wav2VecBertConfig::new(feature_dim, hidden_size),
            vocab_size,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Wav2VecBertCtc<B> {
        let mut classifier =
            LinearConfig::new(self.encoder.hidden_size, self.vocab_size).init(device);
        if let Some(bias) = classifier.bias.as_mut() {
            *bias = Param::from_tensor(Tensor::zeros([self.vocab_size], device));
        }
        Wav2VecBertCtc {
            encoder: self.encoder.init(device),
            classifier,
        }
    }

    pub fn from_huggingface_dir(path: impl AsRef<Path>, vocab_size: Option<usize>) -> Result<Self> {
        let encoder = Wav2VecBertConfig::from_huggingface_dir(path.as_ref())?;
        let vocab_size = vocab_size
            .or_else(|| {
                encoder
                    .model_config
                    .get("vocab_size")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
            })
            .context("vocab size is required when the Hugging Face config has no vocab_size")?;
        Ok(Self {
            encoder,
            vocab_size,
        })
    }

    pub fn init_from_huggingface_dir<B: Backend>(
        path: impl AsRef<Path>,
        vocab_size: Option<usize>,
        device: &B::Device,
    ) -> Result<(Wav2VecBertCtc<B>, Wav2VecBertImportReport)> {
        let path = path.as_ref();
        let config = Self::from_huggingface_dir(path, vocab_size)?;
        let mut model = config.init(device);
        let report = model.load_huggingface_weights(path, device)?;
        Ok((model, report))
    }
}

#[derive(Module, Debug)]
pub struct Wav2VecBertCtc<B: Backend> {
    encoder: Wav2VecBertModel<B>,
    classifier: Linear<B>,
}

#[derive(Debug)]
pub struct Wav2VecBertTrainingOutput<B: Backend> {
    pub encoded: Tensor<B, 3>,
    pub output_lengths: Vec<usize>,
    pub main_logits: Option<Tensor<B, 3>>,
    pub main_log_probs: Option<Tensor<B, 3>>,
    pub main_ctc_loss: Option<Tensor<B, 1>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Wav2VecBertImportReport {
    pub source_files: Vec<PathBuf>,
    pub loaded: Vec<String>,
    pub skipped_missing: Vec<String>,
    pub skipped_shape: Vec<String>,
}

impl Wav2VecBertImportReport {
    pub fn loaded_count(&self) -> usize {
        self.loaded.len()
    }
}

impl<B: Backend> Wav2VecBertCtc<B> {
    pub fn forward(&self, input_features: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = input_features.dims();
        self.forward_with_lengths(input_features, vec![seq_len; batch_size])
            .0
    }

    pub fn forward_with_lengths(
        &self,
        input_features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        let (encoded, lengths) = self.encoder.forward(input_features, lengths);
        (self.classifier.forward(encoded), lengths)
    }

    pub fn encode(
        &self,
        input_features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        self.encoder.forward(input_features, lengths)
    }

    pub fn log_probs(
        &self,
        input_features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        let (logits, output_lengths) = self.forward_with_lengths(input_features, lengths);
        (log_softmax(logits, 2), output_lengths)
    }

    pub fn forward_training(
        &self,
        input_features: Tensor<B, 3>,
        lengths: Vec<usize>,
        targets: Option<Tensor<B, 2, Int>>,
        target_lengths: Option<Vec<usize>>,
        blank_id: Option<usize>,
        return_main_log_probs: bool,
    ) -> Wav2VecBertTrainingOutput<B> {
        let (encoded, output_lengths) = self.encode(input_features, lengths);
        let needs_logits = return_main_log_probs || targets.is_some();
        let logits = needs_logits.then(|| self.classifier.forward(encoded.clone()));
        let log_probs = logits.as_ref().map(|value| log_softmax(value.clone(), 2));
        let main_ctc_loss = match (targets, target_lengths, blank_id, log_probs.as_ref()) {
            (Some(targets), Some(target_lengths), Some(blank_id), Some(log_probs)) => {
                Some(ctc_loss_from_log_probs(
                    log_probs.clone(),
                    targets,
                    output_lengths.clone(),
                    target_lengths,
                    blank_id,
                ))
            }
            _ => None,
        };

        Wav2VecBertTrainingOutput {
            encoded,
            output_lengths,
            main_logits: return_main_log_probs.then(|| logits.expect("logits computed")),
            main_log_probs: return_main_log_probs.then(|| log_probs.expect("log_probs computed")),
            main_ctc_loss,
        }
    }

    pub fn load_huggingface_weights(
        &mut self,
        path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<Wav2VecBertImportReport> {
        let store = HfSafetensors::load(path.as_ref())?;
        let mut report = Wav2VecBertImportReport {
            source_files: store.source_files.clone(),
            ..Default::default()
        };

        load_layer_norm(
            &mut self.encoder.feature_projection.layer_norm,
            &store,
            &[
                "feature_projection.layer_norm",
                "wav2vec2_bert.feature_projection.layer_norm",
                "model.feature_projection.layer_norm",
            ],
            device,
            &mut report,
        );
        load_linear(
            &mut self.encoder.feature_projection.projection,
            &store,
            &[
                "feature_projection.projection",
                "wav2vec2_bert.feature_projection.projection",
                "model.feature_projection.projection",
            ],
            true,
            device,
            &mut report,
        );
        load_conv1d(
            &mut self.encoder.positional_conv,
            &store,
            &[
                "encoder.pos_conv_embed.conv",
                "wav2vec2_bert.encoder.pos_conv_embed.conv",
                "model.encoder.pos_conv_embed.conv",
            ],
            device,
            &mut report,
        );
        load_layer_norm(
            &mut self.encoder.final_norm,
            &store,
            &[
                "encoder.layer_norm",
                "wav2vec2_bert.encoder.layer_norm",
                "model.encoder.layer_norm",
            ],
            device,
            &mut report,
        );
        load_linear(
            &mut self.classifier,
            &store,
            &["lm_head", "classifier", "ctc_head", "projector"],
            true,
            device,
            &mut report,
        );

        for (index, layer) in self.encoder.encoder.layers.iter_mut().enumerate() {
            let prefixes = layer_prefixes(index);
            load_linear(
                &mut layer.mha.query,
                &store,
                &prefix_field(&prefixes, "attention.q_proj"),
                true,
                device,
                &mut report,
            );
            load_linear(
                &mut layer.mha.key,
                &store,
                &prefix_field(&prefixes, "attention.k_proj"),
                true,
                device,
                &mut report,
            );
            load_linear(
                &mut layer.mha.value,
                &store,
                &prefix_field(&prefixes, "attention.v_proj"),
                true,
                device,
                &mut report,
            );
            load_linear(
                &mut layer.mha.output,
                &store,
                &prefix_field(&prefixes, "attention.out_proj"),
                true,
                device,
                &mut report,
            );
            load_linear(
                &mut layer.pwff.linear_inner,
                &store,
                &prefix_field(&prefixes, "feed_forward.intermediate_dense"),
                true,
                device,
                &mut report,
            );
            load_linear(
                &mut layer.pwff.linear_outer,
                &store,
                &prefix_field(&prefixes, "feed_forward.output_dense"),
                true,
                device,
                &mut report,
            );
            load_layer_norm(
                &mut layer.norm_2,
                &store,
                &prefix_field(&prefixes, "layer_norm"),
                device,
                &mut report,
            );
            load_layer_norm(
                &mut layer.norm_1,
                &store,
                &prefix_field(&prefixes, "final_layer_norm"),
                device,
                &mut report,
            );
        }

        Ok(report)
    }
}

#[derive(Clone, Debug)]
struct HfTensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Clone, Debug)]
struct HfSafetensors {
    source_files: Vec<PathBuf>,
    tensors: HashMap<String, HfTensor>,
}

impl HfSafetensors {
    fn load(path: &Path) -> Result<Self> {
        let files = resolve_safetensor_files(path)?;
        if files.is_empty() {
            bail!(
                "no Hugging Face .safetensors files found at {}",
                path.display()
            );
        }
        let mut tensors = HashMap::new();
        for file in &files {
            let bytes = fs::read(file)
                .with_context(|| format!("failed to read safetensors file {}", file.display()))?;
            let safetensors = SafeTensors::deserialize(&bytes)
                .with_context(|| format!("failed to parse {}", file.display()))?;
            for name in safetensors.names() {
                let view = safetensors
                    .tensor(name)
                    .with_context(|| format!("failed to read tensor {name}"))?;
                tensors.insert(name.to_string(), tensor_view_to_f32(&view)?);
            }
        }
        Ok(Self {
            source_files: files,
            tensors,
        })
    }

    fn get_any<'a>(&'a self, prefixes: &[String], suffix: &str) -> Option<(&'a str, &'a HfTensor)> {
        for prefix in prefixes {
            let name = format!("{prefix}.{suffix}");
            if let Some(tensor) = self.tensors.get(&name) {
                return Some((
                    self.tensors.get_key_value(&name).unwrap().0.as_str(),
                    tensor,
                ));
            }
        }
        None
    }
}

fn resolve_safetensor_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
            return Ok(vec![path.to_path_buf()]);
        }
        bail!("{} is not a .safetensors file", path.display());
    }
    let index_path = path.join("model.safetensors.index.json");
    if index_path.exists() {
        let index: Value = serde_json::from_str(&fs::read_to_string(&index_path)?)
            .with_context(|| format!("failed to parse {}", index_path.display()))?;
        let mut files = index
            .get("weight_map")
            .and_then(Value::as_object)
            .map(|weight_map| {
                let mut values = weight_map
                    .values()
                    .filter_map(Value::as_str)
                    .map(|value| path.join(value))
                    .collect::<Vec<_>>();
                values.sort();
                values.dedup();
                values
            })
            .unwrap_or_default();
        files.retain(|file| file.exists());
        return Ok(files);
    }
    let preferred = [
        path.join("model.safetensors"),
        path.join("pytorch_model.safetensors"),
    ];
    let mut files = preferred
        .into_iter()
        .filter(|file| file.exists())
        .collect::<Vec<_>>();
    if files.is_empty() {
        for entry in
            fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))?
        {
            let entry = entry?;
            let file = entry.path();
            if file.extension().and_then(|value| value.to_str()) == Some("safetensors") {
                files.push(file);
            }
        }
        files.sort();
    }
    Ok(files)
}

fn tensor_view_to_f32(view: &TensorView<'_>) -> Result<HfTensor> {
    let shape = view.shape().to_vec();
    let values = match view.dtype() {
        Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 chunk")))
            .collect(),
        Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|bytes| half::f16::from_le_bytes(bytes.try_into().expect("f16 chunk")).to_f32())
            .collect(),
        Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|bytes| half::bf16::from_le_bytes(bytes.try_into().expect("bf16 chunk")).to_f32())
            .collect(),
        other => bail!("unsupported safetensors dtype {other:?}; expected f32/f16/bf16"),
    };
    Ok(HfTensor { shape, values })
}

fn layer_prefixes(index: usize) -> Vec<String> {
    [
        format!("encoder.layers.{index}"),
        format!("wav2vec2_bert.encoder.layers.{index}"),
        format!("model.encoder.layers.{index}"),
    ]
    .into()
}

fn prefix_field(prefixes: &[String], field: &str) -> Vec<String> {
    prefixes
        .iter()
        .map(|prefix| format!("{prefix}.{field}"))
        .collect()
}

fn str_prefixes(prefixes: &[&str]) -> Vec<String> {
    prefixes.iter().map(|value| (*value).to_string()).collect()
}

fn load_linear<B: Backend>(
    linear: &mut Linear<B>,
    store: &HfSafetensors,
    prefixes: &[impl AsRef<str>],
    transpose_weight: bool,
    device: &B::Device,
    report: &mut Wav2VecBertImportReport,
) {
    let prefixes = prefixes
        .iter()
        .map(|value| value.as_ref().to_string())
        .collect::<Vec<_>>();
    let [d_input, d_output] = linear.weight.dims();
    load_param_2d(
        &mut linear.weight,
        store,
        &prefixes,
        "weight",
        [d_input, d_output],
        transpose_weight,
        device,
        report,
    );
    if let Some(bias) = linear.bias.as_mut() {
        load_param_1d(bias, store, &prefixes, "bias", d_output, device, report);
    }
}

fn load_conv1d<B: Backend>(
    conv: &mut Conv1d<B>,
    store: &HfSafetensors,
    prefixes: &[&str],
    device: &B::Device,
    report: &mut Wav2VecBertImportReport,
) {
    let prefixes = str_prefixes(prefixes);
    let [out_channels, in_channels, kernel] = conv.weight.dims();
    load_param_3d(
        &mut conv.weight,
        store,
        &prefixes,
        "weight",
        [out_channels, in_channels, kernel],
        device,
        report,
    );
    if let Some(bias) = conv.bias.as_mut() {
        load_param_1d(bias, store, &prefixes, "bias", out_channels, device, report);
    }
}

fn load_layer_norm<B: Backend>(
    norm: &mut LayerNorm<B>,
    store: &HfSafetensors,
    prefixes: &[impl AsRef<str>],
    device: &B::Device,
    report: &mut Wav2VecBertImportReport,
) {
    let prefixes = prefixes
        .iter()
        .map(|value| value.as_ref().to_string())
        .collect::<Vec<_>>();
    let dim = norm.gamma.dims()[0];
    load_param_1d(
        &mut norm.gamma,
        store,
        &prefixes,
        "weight",
        dim,
        device,
        report,
    );
    if let Some(beta) = norm.beta.as_mut() {
        load_param_1d(beta, store, &prefixes, "bias", dim, device, report);
    }
}

fn load_param_1d<B: Backend>(
    param: &mut Param<Tensor<B, 1>>,
    store: &HfSafetensors,
    prefixes: &[String],
    suffix: &str,
    expected: usize,
    device: &B::Device,
    report: &mut Wav2VecBertImportReport,
) {
    let Some((name, tensor)) = store.get_any(prefixes, suffix) else {
        report.skipped_missing.push(format!("*.{suffix}"));
        return;
    };
    if tensor.shape != [expected] {
        report
            .skipped_shape
            .push(format!("{name}: {:?} != [{expected}]", tensor.shape));
        return;
    }
    *param = Param::from_tensor(Tensor::from_data(
        TensorData::new(tensor.values.clone(), [expected]),
        device,
    ));
    report.loaded.push(name.to_string());
}

fn load_param_2d<B: Backend>(
    param: &mut Param<Tensor<B, 2>>,
    store: &HfSafetensors,
    prefixes: &[String],
    suffix: &str,
    expected: [usize; 2],
    transpose: bool,
    device: &B::Device,
    report: &mut Wav2VecBertImportReport,
) {
    let Some((name, tensor)) = store.get_any(prefixes, suffix) else {
        report.skipped_missing.push(format!("*.{suffix}"));
        return;
    };
    let values = if tensor.shape == expected {
        tensor.values.clone()
    } else if transpose && tensor.shape == [expected[1], expected[0]] {
        transpose_2d(&tensor.values, expected[1], expected[0])
    } else {
        report
            .skipped_shape
            .push(format!("{name}: {:?} != {:?}", tensor.shape, expected));
        return;
    };
    *param = Param::from_tensor(Tensor::from_data(TensorData::new(values, expected), device));
    report.loaded.push(name.to_string());
}

fn load_param_3d<B: Backend>(
    param: &mut Param<Tensor<B, 3>>,
    store: &HfSafetensors,
    prefixes: &[String],
    suffix: &str,
    expected: [usize; 3],
    device: &B::Device,
    report: &mut Wav2VecBertImportReport,
) {
    let Some((name, tensor)) = store.get_any(prefixes, suffix) else {
        report.skipped_missing.push(format!("*.{suffix}"));
        return;
    };
    if tensor.shape != expected {
        report
            .skipped_shape
            .push(format!("{name}: {:?} != {:?}", tensor.shape, expected));
        return;
    }
    *param = Param::from_tensor(Tensor::from_data(
        TensorData::new(tensor.values.clone(), expected),
        device,
    ));
    report.loaded.push(name.to_string());
}

fn transpose_2d(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = values[row * cols + col];
        }
    }
    transposed
}

fn downsample_time<B: Backend>(
    input: Tensor<B, 3>,
    lengths: &[usize],
    stride: usize,
) -> (Tensor<B, 3>, Vec<usize>) {
    if stride <= 1 {
        return (input, lengths.to_vec());
    }
    let [_, seq_len, _] = input.dims();
    let pad = (stride - (seq_len % stride)) % stride;
    let output = if pad > 0 {
        input.pad([(0, pad), (0, 0)], PadMode::Edge)
    } else {
        input
    };
    let [batch_size, padded_len, dim] = output.dims();
    let output = output
        .reshape([batch_size, padded_len / stride, stride, dim])
        .mean_dim(2)
        .reshape([batch_size, padded_len / stride, dim]);
    let output_len = output.dims()[1];
    let lengths = lengths
        .iter()
        .map(|length| ceil_divide(*length, stride).clamp(1, output_len))
        .collect();
    (output, lengths)
}

fn clamp_lengths(lengths: &[usize], max_len: usize) -> Vec<usize> {
    lengths
        .iter()
        .map(|length| (*length).clamp(1, max_len))
        .collect()
}

fn ceil_divide(value: usize, divisor: usize) -> usize {
    value.div_ceil(divisor)
}

fn to_i64(values: Vec<usize>) -> Vec<i64> {
    values.into_iter().map(|value| value as i64).collect()
}

fn ctc_loss_from_log_probs<B: Backend>(
    log_probs: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    input_lengths: Vec<usize>,
    target_lengths: Vec<usize>,
    blank_id: usize,
) -> Tensor<B, 1> {
    let batch_size = target_lengths.len();
    let device = log_probs.device();
    let input_lengths = Tensor::<B, 1, Int>::from_data(
        TensorData::new(to_i64(input_lengths), [batch_size]),
        &device,
    );
    let target_lengths = Tensor::<B, 1, Int>::from_data(
        TensorData::new(to_i64(target_lengths), [batch_size]),
        &device,
    );
    CTCLossConfig::new()
        .with_blank(blank_id)
        .with_zero_infinity(true)
        .init()
        .forward_with_reduction(
            log_probs.swap_dims(0, 1),
            targets,
            input_lengths,
            target_lengths,
            Reduction::Mean,
        )
}

fn padding_mask<B: Backend>(
    lengths: &[usize],
    max_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let mut values = Vec::with_capacity(lengths.len() * max_len);
    for length in lengths {
        for index in 0..max_len {
            values.push(index >= *length);
        }
    }
    Tensor::from_data(TensorData::new(values, [lengths.len(), max_len]), device)
}

fn sequence_mask<B: Backend>(
    lengths: &[usize],
    max_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let mut values = Vec::with_capacity(lengths.len() * max_len);
    for length in lengths {
        for index in 0..max_len {
            values.push(index < *length);
        }
    }
    Tensor::from_data(TensorData::new(values, [lengths.len(), max_len]), device)
}

fn mask_time<B: Backend>(input: Tensor<B, 3>, lengths: &[usize]) -> Tensor<B, 3> {
    let seq_len = input.dims()[1];
    let mask = sequence_mask::<B>(lengths, seq_len, &input.device())
        .float()
        .unsqueeze_dim::<3>(2);
    input * mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;
    use safetensors::serialize_to_file;

    type TestBackend = NdArray<f32>;
    type TestAutodiffBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn wav2vec_ctc_preserves_feature_time_axis() {
        let device = Default::default();
        let config = Wav2VecBertCtcConfig {
            encoder: Wav2VecBertConfig::new(160, 32).with_layers(1),
            vocab_size: 40,
        };
        let model = config.init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 9, 160], &device);

        let (output, lengths) = model.forward_with_lengths(input, vec![9, 7]);

        assert_eq!(output.dims(), [2, 9, 40]);
        assert_eq!(lengths, vec![9, 7]);
    }

    #[test]
    fn wav2vec_adapter_downsamples_time_like_hf_adapter_lengths() {
        let device = Default::default();
        let config = Wav2VecBertCtcConfig {
            encoder: Wav2VecBertConfig::new(8, 16)
                .with_layers(1)
                .with_adapter(2, 2),
            vocab_size: 6,
        };
        let model = config.init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 9, 8], &device);

        let (output, lengths) = model.forward_with_lengths(input, vec![9, 7]);

        assert_eq!(output.dims(), [2, 3, 6]);
        assert_eq!(lengths, vec![3, 2]);
    }

    #[test]
    fn wav2vec_training_outputs_include_ctc_loss_and_log_probs() {
        let device = Default::default();
        let config = Wav2VecBertCtcConfig {
            encoder: Wav2VecBertConfig::new(8, 16).with_layers(1),
            vocab_size: 6,
        };
        let model = config.init::<TestAutodiffBackend>(&device);
        let input = Tensor::<TestAutodiffBackend, 3>::zeros([2, 6, 8], &device);
        let targets = Tensor::<TestAutodiffBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 2, 0], [2, 2]),
            &device,
        );

        let output = model.forward_training(
            input,
            vec![6, 5],
            Some(targets),
            Some(vec![2, 1]),
            Some(0),
            true,
        );

        assert_eq!(output.encoded.dims(), [2, 6, 16]);
        assert_eq!(output.output_lengths, vec![6, 5]);
        assert_eq!(output.main_logits.unwrap().dims(), [2, 6, 6]);
        assert_eq!(output.main_log_probs.unwrap().dims(), [2, 6, 6]);
        assert_eq!(output.main_ctc_loss.unwrap().dims(), [1]);
    }

    #[test]
    fn wav2vec_config_round_trips_python_wrapper_fields() {
        let mut hf_config = Map::new();
        hf_config.insert("hidden_size".to_string(), json!(16));
        hf_config.insert("feature_projection_input_dim".to_string(), json!(8));
        hf_config.insert("num_hidden_layers".to_string(), json!(1));
        hf_config.insert("num_attention_heads".to_string(), json!(2));
        hf_config.insert("intermediate_size".to_string(), json!(32));
        hf_config.insert("add_adapter".to_string(), json!(true));
        hf_config.insert("adapter_stride".to_string(), json!(2));
        hf_config.insert("num_adapter_layers".to_string(), json!(2));
        hf_config.insert("gradient_checkpointing".to_string(), json!(true));
        let mut mapping = Map::new();
        mapping.insert("model_name".to_string(), json!("local-w2v-bert"));
        mapping.insert("sample_rate".to_string(), json!(16_000));
        mapping.insert("model_config".to_string(), Value::Object(hf_config));

        let config = Wav2VecBertConfig::from_mapping(mapping);
        let value = config.to_config_dict();

        assert_eq!(config.model_dim(), 16);
        assert_eq!(config.feature_dim, 8);
        assert_eq!(config.num_attention_heads, 2);
        assert!(config.add_adapter);
        assert!(config.activation_checkpointing);
        assert_eq!(value["architecture"], "w2v_bert");
        assert_eq!(value["model_name"], "local-w2v-bert");
        assert_eq!(value["activation_checkpointing"], true);
    }

    #[test]
    fn wav2vec_config_loads_local_huggingface_config_json() {
        let dir =
            std::env::temp_dir().join(format!("w2v_bert_uk_hf_config_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            serde_json::to_string(&json!({
                "hidden_size": 16,
                "feature_projection_input_dim": 8,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_conv_pos_embeddings": 9,
                "num_conv_pos_embedding_groups": 1,
                "hidden_dropout": 0.0,
                "vocab_size": 6
            }))
            .unwrap(),
        )
        .unwrap();

        let config = Wav2VecBertCtcConfig::from_huggingface_dir(&dir, None).unwrap();

        assert_eq!(config.encoder.hidden_size, 16);
        assert_eq!(config.encoder.feature_dim, 8);
        assert_eq!(config.encoder.conv_pos_kernel_size, 9);
        assert_eq!(config.vocab_size, 6);
        assert_eq!(
            config
                .encoder
                .model_config
                .get("apply_spec_augment")
                .and_then(Value::as_bool),
            Some(false)
        );
    }

    #[test]
    fn wav2vec_imports_matching_huggingface_safetensors() {
        let dir =
            std::env::temp_dir().join(format!("w2v_bert_uk_hf_weights_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let weight_values = (0..(6 * 16))
            .map(|value| value as f32 / 100.0)
            .collect::<Vec<_>>();
        let bias_values = vec![0.0_f32; 6];
        let weight_bytes = f32_bytes(&weight_values);
        let bias_bytes = f32_bytes(&bias_values);
        let weight =
            safetensors::tensor::TensorView::new(Dtype::F32, vec![6, 16], &weight_bytes).unwrap();
        let bias = safetensors::tensor::TensorView::new(Dtype::F32, vec![6], &bias_bytes).unwrap();
        serialize_to_file(
            vec![("lm_head.weight", weight), ("lm_head.bias", bias)],
            None,
            &dir.join("model.safetensors"),
        )
        .unwrap();
        let device = Default::default();
        let mut model = Wav2VecBertCtcConfig {
            encoder: Wav2VecBertConfig::new(8, 16).with_layers(1),
            vocab_size: 6,
        }
        .init::<TestBackend>(&device);

        let report = model.load_huggingface_weights(&dir, &device).unwrap();

        assert!(report.loaded.contains(&"lm_head.weight".to_string()));
        assert!(report.loaded.contains(&"lm_head.bias".to_string()));
        assert_eq!(report.source_files.len(), 1);
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }
}
