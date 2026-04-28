use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::module::Module;
use burn::module::{Initializer, Param};
use burn::tensor::activation::{gelu, log_softmax, sigmoid, softmax};
#[cfg(feature = "asr-cubecl-kernels")]
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::ops::PadMode;
use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};
use burn_nn::conv::{Conv1d, Conv1dConfig};
use burn_nn::loss::{CTCLossConfig, Reduction};
use burn_nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
    LinearConfig, PaddingConfig1d,
};
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};
use serde_json::{Map, Value, json};

pub const DEFAULT_W2V_BERT_MODEL: &str = "facebook/w2v-bert-2.0";

pub trait Wav2VecKernelBackend: Backend + Sized {
    fn mask_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        mask_time_fallback(input, lengths)
    }

    fn sequence_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 2, Bool> {
        sequence_mask_fallback(lengths, max_len, device)
    }

    fn glu_channel_dim(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        glu_fallback(input, 1)
    }
}

impl Wav2VecKernelBackend for burn_ndarray::NdArray<f32> {}

impl<C> Wav2VecKernelBackend for burn_autodiff::Autodiff<burn_ndarray::NdArray<f32>, C> where
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy
{
}

#[cfg(feature = "asr-cubecl-kernels")]
fn inner_bool_to_autodiff<B, C, const D: usize>(
    tensor: Tensor<B, D, Bool>,
) -> Tensor<burn_autodiff::Autodiff<B, C>, D, Bool>
where
    B: Backend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    let tensor = <burn_autodiff::Autodiff<B, C> as AutodiffBackend>::bool_from_inner(
        tensor.into_primitive(),
    );
    Tensor::from_primitive(tensor)
}

#[cfg(all(feature = "asr-cubecl-kernels", feature = "burn-cuda-backend"))]
impl<F, I> Wav2VecKernelBackend for burn_cuda::Cuda<F, I>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
{
    fn mask_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::cubecl_kernels::mask_time(input, lengths)
    }

    fn sequence_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 2, Bool> {
        crate::cubecl_kernels::sequence_mask(lengths, max_len, device)
    }

    fn glu_channel_dim(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        crate::cubecl_kernels::glu_channel_dim(input)
    }
}

#[cfg(all(feature = "asr-cubecl-kernels", feature = "burn-cuda-backend"))]
impl<F, I, C> Wav2VecKernelBackend for burn_autodiff::Autodiff<burn_cuda::Cuda<F, I>, C>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    fn mask_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::asr_autodiff_kernels::mask_time(input, lengths)
    }

    fn sequence_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 2, Bool> {
        let raw = crate::cubecl_kernels::sequence_mask::<_, F, I, u8>(lengths, max_len, device);
        inner_bool_to_autodiff(raw)
    }

    fn glu_channel_dim(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        crate::asr_autodiff_kernels::glu_channel_dim(input)
    }
}

#[cfg(all(feature = "asr-cubecl-kernels", feature = "burn-wgpu-backend"))]
impl<F, I, BT> Wav2VecKernelBackend for burn_wgpu::Wgpu<F, I, BT>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
{
    fn mask_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::cubecl_kernels::mask_time(input, lengths)
    }

    fn sequence_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 2, Bool> {
        crate::cubecl_kernels::sequence_mask(lengths, max_len, device)
    }

    fn glu_channel_dim(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        crate::cubecl_kernels::glu_channel_dim(input)
    }
}

#[cfg(all(feature = "asr-cubecl-kernels", feature = "burn-wgpu-backend"))]
impl<F, I, BT, C> Wav2VecKernelBackend for burn_autodiff::Autodiff<burn_wgpu::Wgpu<F, I, BT>, C>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    fn mask_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::asr_autodiff_kernels::mask_time(input, lengths)
    }

    fn sequence_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 2, Bool> {
        let raw = crate::cubecl_kernels::sequence_mask::<_, F, I, BT>(lengths, max_len, device);
        inner_bool_to_autodiff(raw)
    }

    fn glu_channel_dim(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        crate::asr_autodiff_kernels::glu_channel_dim(input)
    }
}

#[cfg(all(feature = "burn-cuda-backend", not(feature = "asr-cubecl-kernels")))]
impl<F, I> Wav2VecKernelBackend for burn_cuda::Cuda<F, I>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    burn_cuda::Cuda<F, I>: Backend,
{
}

#[cfg(all(feature = "burn-cuda-backend", not(feature = "asr-cubecl-kernels")))]
impl<F, I, C> Wav2VecKernelBackend for burn_autodiff::Autodiff<burn_cuda::Cuda<F, I>, C>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    burn_cuda::Cuda<F, I>: Backend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
}

#[cfg(all(feature = "burn-wgpu-backend", not(feature = "asr-cubecl-kernels")))]
impl<F, I, BT> Wav2VecKernelBackend for burn_wgpu::Wgpu<F, I, BT>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
    burn_wgpu::Wgpu<F, I, BT>: Backend,
{
}

#[cfg(all(feature = "burn-wgpu-backend", not(feature = "asr-cubecl-kernels")))]
impl<F, I, BT, C> Wav2VecKernelBackend for burn_autodiff::Autodiff<burn_wgpu::Wgpu<F, I, BT>, C>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
    burn_wgpu::Wgpu<F, I, BT>: Backend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
}

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
    pub conv_depthwise_kernel_size: usize,
    pub left_max_position_embeddings: usize,
    pub right_max_position_embeddings: usize,
    pub dropout: f64,
    pub sample_rate: usize,
    pub model_config: Map<String, Value>,
    pub add_adapter: bool,
    pub adapter_stride: usize,
    pub adapter_kernel_size: usize,
    pub output_hidden_size: usize,
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
            conv_depthwise_kernel_size: 31,
            left_max_position_embeddings: 72,
            right_max_position_embeddings: 0,
            dropout: 0.1,
            sample_rate: 16_000,
            model_config: Map::new(),
            add_adapter: false,
            adapter_stride: 2,
            adapter_kernel_size: 3,
            output_hidden_size: 1024,
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
        if let Some(kernel_size) = self
            .model_config
            .get("conv_depthwise_kernel_size")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            self.conv_depthwise_kernel_size = kernel_size;
        }
        if let Some(left) = self
            .model_config
            .get("left_max_position_embeddings")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            self.left_max_position_embeddings = left;
        }
        if let Some(right) = self
            .model_config
            .get("right_max_position_embeddings")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            self.right_max_position_embeddings = right;
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
        if let Some(kernel_size) = self
            .model_config
            .get("adapter_kernel_size")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            self.adapter_kernel_size = kernel_size;
        }
        if let Some(output_hidden_size) = self
            .model_config
            .get("output_hidden_size")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        {
            self.output_hidden_size = output_hidden_size;
        } else {
            self.output_hidden_size = self.hidden_size;
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

    pub fn with_adapter_kernel_size(mut self, adapter_kernel_size: usize) -> Self {
        self.adapter_kernel_size = adapter_kernel_size.max(1);
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
            masked_spec_embed: Initializer::Uniform { min: 0.0, max: 1.0 }
                .init([self.hidden_size], device),
            encoder: Wav2VecBertEncoderConfig {
                hidden_size: self.hidden_size,
                intermediate_size: self.intermediate_size,
                num_attention_heads: self.num_attention_heads,
                num_hidden_layers: self.num_hidden_layers,
                conv_depthwise_kernel_size: self.conv_depthwise_kernel_size,
                left_max_position_embeddings: self.left_max_position_embeddings,
                right_max_position_embeddings: self.right_max_position_embeddings,
                dropout: self.dropout,
            }
            .init(device),
            adapter: self.add_adapter.then(|| {
                Wav2VecBertAdapterConfig {
                    hidden_size: self.hidden_size,
                    intermediate_size: self.intermediate_size,
                    num_attention_heads: self.num_attention_heads,
                    num_adapter_layers: self.num_adapter_layers,
                    adapter_stride: self.adapter_stride.max(1),
                    adapter_kernel_size: self.adapter_kernel_size.max(1),
                }
                .init(device)
            }),
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
    masked_spec_embed: Param<Tensor<B, 1>>,
    encoder: Wav2VecBertEncoder<B>,
    adapter: Option<Wav2VecBertAdapter<B>>,
}

impl<B: Wav2VecKernelBackend> Wav2VecBertModel<B> {
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
        hidden = B::mask_time(hidden, &lengths);
        let mut encoded = self.encoder.forward(hidden, &lengths, &device);
        let mut output_lengths = lengths;
        if let Some(adapter) = &self.adapter {
            (encoded, output_lengths) = adapter.forward(encoded, &output_lengths);
        }
        output_lengths = clamp_lengths(&output_lengths, encoded.dims()[1]);
        (B::mask_time(encoded, &output_lengths), output_lengths)
    }
}

#[derive(Clone, Debug)]
struct Wav2VecBertEncoderConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    conv_depthwise_kernel_size: usize,
    left_max_position_embeddings: usize,
    right_max_position_embeddings: usize,
    dropout: f64,
}

impl Wav2VecBertEncoderConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Wav2VecBertEncoder<B> {
        Wav2VecBertEncoder {
            layers: (0..self.num_hidden_layers)
                .map(|_| {
                    Wav2VecBertEncoderLayerConfig {
                        hidden_size: self.hidden_size,
                        intermediate_size: self.intermediate_size,
                        num_attention_heads: self.num_attention_heads,
                        conv_depthwise_kernel_size: self.conv_depthwise_kernel_size,
                        left_max_position_embeddings: self.left_max_position_embeddings,
                        right_max_position_embeddings: self.right_max_position_embeddings,
                        dropout: self.dropout,
                    }
                    .init(device)
                })
                .collect(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
struct Wav2VecBertEncoder<B: Backend> {
    layers: Vec<Wav2VecBertEncoderLayer<B>>,
    dropout: Dropout,
}

impl<B: Wav2VecKernelBackend> Wav2VecBertEncoder<B> {
    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        lengths: &[usize],
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let mut hidden_states = self.dropout.forward(B::mask_time(hidden_states, lengths));
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states, lengths, device);
        }
        B::mask_time(hidden_states, lengths)
    }
}

#[derive(Clone, Debug)]
struct Wav2VecBertEncoderLayerConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    conv_depthwise_kernel_size: usize,
    left_max_position_embeddings: usize,
    right_max_position_embeddings: usize,
    dropout: f64,
}

impl Wav2VecBertEncoderLayerConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Wav2VecBertEncoderLayer<B> {
        Wav2VecBertEncoderLayer {
            ffn1_layer_norm: LayerNormConfig::new(self.hidden_size).init(device),
            ffn1: Wav2VecBertFeedForwardConfig {
                hidden_size: self.hidden_size,
                intermediate_size: self.intermediate_size,
                dropout: self.dropout,
            }
            .init(device),
            self_attn_layer_norm: LayerNormConfig::new(self.hidden_size).init(device),
            self_attn_dropout: DropoutConfig::new(self.dropout).init(),
            self_attn: Wav2VecBertSelfAttentionConfig {
                hidden_size: self.hidden_size,
                num_attention_heads: self.num_attention_heads,
                left_max_position_embeddings: self.left_max_position_embeddings,
                right_max_position_embeddings: self.right_max_position_embeddings,
                use_relative_key: true,
                dropout: self.dropout,
            }
            .init(device),
            conv_module: Wav2VecBertConvolutionModuleConfig {
                hidden_size: self.hidden_size,
                kernel_size: self.conv_depthwise_kernel_size,
                dropout: self.dropout,
            }
            .init(device),
            ffn2_layer_norm: LayerNormConfig::new(self.hidden_size).init(device),
            ffn2: Wav2VecBertFeedForwardConfig {
                hidden_size: self.hidden_size,
                intermediate_size: self.intermediate_size,
                dropout: self.dropout,
            }
            .init(device),
            final_layer_norm: LayerNormConfig::new(self.hidden_size).init(device),
        }
    }
}

#[derive(Module, Debug)]
struct Wav2VecBertEncoderLayer<B: Backend> {
    ffn1_layer_norm: LayerNorm<B>,
    ffn1: Wav2VecBertFeedForward<B>,
    self_attn_layer_norm: LayerNorm<B>,
    self_attn_dropout: Dropout,
    self_attn: Wav2VecBertSelfAttention<B>,
    conv_module: Wav2VecBertConvolutionModule<B>,
    ffn2_layer_norm: LayerNorm<B>,
    ffn2: Wav2VecBertFeedForward<B>,
    final_layer_norm: LayerNorm<B>,
}

impl<B: Wav2VecKernelBackend> Wav2VecBertEncoderLayer<B> {
    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        lengths: &[usize],
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let residual = hidden_states.clone();
        let hidden_states = self
            .ffn1
            .forward(self.ffn1_layer_norm.forward(hidden_states))
            * 0.5
            + residual;

        let residual = hidden_states.clone();
        let hidden_states = self.self_attn.forward(
            self.self_attn_layer_norm.forward(hidden_states),
            lengths,
            device,
        );
        let hidden_states = self.self_attn_dropout.forward(hidden_states) + residual;

        let residual = hidden_states.clone();
        let hidden_states = self.conv_module.forward(hidden_states, lengths) + residual;

        let residual = hidden_states.clone();
        let hidden_states = self
            .ffn2
            .forward(self.ffn2_layer_norm.forward(hidden_states))
            * 0.5
            + residual;
        self.final_layer_norm.forward(hidden_states)
    }
}

#[derive(Clone, Debug)]
struct Wav2VecBertFeedForwardConfig {
    hidden_size: usize,
    intermediate_size: usize,
    dropout: f64,
}

impl Wav2VecBertFeedForwardConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Wav2VecBertFeedForward<B> {
        Wav2VecBertFeedForward {
            intermediate_dense: LinearConfig::new(self.hidden_size, self.intermediate_size)
                .init(device),
            intermediate_dropout: DropoutConfig::new(self.dropout).init(),
            output_dense: LinearConfig::new(self.intermediate_size, self.hidden_size).init(device),
            output_dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
struct Wav2VecBertFeedForward<B: Backend> {
    intermediate_dense: Linear<B>,
    intermediate_dropout: Dropout,
    output_dense: Linear<B>,
    output_dropout: Dropout,
}

impl<B: Backend> Wav2VecBertFeedForward<B> {
    fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden_states = gelu(self.intermediate_dense.forward(hidden_states));
        let hidden_states = self.intermediate_dropout.forward(hidden_states);
        self.output_dropout
            .forward(self.output_dense.forward(hidden_states))
    }
}

#[derive(Clone, Debug)]
struct Wav2VecBertSelfAttentionConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    left_max_position_embeddings: usize,
    right_max_position_embeddings: usize,
    use_relative_key: bool,
    dropout: f64,
}

impl Wav2VecBertSelfAttentionConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Wav2VecBertSelfAttention<B> {
        let head_size = self.hidden_size / self.num_attention_heads;
        let num_positions =
            self.left_max_position_embeddings + self.right_max_position_embeddings + 1;
        Wav2VecBertSelfAttention {
            linear_q: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            linear_k: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            linear_v: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            linear_out: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            distance_embedding: self
                .use_relative_key
                .then(|| EmbeddingConfig::new(num_positions, head_size).init(device)),
            dropout: DropoutConfig::new(self.dropout).init(),
            num_heads: self.num_attention_heads,
            head_size,
            left_max_position_embeddings: self.left_max_position_embeddings,
            right_max_position_embeddings: self.right_max_position_embeddings,
        }
    }
}

#[derive(Module, Debug)]
struct Wav2VecBertSelfAttention<B: Backend> {
    linear_q: Linear<B>,
    linear_k: Linear<B>,
    linear_v: Linear<B>,
    linear_out: Linear<B>,
    distance_embedding: Option<Embedding<B>>,
    dropout: Dropout,
    num_heads: usize,
    head_size: usize,
    left_max_position_embeddings: usize,
    right_max_position_embeddings: usize,
}

impl<B: Wav2VecKernelBackend> Wav2VecBertSelfAttention<B> {
    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        lengths: &[usize],
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, hidden_size] = hidden_states.dims();
        let query = self.project_heads(self.linear_q.forward(hidden_states.clone()));
        let key = self.project_heads(self.linear_k.forward(hidden_states.clone()));
        let value = self.project_heads(self.linear_v.forward(hidden_states));
        let mut scores = query.clone().matmul(key.swap_dims(2, 3)) / (self.head_size as f64).sqrt();

        if let Some(distance_embedding) = &self.distance_embedding {
            let positions = relative_key_positions::<B>(
                seq_len,
                self.left_max_position_embeddings,
                self.right_max_position_embeddings,
                device,
            );
            let positional =
                distance_embedding
                    .forward(positions)
                    .reshape([seq_len, seq_len, self.head_size]);
            let query_flat =
                query
                    .clone()
                    .reshape([batch_size * self.num_heads, seq_len, self.head_size]);
            let positional = positional
                .unsqueeze_dim::<4>(0)
                .repeat_dim(0, batch_size * self.num_heads);
            let relative_scores = query_flat
                .unsqueeze_dim::<4>(2)
                .matmul(positional.swap_dims(2, 3))
                .reshape([batch_size, self.num_heads, seq_len, seq_len]);
            scores = scores + relative_scores / (self.head_size as f64).sqrt();
        }

        let mask = B::sequence_mask(lengths, seq_len, device)
            .unsqueeze_dim::<3>(1)
            .unsqueeze_dim::<4>(2)
            .repeat_dim(1, self.num_heads)
            .repeat_dim(2, seq_len);
        let negative = Tensor::full(
            [batch_size, self.num_heads, seq_len, seq_len],
            -1.0e9,
            device,
        );
        scores = scores.mask_where(mask.bool_not(), negative);

        let attn = self.dropout.forward(softmax(scores, 3));
        self.linear_out
            .forward(
                attn.matmul(value)
                    .swap_dims(1, 2)
                    .reshape([batch_size, seq_len, hidden_size]),
            )
    }

    fn project_heads(&self, input: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch_size, seq_len, _] = input.dims();
        input
            .reshape([batch_size, seq_len, self.num_heads, self.head_size])
            .swap_dims(1, 2)
    }
}

#[derive(Clone, Debug)]
struct Wav2VecBertConvolutionModuleConfig {
    hidden_size: usize,
    kernel_size: usize,
    dropout: f64,
}

impl Wav2VecBertConvolutionModuleConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Wav2VecBertConvolutionModule<B> {
        Wav2VecBertConvolutionModule {
            layer_norm: LayerNormConfig::new(self.hidden_size).init(device),
            pointwise_conv1: Conv1dConfig::new(self.hidden_size, self.hidden_size * 2, 1)
                .with_bias(false)
                .init(device),
            depthwise_conv: Conv1dConfig::new(self.hidden_size, self.hidden_size, self.kernel_size)
                .with_groups(self.hidden_size)
                .with_bias(false)
                .init(device),
            depthwise_layer_norm: LayerNormConfig::new(self.hidden_size).init(device),
            pointwise_conv2: Conv1dConfig::new(self.hidden_size, self.hidden_size, 1)
                .with_bias(false)
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
struct Wav2VecBertConvolutionModule<B: Backend> {
    layer_norm: LayerNorm<B>,
    pointwise_conv1: Conv1d<B>,
    depthwise_conv: Conv1d<B>,
    depthwise_layer_norm: LayerNorm<B>,
    pointwise_conv2: Conv1d<B>,
    dropout: Dropout,
}

impl<B: Wav2VecKernelBackend> Wav2VecBertConvolutionModule<B> {
    fn forward(&self, hidden_states: Tensor<B, 3>, lengths: &[usize]) -> Tensor<B, 3> {
        let [batch_size, seq_len, hidden_size] = hidden_states.dims();
        let mut hidden_states = self
            .layer_norm
            .forward(B::mask_time(hidden_states, lengths));
        hidden_states = hidden_states.swap_dims(1, 2);
        hidden_states = B::glu_channel_dim(self.pointwise_conv1.forward(hidden_states));
        let pad_left = self.depthwise_conv.kernel_size.saturating_sub(1);
        if pad_left > 0 {
            hidden_states = hidden_states.pad([(0, 0), (pad_left, 0)], PadMode::Constant(0.0));
        }
        hidden_states = self.depthwise_conv.forward(hidden_states);
        hidden_states = self
            .depthwise_layer_norm
            .forward(hidden_states.swap_dims(1, 2))
            .swap_dims(1, 2);
        hidden_states = gelu(hidden_states);
        hidden_states = self.pointwise_conv2.forward(hidden_states);
        hidden_states = self.dropout.forward(hidden_states).swap_dims(1, 2);
        B::mask_time(
            hidden_states.reshape([batch_size, seq_len, hidden_size]),
            lengths,
        )
    }
}

#[derive(Clone, Debug)]
struct Wav2VecBertAdapterConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_adapter_layers: usize,
    adapter_stride: usize,
    adapter_kernel_size: usize,
}

impl Wav2VecBertAdapterConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Wav2VecBertAdapter<B> {
        Wav2VecBertAdapter {
            layers: (0..self.num_adapter_layers)
                .map(|_| {
                    Wav2VecBertAdapterLayerConfig {
                        hidden_size: self.hidden_size,
                        intermediate_size: self.intermediate_size,
                        num_attention_heads: self.num_attention_heads,
                        stride: self.adapter_stride,
                        kernel_size: self.adapter_kernel_size,
                    }
                    .init(device)
                })
                .collect(),
            stride: self.adapter_stride,
            kernel_size: self.adapter_kernel_size,
        }
    }
}

#[derive(Module, Debug)]
struct Wav2VecBertAdapter<B: Backend> {
    layers: Vec<Wav2VecBertAdapterLayer<B>>,
    stride: usize,
    kernel_size: usize,
}

impl<B: Wav2VecKernelBackend> Wav2VecBertAdapter<B> {
    fn forward(
        &self,
        mut hidden_states: Tensor<B, 3>,
        lengths: &[usize],
    ) -> (Tensor<B, 3>, Vec<usize>) {
        let mut output_lengths = lengths.to_vec();
        for layer in &self.layers {
            output_lengths = conv1d_output_lengths(
                &output_lengths,
                self.kernel_size,
                self.stride,
                self.stride / 2,
            );
            hidden_states = layer.forward(hidden_states, &output_lengths);
        }
        (hidden_states, output_lengths)
    }
}

#[derive(Clone, Debug)]
struct Wav2VecBertAdapterLayerConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    stride: usize,
    kernel_size: usize,
}

impl Wav2VecBertAdapterLayerConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Wav2VecBertAdapterLayer<B> {
        let padding = PaddingConfig1d::Explicit(self.stride / 2, self.stride / 2);
        Wav2VecBertAdapterLayer {
            residual_layer_norm: LayerNormConfig::new(self.hidden_size).init(device),
            residual_conv: Conv1dConfig::new(
                self.hidden_size,
                self.hidden_size * 2,
                self.kernel_size,
            )
            .with_stride(self.stride)
            .with_padding(padding.clone())
            .init(device),
            self_attn_layer_norm: LayerNormConfig::new(self.hidden_size).init(device),
            self_attn_conv: Conv1dConfig::new(
                self.hidden_size,
                self.hidden_size * 2,
                self.kernel_size,
            )
            .with_stride(self.stride)
            .with_padding(padding)
            .init(device),
            self_attn: Wav2VecBertSelfAttentionConfig {
                hidden_size: self.hidden_size,
                num_attention_heads: self.num_attention_heads,
                left_max_position_embeddings: 0,
                right_max_position_embeddings: 0,
                use_relative_key: false,
                dropout: 0.0,
            }
            .init(device),
            self_attn_dropout: DropoutConfig::new(0.0).init(),
            ffn_layer_norm: LayerNormConfig::new(self.hidden_size).init(device),
            ffn: Wav2VecBertFeedForwardConfig {
                hidden_size: self.hidden_size,
                intermediate_size: self.intermediate_size,
                dropout: 0.0,
            }
            .init(device),
        }
    }
}

#[derive(Module, Debug)]
struct Wav2VecBertAdapterLayer<B: Backend> {
    residual_layer_norm: LayerNorm<B>,
    residual_conv: Conv1d<B>,
    self_attn_layer_norm: LayerNorm<B>,
    self_attn_conv: Conv1d<B>,
    self_attn: Wav2VecBertSelfAttention<B>,
    self_attn_dropout: Dropout,
    ffn_layer_norm: LayerNorm<B>,
    ffn: Wav2VecBertFeedForward<B>,
}

impl<B: Wav2VecKernelBackend> Wav2VecBertAdapterLayer<B> {
    fn forward(&self, hidden_states: Tensor<B, 3>, output_lengths: &[usize]) -> Tensor<B, 3> {
        let device = hidden_states.device();
        let residual = self.residual_conv.forward(
            self.residual_layer_norm
                .forward(hidden_states.clone())
                .swap_dims(1, 2),
        );
        let residual = B::glu_channel_dim(residual).swap_dims(1, 2);

        let hidden_states = self.self_attn_conv.forward(
            self.self_attn_layer_norm
                .forward(hidden_states)
                .swap_dims(1, 2),
        );
        let hidden_states = B::glu_channel_dim(hidden_states).swap_dims(1, 2);
        let hidden_states = self.self_attn_dropout.forward(self.self_attn.forward(
            hidden_states,
            output_lengths,
            &device,
        )) + residual;

        let residual = hidden_states.clone();
        self.ffn.forward(self.ffn_layer_norm.forward(hidden_states)) + residual
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

    pub fn init_from_huggingface_dir<B: Wav2VecKernelBackend>(
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

impl<B: Wav2VecKernelBackend> Wav2VecBertCtc<B> {
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
        let masked_spec_dim = self.encoder.masked_spec_embed.dims()[0];
        load_param_1d(
            &mut self.encoder.masked_spec_embed,
            &store,
            &str_prefixes(&[
                "masked_spec_embed",
                "wav2vec2_bert.masked_spec_embed",
                "model.masked_spec_embed",
            ]),
            "",
            masked_spec_dim,
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
            load_wav2vec_encoder_layer(layer, &store, index, device, &mut report);
        }
        if let Some(adapter) = self.encoder.adapter.as_mut() {
            for (index, layer) in adapter.layers.iter_mut().enumerate() {
                load_wav2vec_adapter_layer(layer, &store, index, device, &mut report);
            }
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
            let name = if suffix.is_empty() {
                prefix.to_string()
            } else {
                format!("{prefix}.{suffix}")
            };
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

fn load_wav2vec_encoder_layer<B: Backend>(
    layer: &mut Wav2VecBertEncoderLayer<B>,
    store: &HfSafetensors,
    index: usize,
    device: &B::Device,
    report: &mut Wav2VecBertImportReport,
) {
    let prefixes = layer_prefixes(index);
    load_layer_norm(
        &mut layer.ffn1_layer_norm,
        store,
        &prefix_field(&prefixes, "ffn1_layer_norm"),
        device,
        report,
    );
    load_feed_forward(
        &mut layer.ffn1,
        store,
        &prefix_field(&prefixes, "ffn1"),
        device,
        report,
    );
    load_layer_norm(
        &mut layer.self_attn_layer_norm,
        store,
        &prefix_field(&prefixes, "self_attn_layer_norm"),
        device,
        report,
    );
    load_self_attention(
        &mut layer.self_attn,
        store,
        &prefix_field(&prefixes, "self_attn"),
        device,
        report,
    );
    load_layer_norm(
        &mut layer.conv_module.layer_norm,
        store,
        &prefix_field(&prefixes, "conv_module.layer_norm"),
        device,
        report,
    );
    load_conv1d(
        &mut layer.conv_module.pointwise_conv1,
        store,
        &prefix_field(&prefixes, "conv_module.pointwise_conv1"),
        device,
        report,
    );
    load_conv1d(
        &mut layer.conv_module.depthwise_conv,
        store,
        &prefix_field(&prefixes, "conv_module.depthwise_conv"),
        device,
        report,
    );
    load_layer_norm(
        &mut layer.conv_module.depthwise_layer_norm,
        store,
        &prefix_field(&prefixes, "conv_module.depthwise_layer_norm"),
        device,
        report,
    );
    load_conv1d(
        &mut layer.conv_module.pointwise_conv2,
        store,
        &prefix_field(&prefixes, "conv_module.pointwise_conv2"),
        device,
        report,
    );
    load_layer_norm(
        &mut layer.ffn2_layer_norm,
        store,
        &prefix_field(&prefixes, "ffn2_layer_norm"),
        device,
        report,
    );
    load_feed_forward(
        &mut layer.ffn2,
        store,
        &prefix_field(&prefixes, "ffn2"),
        device,
        report,
    );
    load_layer_norm(
        &mut layer.final_layer_norm,
        store,
        &prefix_field(&prefixes, "final_layer_norm"),
        device,
        report,
    );
}

fn load_wav2vec_adapter_layer<B: Backend>(
    layer: &mut Wav2VecBertAdapterLayer<B>,
    store: &HfSafetensors,
    index: usize,
    device: &B::Device,
    report: &mut Wav2VecBertImportReport,
) {
    let prefixes = [
        format!("adapter.layers.{index}"),
        format!("wav2vec2_bert.adapter.layers.{index}"),
        format!("model.adapter.layers.{index}"),
    ];
    load_layer_norm(
        &mut layer.residual_layer_norm,
        store,
        &prefix_field(&prefixes, "residual_layer_norm"),
        device,
        report,
    );
    load_conv1d(
        &mut layer.residual_conv,
        store,
        &prefix_field(&prefixes, "residual_conv"),
        device,
        report,
    );
    load_layer_norm(
        &mut layer.self_attn_layer_norm,
        store,
        &prefix_field(&prefixes, "self_attn_layer_norm"),
        device,
        report,
    );
    load_conv1d(
        &mut layer.self_attn_conv,
        store,
        &prefix_field(&prefixes, "self_attn_conv"),
        device,
        report,
    );
    load_self_attention(
        &mut layer.self_attn,
        store,
        &prefix_field(&prefixes, "self_attn"),
        device,
        report,
    );
    load_layer_norm(
        &mut layer.ffn_layer_norm,
        store,
        &prefix_field(&prefixes, "ffn_layer_norm"),
        device,
        report,
    );
    load_feed_forward(
        &mut layer.ffn,
        store,
        &prefix_field(&prefixes, "ffn"),
        device,
        report,
    );
}

fn load_self_attention<B: Backend>(
    attention: &mut Wav2VecBertSelfAttention<B>,
    store: &HfSafetensors,
    prefixes: &[String],
    device: &B::Device,
    report: &mut Wav2VecBertImportReport,
) {
    load_linear(
        &mut attention.linear_q,
        store,
        &prefix_field(prefixes, "linear_q"),
        true,
        device,
        report,
    );
    load_linear(
        &mut attention.linear_k,
        store,
        &prefix_field(prefixes, "linear_k"),
        true,
        device,
        report,
    );
    load_linear(
        &mut attention.linear_v,
        store,
        &prefix_field(prefixes, "linear_v"),
        true,
        device,
        report,
    );
    load_linear(
        &mut attention.linear_out,
        store,
        &prefix_field(prefixes, "linear_out"),
        true,
        device,
        report,
    );
    if let Some(distance_embedding) = attention.distance_embedding.as_mut() {
        let [positions, head_size] = distance_embedding.weight.dims();
        load_param_2d(
            &mut distance_embedding.weight,
            store,
            &prefix_field(prefixes, "distance_embedding"),
            "weight",
            [positions, head_size],
            false,
            device,
            report,
        );
    }
}

fn load_feed_forward<B: Backend>(
    ffn: &mut Wav2VecBertFeedForward<B>,
    store: &HfSafetensors,
    prefixes: &[String],
    device: &B::Device,
    report: &mut Wav2VecBertImportReport,
) {
    load_linear(
        &mut ffn.intermediate_dense,
        store,
        &prefix_field(prefixes, "intermediate_dense"),
        true,
        device,
        report,
    );
    load_linear(
        &mut ffn.output_dense,
        store,
        &prefix_field(prefixes, "output_dense"),
        true,
        device,
        report,
    );
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
    prefixes: &[impl AsRef<str>],
    device: &B::Device,
    report: &mut Wav2VecBertImportReport,
) {
    let prefixes = prefixes
        .iter()
        .map(|value| value.as_ref().to_string())
        .collect::<Vec<_>>();
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

fn glu_fallback<B: Backend>(input: Tensor<B, 3>, dim: usize) -> Tensor<B, 3> {
    let mut chunks = input.chunk(2, dim);
    let gate = chunks.remove(1);
    let value = chunks.remove(0);
    value * sigmoid(gate)
}

fn relative_key_positions<B: Backend>(
    seq_len: usize,
    left_max: usize,
    right_max: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let mut values = Vec::with_capacity(seq_len * seq_len);
    for left in 0..seq_len {
        for right in 0..seq_len {
            let distance = right as isize - left as isize;
            let distance = distance.clamp(-(left_max as isize), right_max as isize);
            values.push((distance + left_max as isize) as i64);
        }
    }
    Tensor::from_data(TensorData::new(values, [seq_len, seq_len]), device)
}

fn conv1d_output_lengths(
    lengths: &[usize],
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Vec<usize> {
    lengths
        .iter()
        .map(|length| ((*length + 2 * padding).saturating_sub(kernel_size) / stride + 1).max(1))
        .collect()
}

fn clamp_lengths(lengths: &[usize], max_len: usize) -> Vec<usize> {
    lengths
        .iter()
        .map(|length| (*length).clamp(1, max_len))
        .collect()
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

fn sequence_mask_fallback<B: Backend>(
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

fn mask_time_fallback<B: Backend>(input: Tensor<B, 3>, lengths: &[usize]) -> Tensor<B, 3> {
    let seq_len = input.dims()[1];
    let mask = sequence_mask_fallback::<B>(lengths, seq_len, &input.device())
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
            std::env::temp_dir().join(format!("rust_asr_hf_config_{}", std::process::id()));
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
            std::env::temp_dir().join(format!("rust_asr_hf_weights_{}", std::process::id()));
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
