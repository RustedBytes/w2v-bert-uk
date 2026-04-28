use burn::module::Module;
#[cfg(feature = "asr-cubecl-kernels")]
use burn::tensor::TensorPrimitive;
use burn::tensor::activation::{log_softmax, relu, sigmoid, silu, softmax};
#[cfg(feature = "asr-cubecl-kernels")]
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};
use burn_nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn_nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn_nn::transformer::{TransformerDecoder, TransformerDecoderConfig, TransformerDecoderInput};
use burn_nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
    LinearConfig, PaddingConfig1d, PaddingConfig2d,
};

use crate::squeezeformer::PositiveLossBatchNorm1d;

pub trait ParaformerKernelBackend: Backend + Sized {
    fn sequence_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 2, Bool> {
        sequence_mask_fallback(lengths, max_len, device)
    }

    fn padding_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 2, Bool> {
        padding_mask_fallback(lengths, max_len, device)
    }

    fn glu_channel_dim(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        glu_channel_dim_fallback(input)
    }
}

impl ParaformerKernelBackend for burn_ndarray::NdArray<f32> {}

impl<C> ParaformerKernelBackend for burn_autodiff::Autodiff<burn_ndarray::NdArray<f32>, C> where
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy
{
}

#[cfg(feature = "asr-cubecl-kernels")]
fn autodiff_to_inner<B, C, const D: usize>(
    tensor: Tensor<burn_autodiff::Autodiff<B, C>, D>,
) -> Tensor<B, D>
where
    B: Backend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    let tensor =
        <burn_autodiff::Autodiff<B, C> as AutodiffBackend>::inner(tensor.into_primitive().tensor());
    Tensor::from_primitive(TensorPrimitive::Float(tensor))
}

#[cfg(feature = "asr-cubecl-kernels")]
fn inner_to_autodiff<B, C, const D: usize>(
    tensor: Tensor<B, D>,
) -> Tensor<burn_autodiff::Autodiff<B, C>, D>
where
    B: Backend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    let tensor = <burn_autodiff::Autodiff<B, C> as AutodiffBackend>::from_inner(
        tensor.into_primitive().tensor(),
    );
    Tensor::from_primitive(TensorPrimitive::Float(tensor))
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

#[cfg(feature = "asr-cubecl-kernels")]
fn attach_autodiff_gradient<B, C, const D: usize>(
    raw: Tensor<burn_autodiff::Autodiff<B, C>, D>,
    portable: Tensor<burn_autodiff::Autodiff<B, C>, D>,
) -> Tensor<burn_autodiff::Autodiff<B, C>, D>
where
    B: Backend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    raw + portable.clone() - portable.detach()
}

#[cfg(all(feature = "asr-cubecl-kernels", feature = "burn-cuda-backend"))]
impl<F, I> ParaformerKernelBackend for burn_cuda::Cuda<F, I>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
{
    fn sequence_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 2, Bool> {
        crate::cubecl_kernels::sequence_mask(lengths, max_len, device)
    }

    fn padding_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 2, Bool> {
        Self::sequence_mask(lengths, max_len, device).bool_not()
    }

    fn glu_channel_dim(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        crate::cubecl_kernels::glu_channel_dim(input)
    }
}

#[cfg(all(feature = "asr-cubecl-kernels", feature = "burn-cuda-backend"))]
impl<F, I, C> ParaformerKernelBackend for burn_autodiff::Autodiff<burn_cuda::Cuda<F, I>, C>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    fn sequence_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 2, Bool> {
        let raw = crate::cubecl_kernels::sequence_mask::<_, F, I, u8>(lengths, max_len, device);
        inner_bool_to_autodiff(raw)
    }

    fn padding_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 2, Bool> {
        Self::sequence_mask(lengths, max_len, device).bool_not()
    }

    fn glu_channel_dim(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        let portable = glu_channel_dim_fallback(input.clone());
        let raw = crate::cubecl_kernels::glu_channel_dim(autodiff_to_inner(input));
        attach_autodiff_gradient(inner_to_autodiff(raw), portable)
    }
}

#[cfg(all(feature = "asr-cubecl-kernels", feature = "burn-wgpu-backend"))]
impl<F, I, BT> ParaformerKernelBackend for burn_wgpu::Wgpu<F, I, BT>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
{
    fn sequence_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 2, Bool> {
        crate::cubecl_kernels::sequence_mask(lengths, max_len, device)
    }

    fn padding_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 2, Bool> {
        Self::sequence_mask(lengths, max_len, device).bool_not()
    }

    fn glu_channel_dim(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        crate::cubecl_kernels::glu_channel_dim(input)
    }
}

#[cfg(all(feature = "asr-cubecl-kernels", feature = "burn-wgpu-backend"))]
impl<F, I, BT, C> ParaformerKernelBackend for burn_autodiff::Autodiff<burn_wgpu::Wgpu<F, I, BT>, C>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    fn sequence_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 2, Bool> {
        let raw = crate::cubecl_kernels::sequence_mask::<_, F, I, BT>(lengths, max_len, device);
        inner_bool_to_autodiff(raw)
    }

    fn padding_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 2, Bool> {
        Self::sequence_mask(lengths, max_len, device).bool_not()
    }

    fn glu_channel_dim(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        let portable = glu_channel_dim_fallback(input.clone());
        let raw = crate::cubecl_kernels::glu_channel_dim(autodiff_to_inner(input));
        attach_autodiff_gradient(inner_to_autodiff(raw), portable)
    }
}

#[cfg(all(feature = "burn-cuda-backend", not(feature = "asr-cubecl-kernels")))]
impl<F, I> ParaformerKernelBackend for burn_cuda::Cuda<F, I>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    burn_cuda::Cuda<F, I>: Backend,
{
}

#[cfg(all(feature = "burn-cuda-backend", not(feature = "asr-cubecl-kernels")))]
impl<F, I, C> ParaformerKernelBackend for burn_autodiff::Autodiff<burn_cuda::Cuda<F, I>, C>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    burn_cuda::Cuda<F, I>: Backend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
}

#[cfg(all(feature = "burn-wgpu-backend", not(feature = "asr-cubecl-kernels")))]
impl<F, I, BT> ParaformerKernelBackend for burn_wgpu::Wgpu<F, I, BT>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
    burn_wgpu::Wgpu<F, I, BT>: Backend,
{
}

#[cfg(all(feature = "burn-wgpu-backend", not(feature = "asr-cubecl-kernels")))]
impl<F, I, BT, C> ParaformerKernelBackend for burn_autodiff::Autodiff<burn_wgpu::Wgpu<F, I, BT>, C>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
    burn_wgpu::Wgpu<F, I, BT>: Backend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
}

#[derive(Clone, Debug)]
pub struct ParaformerV2Config {
    pub input_dim: usize,
    pub vocab_size: usize,
    pub encoder_dim: usize,
    pub decoder_dim: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_ff_dim: usize,
    pub decoder_ff_dim: usize,
    pub attention_heads: usize,
    pub conv_kernel_size: usize,
    pub dropout: f64,
    pub blank_id: Option<usize>,
}

impl ParaformerV2Config {
    pub fn new(input_dim: usize, vocab_size: usize) -> Self {
        Self {
            input_dim,
            vocab_size,
            encoder_dim: 256,
            decoder_dim: 256,
            encoder_layers: 12,
            decoder_layers: 6,
            encoder_ff_dim: 2048,
            decoder_ff_dim: 2048,
            attention_heads: 4,
            conv_kernel_size: 15,
            dropout: 0.1,
            blank_id: None,
        }
    }

    pub fn variant(
        name: &str,
        input_dim: usize,
        vocab_size: usize,
        blank_id: usize,
    ) -> Option<Self> {
        let mut config = Self::new(input_dim, vocab_size).with_blank_id(blank_id);
        match name {
            "xs" | "s" | "small" => {}
            "sm" | "m" | "medium" => {
                config.encoder_dim = 384;
                config.decoder_dim = 384;
                config.attention_heads = 6;
            }
            "ml" | "l" | "large" => {
                config.encoder_dim = 512;
                config.decoder_dim = 512;
                config.attention_heads = 8;
            }
            _ => return None,
        }
        Some(config)
    }

    pub fn ctc_vocab_size(&self) -> usize {
        if self
            .blank_id
            .is_some_and(|blank_id| blank_id < self.vocab_size)
        {
            self.vocab_size
        } else {
            self.vocab_size + 1
        }
    }

    pub fn resolved_blank_id(&self) -> usize {
        self.blank_id.unwrap_or(self.vocab_size)
    }

    pub fn with_blank_id(mut self, blank_id: usize) -> Self {
        self.blank_id = Some(blank_id);
        self
    }

    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> ParaformerV2<B> {
        ParaformerV2 {
            encoder: ConformerEncoderConfig {
                input_dim: self.input_dim,
                encoder_dim: self.encoder_dim,
                encoder_layers: self.encoder_layers,
                encoder_ff_dim: self.encoder_ff_dim,
                attention_heads: self.attention_heads,
                conv_kernel_size: self.conv_kernel_size,
                dropout: self.dropout,
            }
            .init(device),
            ctc_projection: LinearConfig::new(self.encoder_dim, self.ctc_vocab_size()).init(device),
            posterior_embed: LinearConfig::new(self.ctc_vocab_size(), self.decoder_dim)
                .with_bias(false)
                .init(device),
            memory_projection: if self.encoder_dim == self.decoder_dim {
                None
            } else {
                Some(LinearConfig::new(self.encoder_dim, self.decoder_dim).init(device))
            },
            decoder: TransformerDecoderConfig::new(
                self.decoder_dim,
                self.decoder_ff_dim,
                self.attention_heads,
                self.decoder_layers,
            )
            .with_dropout(self.dropout)
            .with_norm_first(true)
            .init(device),
            decoder_projection: LinearConfig::new(self.decoder_dim, self.vocab_size).init(device),
            blank_id: self.resolved_blank_id(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EnhancedParaformerV2Config {
    pub base: ParaformerV2Config,
    pub shallow_ctc_loss_weight: f64,
    pub boundary_loss_weight: f64,
    pub refinement_loss_weight: f64,
    pub confidence_threshold: f64,
    pub low_confidence_threshold: f64,
}

impl EnhancedParaformerV2Config {
    pub fn new(input_dim: usize, vocab_size: usize) -> Self {
        Self {
            base: ParaformerV2Config::new(input_dim, vocab_size),
            shallow_ctc_loss_weight: 0.3,
            boundary_loss_weight: 0.1,
            refinement_loss_weight: 0.3,
            confidence_threshold: 0.55,
            low_confidence_threshold: 0.7,
        }
    }

    pub fn variant(
        name: &str,
        input_dim: usize,
        vocab_size: usize,
        blank_id: usize,
    ) -> Option<Self> {
        Some(Self {
            base: ParaformerV2Config::variant(name, input_dim, vocab_size, blank_id)?,
            ..Self::new(input_dim, vocab_size)
        })
    }

    pub fn with_blank_id(mut self, blank_id: usize) -> Self {
        self.base = self.base.with_blank_id(blank_id);
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> EnhancedParaformerV2<B> {
        let base = &self.base;
        EnhancedParaformerV2 {
            encoder: MultiResolutionConformerEncoderConfig {
                input_dim: base.input_dim,
                encoder_dim: base.encoder_dim,
                encoder_layers: base.encoder_layers,
                encoder_ff_dim: base.encoder_ff_dim,
                attention_heads: base.attention_heads,
                conv_kernel_size: base.conv_kernel_size,
                dropout: base.dropout,
            }
            .init(device),
            shallow_ctc_projection: LinearConfig::new(base.encoder_dim, base.ctc_vocab_size())
                .init(device),
            final_ctc_projection: LinearConfig::new(base.encoder_dim, base.ctc_vocab_size())
                .init(device),
            boundary_in: LinearConfig::new(base.encoder_dim * 2, base.encoder_dim).init(device),
            boundary_out: LinearConfig::new(base.encoder_dim, 1).init(device),
            query_projection: LinearConfig::new(base.ctc_vocab_size() * 2 + 4, base.decoder_dim)
                .init(device),
            query_norm: LayerNormConfig::new(base.decoder_dim).init(device),
            memory_projection: if base.encoder_dim == base.decoder_dim {
                None
            } else {
                Some(LinearConfig::new(base.encoder_dim, base.decoder_dim).init(device))
            },
            decoder: TransformerDecoderConfig::new(
                base.decoder_dim,
                base.decoder_ff_dim,
                base.attention_heads,
                base.decoder_layers,
            )
            .with_dropout(base.dropout)
            .with_norm_first(true)
            .init(device),
            decoder_projection: LinearConfig::new(base.decoder_dim, base.vocab_size).init(device),
            token_embedding: EmbeddingConfig::new(base.vocab_size, base.decoder_dim).init(device),
            refinement_decoder: TransformerDecoderConfig::new(
                base.decoder_dim,
                base.decoder_ff_dim,
                base.attention_heads,
                1,
            )
            .with_dropout(base.dropout)
            .with_norm_first(true)
            .init(device),
            refinement_projection: LinearConfig::new(base.decoder_dim, base.vocab_size)
                .init(device),
            blank_id: base.resolved_blank_id(),
            shallow_ctc_loss_weight: self.shallow_ctc_loss_weight,
            boundary_loss_weight: self.boundary_loss_weight,
            refinement_loss_weight: self.refinement_loss_weight,
            confidence_threshold: self.confidence_threshold,
            low_confidence_threshold: self.low_confidence_threshold,
        }
    }
}

#[derive(Clone, Debug)]
struct ConformerEncoderConfig {
    input_dim: usize,
    encoder_dim: usize,
    encoder_layers: usize,
    encoder_ff_dim: usize,
    attention_heads: usize,
    conv_kernel_size: usize,
    dropout: f64,
}

impl ConformerEncoderConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ConformerEncoder<B> {
        ConformerEncoder {
            subsampling: ConvSubsamplingConfig::new(self.input_dim, self.encoder_dim).init(device),
            layers: (0..self.encoder_layers)
                .map(|_| {
                    ConformerBlockConfig {
                        dim: self.encoder_dim,
                        ff_dim: self.encoder_ff_dim,
                        heads: self.attention_heads,
                        kernel_size: self.conv_kernel_size,
                        dropout: self.dropout,
                    }
                    .init(device)
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
struct ConvSubsamplingConfig {
    input_dim: usize,
    output_dim: usize,
}

impl ConvSubsamplingConfig {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> ConvSubsampling<B> {
        ConvSubsampling {
            conv1: Conv2dConfig::new([1, self.output_dim], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv2: Conv2dConfig::new([self.output_dim, self.output_dim], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            projection: LinearConfig::new(
                self.output_dim * ((self.input_dim + 3) / 4),
                self.output_dim,
            )
            .init(device),
        }
    }
}

#[derive(Module, Debug)]
struct ConvSubsampling<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    projection: Linear<B>,
}

impl<B: Backend> ConvSubsampling<B> {
    fn forward(&self, features: Tensor<B, 3>, lengths: Vec<usize>) -> (Tensor<B, 3>, Vec<usize>) {
        let output = relu(self.conv1.forward(features.unsqueeze_dim::<4>(1)));
        let output = relu(self.conv2.forward(output));
        let [batch_size, channels, time, freq] = output.dims();
        let output = output
            .swap_dims(1, 2)
            .reshape([batch_size, time, channels * freq]);
        let lengths = lengths
            .into_iter()
            .map(|length| ((length + 3) / 4).min(time))
            .collect();
        (self.projection.forward(output), lengths)
    }
}

#[derive(Clone, Debug)]
struct FeedForwardConfig {
    dim: usize,
    hidden_dim: usize,
    dropout: f64,
}

impl FeedForwardConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        FeedForward {
            norm: LayerNormConfig::new(self.dim).init(device),
            linear_in: LinearConfig::new(self.dim, self.hidden_dim).init(device),
            linear_out: LinearConfig::new(self.hidden_dim, self.dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
struct FeedForward<B: Backend> {
    norm: LayerNorm<B>,
    linear_in: Linear<B>,
    linear_out: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> FeedForward<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let output = self.norm.forward(input);
        let output = silu(self.linear_in.forward(output));
        let output = self.dropout.forward(output);
        self.dropout.forward(self.linear_out.forward(output))
    }
}

#[derive(Clone, Debug)]
struct ConformerBlockConfig {
    dim: usize,
    ff_dim: usize,
    heads: usize,
    kernel_size: usize,
    dropout: f64,
}

impl ConformerBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ConformerBlock<B> {
        ConformerBlock {
            ff1: FeedForwardConfig {
                dim: self.dim,
                hidden_dim: self.ff_dim,
                dropout: self.dropout,
            }
            .init(device),
            self_attn_norm: LayerNormConfig::new(self.dim).init(device),
            self_attn: MultiHeadAttentionConfig::new(self.dim, self.heads)
                .with_dropout(self.dropout)
                .init(device),
            self_attn_dropout: DropoutConfig::new(self.dropout).init(),
            conv_norm: LayerNormConfig::new(self.dim).init(device),
            conv_in: Conv1dConfig::new(self.dim, self.dim * 2, 1).init(device),
            depthwise: Conv1dConfig::new(self.dim, self.dim, self.kernel_size)
                .with_groups(self.dim)
                .with_padding(PaddingConfig1d::Explicit(
                    self.kernel_size / 2,
                    self.kernel_size / 2,
                ))
                .init(device),
            batch_norm: PositiveLossBatchNorm1d::new(self.dim, device),
            conv_out: Conv1dConfig::new(self.dim, self.dim, 1).init(device),
            conv_dropout: DropoutConfig::new(self.dropout).init(),
            ff2: FeedForwardConfig {
                dim: self.dim,
                hidden_dim: self.ff_dim,
                dropout: self.dropout,
            }
            .init(device),
            final_norm: LayerNormConfig::new(self.dim).init(device),
        }
    }
}

#[derive(Module, Debug)]
struct ConformerBlock<B: Backend> {
    ff1: FeedForward<B>,
    self_attn_norm: LayerNorm<B>,
    self_attn: MultiHeadAttention<B>,
    self_attn_dropout: Dropout,
    conv_norm: LayerNorm<B>,
    conv_in: Conv1d<B>,
    depthwise: Conv1d<B>,
    batch_norm: PositiveLossBatchNorm1d<B>,
    conv_out: Conv1d<B>,
    conv_dropout: Dropout,
    ff2: FeedForward<B>,
    final_norm: LayerNorm<B>,
}

impl<B: ParaformerKernelBackend> ConformerBlock<B> {
    fn forward(&self, input: Tensor<B, 3>, key_padding_mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let output = input.clone() + self.ff1.forward(input) * 0.5;

        let attn_input = self.self_attn_norm.forward(output.clone());
        let attn = self
            .self_attn
            .forward(MhaInput::self_attn(attn_input).mask_pad(key_padding_mask));
        let output = output + self.self_attn_dropout.forward(attn.context);

        let [batch_size, seq_len, dim] = output.dims();
        let conv_input = self.conv_norm.forward(output.clone()).swap_dims(1, 2);
        let conv = B::glu_channel_dim(self.conv_in.forward(conv_input));
        let conv = silu(self.batch_norm.forward(self.depthwise.forward(conv)));
        let conv = self.conv_dropout.forward(self.conv_out.forward(conv));
        let output = output + conv.swap_dims(1, 2).reshape([batch_size, seq_len, dim]);

        let residual = output.clone();
        self.final_norm
            .forward(output + self.ff2.forward(residual) * 0.5)
    }
}

#[derive(Module, Debug)]
struct ConformerEncoder<B: Backend> {
    subsampling: ConvSubsampling<B>,
    layers: Vec<ConformerBlock<B>>,
}

impl<B: ParaformerKernelBackend> ConformerEncoder<B> {
    fn forward(&self, features: Tensor<B, 3>, lengths: Vec<usize>) -> (Tensor<B, 3>, Vec<usize>) {
        let device = features.device();
        let (mut output, lengths) = self.subsampling.forward(features, lengths);
        let mask = B::padding_mask(&lengths, output.dims()[1], &device);
        for layer in self.layers.iter() {
            output = layer.forward(output, mask.clone());
        }
        (output, lengths)
    }
}

#[derive(Clone, Debug)]
struct MultiResolutionConformerEncoderConfig {
    input_dim: usize,
    encoder_dim: usize,
    encoder_layers: usize,
    encoder_ff_dim: usize,
    attention_heads: usize,
    conv_kernel_size: usize,
    dropout: f64,
}

impl MultiResolutionConformerEncoderConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> MultiResolutionConformerEncoder<B> {
        MultiResolutionConformerEncoder {
            subsampling: ConvSubsamplingConfig::new(self.input_dim, self.encoder_dim).init(device),
            layers: (0..self.encoder_layers)
                .map(|_| {
                    ConformerBlockConfig {
                        dim: self.encoder_dim,
                        ff_dim: self.encoder_ff_dim,
                        heads: self.attention_heads,
                        kernel_size: self.conv_kernel_size,
                        dropout: self.dropout,
                    }
                    .init(device)
                })
                .collect(),
            shallow_index: self.encoder_layers.saturating_div(2).saturating_sub(1),
        }
    }
}

#[derive(Module, Debug)]
struct MultiResolutionConformerEncoder<B: Backend> {
    subsampling: ConvSubsampling<B>,
    layers: Vec<ConformerBlock<B>>,
    shallow_index: usize,
}

impl<B: ParaformerKernelBackend> MultiResolutionConformerEncoder<B> {
    fn forward(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Vec<usize>) {
        let device = features.device();
        let (mut output, lengths) = self.subsampling.forward(features, lengths);
        let mask = B::padding_mask(&lengths, output.dims()[1], &device);
        let mut shallow = None;
        for (index, layer) in self.layers.iter().enumerate() {
            output = layer.forward(output, mask.clone());
            if index == self.shallow_index {
                shallow = Some(output.clone());
            }
        }
        (shallow.unwrap_or_else(|| output.clone()), output, lengths)
    }
}

#[derive(Module, Debug)]
pub struct ParaformerV2<B: Backend> {
    encoder: ConformerEncoder<B>,
    ctc_projection: Linear<B>,
    posterior_embed: Linear<B>,
    memory_projection: Option<Linear<B>>,
    decoder: TransformerDecoder<B>,
    decoder_projection: Linear<B>,
    blank_id: usize,
}

#[derive(Debug)]
pub struct ParaformerOutput<B: Backend> {
    pub decoder_logits: Tensor<B, 3>,
    pub ctc_log_probs: Tensor<B, 3>,
    pub encoder_lengths: Vec<usize>,
    pub query_lengths: Vec<usize>,
    pub alignments: Vec<i64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParaformerAlignmentMode {
    Viterbi,
    Uniform,
    Greedy,
}

#[derive(Debug)]
pub struct ParaformerLossOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub ctc_loss: Tensor<B, 1>,
    pub ce_loss: Tensor<B, 1>,
    pub output: ParaformerOutput<B>,
}

#[derive(Module, Debug)]
pub struct EnhancedParaformerV2<B: Backend> {
    encoder: MultiResolutionConformerEncoder<B>,
    shallow_ctc_projection: Linear<B>,
    final_ctc_projection: Linear<B>,
    boundary_in: Linear<B>,
    boundary_out: Linear<B>,
    query_projection: Linear<B>,
    query_norm: LayerNorm<B>,
    memory_projection: Option<Linear<B>>,
    decoder: TransformerDecoder<B>,
    decoder_projection: Linear<B>,
    token_embedding: Embedding<B>,
    refinement_decoder: TransformerDecoder<B>,
    refinement_projection: Linear<B>,
    blank_id: usize,
    shallow_ctc_loss_weight: f64,
    boundary_loss_weight: f64,
    refinement_loss_weight: f64,
    confidence_threshold: f64,
    low_confidence_threshold: f64,
}

#[derive(Debug)]
pub struct EnhancedParaformerOutput<B: Backend> {
    pub decoder_logits: Tensor<B, 3>,
    pub initial_decoder_logits: Tensor<B, 3>,
    pub ctc_log_probs: Tensor<B, 3>,
    pub shallow_ctc_log_probs: Tensor<B, 3>,
    pub encoder_lengths: Vec<usize>,
    pub query_lengths: Vec<usize>,
    pub query_confidences: Tensor<B, 2>,
    pub alignments: Vec<i64>,
    pub boundary_logits: Tensor<B, 2>,
}

#[derive(Debug)]
pub struct EnhancedParaformerLossOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub ctc_loss: Tensor<B, 1>,
    pub shallow_ctc_loss: Tensor<B, 1>,
    pub ce_loss: Tensor<B, 1>,
    pub boundary_loss: Tensor<B, 1>,
    pub output: EnhancedParaformerOutput<B>,
}

impl<B: ParaformerKernelBackend> ParaformerV2<B> {
    pub fn forward(&self, features: Tensor<B, 3>, lengths: Vec<usize>) -> ParaformerOutput<B> {
        let (encoder_out, encoder_lengths) = self.encoder.forward(features, lengths);
        let ctc_logits = self.ctc_projection.forward(encoder_out.clone());
        let ctc_log_probs = softmax(ctc_logits.clone(), 2).log();
        self.forward_with_posterior_queries(
            encoder_out,
            encoder_lengths,
            softmax(ctc_logits, 2),
            None,
            None,
        )
        .with_ctc_log_probs(ctc_log_probs)
    }

    pub fn forward_train(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
        targets: &[i64],
        target_lengths: &[usize],
        alignment_mode: ParaformerAlignmentMode,
    ) -> ParaformerOutput<B> {
        let (encoder_out, encoder_lengths) = self.encoder.forward(features, lengths);
        let ctc_logits = self.ctc_projection.forward(encoder_out.clone());
        let ctc_log_probs = log_softmax(ctc_logits.clone(), 2);
        let posteriors = softmax(ctc_logits, 2);
        let alignments = match alignment_mode {
            ParaformerAlignmentMode::Viterbi => batch_ctc_viterbi_alignments(
                ctc_log_probs.clone(),
                &encoder_lengths,
                targets,
                target_lengths,
                self.resolved_blank_id(),
            ),
            ParaformerAlignmentMode::Uniform => batch_uniform_alignments(
                &encoder_lengths,
                targets,
                target_lengths,
                posteriors.dims()[1],
                self.resolved_blank_id(),
            ),
            ParaformerAlignmentMode::Greedy => {
                greedy_alignments(ctc_log_probs.clone(), &encoder_lengths)
            }
        };
        let (compressed, query_lengths) = compress_posteriors(
            posteriors,
            &alignments,
            &encoder_lengths,
            self.resolved_blank_id(),
        );
        self.forward_with_posterior_queries(
            encoder_out,
            encoder_lengths,
            compressed,
            Some(query_lengths),
            Some(alignments),
        )
        .with_ctc_log_probs(ctc_log_probs)
    }

    pub fn loss(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
        targets: Tensor<B, 2, Int>,
        targets_flat: &[i64],
        target_lengths: Vec<usize>,
        blank_id: usize,
        alignment_mode: ParaformerAlignmentMode,
    ) -> ParaformerLossOutput<B> {
        let output = self.forward_train(
            features,
            lengths,
            targets_flat,
            &target_lengths,
            alignment_mode,
        );
        let input_lengths = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                to_i64(output.encoder_lengths.clone()),
                [target_lengths.len()],
            ),
            &output.ctc_log_probs.device(),
        );
        let target_length_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::new(to_i64(target_lengths.clone()), [target_lengths.len()]),
            &output.ctc_log_probs.device(),
        );
        let ctc_loss = burn_nn::loss::CTCLossConfig::new()
            .with_blank(blank_id)
            .with_zero_infinity(true)
            .init()
            .forward_with_reduction(
                output.ctc_log_probs.clone().swap_dims(0, 1),
                targets.clone(),
                input_lengths,
                target_length_tensor,
                burn_nn::loss::Reduction::Mean,
            );
        let ce_loss = masked_cross_entropy(output.decoder_logits.clone(), targets, target_lengths);
        ParaformerLossOutput {
            loss: ctc_loss.clone() + ce_loss.clone(),
            ctc_loss,
            ce_loss,
            output,
        }
    }

    pub fn forward_with_posterior_queries(
        &self,
        encoder_out: Tensor<B, 3>,
        encoder_lengths: Vec<usize>,
        posterior_queries: Tensor<B, 3>,
        query_lengths: Option<Vec<usize>>,
        alignments: Option<Vec<i64>>,
    ) -> ParaformerOutput<B> {
        let device = encoder_out.device();
        let query_lengths = query_lengths.unwrap_or_else(|| encoder_lengths.clone());
        let decoder_in = self.posterior_embed.forward(posterior_queries);
        let memory = match &self.memory_projection {
            Some(projection) => projection.forward(encoder_out),
            None => encoder_out,
        };
        let target_mask = B::padding_mask(&query_lengths, decoder_in.dims()[1], &device);
        let memory_mask = B::padding_mask(&encoder_lengths, memory.dims()[1], &device);
        let decoder_out = self.decoder.forward(
            TransformerDecoderInput::new(decoder_in, memory)
                .target_mask_pad(target_mask)
                .memory_mask_pad(memory_mask),
        );
        ParaformerOutput {
            decoder_logits: self.decoder_projection.forward(decoder_out),
            ctc_log_probs: Tensor::zeros([1, 1, 1], &device),
            encoder_lengths,
            query_lengths,
            alignments: alignments.unwrap_or_default(),
        }
    }

    fn resolved_blank_id(&self) -> usize {
        self.blank_id
    }
}

impl<B: Backend> ParaformerOutput<B> {
    fn with_ctc_log_probs(mut self, ctc_log_probs: Tensor<B, 3>) -> Self {
        self.ctc_log_probs = ctc_log_probs;
        self
    }
}

impl<B: ParaformerKernelBackend> EnhancedParaformerV2<B> {
    pub fn ctc_log_probs(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        let (_, final_out, encoder_lengths) = self.encoder.forward(features, lengths);
        (
            log_softmax(self.final_ctc_projection.forward(final_out), 2),
            encoder_lengths,
        )
    }

    pub fn forward_train(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
        targets: &[i64],
        target_lengths: &[usize],
        alignment_mode: ParaformerAlignmentMode,
    ) -> EnhancedParaformerOutput<B> {
        let (shallow_out, final_out, encoder_lengths) = self.encoder.forward(features, lengths);
        let shallow_logits = self.shallow_ctc_projection.forward(shallow_out.clone());
        let final_logits = self.final_ctc_projection.forward(final_out.clone());
        let shallow_ctc_log_probs = log_softmax(shallow_logits.clone(), 2);
        let ctc_log_probs = log_softmax(final_logits.clone(), 2);
        let final_posteriors = softmax(final_logits.clone(), 2);
        let alignments = match alignment_mode {
            ParaformerAlignmentMode::Viterbi => batch_ctc_viterbi_alignments(
                ctc_log_probs.clone(),
                &encoder_lengths,
                targets,
                target_lengths,
                self.blank_id,
            ),
            ParaformerAlignmentMode::Uniform => batch_uniform_alignments(
                &encoder_lengths,
                targets,
                target_lengths,
                final_posteriors.dims()[1],
                self.blank_id,
            ),
            ParaformerAlignmentMode::Greedy => {
                greedy_alignments(ctc_log_probs.clone(), &encoder_lengths)
            }
        };

        let boundary_logits = self
            .boundary_out
            .forward(silu(self.boundary_in.forward(Tensor::cat(
                vec![shallow_out.clone(), final_out.clone()],
                2,
            ))))
            .reshape([final_out.dims()[0], final_out.dims()[1]]);
        let (query_features, query_lengths, query_confidences) = compress_confidence_gated_queries(
            softmax(shallow_logits, 2),
            final_posteriors,
            &alignments,
            &encoder_lengths,
            self.blank_id,
            sigmoid(boundary_logits.clone()),
            self.confidence_threshold,
        );
        let decoder_in = silu(
            self.query_norm
                .forward(self.query_projection.forward(query_features)),
        ) * query_confidences.clone().unsqueeze_dim::<3>(2);
        let memory = match &self.memory_projection {
            Some(projection) => projection.forward(final_out),
            None => final_out,
        };
        let target_mask = B::padding_mask(&query_lengths, decoder_in.dims()[1], &memory.device());
        let memory_mask = B::padding_mask(&encoder_lengths, memory.dims()[1], &memory.device());
        let decoder_out = self.decoder.forward(
            TransformerDecoderInput::new(decoder_in, memory.clone())
                .target_mask_pad(target_mask.clone())
                .memory_mask_pad(memory_mask.clone()),
        );
        let initial_decoder_logits = self.decoder_projection.forward(decoder_out.clone());
        let decoder_probs = softmax(initial_decoder_logits.clone(), 2);
        let token_confidences = decoder_probs
            .clone()
            .max_dim(2)
            .reshape([decoder_probs.dims()[0], decoder_probs.dims()[1]]);
        let token_ids = decoder_probs
            .clone()
            .argmax(2)
            .reshape([decoder_probs.dims()[0], decoder_probs.dims()[1]]);
        let correction_seed = decoder_out.clone() + self.token_embedding.forward(token_ids);
        let refinement_out = self.refinement_decoder.forward(
            TransformerDecoderInput::new(correction_seed, memory)
                .target_mask_pad(target_mask)
                .memory_mask_pad(memory_mask),
        );
        let low_confidence = token_confidences
            .lower_elem(self.low_confidence_threshold)
            .unsqueeze_dim::<3>(2)
            .repeat_dim(2, decoder_out.dims()[2]);
        let refined_states = decoder_out.mask_where(low_confidence, refinement_out);
        let decoder_logits = self.refinement_projection.forward(refined_states);

        EnhancedParaformerOutput {
            decoder_logits,
            initial_decoder_logits,
            ctc_log_probs,
            shallow_ctc_log_probs,
            encoder_lengths,
            query_lengths,
            query_confidences,
            alignments,
            boundary_logits,
        }
    }

    pub fn loss(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
        targets: Tensor<B, 2, Int>,
        targets_flat: &[i64],
        target_lengths: Vec<usize>,
        blank_id: usize,
        alignment_mode: ParaformerAlignmentMode,
    ) -> EnhancedParaformerLossOutput<B> {
        let output = self.forward_train(
            features,
            lengths,
            targets_flat,
            &target_lengths,
            alignment_mode,
        );
        let ctc_loss = ctc_loss_from_log_probs(
            output.ctc_log_probs.clone(),
            targets.clone(),
            output.encoder_lengths.clone(),
            target_lengths.clone(),
            blank_id,
        );
        let shallow_ctc_loss = ctc_loss_from_log_probs(
            output.shallow_ctc_log_probs.clone(),
            targets.clone(),
            output.encoder_lengths.clone(),
            target_lengths.clone(),
            blank_id,
        );
        let base_ce_loss = masked_cross_entropy(
            output.initial_decoder_logits.clone(),
            targets.clone(),
            target_lengths.clone(),
        );
        let ce_loss = masked_cross_entropy(output.decoder_logits.clone(), targets, target_lengths);
        let boundary_loss = boundary_bce_loss(
            output.boundary_logits.clone(),
            &output.alignments,
            &output.encoder_lengths,
        );
        let loss = ctc_loss.clone()
            + shallow_ctc_loss.clone() * self.shallow_ctc_loss_weight
            + base_ce_loss
            + ce_loss.clone() * self.refinement_loss_weight
            + boundary_loss.clone() * self.boundary_loss_weight;

        EnhancedParaformerLossOutput {
            loss,
            ctc_loss,
            shallow_ctc_loss,
            ce_loss,
            boundary_loss,
            output,
        }
    }
}

fn compress_posteriors<B: Backend>(
    posteriors: Tensor<B, 3>,
    alignments: &[i64],
    lengths: &[usize],
    blank_id: usize,
) -> (Tensor<B, 3>, Vec<usize>) {
    let [batch_size, max_time, vocab_size] = posteriors.dims();
    let mut pieces = Vec::with_capacity(batch_size);
    let mut piece_lengths = Vec::with_capacity(batch_size);
    let device = posteriors.device();

    for (batch, &length) in lengths.iter().enumerate().take(batch_size) {
        let mut utterance = Vec::new();
        let mut start = 0usize;
        while start < length {
            let label = alignments[batch * max_time + start];
            let mut end = start + 1;
            while end < length && alignments[batch * max_time + end] == label {
                end += 1;
            }
            if label != blank_id as i64 {
                let segment = posteriors
                    .clone()
                    .slice([batch..batch + 1, start..end, 0..vocab_size])
                    .mean_dim(1);
                utterance.push(segment);
            }
            start = end;
        }
        if utterance.is_empty() {
            utterance.push(Tensor::zeros([1, 1, vocab_size], &device));
        }
        let piece = Tensor::cat(utterance, 1);
        piece_lengths.push(piece.dims()[1]);
        pieces.push(piece);
    }

    let max_piece_len = piece_lengths.iter().copied().max().unwrap_or(1);
    let padded = pieces
        .into_iter()
        .map(|piece| {
            let piece_len = piece.dims()[1];
            if piece_len < max_piece_len {
                Tensor::cat(
                    vec![
                        piece,
                        Tensor::zeros([1, max_piece_len - piece_len, vocab_size], &device),
                    ],
                    1,
                )
            } else {
                piece
            }
        })
        .collect();
    (Tensor::cat(padded, 0), piece_lengths)
}

fn compress_confidence_gated_queries<B: Backend>(
    shallow_posteriors: Tensor<B, 3>,
    final_posteriors: Tensor<B, 3>,
    alignments: &[i64],
    lengths: &[usize],
    blank_id: usize,
    boundary_probs: Tensor<B, 2>,
    confidence_threshold: f64,
) -> (Tensor<B, 3>, Vec<usize>, Tensor<B, 2>) {
    let [batch_size, max_time, vocab_size] = shallow_posteriors.dims();
    let feature_dim = vocab_size * 2 + 4;
    let device = shallow_posteriors.device();
    let mut pieces = Vec::with_capacity(batch_size);
    let mut confidence_pieces = Vec::with_capacity(batch_size);
    let mut piece_lengths = Vec::with_capacity(batch_size);

    for (batch, &length) in lengths.iter().enumerate().take(batch_size) {
        let segments = nonblank_segments(
            &alignments[batch * max_time..batch * max_time + length],
            blank_id,
        );
        if segments.is_empty() {
            pieces.push(Tensor::zeros([1, 1, feature_dim], &device));
            confidence_pieces.push(Tensor::ones([1, 1], &device));
            piece_lengths.push(1);
            continue;
        }

        let mut segment_features = Vec::with_capacity(segments.len());
        let mut segment_confidences = Vec::with_capacity(segments.len());
        for (segment_index, (start, end, label)) in segments.iter().copied().enumerate() {
            let shallow_frames =
                shallow_posteriors
                    .clone()
                    .slice([batch..batch + 1, start..end, 0..vocab_size]);
            let final_frames =
                final_posteriors
                    .clone()
                    .slice([batch..batch + 1, start..end, 0..vocab_size]);
            let label_indices =
                Tensor::<B, 3, Int>::full([1, end - start, 1], label as i64, &device);
            let shallow_label_confidence = shallow_frames.clone().gather(2, label_indices.clone());
            let final_label_confidence = final_frames.clone().gather(2, label_indices);
            let frame_confidence =
                (shallow_label_confidence.clone() + final_label_confidence.clone()) * 0.5;
            let gate = sigmoid((frame_confidence.clone() - confidence_threshold) * 12.0) + 0.05;
            let gate_sum = gate.clone().sum().clamp_min(1.0e-6).reshape([1, 1, 1]);
            let shallow_pool = (shallow_frames * gate.clone())
                .sum_dim(1)
                .reshape([1, 1, vocab_size])
                / gate_sum.clone();
            let final_pool =
                (final_frames * gate).sum_dim(1).reshape([1, 1, vocab_size]) / gate_sum;
            let segment_confidence = frame_confidence.clone().mean().reshape([1, 1, 1]);
            let shallow_mean = shallow_label_confidence.mean().reshape([1, 1, 1]);
            let left_boundary = if segment_index > 0 {
                boundary_probs
                    .clone()
                    .slice([batch..batch + 1, start - 1..start])
                    .reshape([1, 1, 1])
            } else {
                Tensor::ones([1, 1, 1], &device)
            };
            let right_boundary = boundary_probs
                .clone()
                .slice([batch..batch + 1, end - 1..end])
                .reshape([1, 1, 1]);
            let smooth_left = (Tensor::ones([1, 1, 1], &device) - left_boundary.clone()) * 0.25;
            let smooth_right = (Tensor::ones([1, 1, 1], &device) - right_boundary.clone()) * 0.25;
            let meta = Tensor::cat(
                vec![
                    segment_confidence.clone(),
                    shallow_mean,
                    left_boundary,
                    right_boundary,
                ],
                2,
            );
            let feature = Tensor::cat(vec![shallow_pool, final_pool, meta], 2)
                * (Tensor::ones([1, 1, 1], &device) + smooth_left + smooth_right);
            segment_features.push(feature);
            segment_confidences.push(segment_confidence.reshape([1, 1]).clamp(0.05, 1.0));
        }
        let piece = Tensor::cat(segment_features, 1);
        let confidences = Tensor::cat(segment_confidences, 1);
        piece_lengths.push(piece.dims()[1]);
        pieces.push(piece);
        confidence_pieces.push(confidences);
    }

    let max_piece_len = piece_lengths.iter().copied().max().unwrap_or(1);
    let padded = pieces
        .into_iter()
        .map(|piece| {
            let piece_len = piece.dims()[1];
            if piece_len < max_piece_len {
                Tensor::cat(
                    vec![
                        piece,
                        Tensor::zeros([1, max_piece_len - piece_len, feature_dim], &device),
                    ],
                    1,
                )
            } else {
                piece
            }
        })
        .collect();
    let padded_confidences = confidence_pieces
        .into_iter()
        .map(|piece| {
            let piece_len = piece.dims()[1];
            if piece_len < max_piece_len {
                Tensor::cat(
                    vec![
                        piece,
                        Tensor::zeros([1, max_piece_len - piece_len], &device),
                    ],
                    1,
                )
            } else {
                piece
            }
        })
        .collect();
    (
        Tensor::cat(padded, 0),
        piece_lengths,
        Tensor::cat(padded_confidences, 0),
    )
}

fn nonblank_segments(labels: &[i64], blank_id: usize) -> Vec<(usize, usize, usize)> {
    let mut segments = Vec::new();
    let mut start = 0usize;
    while start < labels.len() {
        let label = labels[start];
        let mut end = start + 1;
        while end < labels.len() && labels[end] == label {
            end += 1;
        }
        if label != blank_id as i64 {
            segments.push((start, end, label as usize));
        }
        start = end;
    }
    segments
}

fn masked_cross_entropy<B: ParaformerKernelBackend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    target_lengths: Vec<usize>,
) -> Tensor<B, 1> {
    let [batch_size, query_len, vocab_size] = logits.dims();
    let max_target_len = target_lengths.iter().copied().max().unwrap_or(0);
    let usable_len = max_target_len.min(query_len);
    let device = logits.device();
    if usable_len == 0 {
        return Tensor::zeros([1], &device);
    }

    let trimmed_logits = logits.slice([0..batch_size, 0..usable_len, 0..vocab_size]);
    let trimmed_targets = targets.slice([0..batch_size, 0..usable_len]);
    let log_probs = log_softmax(trimmed_logits, 2);
    let gathered = log_probs
        .gather(2, trimmed_targets.clone().unsqueeze_dim::<3>(2))
        .reshape([batch_size, usable_len]);
    let mask = B::sequence_mask(&target_lengths, usable_len, &device).float();
    let denom = mask.clone().sum().clamp_min(1.0);
    -(gathered * mask).sum() / denom
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
    burn_nn::loss::CTCLossConfig::new()
        .with_blank(blank_id)
        .with_zero_infinity(true)
        .init()
        .forward_with_reduction(
            log_probs.swap_dims(0, 1),
            targets,
            input_lengths,
            target_lengths,
            burn_nn::loss::Reduction::Mean,
        )
}

fn boundary_bce_loss<B: ParaformerKernelBackend>(
    logits: Tensor<B, 2>,
    alignments: &[i64],
    lengths: &[usize],
) -> Tensor<B, 1> {
    let [batch_size, max_time] = logits.dims();
    let targets = build_boundary_targets::<B>(alignments, lengths, max_time, &logits.device());
    let mask = B::sequence_mask(lengths, max_time, &logits.device()).float();
    let probs = sigmoid(logits).clamp(1.0e-6, 1.0 - 1.0e-6);
    let one = Tensor::ones([batch_size, max_time], &probs.device());
    let loss =
        -(targets.clone() * probs.clone().log() + (one.clone() - targets) * (one - probs).log());
    (loss * mask.clone()).sum() / mask.sum().clamp_min(1.0)
}

fn build_boundary_targets<B: Backend>(
    alignments: &[i64],
    lengths: &[usize],
    max_time: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let batch_size = lengths.len();
    let mut values = vec![0.0f32; batch_size * max_time];
    for batch in 0..batch_size {
        let length = lengths[batch].min(max_time);
        if length > 1 {
            for time in 0..length - 1 {
                if alignments[batch * max_time + time] != alignments[batch * max_time + time + 1] {
                    values[batch * max_time + time] = 1.0;
                }
            }
        }
    }
    Tensor::from_data(TensorData::new(values, [batch_size, max_time]), device)
}

fn batch_uniform_alignments(
    input_lengths: &[usize],
    targets: &[i64],
    target_lengths: &[usize],
    max_input_len: usize,
    blank_id: usize,
) -> Vec<i64> {
    let batch_size = input_lengths.len();
    let max_target_len = target_lengths.iter().copied().max().unwrap_or(0);
    let mut alignments = vec![blank_id as i64; batch_size * max_input_len];
    for batch in 0..batch_size {
        let t_len = input_lengths[batch].min(max_input_len);
        let y_len = target_lengths[batch];
        if t_len == 0 || y_len == 0 {
            continue;
        }
        let mut extended = vec![blank_id as i64; y_len * 2 + 1];
        for index in 0..y_len {
            extended[index * 2 + 1] = targets[batch * max_target_len + index];
        }
        for time in 0..t_len {
            let state = ((time * extended.len()) / t_len).min(extended.len() - 1);
            alignments[batch * max_input_len + time] = extended[state];
        }
    }
    alignments
}

fn greedy_alignments<B: Backend>(log_probs: Tensor<B, 3>, lengths: &[usize]) -> Vec<i64> {
    let [batch_size, max_time, _] = log_probs.dims();
    let mut values = log_probs
        .argmax(2)
        .into_data()
        .to_vec::<i64>()
        .expect("failed to read Paraformer greedy alignments");
    for batch in 0..batch_size {
        for time in lengths[batch]..max_time {
            values[batch * max_time + time] = 0;
        }
    }
    values
}

fn batch_ctc_viterbi_alignments<B: Backend>(
    log_probs: Tensor<B, 3>,
    input_lengths: &[usize],
    targets: &[i64],
    target_lengths: &[usize],
    blank_id: usize,
) -> Vec<i64> {
    let [batch_size, max_time, vocab_size] = log_probs.dims();
    let data = log_probs
        .detach()
        .into_data()
        .to_vec::<f32>()
        .expect("failed to read Paraformer CTC log probabilities");
    let max_target_len = target_lengths.iter().copied().max().unwrap_or(0);
    let mut alignments = vec![blank_id as i64; batch_size * max_time];
    for batch in 0..batch_size {
        let t_len = input_lengths[batch].min(max_time);
        let y_len = target_lengths[batch];
        let target = &targets[batch * max_target_len..batch * max_target_len + y_len];
        let alignment = ctc_viterbi_alignment(
            &data[batch * max_time * vocab_size..(batch + 1) * max_time * vocab_size],
            t_len,
            vocab_size,
            target,
            blank_id,
        );
        alignments[batch * max_time..batch * max_time + t_len].copy_from_slice(&alignment);
    }
    alignments
}

fn ctc_viterbi_alignment(
    log_probs: &[f32],
    time: usize,
    vocab_size: usize,
    targets: &[i64],
    blank_id: usize,
) -> Vec<i64> {
    if time == 0 {
        return Vec::new();
    }
    if targets.is_empty() {
        return vec![blank_id as i64; time];
    }

    let mut extended = vec![blank_id as i64; targets.len() * 2 + 1];
    for (index, token) in targets.iter().enumerate() {
        extended[index * 2 + 1] = *token;
    }
    let states = extended.len();
    let neg_inf = f32::NEG_INFINITY;
    let mut dp = vec![neg_inf; time * states];
    let mut back = vec![0usize; time * states];
    dp[0] = log_probs[blank_id];
    if states > 1 {
        dp[1] = log_probs[extended[1] as usize];
    }

    for t in 1..time {
        for s in 0..states {
            let mut best_score = dp[(t - 1) * states + s];
            let mut best_state = s;
            if s >= 1 && dp[(t - 1) * states + s - 1] > best_score {
                best_score = dp[(t - 1) * states + s - 1];
                best_state = s - 1;
            }
            if s >= 2
                && extended[s] != blank_id as i64
                && extended[s] != extended[s - 2]
                && dp[(t - 1) * states + s - 2] > best_score
            {
                best_score = dp[(t - 1) * states + s - 2];
                best_state = s - 2;
            }
            dp[t * states + s] = best_score + log_probs[t * vocab_size + extended[s] as usize];
            back[t * states + s] = best_state;
        }
    }

    let mut state = states - 1;
    if states > 1 && dp[(time - 1) * states + states - 2] > dp[(time - 1) * states + state] {
        state = states - 2;
    }
    let mut path = vec![blank_id as i64; time];
    for t in (0..time).rev() {
        path[t] = extended[state];
        if t > 0 {
            state = back[t * states + state];
        }
    }
    path
}

fn glu_channel_dim_fallback<B: Backend>(input: Tensor<B, 3>) -> Tensor<B, 3> {
    let mut chunks = input.chunk(2, 1);
    let gate = chunks.remove(1);
    let value = chunks.remove(0);
    value * sigmoid(gate)
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

fn to_i64(values: Vec<usize>) -> Vec<i64> {
    values.into_iter().map(|value| value as i64).collect()
}

fn padding_mask_fallback<B: Backend>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn paraformer_forward_matches_expected_axes() {
        let device = Default::default();
        let mut config = ParaformerV2Config::new(80, 32);
        config.encoder_dim = 16;
        config.decoder_dim = 16;
        config.encoder_layers = 1;
        config.decoder_layers = 1;
        config.encoder_ff_dim = 32;
        config.decoder_ff_dim = 32;
        config.attention_heads = 4;
        let model = config.init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 16, 80], &device);

        let output = model.forward(input, vec![16, 12]);

        assert_eq!(output.decoder_logits.dims(), [2, 4, 32]);
        assert_eq!(output.encoder_lengths, vec![4, 3]);
    }

    #[test]
    fn uniform_alignment_and_compression_match_python_contract() {
        let alignments = batch_uniform_alignments(&[6], &[1, 2], &[2], 6, 0);
        assert_eq!(alignments, vec![0, 0, 1, 0, 2, 0]);

        let device = Default::default();
        let posteriors = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0,
                    0.0, 4.0,
                ],
                [1, 6, 3],
            ),
            &device,
        );
        let (compressed, lengths) = compress_posteriors(posteriors, &alignments, &[6], 0);

        assert_eq!(compressed.dims(), [1, 2, 3]);
        assert_eq!(lengths, vec![2]);
    }

    #[test]
    fn paraformer_training_forward_uses_target_aligned_queries() {
        let device = Default::default();
        let mut config = ParaformerV2Config::new(8, 6).with_blank_id(0);
        config.encoder_dim = 16;
        config.decoder_dim = 16;
        config.encoder_layers = 1;
        config.decoder_layers = 1;
        config.encoder_ff_dim = 32;
        config.decoder_ff_dim = 32;
        config.attention_heads = 2;
        let model = config.init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 24, 8], &device);
        let targets = vec![1, 2, 3, 2, 4, 0];
        let output = model.forward_train(
            input,
            vec![24, 20],
            &targets,
            &[3, 2],
            ParaformerAlignmentMode::Uniform,
        );

        assert_eq!(output.encoder_lengths, vec![6, 5]);
        assert_eq!(output.query_lengths, vec![3, 2]);
        assert_eq!(output.decoder_logits.dims(), [2, 3, 6]);
        assert_eq!(output.alignments.len(), 12);
    }

    #[test]
    fn enhanced_paraformer_forward_exposes_auxiliary_heads() {
        let device = Default::default();
        let mut config = EnhancedParaformerV2Config::new(8, 6).with_blank_id(0);
        config.base.encoder_dim = 16;
        config.base.decoder_dim = 16;
        config.base.encoder_layers = 2;
        config.base.decoder_layers = 1;
        config.base.encoder_ff_dim = 32;
        config.base.decoder_ff_dim = 32;
        config.base.attention_heads = 2;
        let model = config.init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 24, 8], &device);
        let targets = vec![1, 2, 3, 2, 4, 0];

        let output = model.forward_train(
            input,
            vec![24, 20],
            &targets,
            &[3, 2],
            ParaformerAlignmentMode::Uniform,
        );

        assert_eq!(output.encoder_lengths, vec![6, 5]);
        assert_eq!(output.ctc_log_probs.dims(), [2, 6, 6]);
        assert_eq!(output.shallow_ctc_log_probs.dims(), [2, 6, 6]);
        assert_eq!(output.boundary_logits.dims(), [2, 6]);
        assert_eq!(output.initial_decoder_logits.dims()[2], 6);
        assert_eq!(output.decoder_logits.dims()[2], 6);
    }
}
