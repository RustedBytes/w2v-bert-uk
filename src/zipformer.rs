use burn::module::{Initializer, Module, Param};
#[cfg(feature = "asr-cubecl-kernels")]
use burn::tensor::Int;
#[cfg(feature = "asr-cubecl-kernels")]
use burn::tensor::TensorPrimitive;
use burn::tensor::activation::{sigmoid, silu, softmax, tanh};
#[cfg(feature = "asr-cubecl-kernels")]
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::ops::PadMode;
use burn::tensor::{Bool, Tensor, TensorData, backend::Backend};
use burn_nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn_nn::{Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig1d, PaddingConfig2d};

const DEFAULT_DOWNSAMPLING: [usize; 6] = [1, 2, 4, 8, 4, 2];
const DEFAULT_ENCODER_DIM: [usize; 6] = [192, 256, 384, 512, 384, 256];
const DEFAULT_NUM_LAYERS: [usize; 6] = [2, 2, 3, 4, 3, 2];
const DEFAULT_NUM_HEADS: [usize; 6] = [4, 4, 4, 8, 4, 4];
const DEFAULT_FEEDFORWARD_DIM: [usize; 6] = [512, 768, 1024, 1536, 1024, 768];
const DEFAULT_CNN_KERNELS: [usize; 6] = [31, 31, 15, 15, 15, 31];

pub trait ZipformerKernelBackend: Backend + Sized {
    fn relative_shift(input: Tensor<Self, 4>, seq_len: usize) -> Tensor<Self, 4> {
        relative_shift_fallback(input, seq_len)
    }

    fn mask_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        mask_time_fallback(input, lengths)
    }

    fn mask_add_time(
        residual: Tensor<Self, 3>,
        update: Tensor<Self, 3>,
        lengths: &[usize],
    ) -> Tensor<Self, 3> {
        mask_time_fallback(residual + update, lengths)
    }

    fn glu_last_dim(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        glu_last_dim_fallback(input)
    }

    fn attention_mask_4d(
        lengths: &[usize],
        heads: usize,
        query_len: usize,
        key_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 4, Bool> {
        attention_key_mask_4d_fallback(lengths, heads, query_len, key_len, device)
    }

    fn pairwise_downsample(
        input: Tensor<Self, 3>,
        lengths: &[usize],
        weights: Tensor<Self, 1>,
    ) -> (Tensor<Self, 3>, Vec<usize>) {
        pairwise_downsample_fallback(input, lengths, weights)
    }
}

impl ZipformerKernelBackend for burn_ndarray::NdArray<f32> {}

impl<C> ZipformerKernelBackend for burn_autodiff::Autodiff<burn_ndarray::NdArray<f32>, C> where
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
impl<F, I> ZipformerKernelBackend for burn_cuda::Cuda<F, I>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
{
    fn relative_shift(input: Tensor<Self, 4>, seq_len: usize) -> Tensor<Self, 4> {
        crate::cubecl_kernels::relative_shift(input, seq_len)
    }

    fn mask_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::cubecl_kernels::mask_time(input, lengths)
    }

    fn mask_add_time(
        residual: Tensor<Self, 3>,
        update: Tensor<Self, 3>,
        lengths: &[usize],
    ) -> Tensor<Self, 3> {
        let lengths = lengths_tensor::<Self>(lengths, &residual.device());
        crate::cubecl_kernels::residual_add_mask_time(residual, update, lengths)
    }

    fn glu_last_dim(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        crate::cubecl_kernels::glu_last_dim(input)
    }

    fn attention_mask_4d(
        lengths: &[usize],
        heads: usize,
        query_len: usize,
        key_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 4, Bool> {
        let lengths = lengths_tensor::<Self>(lengths, device);
        crate::cubecl_kernels::attention_mask_4d_with_lengths(lengths, heads, query_len, key_len)
    }

    fn pairwise_downsample(
        input: Tensor<Self, 3>,
        lengths: &[usize],
        weights: Tensor<Self, 1>,
    ) -> (Tensor<Self, 3>, Vec<usize>) {
        let output_lengths = downsample_lengths(lengths, 2);
        let length_tensor = lengths_tensor::<Self>(lengths, &input.device());
        let output = crate::cubecl_kernels::pairwise_downsample(input, length_tensor, weights);
        (Self::mask_time(output, &output_lengths), output_lengths)
    }
}

#[cfg(all(feature = "asr-cubecl-kernels", feature = "burn-cuda-backend"))]
impl<F, I, C> ZipformerKernelBackend for burn_autodiff::Autodiff<burn_cuda::Cuda<F, I>, C>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    fn relative_shift(input: Tensor<Self, 4>, seq_len: usize) -> Tensor<Self, 4> {
        let portable = relative_shift_fallback(input.clone(), seq_len);
        let raw = crate::cubecl_kernels::relative_shift(autodiff_to_inner(input), seq_len);
        attach_autodiff_gradient(inner_to_autodiff(raw), portable)
    }

    fn mask_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::asr_autodiff_kernels::mask_time(input, lengths)
    }

    fn mask_add_time(
        residual: Tensor<Self, 3>,
        update: Tensor<Self, 3>,
        lengths: &[usize],
    ) -> Tensor<Self, 3> {
        crate::asr_autodiff_kernels::residual_add_mask_time(residual, update, lengths)
    }

    fn glu_last_dim(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        let portable = glu_last_dim_fallback(input.clone());
        let raw = crate::cubecl_kernels::glu_last_dim(autodiff_to_inner(input));
        attach_autodiff_gradient(inner_to_autodiff(raw), portable)
    }

    fn attention_mask_4d(
        lengths: &[usize],
        heads: usize,
        query_len: usize,
        key_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 4, Bool> {
        let lengths = lengths_tensor::<burn_cuda::Cuda<F, I>>(lengths, device);
        let raw = crate::cubecl_kernels::attention_mask_4d_with_lengths(
            lengths, heads, query_len, key_len,
        );
        inner_bool_to_autodiff(raw)
    }

    fn pairwise_downsample(
        input: Tensor<Self, 3>,
        lengths: &[usize],
        weights: Tensor<Self, 1>,
    ) -> (Tensor<Self, 3>, Vec<usize>) {
        let (portable, output_lengths) =
            pairwise_downsample_fallback(input.clone(), lengths, weights.clone());
        let input_inner = autodiff_to_inner(input);
        let length_tensor = lengths_tensor::<burn_cuda::Cuda<F, I>>(lengths, &input_inner.device());
        let weights_inner = autodiff_to_inner(weights);
        let raw =
            crate::cubecl_kernels::pairwise_downsample(input_inner, length_tensor, weights_inner);
        let raw = crate::cubecl_kernels::mask_time(raw, &output_lengths);
        (
            attach_autodiff_gradient(inner_to_autodiff(raw), portable),
            output_lengths,
        )
    }
}

#[cfg(all(feature = "asr-cubecl-kernels", feature = "burn-wgpu-backend"))]
impl<F, I, BT> ZipformerKernelBackend for burn_wgpu::Wgpu<F, I, BT>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
{
    fn relative_shift(input: Tensor<Self, 4>, seq_len: usize) -> Tensor<Self, 4> {
        crate::cubecl_kernels::relative_shift(input, seq_len)
    }

    fn mask_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::cubecl_kernels::mask_time(input, lengths)
    }

    fn mask_add_time(
        residual: Tensor<Self, 3>,
        update: Tensor<Self, 3>,
        lengths: &[usize],
    ) -> Tensor<Self, 3> {
        let lengths = lengths_tensor::<Self>(lengths, &residual.device());
        crate::cubecl_kernels::residual_add_mask_time(residual, update, lengths)
    }

    fn glu_last_dim(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        crate::cubecl_kernels::glu_last_dim(input)
    }

    fn attention_mask_4d(
        lengths: &[usize],
        heads: usize,
        query_len: usize,
        key_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 4, Bool> {
        let lengths = lengths_tensor::<Self>(lengths, device);
        crate::cubecl_kernels::attention_mask_4d_with_lengths(lengths, heads, query_len, key_len)
    }

    fn pairwise_downsample(
        input: Tensor<Self, 3>,
        lengths: &[usize],
        weights: Tensor<Self, 1>,
    ) -> (Tensor<Self, 3>, Vec<usize>) {
        let output_lengths = downsample_lengths(lengths, 2);
        let length_tensor = lengths_tensor::<Self>(lengths, &input.device());
        let output = crate::cubecl_kernels::pairwise_downsample(input, length_tensor, weights);
        (Self::mask_time(output, &output_lengths), output_lengths)
    }
}

#[cfg(all(feature = "asr-cubecl-kernels", feature = "burn-wgpu-backend"))]
impl<F, I, BT, C> ZipformerKernelBackend for burn_autodiff::Autodiff<burn_wgpu::Wgpu<F, I, BT>, C>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    fn relative_shift(input: Tensor<Self, 4>, seq_len: usize) -> Tensor<Self, 4> {
        let portable = relative_shift_fallback(input.clone(), seq_len);
        let raw = crate::cubecl_kernels::relative_shift(autodiff_to_inner(input), seq_len);
        attach_autodiff_gradient(inner_to_autodiff(raw), portable)
    }

    fn mask_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::asr_autodiff_kernels::mask_time(input, lengths)
    }

    fn mask_add_time(
        residual: Tensor<Self, 3>,
        update: Tensor<Self, 3>,
        lengths: &[usize],
    ) -> Tensor<Self, 3> {
        crate::asr_autodiff_kernels::residual_add_mask_time(residual, update, lengths)
    }

    fn glu_last_dim(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        let portable = glu_last_dim_fallback(input.clone());
        let raw = crate::cubecl_kernels::glu_last_dim(autodiff_to_inner(input));
        attach_autodiff_gradient(inner_to_autodiff(raw), portable)
    }

    fn attention_mask_4d(
        lengths: &[usize],
        heads: usize,
        query_len: usize,
        key_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 4, Bool> {
        let lengths = lengths_tensor::<burn_wgpu::Wgpu<F, I, BT>>(lengths, device);
        let raw = crate::cubecl_kernels::attention_mask_4d_with_lengths(
            lengths, heads, query_len, key_len,
        );
        inner_bool_to_autodiff(raw)
    }

    fn pairwise_downsample(
        input: Tensor<Self, 3>,
        lengths: &[usize],
        weights: Tensor<Self, 1>,
    ) -> (Tensor<Self, 3>, Vec<usize>) {
        let (portable, output_lengths) =
            pairwise_downsample_fallback(input.clone(), lengths, weights.clone());
        let input_inner = autodiff_to_inner(input);
        let length_tensor =
            lengths_tensor::<burn_wgpu::Wgpu<F, I, BT>>(lengths, &input_inner.device());
        let weights_inner = autodiff_to_inner(weights);
        let raw =
            crate::cubecl_kernels::pairwise_downsample(input_inner, length_tensor, weights_inner);
        let raw = crate::cubecl_kernels::mask_time(raw, &output_lengths);
        (
            attach_autodiff_gradient(inner_to_autodiff(raw), portable),
            output_lengths,
        )
    }
}

#[cfg(all(feature = "burn-cuda-backend", not(feature = "asr-cubecl-kernels")))]
impl<F, I> ZipformerKernelBackend for burn_cuda::Cuda<F, I>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    burn_cuda::Cuda<F, I>: Backend,
{
}

#[cfg(all(feature = "burn-cuda-backend", not(feature = "asr-cubecl-kernels")))]
impl<F, I, C> ZipformerKernelBackend for burn_autodiff::Autodiff<burn_cuda::Cuda<F, I>, C>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    burn_cuda::Cuda<F, I>: Backend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
}

#[cfg(all(feature = "burn-wgpu-backend", not(feature = "asr-cubecl-kernels")))]
impl<F, I, BT> ZipformerKernelBackend for burn_wgpu::Wgpu<F, I, BT>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
    burn_wgpu::Wgpu<F, I, BT>: Backend,
{
}

#[cfg(all(feature = "burn-wgpu-backend", not(feature = "asr-cubecl-kernels")))]
impl<F, I, BT, C> ZipformerKernelBackend for burn_autodiff::Autodiff<burn_wgpu::Wgpu<F, I, BT>, C>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
    burn_wgpu::Wgpu<F, I, BT>: Backend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
}

#[derive(Clone, Debug)]
pub struct ZipformerConfig {
    pub input_dim: usize,
    pub output_downsampling_factor: usize,
    pub downsampling_factor: Vec<usize>,
    pub encoder_dim: Vec<usize>,
    pub num_encoder_layers: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub query_head_dim: Vec<usize>,
    pub value_head_dim: Vec<usize>,
    pub pos_head_dim: Vec<usize>,
    pub feedforward_dim: Vec<usize>,
    pub cnn_module_kernel: Vec<usize>,
    pub pos_dim: usize,
    pub dropout: f64,
}

impl Default for ZipformerConfig {
    fn default() -> Self {
        Self {
            input_dim: 80,
            output_downsampling_factor: 2,
            downsampling_factor: DEFAULT_DOWNSAMPLING.to_vec(),
            encoder_dim: DEFAULT_ENCODER_DIM.to_vec(),
            num_encoder_layers: DEFAULT_NUM_LAYERS.to_vec(),
            num_heads: DEFAULT_NUM_HEADS.to_vec(),
            query_head_dim: vec![32],
            value_head_dim: vec![12],
            pos_head_dim: vec![4],
            feedforward_dim: DEFAULT_FEEDFORWARD_DIM.to_vec(),
            cnn_module_kernel: DEFAULT_CNN_KERNELS.to_vec(),
            pos_dim: 48,
            dropout: 0.1,
        }
    }
}

impl ZipformerConfig {
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            ..Self::default()
        }
    }

    pub fn variant(name: &str) -> Option<Self> {
        match name {
            "xs" => Some(Self {
                encoder_dim: vec![64, 96, 128, 160, 128, 96],
                num_encoder_layers: vec![1, 1, 1, 1, 1, 1],
                num_heads: vec![4, 4, 4, 4, 4, 4],
                query_head_dim: vec![16],
                value_head_dim: vec![8],
                feedforward_dim: vec![192, 256, 384, 512, 384, 256],
                pos_dim: 24,
                ..Self::default()
            }),
            "s" => Some(Self {
                encoder_dim: vec![192, 256, 256, 256, 256, 256],
                num_encoder_layers: vec![2, 2, 2, 2, 2, 2],
                num_heads: vec![4, 4, 4, 8, 4, 4],
                query_head_dim: vec![32],
                value_head_dim: vec![12],
                feedforward_dim: vec![512, 768, 768, 768, 768, 768],
                pos_dim: 48,
                ..Self::default()
            }),
            "m" | "sm" => Some(Self::default()),
            "l" | "ml" => Some(Self {
                encoder_dim: vec![192, 256, 512, 768, 512, 256],
                num_encoder_layers: vec![2, 2, 4, 5, 4, 2],
                num_heads: vec![4, 4, 4, 8, 4, 4],
                query_head_dim: vec![32],
                value_head_dim: vec![12],
                feedforward_dim: vec![512, 768, 1536, 2048, 1536, 768],
                pos_dim: 48,
                ..Self::default()
            }),
            _ => None,
        }
    }

    pub fn model_dim(&self) -> usize {
        self.encoder_dim.iter().copied().max().unwrap_or(0)
    }

    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> ZipformerEncoder<B> {
        assert_eq!(self.downsampling_factor.len(), self.encoder_dim.len());
        assert_eq!(self.num_encoder_layers.len(), self.encoder_dim.len());
        assert_eq!(self.num_heads.len(), self.encoder_dim.len());
        assert_eq!(self.feedforward_dim.len(), self.encoder_dim.len());
        assert_eq!(self.cnn_module_kernel.len(), self.encoder_dim.len());
        let query_head_dim = expand_stack_values(&self.query_head_dim, self.encoder_dim.len());
        let value_head_dim = expand_stack_values(&self.value_head_dim, self.encoder_dim.len());
        let pos_head_dim = expand_stack_values(&self.pos_head_dim, self.encoder_dim.len());

        let stacks = self
            .encoder_dim
            .iter()
            .enumerate()
            .map(|(index, dim)| {
                ZipformerStackConfig {
                    dim: *dim,
                    num_layers: self.num_encoder_layers[index],
                    num_heads: self.num_heads[index],
                    query_head_dim: query_head_dim[index],
                    value_head_dim: value_head_dim[index],
                    pos_head_dim: pos_head_dim[index],
                    feedforward_dim: self.feedforward_dim[index],
                    conv_kernel_size: self.cnn_module_kernel[index],
                    pos_dim: self.pos_dim,
                    dropout: self.dropout,
                    downsample: self.downsampling_factor[index],
                }
                .init(device)
            })
            .collect();

        ZipformerEncoder {
            conv_embed: ConvEmbedConfig::new(self.input_dim, self.encoder_dim[0]).init(device),
            stacks,
            encoder_dim: self.encoder_dim.clone(),
            output_dim: self.model_dim(),
            output_downsample: PairwiseDownsample::new(self.output_downsampling_factor, device),
        }
    }
}

#[derive(Clone, Debug)]
struct BiasNormConfig {
    num_features: usize,
    eps: f64,
}

impl BiasNormConfig {
    fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1.0e-8,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> BiasNorm<B> {
        BiasNorm {
            bias: Initializer::Zeros.init([self.num_features], device),
            log_scale: Initializer::Zeros.init([1], device),
            num_features: self.num_features,
            eps: self.eps,
        }
    }
}

#[derive(Module, Debug)]
struct BiasNorm<B: Backend> {
    bias: Param<Tensor<B, 1>>,
    log_scale: Param<Tensor<B, 1>>,
    num_features: usize,
    eps: f64,
}

impl<B: Backend> BiasNorm<B> {
    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let mut channel_shape = [1; D];
        channel_shape[D - 1] = self.num_features;
        let bias = self.bias.val().reshape(channel_shape);
        let centered = input - bias;
        let rms = (centered.clone() * centered.clone())
            .mean_dim(D - 1)
            .add_scalar(self.eps)
            .sqrt();
        let scale = self.log_scale.val().exp().reshape([1; D]);
        (centered / rms) * scale
    }
}

#[derive(Module, Debug)]
struct ActivationBalancer<B: Backend> {
    count: Param<Tensor<B, 1>>,
    min_positive: f64,
    max_positive: f64,
    sign_gain_factor: f64,
    scale_gain_factor: f64,
    min_abs: f64,
    max_abs: f64,
}

impl<B: Backend> ActivationBalancer<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            count: Initializer::Zeros.init([1], device),
            min_positive: 0.05,
            max_positive: 0.95,
            sign_gain_factor: 0.01,
            scale_gain_factor: 0.02,
            min_abs: 0.2,
            max_abs: 100.0,
        }
    }

    fn with_positive_range(mut self, min_positive: f64, max_positive: f64) -> Self {
        self.min_positive = min_positive;
        self.max_positive = max_positive;
        self
    }

    fn with_abs_range(mut self, min_abs: f64, max_abs: f64) -> Self {
        self.min_abs = min_abs;
        self.max_abs = max_abs;
        self
    }

    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // Burn 0.21.0-pre.3 does not expose a clean PyTorch-style custom backward hook
        // for generic model code. Keep the forward value unchanged while attaching a
        // differentiable penalty to the graph.
        let abs_mean = input.clone().abs().mean();
        let too_small = (Tensor::full([1], self.min_abs, &input.device()) - abs_mean.clone())
            .clamp_min(0.0)
            .powf_scalar(2.0);
        let too_large = (abs_mean - self.max_abs).clamp_min(0.0).powf_scalar(2.0);

        let positive_mean = input.clone().greater_elem(0.0).float().mean();
        let too_negative = (Tensor::full([1], self.min_positive, &input.device())
            - positive_mean.clone())
        .clamp_min(0.0)
        .powf_scalar(2.0);
        let too_positive = (positive_mean - self.max_positive)
            .clamp_min(0.0)
            .powf_scalar(2.0);

        let penalty = (too_small + too_large) * self.scale_gain_factor
            + (too_negative + too_positive) * self.sign_gain_factor;
        add_zero_forward_penalty(input, penalty)
    }
}

#[derive(Module, Debug, Clone)]
struct Whiten {
    whitening_limit: f64,
    grad_scale: f64,
}

impl Whiten {
    fn new() -> Self {
        Self {
            whitening_limit: 5.0,
            grad_scale: 0.01,
        }
    }

    fn attention() -> Self {
        Self {
            whitening_limit: 2.0,
            grad_scale: 0.025,
        }
    }

    fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // This approximates Zipformer's custom whitening backward as a zero-forward
        // regularizer. It is intentionally value-preserving during inference.
        let centered = input.clone() - input.clone().mean_dim(D - 1);
        let variance = (centered.clone() * centered.clone())
            .mean_dim(D - 1)
            .add_scalar(1.0e-6);
        let normalized = centered / variance.sqrt();
        let metric = normalized.powf_scalar(2.0).mean();
        let penalty = (metric - self.whitening_limit)
            .clamp_min(0.0)
            .powf_scalar(2.0)
            * self.grad_scale;
        add_zero_forward_penalty(input, penalty)
    }
}

fn add_zero_forward_penalty<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    penalty: Tensor<B, 1>,
) -> Tensor<B, D> {
    let zero_forward = penalty.clone() - penalty.detach();
    input + zero_forward.reshape([1; D])
}

#[derive(Module, Debug, Clone)]
struct CompactRelPositionalEncoding {
    embed_dim: usize,
}

impl CompactRelPositionalEncoding {
    fn new(embed_dim: usize) -> Self {
        assert!(embed_dim.is_multiple_of(2));
        Self { embed_dim }
    }

    fn forward<B: Backend>(&self, sequence_length: usize, device: &B::Device) -> Tensor<B, 2> {
        let pos_len = sequence_length.saturating_mul(2).saturating_sub(1);
        let half = self.embed_dim / 2;
        let compression_length = (self.embed_dim as f32).sqrt();
        let length_scale = self.embed_dim as f32 / (2.0 * std::f32::consts::PI);
        let mut values = Vec::with_capacity(pos_len * self.embed_dim);
        for offset in 0..pos_len {
            let position = offset as isize - sequence_length as isize + 1;
            let abs_position = position.unsigned_abs() as f32;
            let sign = if position < 0 {
                -1.0
            } else if position > 0 {
                1.0
            } else {
                0.0
            };
            let compressed = compression_length
                * sign
                * ((abs_position + compression_length).ln() - compression_length.ln());
            let angle = (compressed / length_scale).atan();
            for channel in 0..half {
                let frequency = (channel + 1) as f32;
                values.push((angle * frequency).cos());
                values.push((angle * frequency).sin());
            }
            if let Some(last) = values.last_mut() {
                *last = 1.0;
            }
        }
        Tensor::from_data(TensorData::new(values, [pos_len, self.embed_dim]), device)
    }
}

#[derive(Clone, Debug)]
struct MultiHeadAttentionWeightsConfig {
    embed_dim: usize,
    pos_dim: usize,
    num_heads: usize,
    query_head_dim: usize,
    pos_head_dim: usize,
    dropout: f64,
}

impl MultiHeadAttentionWeightsConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttentionWeights<B> {
        MultiHeadAttentionWeights {
            query_proj: LinearConfig::new(self.embed_dim, self.num_heads * self.query_head_dim)
                .init(device),
            key_proj: LinearConfig::new(self.embed_dim, self.num_heads * self.query_head_dim)
                .init(device),
            key_whiten: Whiten::attention(),
            pos_query_proj: LinearConfig::new(self.embed_dim, self.num_heads * self.pos_head_dim)
                .init(device),
            pos_proj: LinearConfig::new(self.pos_dim, self.num_heads * self.pos_head_dim)
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            num_heads: self.num_heads,
            query_head_dim: self.query_head_dim,
            pos_head_dim: self.pos_head_dim,
        }
    }
}

#[derive(Module, Debug)]
struct MultiHeadAttentionWeights<B: Backend> {
    query_proj: Linear<B>,
    key_proj: Linear<B>,
    key_whiten: Whiten,
    pos_query_proj: Linear<B>,
    pos_proj: Linear<B>,
    dropout: Dropout,
    num_heads: usize,
    query_head_dim: usize,
    pos_head_dim: usize,
}

impl<B: ZipformerKernelBackend> MultiHeadAttentionWeights<B> {
    fn forward(
        &self,
        input: Tensor<B, 3>,
        pos_emb: Tensor<B, 2>,
        lengths: &[usize],
    ) -> Tensor<B, 4> {
        let [batch_size, seq_len, _] = input.dims();
        let q = self
            .query_proj
            .forward(input.clone())
            .reshape([batch_size, seq_len, self.num_heads, self.query_head_dim])
            .swap_dims(1, 2);
        let k = self
            .key_whiten
            .forward(self.key_proj.forward(input.clone()))
            .reshape([batch_size, seq_len, self.num_heads, self.query_head_dim])
            .swap_dims(1, 2);
        let p_query = self
            .pos_query_proj
            .forward(input)
            .reshape([batch_size, seq_len, self.num_heads, self.pos_head_dim])
            .swap_dims(1, 2);

        let content_scores = q.matmul(k.swap_dims(2, 3)) / (self.query_head_dim as f64).sqrt();
        let rel = self
            .pos_proj
            .forward(pos_emb.unsqueeze_dim::<3>(0))
            .reshape([1, seq_len * 2 - 1, self.num_heads, self.pos_head_dim])
            .swap_dims(1, 2);
        let pos_scores = B::relative_shift(p_query.matmul(rel.swap_dims(2, 3)), seq_len);
        let mut scores = content_scores + pos_scores;

        let key_mask =
            B::attention_mask_4d(lengths, self.num_heads, seq_len, seq_len, &scores.device());
        let negative = Tensor::full(
            [batch_size, self.num_heads, seq_len, seq_len],
            -1.0e4,
            &scores.device(),
        );
        scores = scores.mask_where(key_mask.bool_not(), negative);
        self.dropout.forward(softmax(scores, 3))
    }
}

#[derive(Clone, Debug)]
struct SelfAttentionConfig {
    embed_dim: usize,
    num_heads: usize,
    value_head_dim: usize,
    dropout: f64,
}

impl SelfAttentionConfig {
    fn new(embed_dim: usize, num_heads: usize, value_head_dim: usize, dropout: f64) -> Self {
        Self {
            embed_dim,
            num_heads,
            value_head_dim,
            dropout,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> SelfAttention<B> {
        SelfAttention {
            value_proj: LinearConfig::new(self.embed_dim, self.num_heads * self.value_head_dim)
                .init(device),
            value_whiten: Whiten::attention(),
            output_proj: LinearConfig::new(self.num_heads * self.value_head_dim, self.embed_dim)
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            num_heads: self.num_heads,
            value_head_dim: self.value_head_dim,
        }
    }
}

#[derive(Module, Debug)]
struct SelfAttention<B: Backend> {
    value_proj: Linear<B>,
    value_whiten: Whiten,
    output_proj: Linear<B>,
    dropout: Dropout,
    num_heads: usize,
    value_head_dim: usize,
}

impl<B: Backend> SelfAttention<B> {
    fn forward(&self, input: Tensor<B, 3>, attn_weights: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = input.dims();
        let value = self
            .value_whiten
            .forward(self.value_proj.forward(input))
            .reshape([batch_size, seq_len, self.num_heads, self.value_head_dim])
            .swap_dims(1, 2);
        let output = attn_weights.matmul(value).swap_dims(1, 2).reshape([
            batch_size,
            seq_len,
            self.num_heads * self.value_head_dim,
        ]);
        self.dropout.forward(self.output_proj.forward(output))
    }
}

#[derive(Clone, Debug)]
struct NonLinearAttentionConfig {
    embed_dim: usize,
    hidden_dim: usize,
    dropout: f64,
}

impl NonLinearAttentionConfig {
    fn new(embed_dim: usize, hidden_dim: usize, dropout: f64) -> Self {
        Self {
            embed_dim,
            hidden_dim,
            dropout,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> NonLinearAttention<B> {
        NonLinearAttention {
            proj: LinearConfig::new(self.embed_dim, self.hidden_dim * 3).init(device),
            output_proj: LinearConfig::new(self.hidden_dim, self.embed_dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            hidden_dim: self.hidden_dim,
        }
    }
}

#[derive(Module, Debug)]
struct NonLinearAttention<B: Backend> {
    proj: Linear<B>,
    output_proj: Linear<B>,
    dropout: Dropout,
    hidden_dim: usize,
}

impl<B: Backend> NonLinearAttention<B> {
    fn forward(&self, input: Tensor<B, 3>, attn_weights: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = input.dims();
        let projected = self.proj.forward(input);
        let mut chunks = projected.chunk(3, 2);
        let c = chunks.pop().unwrap();
        let b = chunks.pop().unwrap();
        let a = chunks.pop().unwrap();
        let values = tanh(b) * c;
        let weights = attn_weights
            .slice_dim(1, 0..1)
            .reshape([batch_size, seq_len, seq_len]);
        let attended = weights.matmul(values);
        let output = self.output_proj.forward((a * attended).reshape([
            batch_size,
            seq_len,
            self.hidden_dim,
        ]));
        self.dropout.forward(output)
    }
}

fn relative_shift_fallback<B: Backend>(input: Tensor<B, 4>, seq_len: usize) -> Tensor<B, 4> {
    let [batch_size, n_heads, _, pos_len] = input.dims();
    let padded = input.pad([(0, 0), (0, 1)], PadMode::Constant(0.0));
    padded
        .reshape([batch_size, n_heads, pos_len + 1, seq_len])
        .slice_dim(2, 1..pos_len + 1)
        .reshape([batch_size, n_heads, seq_len, pos_len])
        .slice_dim(3, 0..seq_len)
}

#[derive(Module, Debug)]
struct BypassModule<B: Backend> {
    scale: Param<Tensor<B, 1>>,
    num_features: usize,
    min_value: f64,
    max_value: f64,
}

impl<B: Backend> BypassModule<B> {
    fn new(num_features: usize, device: &B::Device) -> Self {
        Self {
            scale: Initializer::Ones.init([num_features], device),
            num_features,
            min_value: 0.2,
            max_value: 1.0,
        }
    }

    fn forward(&self, residual: Tensor<B, 3>, update: Tensor<B, 3>) -> Tensor<B, 3> {
        let scale = self
            .scale
            .val()
            .clamp(self.min_value, self.max_value)
            .reshape([1, 1, self.num_features]);
        residual.clone() + (update - residual) * scale
    }
}

fn swoosh_l<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    input.clone() * sigmoid(input - 4.0)
}

fn swoosh_r<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    input.clone() * sigmoid(input - 1.0)
}

#[derive(Clone, Debug)]
struct ConvEmbedConfig {
    input_dim: usize,
    output_dim: usize,
}

impl ConvEmbedConfig {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> ConvEmbed<B> {
        let freq_dim = ceil_divide(ceil_divide(ceil_divide(self.input_dim, 2), 2), 2);
        ConvEmbed {
            conv1: Conv2dConfig::new([1, 8], [3, 3])
                .with_stride([1, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv2: Conv2dConfig::new([8, 32], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv3: Conv2dConfig::new([32, 128], [3, 3])
                .with_stride([1, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            convnext_depthwise: Conv2dConfig::new([128, 128], [7, 7])
                .with_groups(128)
                .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
                .init(device),
            convnext_in: LinearConfig::new(128, 384).init(device),
            convnext_out: LinearConfig::new(384, 128).init(device),
            output_projection: LinearConfig::new(128 * freq_dim, self.output_dim).init(device),
            output_norm: BiasNormConfig::new(self.output_dim).init(device),
            conv1_balancer: ActivationBalancer::new(device),
            conv2_balancer: ActivationBalancer::new(device),
            conv3_balancer: ActivationBalancer::new(device),
        }
    }
}

#[derive(Module, Debug)]
struct ConvEmbed<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    convnext_depthwise: Conv2d<B>,
    convnext_in: Linear<B>,
    convnext_out: Linear<B>,
    output_projection: Linear<B>,
    output_norm: BiasNorm<B>,
    conv1_balancer: ActivationBalancer<B>,
    conv2_balancer: ActivationBalancer<B>,
    conv3_balancer: ActivationBalancer<B>,
}

impl<B: ZipformerKernelBackend> ConvEmbed<B> {
    fn forward(&self, input: Tensor<B, 3>, lengths: Vec<usize>) -> (Tensor<B, 3>, Vec<usize>) {
        let mut output = input.unsqueeze_dim::<4>(1);
        output = swoosh_r(self.conv1_balancer.forward(self.conv1.forward(output)));
        output = swoosh_r(self.conv2_balancer.forward(self.conv2.forward(output)));
        output = swoosh_r(self.conv3_balancer.forward(self.conv3.forward(output)));

        let residual = output.clone();
        output = self.convnext_depthwise.forward(output);
        output = output.swap_dims(1, 3);
        output = silu(self.convnext_in.forward(output));
        output = self.convnext_out.forward(output).swap_dims(1, 3);
        output = output + residual;

        let [batch_size, channels, time, freq] = output.dims();
        let output = output
            .swap_dims(1, 2)
            .reshape([batch_size, time, channels * freq]);
        let output = self
            .output_norm
            .forward(self.output_projection.forward(output));
        let lengths: Vec<usize> = lengths
            .into_iter()
            .map(|length| {
                conv_out_length(
                    conv_out_length(conv_out_length(length, 3, 1, 1), 3, 2, 1),
                    3,
                    1,
                    1,
                )
                .min(time)
            })
            .collect();
        (B::mask_time(output, &lengths), lengths)
    }
}

#[derive(Clone, Debug)]
struct ZipformerStackConfig {
    dim: usize,
    num_layers: usize,
    num_heads: usize,
    query_head_dim: usize,
    value_head_dim: usize,
    pos_head_dim: usize,
    feedforward_dim: usize,
    conv_kernel_size: usize,
    pos_dim: usize,
    dropout: f64,
    downsample: usize,
}

impl ZipformerStackConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ZipformerStack<B> {
        ZipformerStack {
            downsample: PairwiseDownsample::new(self.downsample, device),
            upsample: PairwiseUpsample::new(self.downsample),
            output_bypass: (self.downsample > 1).then(|| BypassModule::new(self.dim, device)),
            positional_encoding: CompactRelPositionalEncoding::new(self.pos_dim),
            blocks: (0..self.num_layers)
                .map(|_| {
                    ZipformerBlockConfig {
                        dim: self.dim,
                        heads: self.num_heads,
                        query_head_dim: self.query_head_dim,
                        value_head_dim: self.value_head_dim,
                        pos_head_dim: self.pos_head_dim,
                        pos_dim: self.pos_dim,
                        feedforward_dim: self.feedforward_dim,
                        conv_kernel_size: self.conv_kernel_size,
                        dropout: self.dropout,
                    }
                    .init(device)
                })
                .collect(),
        }
    }
}

#[derive(Module, Debug)]
struct ZipformerStack<B: Backend> {
    downsample: PairwiseDownsample<B>,
    upsample: PairwiseUpsample,
    output_bypass: Option<BypassModule<B>>,
    positional_encoding: CompactRelPositionalEncoding,
    blocks: Vec<ZipformerBlock<B>>,
}

impl<B: ZipformerKernelBackend> ZipformerStack<B> {
    fn forward(&self, input: Tensor<B, 3>, lengths: &[usize]) -> Tensor<B, 3> {
        let target_len = input.dims()[1];
        let residual = input.clone();
        let (mut output, down_lengths) = self.downsample.forward(input, lengths);
        let pos_emb = self
            .positional_encoding
            .forward::<B>(output.dims()[1], &output.device());
        for block in self.blocks.iter() {
            output = block.forward(output, pos_emb.clone(), &down_lengths);
        }
        output = self.upsample.forward(output, target_len);
        let output = if let Some(output_bypass) = &self.output_bypass {
            output_bypass.forward(residual, output)
        } else {
            output
        };
        B::mask_time(output, lengths)
    }
}

#[derive(Clone, Debug)]
struct ZipformerBlockConfig {
    dim: usize,
    heads: usize,
    query_head_dim: usize,
    value_head_dim: usize,
    pos_head_dim: usize,
    pos_dim: usize,
    feedforward_dim: usize,
    conv_kernel_size: usize,
    dropout: f64,
}

impl ZipformerBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ZipformerBlock<B> {
        ZipformerBlock {
            attention_weights: MultiHeadAttentionWeightsConfig {
                embed_dim: self.dim,
                pos_dim: self.pos_dim,
                num_heads: self.heads,
                query_head_dim: self.query_head_dim,
                pos_head_dim: self.pos_head_dim,
                dropout: self.dropout,
            }
            .init(device),
            feed_forward1: FeedForwardConfig::new(
                self.dim,
                (self.feedforward_dim * 3) / 4,
                self.dropout,
            )
            .init(device),
            non_linear_attention: NonLinearAttentionConfig::new(
                self.dim,
                (self.dim * 3) / 4,
                self.dropout,
            )
            .init(device),
            self_attention1: SelfAttentionConfig::new(
                self.dim,
                self.heads,
                self.value_head_dim,
                self.dropout,
            )
            .init(device),
            conv1: ZipformerConvModuleConfig::new(self.dim, self.conv_kernel_size, self.dropout)
                .init(device),
            feed_forward2: FeedForwardConfig::new(self.dim, self.feedforward_dim, self.dropout)
                .init(device),
            mid_bypass: BypassModule::new(self.dim, device),
            self_attention2: SelfAttentionConfig::new(
                self.dim,
                self.heads,
                self.value_head_dim,
                self.dropout,
            )
            .init(device),
            conv2: ZipformerConvModuleConfig::new(self.dim, self.conv_kernel_size, self.dropout)
                .init(device),
            feed_forward3: FeedForwardConfig::new(
                self.dim,
                (self.feedforward_dim * 5) / 4,
                self.dropout,
            )
            .init(device),
            block_balancer: ActivationBalancer::new(device)
                .with_positive_range(0.45, 0.55)
                .with_abs_range(0.2, 6.0),
            output_norm: BiasNormConfig::new(self.dim).init(device),
            output_bypass: BypassModule::new(self.dim, device),
            whiten: Whiten::new(),
        }
    }
}

#[derive(Module, Debug)]
struct ZipformerBlock<B: Backend> {
    attention_weights: MultiHeadAttentionWeights<B>,
    feed_forward1: FeedForward<B>,
    non_linear_attention: NonLinearAttention<B>,
    self_attention1: SelfAttention<B>,
    conv1: ZipformerConvModule<B>,
    feed_forward2: FeedForward<B>,
    mid_bypass: BypassModule<B>,
    self_attention2: SelfAttention<B>,
    conv2: ZipformerConvModule<B>,
    feed_forward3: FeedForward<B>,
    block_balancer: ActivationBalancer<B>,
    output_norm: BiasNorm<B>,
    output_bypass: BypassModule<B>,
    whiten: Whiten,
}

impl<B: ZipformerKernelBackend> ZipformerBlock<B> {
    fn forward(
        &self,
        input: Tensor<B, 3>,
        pos_emb: Tensor<B, 2>,
        lengths: &[usize],
    ) -> Tensor<B, 3> {
        let attn_weights = self
            .attention_weights
            .forward(input.clone(), pos_emb, lengths);

        let residual = input.clone();
        let mut output =
            B::mask_add_time(input.clone(), self.feed_forward1.forward(input), lengths);
        output = B::mask_add_time(
            output.clone(),
            self.non_linear_attention
                .forward(output.clone(), attn_weights.clone()),
            lengths,
        );
        output = B::mask_add_time(
            output.clone(),
            self.self_attention1
                .forward(output.clone(), attn_weights.clone()),
            lengths,
        );
        output = B::mask_add_time(output.clone(), self.conv1.forward(output, lengths), lengths);
        output = B::mask_add_time(output.clone(), self.feed_forward2.forward(output), lengths);
        output = B::mask_time(self.mid_bypass.forward(residual.clone(), output), lengths);

        output = B::mask_add_time(
            output.clone(),
            self.self_attention2.forward(output.clone(), attn_weights),
            lengths,
        );
        output = B::mask_add_time(output.clone(), self.conv2.forward(output, lengths), lengths);
        output = output.clone() + self.feed_forward3.forward(output);
        output = self
            .output_norm
            .forward(self.block_balancer.forward(output));
        self.whiten.forward(B::mask_time(
            self.output_bypass.forward(residual, output),
            lengths,
        ))
    }
}

#[derive(Clone, Debug)]
struct FeedForwardConfig {
    dim: usize,
    hidden: usize,
    dropout: f64,
}

impl FeedForwardConfig {
    fn new(dim: usize, hidden: usize, dropout: f64) -> Self {
        Self {
            dim,
            hidden,
            dropout,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        FeedForward {
            linear_in: LinearConfig::new(self.dim, self.hidden).init(device),
            balancer: ActivationBalancer::new(device).with_abs_range(0.2, 10.0),
            linear_out: LinearConfig::new(self.hidden, self.dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            out_dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
struct FeedForward<B: Backend> {
    linear_in: Linear<B>,
    balancer: ActivationBalancer<B>,
    linear_out: Linear<B>,
    dropout: Dropout,
    out_dropout: Dropout,
}

impl<B: Backend> FeedForward<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let output = self.linear_in.forward(input);
        let output = swoosh_l(self.balancer.forward(output));
        let output = self.dropout.forward(output);
        self.out_dropout.forward(self.linear_out.forward(output))
    }
}

#[derive(Clone, Debug)]
struct ZipformerConvModuleConfig {
    dim: usize,
    kernel_size: usize,
    dropout: f64,
}

impl ZipformerConvModuleConfig {
    fn new(dim: usize, kernel_size: usize, dropout: f64) -> Self {
        Self {
            dim,
            kernel_size,
            dropout,
        }
    }

    fn init<B: Backend>(&self, device: &B::Device) -> ZipformerConvModule<B> {
        ZipformerConvModule {
            input_proj: LinearConfig::new(self.dim, self.dim * 2).init(device),
            input_balancer: ActivationBalancer::new(device)
                .with_positive_range(0.05, 1.0)
                .with_abs_range(0.2, 10.0),
            depthwise: Conv1dConfig::new(self.dim, self.dim, self.kernel_size)
                .with_groups(self.dim)
                .with_padding(PaddingConfig1d::Explicit(
                    self.kernel_size / 2,
                    self.kernel_size / 2,
                ))
                .init(device),
            depthwise_balancer: ActivationBalancer::new(device)
                .with_positive_range(0.05, 1.0)
                .with_abs_range(0.2, 20.0),
            output_proj: LinearConfig::new(self.dim, self.dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
struct ZipformerConvModule<B: Backend> {
    input_proj: Linear<B>,
    input_balancer: ActivationBalancer<B>,
    depthwise: Conv1d<B>,
    depthwise_balancer: ActivationBalancer<B>,
    output_proj: Linear<B>,
    dropout: Dropout,
}

impl<B: ZipformerKernelBackend> ZipformerConvModule<B> {
    fn forward(&self, input: Tensor<B, 3>, lengths: &[usize]) -> Tensor<B, 3> {
        let [batch_size, seq_len, dim] = input.dims();
        let output = self.input_balancer.forward(self.input_proj.forward(input));
        let output = B::glu_last_dim(output).swap_dims(1, 2);
        let output = silu(self.depthwise.forward(output));
        let output = self.depthwise_balancer.forward(output.swap_dims(1, 2));
        let output = self
            .dropout
            .forward(self.output_proj.forward(swoosh_r(output)));
        let output = output.reshape([batch_size, seq_len, dim]);
        B::mask_time(output, lengths)
    }
}

#[derive(Module, Debug)]
struct Downsample<B: Backend> {
    weights: Param<Tensor<B, 1>>,
    factor: usize,
}

impl<B: Backend> Downsample<B> {
    fn new(factor: usize, device: &B::Device) -> Self {
        assert!(factor == 1 || factor == 2);
        Self {
            weights: Initializer::Zeros.init([factor], device),
            factor,
        }
    }
}

impl<B: ZipformerKernelBackend> Downsample<B> {
    fn forward(&self, input: Tensor<B, 3>, lengths: &[usize]) -> (Tensor<B, 3>, Vec<usize>) {
        if self.factor == 1 {
            return (input, lengths.to_vec());
        }

        B::pairwise_downsample(input, lengths, self.weights.val())
    }
}

#[derive(Module, Debug)]
struct PairwiseDownsample<B: Backend> {
    stages: Vec<Downsample<B>>,
    factor: usize,
}

impl<B: Backend> PairwiseDownsample<B> {
    fn new(factor: usize, device: &B::Device) -> Self {
        assert!(factor >= 1 && factor.is_power_of_two());
        let levels = factor.ilog2() as usize;
        Self {
            stages: (0..levels).map(|_| Downsample::new(2, device)).collect(),
            factor,
        }
    }
}

impl<B: ZipformerKernelBackend> PairwiseDownsample<B> {
    fn forward(&self, input: Tensor<B, 3>, lengths: &[usize]) -> (Tensor<B, 3>, Vec<usize>) {
        let mut output = input;
        let mut output_lengths = lengths.to_vec();
        for stage in &self.stages {
            let (next, next_lengths) = stage.forward(output, &output_lengths);
            output = next;
            output_lengths = next_lengths;
        }
        (output, output_lengths)
    }
}

#[derive(Module, Debug, Clone)]
struct PairwiseUpsample {
    factor: usize,
}

impl PairwiseUpsample {
    fn new(factor: usize) -> Self {
        assert!(factor >= 1 && factor.is_power_of_two());
        Self { factor }
    }

    fn forward<B: Backend>(&self, input: Tensor<B, 3>, target_len: usize) -> Tensor<B, 3> {
        let mut output = input.repeat_dim(1, self.factor);
        let current_len = output.dims()[1];
        output = if current_len < target_len {
            output.pad([(0, target_len - current_len), (0, 0)], PadMode::Edge)
        } else {
            output.slice_dim(1, 0..target_len)
        };
        output
    }
}

#[derive(Module, Debug)]
pub struct ZipformerEncoder<B: Backend> {
    conv_embed: ConvEmbed<B>,
    stacks: Vec<ZipformerStack<B>>,
    encoder_dim: Vec<usize>,
    output_dim: usize,
    output_downsample: PairwiseDownsample<B>,
}

impl<B: ZipformerKernelBackend> ZipformerEncoder<B> {
    pub fn forward(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        let (mut output, lengths) = self.conv_embed.forward(features, lengths);
        let mut outputs = Vec::with_capacity(self.stacks.len());
        for (index, stack) in self.stacks.iter().enumerate() {
            output = convert_num_channels(output, self.encoder_dim[index]);
            output = B::mask_time(output, &lengths);
            output = stack.forward(output, &lengths);
            outputs.push(output.clone());
        }

        let mut output = Tensor::zeros(
            [
                outputs.last().unwrap().dims()[0],
                outputs.last().unwrap().dims()[1],
                self.output_dim,
            ],
            &outputs.last().unwrap().device(),
        );
        for piece in outputs {
            output = add_channel_prefix(output, piece);
        }
        let (output, lengths) = self.output_downsample.forward(output, &lengths);
        (B::mask_time(output, &lengths), lengths)
    }
}

#[derive(Clone, Debug)]
pub struct ZipformerCtcConfig {
    pub encoder: ZipformerConfig,
    pub vocab_size: usize,
}

impl ZipformerCtcConfig {
    pub fn new(input_dim: usize, vocab_size: usize) -> Self {
        Self {
            encoder: ZipformerConfig::new(input_dim),
            vocab_size,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> ZipformerCtc<B> {
        ZipformerCtc {
            encoder: self.encoder.init(device),
            classifier: LinearConfig::new(self.encoder.model_dim(), self.vocab_size).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct ZipformerCtc<B: Backend> {
    encoder: ZipformerEncoder<B>,
    classifier: Linear<B>,
}

impl<B: ZipformerKernelBackend> ZipformerCtc<B> {
    pub fn forward(&self, features: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = features.dims();
        self.forward_with_lengths(features, vec![seq_len; batch_size])
            .0
    }

    pub fn forward_with_lengths(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        let (encoded, lengths) = self.encoder.forward(features, lengths);
        (self.classifier.forward(encoded), lengths)
    }
}

fn convert_num_channels<B: Backend>(input: Tensor<B, 3>, channels: usize) -> Tensor<B, 3> {
    let [batch_size, seq_len, current] = input.dims();
    if current == channels {
        input
    } else if current > channels {
        input.slice_dim(2, 0..channels)
    } else {
        let device = input.device();
        Tensor::cat(
            vec![
                input,
                Tensor::zeros([batch_size, seq_len, channels - current], &device),
            ],
            2,
        )
    }
}

fn add_channel_prefix<B: Backend>(target: Tensor<B, 3>, piece: Tensor<B, 3>) -> Tensor<B, 3> {
    let channels = piece.dims()[2];
    let prefix = target.clone().slice_dim(2, 0..channels) + piece;
    let suffix = target.slice_dim(2, channels..);
    if suffix.dims()[2] == 0 {
        prefix
    } else {
        Tensor::cat(vec![prefix, suffix], 2)
    }
}

fn conv_out_length(length: usize, kernel_size: usize, stride: usize, padding: usize) -> usize {
    ((length + 2 * padding - kernel_size) / stride) + 1
}

fn ceil_divide(value: usize, factor: usize) -> usize {
    if value == 0 {
        0
    } else {
        value.div_ceil(factor)
    }
}

fn downsample_lengths(lengths: &[usize], factor: usize) -> Vec<usize> {
    lengths
        .iter()
        .map(|length| ceil_divide(*length, factor))
        .collect()
}

#[cfg(feature = "asr-cubecl-kernels")]
fn lengths_tensor<B: Backend>(lengths: &[usize], device: &B::Device) -> Tensor<B, 1, Int> {
    let values = lengths
        .iter()
        .map(|length| i32::try_from(*length).expect("Zipformer lengths must fit in i32"))
        .collect::<Vec<_>>();
    Tensor::from_ints(values.as_slice(), device)
}

fn expand_stack_values(values: &[usize], num_stacks: usize) -> Vec<usize> {
    match values.len() {
        1 => vec![values[0]; num_stacks],
        len if len == num_stacks => values.to_vec(),
        len => panic!("Zipformer stack value must have length 1 or {num_stacks}, got {len}."),
    }
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

fn attention_key_mask_4d_fallback<B: Backend>(
    lengths: &[usize],
    heads: usize,
    query_len: usize,
    key_len: usize,
    device: &B::Device,
) -> Tensor<B, 4, Bool> {
    let mut values = Vec::with_capacity(lengths.len() * heads * query_len * key_len);
    for length in lengths {
        for _ in 0..heads {
            for _ in 0..query_len {
                for key in 0..key_len {
                    values.push(key < *length);
                }
            }
        }
    }
    Tensor::from_data(
        TensorData::new(values, [lengths.len(), heads, query_len, key_len]),
        device,
    )
}

fn padded_sequence_mask<B: Backend>(
    lengths: &[usize],
    original_len: usize,
    padded_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let mut values = Vec::with_capacity(lengths.len() * padded_len);
    for length in lengths {
        for index in 0..padded_len {
            let source_index = index.min(original_len.saturating_sub(1));
            values.push(source_index < *length);
        }
    }
    Tensor::from_data(TensorData::new(values, [lengths.len(), padded_len]), device)
}

fn mask_time_fallback<B: Backend>(input: Tensor<B, 3>, lengths: &[usize]) -> Tensor<B, 3> {
    let seq_len = input.dims()[1];
    let mask = sequence_mask::<B>(lengths, seq_len, &input.device())
        .float()
        .unsqueeze_dim::<3>(2);
    input * mask
}

fn glu_last_dim_fallback<B: Backend>(input: Tensor<B, 3>) -> Tensor<B, 3> {
    let mut chunks = input.chunk(2, 2);
    let gate = chunks.remove(1);
    let value = chunks.remove(0);
    value * sigmoid(gate)
}

fn pairwise_downsample_fallback<B: Backend>(
    input: Tensor<B, 3>,
    lengths: &[usize],
    weights: Tensor<B, 1>,
) -> (Tensor<B, 3>, Vec<usize>) {
    let [batch_size, seq_len, dim] = input.dims();
    let output_len = ceil_divide(seq_len, 2);
    let padded_len = output_len * 2;
    let pad = padded_len - seq_len;
    let output = if pad > 0 {
        input.pad([(0, pad), (0, 0)], PadMode::Edge)
    } else {
        input
    };

    let window = output.reshape([batch_size, output_len, 2, dim]);
    let weights = softmax(weights, 0).reshape([1, 1, 2, 1]);
    let mask = padded_sequence_mask::<B>(lengths, seq_len, padded_len, &window.device())
        .float()
        .reshape([batch_size, output_len, 2, 1]);
    let masked_weights = weights * mask;
    let denom = masked_weights
        .clone()
        .sum_dim(2)
        .reshape([batch_size, output_len, 1])
        .clamp_min(1.0e-8);
    let output = (window * masked_weights)
        .sum_dim(2)
        .reshape([batch_size, output_len, dim])
        / denom;
    let lengths = downsample_lengths(lengths, 2);
    (mask_time_fallback(output, &lengths), lengths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn zipformer_ctc_downsamples_to_output_factor() {
        let device = Default::default();
        let mut encoder = ZipformerConfig::variant("xs").unwrap();
        encoder.input_dim = 80;
        encoder.encoder_dim = vec![16, 16];
        encoder.num_encoder_layers = vec![1, 1];
        encoder.num_heads = vec![4, 4];
        encoder.feedforward_dim = vec![32, 32];
        encoder.cnn_module_kernel = vec![15, 15];
        encoder.downsampling_factor = vec![1, 2];
        let model = ZipformerCtcConfig {
            encoder,
            vocab_size: 32,
        }
        .init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 16, 80], &device);

        let (output, lengths) = model.forward_with_lengths(input, vec![16, 12]);

        assert_eq!(output.dims(), [2, 4, 32]);
        assert_eq!(lengths, vec![4, 3]);
    }

    #[test]
    fn zipformer_regularizers_preserve_forward_values() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.0, 1.0, -2.0, 3.0], [1, 2, 2]),
            &device,
        );

        let balanced = ActivationBalancer::new().forward(input.clone());
        let whitened = Whiten::new().forward(input.clone());

        assert!(balanced.all_close(input.clone(), Some(1e-6), Some(1e-6)));
        assert!(whitened.all_close(input, Some(1e-6), Some(1e-6)));
    }
}
