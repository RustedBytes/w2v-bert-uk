use burn::module::{Initializer, Module, Param, RunningState};
use burn::tensor::activation::{relu, silu, softmax};
#[cfg(feature = "asr-cubecl-kernels")]
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::ops::PadMode;
use burn::tensor::{Bool, Tensor, TensorData, backend::Backend};
use burn_nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn_nn::{
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig1d,
};

pub mod transcribe;

const DEFAULT_MAX_POSITION: usize = 5000;

pub trait SqueezeformerKernelBackend: Backend + Sized {
    fn relative_shift(input: Tensor<Self, 4>, seq_len: usize) -> Tensor<Self, 4> {
        relative_shift_fallback(input, seq_len)
    }

    fn mask_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        apply_time_mask_fallback(input, lengths)
    }

    fn mask_channel_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        apply_channel_time_mask_fallback(input, lengths)
    }

    fn attention_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 3, Bool> {
        attention_mask_fallback(lengths, max_len, device)
    }
}

impl SqueezeformerKernelBackend for burn_ndarray::NdArray<f32> {}

impl<C> SqueezeformerKernelBackend for burn_autodiff::Autodiff<burn_ndarray::NdArray<f32>, C> where
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
impl<F, I> SqueezeformerKernelBackend for burn_cuda::Cuda<F, I>
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

    fn mask_channel_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::cubecl_kernels::mask_channel_time(input, lengths)
    }

    fn attention_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 3, Bool> {
        crate::cubecl_kernels::attention_mask(lengths, max_len, max_len, device)
    }
}

#[cfg(all(feature = "asr-cubecl-kernels", feature = "burn-cuda-backend"))]
impl<F, I, C> SqueezeformerKernelBackend for burn_autodiff::Autodiff<burn_cuda::Cuda<F, I>, C>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    fn relative_shift(input: Tensor<Self, 4>, seq_len: usize) -> Tensor<Self, 4> {
        crate::asr_autodiff_kernels::relative_shift(input, seq_len)
    }

    fn mask_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::asr_autodiff_kernels::mask_time(input, lengths)
    }

    fn mask_channel_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::asr_autodiff_kernels::mask_channel_time(input, lengths)
    }

    fn attention_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 3, Bool> {
        let raw =
            crate::cubecl_kernels::attention_mask::<_, F, I, u8>(lengths, max_len, max_len, device);
        inner_bool_to_autodiff(raw)
    }
}

#[cfg(all(feature = "asr-cubecl-kernels", feature = "burn-wgpu-backend"))]
impl<F, I, BT> SqueezeformerKernelBackend for burn_wgpu::Wgpu<F, I, BT>
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

    fn mask_channel_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::cubecl_kernels::mask_channel_time(input, lengths)
    }

    fn attention_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 3, Bool> {
        crate::cubecl_kernels::attention_mask(lengths, max_len, max_len, device)
    }
}

#[cfg(all(feature = "asr-cubecl-kernels", feature = "burn-wgpu-backend"))]
impl<F, I, BT, C> SqueezeformerKernelBackend
    for burn_autodiff::Autodiff<burn_wgpu::Wgpu<F, I, BT>, C>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    fn relative_shift(input: Tensor<Self, 4>, seq_len: usize) -> Tensor<Self, 4> {
        crate::asr_autodiff_kernels::relative_shift(input, seq_len)
    }

    fn mask_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::asr_autodiff_kernels::mask_time(input, lengths)
    }

    fn mask_channel_time(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::asr_autodiff_kernels::mask_channel_time(input, lengths)
    }

    fn attention_mask(
        lengths: &[usize],
        max_len: usize,
        device: &Self::Device,
    ) -> Tensor<Self, 3, Bool> {
        let raw =
            crate::cubecl_kernels::attention_mask::<_, F, I, BT>(lengths, max_len, max_len, device);
        inner_bool_to_autodiff(raw)
    }
}

#[cfg(all(feature = "burn-cuda-backend", not(feature = "asr-cubecl-kernels")))]
impl<F, I> SqueezeformerKernelBackend for burn_cuda::Cuda<F, I>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    burn_cuda::Cuda<F, I>: Backend,
{
}

#[cfg(all(feature = "burn-cuda-backend", not(feature = "asr-cubecl-kernels")))]
impl<F, I, C> SqueezeformerKernelBackend for burn_autodiff::Autodiff<burn_cuda::Cuda<F, I>, C>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    burn_cuda::Cuda<F, I>: Backend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
}

#[cfg(all(feature = "burn-wgpu-backend", not(feature = "asr-cubecl-kernels")))]
impl<F, I, BT> SqueezeformerKernelBackend for burn_wgpu::Wgpu<F, I, BT>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
    burn_wgpu::Wgpu<F, I, BT>: Backend,
{
}

#[cfg(all(feature = "burn-wgpu-backend", not(feature = "asr-cubecl-kernels")))]
impl<F, I, BT, C> SqueezeformerKernelBackend
    for burn_autodiff::Autodiff<burn_wgpu::Wgpu<F, I, BT>, C>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
    burn_wgpu::Wgpu<F, I, BT>: Backend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
}

#[derive(Clone, Debug)]
pub struct ScaleBiasLayerConfig {
    pub dim: usize,
}

impl ScaleBiasLayerConfig {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> ScaleBiasLayer<B> {
        ScaleBiasLayer {
            scale: Initializer::Ones.init([self.dim], device),
            bias: Initializer::Zeros.init([self.dim], device),
        }
    }
}

#[derive(Module, Debug)]
pub struct ScaleBiasLayer<B: Backend> {
    scale: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> ScaleBiasLayer<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, _, dim] = input.dims();
        input * self.scale.val().reshape([1, 1, dim]) + self.bias.val().reshape([1, 1, dim])
    }
}

#[derive(Clone, Debug)]
pub struct RelativePositionalEncoding {
    dim: usize,
    max_length: usize,
}

impl RelativePositionalEncoding {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            max_length: DEFAULT_MAX_POSITION,
        }
    }

    pub fn forward<B: Backend>(&self, length: usize, device: &B::Device) -> Tensor<B, 3> {
        let center = self.max_length - 1;
        let start = center - length + 1;
        let end = center + length;
        let mut values = Vec::with_capacity((end - start) * self.dim);

        for index in start..end {
            let position = center as isize - index as isize;
            for channel in 0..self.dim {
                let div_term =
                    (-(10000.0f32).ln() * (2 * (channel / 2)) as f32 / self.dim as f32).exp();
                let value = position as f32 * div_term;
                values.push(if channel % 2 == 0 {
                    value.sin()
                } else {
                    value.cos()
                });
            }
        }

        Tensor::from_data(
            TensorData::new(values, [1, 2 * length - 1, self.dim]),
            device,
        )
    }
}

#[derive(Clone, Debug)]
pub struct RelPositionMultiHeadAttentionConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub dropout: f64,
}

impl RelPositionMultiHeadAttentionConfig {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        Self {
            d_model,
            n_heads,
            dropout: 0.1,
        }
    }

    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> RelPositionMultiHeadAttention<B> {
        assert!(
            self.d_model % self.n_heads == 0,
            "d_model must be divisible by n_heads"
        );

        let head_dim = self.d_model / self.n_heads;
        RelPositionMultiHeadAttention {
            query_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            key_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            value_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            pos_proj: LinearConfig::new(self.d_model, self.d_model)
                .with_bias(false)
                .init(device),
            out_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            pos_bias_u: Initializer::Zeros.init([self.n_heads, head_dim], device),
            pos_bias_v: Initializer::Zeros.init([self.n_heads, head_dim], device),
            dropout: DropoutConfig::new(self.dropout).init(),
            n_heads: self.n_heads,
            head_dim,
        }
    }
}

#[derive(Module, Debug)]
pub struct RelPositionMultiHeadAttention<B: Backend> {
    query_proj: Linear<B>,
    key_proj: Linear<B>,
    value_proj: Linear<B>,
    pos_proj: Linear<B>,
    out_proj: Linear<B>,
    pos_bias_u: Param<Tensor<B, 2>>,
    pos_bias_v: Param<Tensor<B, 2>>,
    dropout: Dropout,
    n_heads: usize,
    head_dim: usize,
}

impl<B: SqueezeformerKernelBackend> RelPositionMultiHeadAttention<B> {
    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        pos_embedding: Tensor<B, 3>,
        mask: Tensor<B, 3, Bool>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_model] = query.dims();

        let q = self.project_heads(self.query_proj.forward(query.clone()));
        let k = self.project_heads(self.key_proj.forward(query.clone()));
        let v = self.project_heads(self.value_proj.forward(query));
        let p = self.project_heads(self.pos_proj.forward(pos_embedding));

        let q_u = q.clone()
            + self
                .pos_bias_u
                .val()
                .reshape([1, self.n_heads, 1, self.head_dim]);
        let q_v = q + self
            .pos_bias_v
            .val()
            .reshape([1, self.n_heads, 1, self.head_dim]);

        let content_scores = q_u.matmul(k.swap_dims(2, 3));
        let position_scores = B::relative_shift(q_v.matmul(p.swap_dims(2, 3)), seq_len);
        let mut scores = (content_scores + position_scores) / (self.head_dim as f64).sqrt();

        let expanded_mask = mask.unsqueeze_dim::<4>(1).repeat_dim(1, self.n_heads);
        let scores_device = scores.device();
        let negative = Tensor::full(
            [batch_size, self.n_heads, seq_len, seq_len],
            -1.0e9,
            &scores_device,
        );
        scores = scores.mask_where(expanded_mask.bool_not(), negative);

        let attn = self.dropout.forward(softmax(scores, 3));
        let context = attn
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, d_model]);

        self.out_proj.forward(context)
    }

    fn project_heads(&self, input: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch_size, seq_len, _] = input.dims();
        input
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2)
    }
}

#[derive(Clone, Debug)]
pub struct FeedForwardModuleConfig {
    pub d_model: usize,
    pub expansion_factor: usize,
    pub dropout: f64,
}

impl FeedForwardModuleConfig {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            expansion_factor: 4,
            dropout: 0.1,
        }
    }

    pub fn with_expansion_factor(mut self, expansion_factor: usize) -> Self {
        self.expansion_factor = expansion_factor;
        self
    }

    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForwardModule<B> {
        let hidden = self.d_model * self.expansion_factor;
        FeedForwardModule {
            input_transform: ScaleBiasLayerConfig::new(self.d_model).init(device),
            linear_in: LinearConfig::new(self.d_model, hidden).init(device),
            linear_out: LinearConfig::new(hidden, self.d_model).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct FeedForwardModule<B: Backend> {
    input_transform: ScaleBiasLayer<B>,
    linear_in: Linear<B>,
    linear_out: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> FeedForwardModule<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = input.clone();
        let output = self.input_transform.forward(input);
        let output = self.linear_in.forward(output);
        let output = silu(output);
        let output = self.dropout.forward(output);
        let output = self.linear_out.forward(output);
        residual + self.dropout.forward(output)
    }
}

#[derive(Clone, Debug)]
pub struct AttentionModuleConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub dropout: f64,
}

impl AttentionModuleConfig {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        Self {
            d_model,
            n_heads,
            dropout: 0.1,
        }
    }

    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> AttentionModule<B> {
        AttentionModule {
            input_transform: ScaleBiasLayerConfig::new(self.d_model).init(device),
            attention: RelPositionMultiHeadAttentionConfig::new(self.d_model, self.n_heads)
                .with_dropout(self.dropout)
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct AttentionModule<B: Backend> {
    input_transform: ScaleBiasLayer<B>,
    attention: RelPositionMultiHeadAttention<B>,
    dropout: Dropout,
}

impl<B: SqueezeformerKernelBackend> AttentionModule<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        pos_embedding: Tensor<B, 3>,
        mask: Tensor<B, 3, Bool>,
    ) -> Tensor<B, 3> {
        let residual = input.clone();
        let output = self.input_transform.forward(input);
        let output = self.attention.forward(output, pos_embedding, mask);
        residual + self.dropout.forward(output)
    }
}

#[derive(Clone, Debug)]
pub struct ConvolutionModuleConfig {
    pub d_model: usize,
    pub kernel_size: usize,
    pub expansion_factor: usize,
    pub dropout: f64,
}

impl ConvolutionModuleConfig {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            kernel_size: 31,
            expansion_factor: 2,
            dropout: 0.1,
        }
    }

    pub fn with_kernel_size(mut self, kernel_size: usize) -> Self {
        self.kernel_size = kernel_size;
        self
    }

    pub fn with_expansion_factor(mut self, expansion_factor: usize) -> Self {
        self.expansion_factor = expansion_factor;
        self
    }

    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvolutionModule<B> {
        let hidden = self.d_model * self.expansion_factor;
        ConvolutionModule {
            input_transform: ScaleBiasLayerConfig::new(self.d_model).init(device),
            pointwise_in: Conv1dConfig::new(self.d_model, hidden, 1).init(device),
            depthwise: Conv1dConfig::new(hidden, hidden, self.kernel_size)
                .with_groups(hidden)
                .with_padding(PaddingConfig1d::Explicit(
                    self.kernel_size / 2,
                    self.kernel_size / 2,
                ))
                .init(device),
            batch_norm: PositiveLossBatchNorm1d::new(hidden, device),
            pointwise_out: Conv1dConfig::new(hidden, self.d_model, 1).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct ConvolutionModule<B: Backend> {
    input_transform: ScaleBiasLayer<B>,
    pointwise_in: Conv1d<B>,
    depthwise: Conv1d<B>,
    batch_norm: PositiveLossBatchNorm1d<B>,
    pointwise_out: Conv1d<B>,
    dropout: Dropout,
}

impl<B: SqueezeformerKernelBackend> ConvolutionModule<B> {
    pub fn forward(&self, input: Tensor<B, 3>, lengths: &[usize]) -> Tensor<B, 3> {
        let residual = input.clone();
        let [batch_size, seq_len, d_model] = input.dims();

        let output = self.input_transform.forward(input);
        let output = output.swap_dims(1, 2);
        let output = self.pointwise_in.forward(output);
        let output = silu(output);
        let output = B::mask_channel_time(output, lengths);
        let output = self.depthwise.forward(output);
        let output = self.batch_norm.forward(output);
        let output = silu(output);
        let output = B::mask_channel_time(output, lengths);
        let output = self.pointwise_out.forward(output);
        let output = self.dropout.forward(output);

        residual
            + output
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, d_model])
    }
}

#[derive(Module, Debug)]
pub struct PositiveLossBatchNorm1d<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
    running_mean: RunningState<Tensor<B, 1>>,
    running_var: RunningState<Tensor<B, 1>>,
    num_batches_tracked: Param<Tensor<B, 1>>,
    momentum: f64,
    epsilon: f64,
}

impl<B: Backend> PositiveLossBatchNorm1d<B> {
    pub(crate) fn new(num_features: usize, device: &B::Device) -> Self {
        Self {
            gamma: Initializer::Ones.init([num_features], device),
            beta: Initializer::Zeros.init([num_features], device),
            running_mean: RunningState::new(Tensor::zeros([num_features], device)),
            running_var: RunningState::new(Tensor::ones([num_features], device)),
            num_batches_tracked: Param::from_tensor(Tensor::from_data(
                TensorData::new(vec![0.0f32], [1]),
                device,
            )),
            momentum: 0.1,
            epsilon: 1.0e-5,
        }
    }

    pub(crate) fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        if B::ad_enabled(&input.device()) {
            self.forward_train(input)
        } else {
            self.forward_inference(input)
        }
    }

    fn forward_inference(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = input.device();
        let channels = input.dims()[1];
        let mean = self.running_mean.value().to_device(&device);
        let var = self.running_var.value().to_device(&device);
        self.forward_shared(
            input,
            mean.reshape([1, channels, 1]),
            var.reshape([1, channels, 1]),
        )
    }

    fn forward_train(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = input.device();
        let [batch_size, channels, seq_len] = input.dims();
        let flatten_size = batch_size * seq_len;
        let mean = input
            .clone()
            .swap_dims(0, 1)
            .reshape([channels, flatten_size])
            .mean_dim(1)
            .reshape([1, channels, 1]);
        let var = input
            .clone()
            .sub(mean.clone())
            .square()
            .swap_dims(0, 1)
            .reshape([channels, flatten_size])
            .mean_dim(1)
            .reshape([1, channels, 1]);

        let running_mean = self.running_mean.value_sync().to_device(&device);
        let running_var = self.running_var.value_sync().to_device(&device);
        self.running_mean.update(
            running_mean
                .mul_scalar(1.0 - self.momentum)
                .add(
                    mean.clone()
                        .detach()
                        .mul_scalar(self.momentum)
                        .reshape([channels]),
                )
                .detach(),
        );
        self.running_var.update(
            running_var
                .mul_scalar(1.0 - self.momentum)
                .add(
                    var.clone()
                        .detach()
                        .mul_scalar(self.momentum)
                        .reshape([channels]),
                )
                .detach(),
        );

        self.forward_shared(input, mean, var)
    }

    fn forward_shared(
        &self,
        input: Tensor<B, 3>,
        mean: Tensor<B, 3>,
        var: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let channels = input.dims()[1];
        let std = var.add_scalar(self.epsilon).sqrt();
        let output = input.sub(mean).div(std);
        output
            .mul(self.gamma.val().reshape([1, channels, 1]))
            .add(self.beta.val().reshape([1, channels, 1]))
    }
}

#[derive(Clone, Debug)]
pub struct SqueezeformerBlockConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub ff_expansion_factor: usize,
    pub conv_expansion_factor: usize,
    pub conv_kernel_size: usize,
    pub dropout: f64,
}

impl SqueezeformerBlockConfig {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        Self {
            d_model,
            n_heads,
            ff_expansion_factor: 4,
            conv_expansion_factor: 2,
            conv_kernel_size: 31,
            dropout: 0.1,
        }
    }

    pub fn with_ff_expansion_factor(mut self, ff_expansion_factor: usize) -> Self {
        self.ff_expansion_factor = ff_expansion_factor;
        self
    }

    pub fn with_conv_expansion_factor(mut self, conv_expansion_factor: usize) -> Self {
        self.conv_expansion_factor = conv_expansion_factor;
        self
    }

    pub fn with_conv_kernel_size(mut self, conv_kernel_size: usize) -> Self {
        self.conv_kernel_size = conv_kernel_size;
        self
    }

    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> SqueezeformerBlock<B> {
        SqueezeformerBlock {
            mhsa_ff: MhsaFfModule {
                attention: AttentionModuleConfig::new(self.d_model, self.n_heads)
                    .with_dropout(self.dropout)
                    .init(device),
                mid_norm: LayerNormConfig::new(self.d_model).init(device),
                feed_forward: FeedForwardModuleConfig::new(self.d_model)
                    .with_expansion_factor(self.ff_expansion_factor)
                    .with_dropout(self.dropout)
                    .init(device),
                out_norm: LayerNormConfig::new(self.d_model).init(device),
            },
            conv_ff: ConvFfModule {
                convolution: ConvolutionModuleConfig::new(self.d_model)
                    .with_kernel_size(self.conv_kernel_size)
                    .with_expansion_factor(self.conv_expansion_factor)
                    .with_dropout(self.dropout)
                    .init(device),
                mid_norm: LayerNormConfig::new(self.d_model).init(device),
                feed_forward: FeedForwardModuleConfig::new(self.d_model)
                    .with_expansion_factor(self.ff_expansion_factor)
                    .with_dropout(self.dropout)
                    .init(device),
                out_norm: LayerNormConfig::new(self.d_model).init(device),
            },
        }
    }
}

#[derive(Module, Debug)]
pub struct MhsaFfModule<B: Backend> {
    attention: AttentionModule<B>,
    mid_norm: LayerNorm<B>,
    feed_forward: FeedForwardModule<B>,
    out_norm: LayerNorm<B>,
}

impl<B: SqueezeformerKernelBackend> MhsaFfModule<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        pos_embedding: Tensor<B, 3>,
        mask: Tensor<B, 3, Bool>,
    ) -> Tensor<B, 3> {
        let output = self.attention.forward(input, pos_embedding, mask);
        let output = self.mid_norm.forward(output);
        let output = self.feed_forward.forward(output);
        self.out_norm.forward(output)
    }
}

#[derive(Module, Debug)]
pub struct ConvFfModule<B: Backend> {
    convolution: ConvolutionModule<B>,
    mid_norm: LayerNorm<B>,
    feed_forward: FeedForwardModule<B>,
    out_norm: LayerNorm<B>,
}

impl<B: SqueezeformerKernelBackend> ConvFfModule<B> {
    pub fn forward(&self, input: Tensor<B, 3>, lengths: &[usize]) -> Tensor<B, 3> {
        let output = self.convolution.forward(input, lengths);
        let output = self.mid_norm.forward(output);
        let output = self.feed_forward.forward(output);
        self.out_norm.forward(output)
    }
}

#[derive(Module, Debug)]
pub struct SqueezeformerBlock<B: Backend> {
    mhsa_ff: MhsaFfModule<B>,
    conv_ff: ConvFfModule<B>,
}

impl<B: SqueezeformerKernelBackend> SqueezeformerBlock<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        lengths: &[usize],
        pos_embedding: Tensor<B, 3>,
        mask: Tensor<B, 3, Bool>,
    ) -> Tensor<B, 3> {
        let output = self.mhsa_ff.forward(input, pos_embedding, mask);
        self.conv_ff.forward(output, lengths)
    }
}

#[derive(Clone, Debug)]
pub struct Conv2dSubsamplingConfig {
    pub input_features: usize,
    pub d_model: usize,
}

impl Conv2dSubsamplingConfig {
    pub fn new(input_features: usize, d_model: usize) -> Self {
        Self {
            input_features,
            d_model,
        }
    }

    pub fn output_dim(&self) -> usize {
        self.d_model * subsample_once(subsample_once(self.input_features))
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Conv2dSubsampling<B> {
        Conv2dSubsampling {
            conv1: Conv2dConfig::new([1, self.d_model], [3, 3])
                .with_stride([2, 2])
                .init(device),
            depthwise: Conv2dConfig::new([self.d_model, self.d_model], [3, 3])
                .with_stride([2, 2])
                .with_groups(self.d_model)
                .init(device),
            pointwise: Conv2dConfig::new([self.d_model, self.d_model], [1, 1]).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Conv2dSubsampling<B: Backend> {
    conv1: Conv2d<B>,
    depthwise: Conv2d<B>,
    pointwise: Conv2d<B>,
}

impl<B: Backend> Conv2dSubsampling<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut output = input.unsqueeze_dim::<4>(1);
        output = pad_for_stride2_conv(output);
        output = relu(self.conv1.forward(output));
        output = pad_for_stride2_conv(output);
        output = self.depthwise.forward(output);
        output = relu(self.pointwise.forward(output));

        let [batch_size, channels, time, features] = output.dims();
        output
            .swap_dims(1, 2)
            .reshape([batch_size, time, channels * features])
    }
}

#[derive(Clone, Debug)]
pub struct TimeReductionLayerConfig {
    pub d_model: usize,
    pub kernel_size: usize,
    pub stride: usize,
}

impl TimeReductionLayerConfig {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            kernel_size: 3,
            stride: 2,
        }
    }

    pub fn with_kernel_size(mut self, kernel_size: usize) -> Self {
        self.kernel_size = kernel_size;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TimeReductionLayer<B> {
        TimeReductionLayer {
            depthwise: Conv1dConfig::new(self.d_model, self.d_model, self.kernel_size)
                .with_stride(self.stride)
                .with_groups(self.d_model)
                .init(device),
            pointwise: Conv1dConfig::new(self.d_model, self.d_model, 1).init(device),
            kernel_size: self.kernel_size,
            stride: self.stride,
        }
    }
}

#[derive(Module, Debug)]
pub struct TimeReductionLayer<B: Backend> {
    depthwise: Conv1d<B>,
    pointwise: Conv1d<B>,
    kernel_size: usize,
    stride: usize,
}

impl<B: SqueezeformerKernelBackend> TimeReductionLayer<B> {
    pub fn forward(&self, input: Tensor<B, 3>, lengths: &[usize]) -> (Tensor<B, 3>, Vec<usize>) {
        let output = B::mask_time(input, lengths).swap_dims(1, 2);
        let output = output.pad(
            [(0, self.kernel_size.saturating_sub(self.stride))],
            PadMode::Constant(0.0),
        );
        let output = self
            .pointwise
            .forward(self.depthwise.forward(output))
            .swap_dims(1, 2);
        let next_lengths = lengths
            .iter()
            .map(|length| {
                if *length == 0 {
                    0
                } else {
                    (length / self.stride).max(1)
                }
            })
            .collect();

        (output, next_lengths)
    }
}

#[derive(Clone, Debug)]
pub struct TimeRecoveryLayerConfig {
    pub d_model: usize,
    pub stride: usize,
}

impl TimeRecoveryLayerConfig {
    pub fn new(d_model: usize) -> Self {
        Self { d_model, stride: 2 }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TimeRecoveryLayer<B> {
        TimeRecoveryLayer {
            projection: LinearConfig::new(self.d_model, self.d_model).init(device),
            stride: self.stride,
        }
    }
}

#[derive(Module, Debug)]
pub struct TimeRecoveryLayer<B: Backend> {
    projection: Linear<B>,
    stride: usize,
}

impl<B: Backend> TimeRecoveryLayer<B> {
    pub fn forward(&self, input: Tensor<B, 3>, skip: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, target_len, _] = skip.dims();
        let mut output = input.repeat_dim(1, self.stride);
        let [_, current_len, _] = output.dims();
        output = if current_len < target_len {
            output.pad([(0, target_len - current_len), (0, 0)], PadMode::Edge)
        } else {
            output.slice_dim(1, 0..target_len)
        };

        skip + self.projection.forward(output)
    }
}

#[derive(Clone, Debug)]
pub struct SqueezeformerEncoderConfig {
    pub input_features: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub kernel_size: usize,
    pub ff_expansion_factor: usize,
    pub conv_expansion_factor: usize,
    pub dropout: f64,
    pub time_reduction_kernel_size: usize,
    pub time_reduce_idx: Vec<usize>,
    pub time_recover_idx: Vec<usize>,
}

impl SqueezeformerEncoderConfig {
    pub fn new(input_features: usize, d_model: usize, num_layers: usize, num_heads: usize) -> Self {
        Self {
            input_features,
            d_model,
            num_layers,
            num_heads,
            kernel_size: 31,
            ff_expansion_factor: 4,
            conv_expansion_factor: 2,
            dropout: 0.1,
            time_reduction_kernel_size: 3,
            time_reduce_idx: vec![7],
            time_recover_idx: vec![15],
        }
    }

    pub fn variant(name: &str) -> Option<Self> {
        let config = match name {
            "xs" => Self::new(80, 144, 16, 4),
            "s" => Self::new(80, 196, 18, 4).with_time_indices(vec![8], vec![17]),
            "sm" => Self::new(80, 256, 16, 4),
            "m" => Self::new(80, 324, 20, 4).with_time_indices(vec![9], vec![19]),
            "ml" => Self::new(80, 512, 18, 8).with_time_indices(vec![8], vec![17]),
            "l" => Self::new(80, 640, 22, 8).with_time_indices(vec![10], vec![21]),
            _ => return None,
        };

        Some(config)
    }

    pub fn with_ff_expansion_factor(mut self, ff_expansion_factor: usize) -> Self {
        self.ff_expansion_factor = ff_expansion_factor;
        self
    }

    pub fn with_conv_expansion_factor(mut self, conv_expansion_factor: usize) -> Self {
        self.conv_expansion_factor = conv_expansion_factor;
        self
    }

    pub fn with_conv_kernel_size(mut self, kernel_size: usize) -> Self {
        self.kernel_size = kernel_size;
        self
    }

    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn with_time_indices(mut self, reduce: Vec<usize>, recover: Vec<usize>) -> Self {
        self.time_reduce_idx = reduce;
        self.time_recover_idx = recover;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> SqueezeformerEncoder<B> {
        let subsampling_config = Conv2dSubsamplingConfig::new(self.input_features, self.d_model);
        let blocks = (0..self.num_layers)
            .map(|_| {
                SqueezeformerBlockConfig::new(self.d_model, self.num_heads)
                    .with_ff_expansion_factor(self.ff_expansion_factor)
                    .with_conv_expansion_factor(self.conv_expansion_factor)
                    .with_conv_kernel_size(self.kernel_size)
                    .with_dropout(self.dropout)
                    .init(device)
            })
            .collect();

        SqueezeformerEncoder {
            subsampling: subsampling_config.init(device),
            input_projection: LinearConfig::new(subsampling_config.output_dim(), self.d_model)
                .init(device),
            input_dropout: DropoutConfig::new(self.dropout).init(),
            input_norm: LayerNormConfig::new(self.d_model).init(device),
            pos_encoding: RelativePositionalEncoding::new(self.d_model),
            blocks,
            time_reduction: TimeReductionLayerConfig::new(self.d_model)
                .with_kernel_size(self.time_reduction_kernel_size)
                .init(device),
            time_recovery: TimeRecoveryLayerConfig::new(self.d_model).init(device),
            time_reduce_idx: self.time_reduce_idx.clone(),
            time_recover_idx: self.time_recover_idx.clone(),
            d_model: self.d_model,
        }
    }
}

#[derive(Module, Debug)]
pub struct SqueezeformerEncoder<B: Backend> {
    subsampling: Conv2dSubsampling<B>,
    input_projection: Linear<B>,
    input_dropout: Dropout,
    input_norm: LayerNorm<B>,
    pos_encoding: RelativePositionalEncoding,
    blocks: Vec<SqueezeformerBlock<B>>,
    time_reduction: TimeReductionLayer<B>,
    time_recovery: TimeRecoveryLayer<B>,
    time_reduce_idx: Vec<usize>,
    time_recover_idx: Vec<usize>,
    d_model: usize,
}

impl<B: SqueezeformerKernelBackend> SqueezeformerEncoder<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = input.dims();
        self.forward_with_lengths(input, vec![seq_len; batch_size])
            .0
    }

    pub fn forward_with_lengths(
        &self,
        input: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        let device = input.device();
        let mut lengths = subsample_lengths(&lengths);
        let mut output = self.subsampling.forward(input);
        output = self.input_projection.forward(output) * (self.d_model as f64).sqrt();
        output = self.input_dropout.forward(output);
        output = self.input_norm.forward(output);
        output = B::mask_time(output, &lengths);

        let mut stack = Vec::new();
        for (index, block) in self.blocks.iter().enumerate() {
            if self.time_reduce_idx.contains(&index) {
                stack.push((output.clone(), lengths.clone()));
                (output, lengths) = self.time_reduction.forward(output, &lengths);
            }

            if self.time_recover_idx.contains(&index) {
                if let Some((skip, skip_lengths)) = stack.pop() {
                    output = self.time_recovery.forward(output, skip);
                    lengths = skip_lengths;
                } else {
                    panic!("encountered a recovery layer without a matching reduced activation");
                }
            }

            let max_len = lengths.iter().copied().max().unwrap_or(0);
            let current_len = output.dims()[1];
            if max_len > 0 && current_len > max_len {
                output = output.slice_dim(1, 0..max_len);
            }

            let seq_len = output.dims()[1];
            let pos_embedding = self.pos_encoding.forward::<B>(seq_len, &device);
            let mask = B::attention_mask(&lengths, seq_len, &device);
            output = block.forward(output, &lengths, pos_embedding, mask);
            output = B::mask_time(output, &lengths);
        }

        (output, lengths)
    }
}

#[derive(Clone, Debug)]
pub struct SqueezeformerCtcConfig {
    pub encoder: SqueezeformerEncoderConfig,
    pub vocab_size: usize,
}

impl SqueezeformerCtcConfig {
    pub fn new(
        input_features: usize,
        d_model: usize,
        num_layers: usize,
        num_heads: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            encoder: SqueezeformerEncoderConfig::new(
                input_features,
                d_model,
                num_layers,
                num_heads,
            ),
            vocab_size,
        }
    }

    pub fn with_encoder(mut self, encoder: SqueezeformerEncoderConfig) -> Self {
        self.encoder = encoder;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> SqueezeformerCtc<B> {
        SqueezeformerCtc {
            encoder: self.encoder.init(device),
            classifier: LinearConfig::new(self.encoder.d_model, self.vocab_size).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct SqueezeformerCtc<B: Backend> {
    encoder: SqueezeformerEncoder<B>,
    classifier: Linear<B>,
}

impl<B: SqueezeformerKernelBackend> SqueezeformerCtc<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_with_lengths(input, None).0
    }

    pub fn forward_with_lengths(
        &self,
        input: Tensor<B, 3>,
        lengths: Option<Vec<usize>>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        let [batch_size, seq_len, _] = input.dims();
        let lengths = lengths.unwrap_or_else(|| vec![seq_len; batch_size]);
        let (encoded, lengths) = self.encoder.forward_with_lengths(input, lengths);
        (self.classifier.forward(encoded), lengths)
    }
}

fn pad_for_stride2_conv<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, height, width] = input.dims();
    let pad_bottom = 1.max(3usize.saturating_sub(height));
    let pad_right = 1.max(3usize.saturating_sub(width));
    input.pad((0, pad_right, 0, pad_bottom), PadMode::Constant(0.0))
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

fn relative_shift_fallback<B: Backend>(input: Tensor<B, 4>, seq_len: usize) -> Tensor<B, 4> {
    let [batch_size, n_heads, _, pos_len] = input.dims();
    let padded = input.pad([(0, 0), (0, 1)], PadMode::Constant(0.0));
    padded
        .reshape([batch_size, n_heads, pos_len + 1, seq_len])
        .slice_dim(2, 1..pos_len + 1)
        .reshape([batch_size, n_heads, seq_len, pos_len])
        .slice_dim(3, 0..seq_len)
}

fn attention_mask_fallback<B: Backend>(
    lengths: &[usize],
    max_len: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    let mask = sequence_mask::<B>(lengths, max_len, device);
    mask.clone()
        .unsqueeze_dim::<3>(1)
        .repeat_dim(1, max_len)
        .bool_and(mask.unsqueeze_dim::<3>(2).repeat_dim(2, max_len))
}

fn apply_time_mask_fallback<B: Backend>(input: Tensor<B, 3>, lengths: &[usize]) -> Tensor<B, 3> {
    let [_, seq_len, _] = input.dims();
    let mask = sequence_mask::<B>(lengths, seq_len, &input.device())
        .float()
        .unsqueeze_dim::<3>(2);
    input * mask
}

fn apply_channel_time_mask_fallback<B: Backend>(
    input: Tensor<B, 3>,
    lengths: &[usize],
) -> Tensor<B, 3> {
    let [_, _, seq_len] = input.dims();
    let mask = sequence_mask::<B>(lengths, seq_len, &input.device())
        .float()
        .unsqueeze_dim::<3>(1);
    input * mask
}

fn subsample_once(length: usize) -> usize {
    if length == 0 {
        0
    } else {
        ((length - 1) / 2) + 1
    }
}

fn subsample_lengths(lengths: &[usize]) -> Vec<usize> {
    lengths
        .iter()
        .map(|length| subsample_once(subsample_once(*length)).max(usize::from(*length > 0)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn ctc_model_subsamples_time_and_preserves_batch_axis() {
        let device = Default::default();
        let model = SqueezeformerCtcConfig::new(80, 16, 2, 4, 32).init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 16, 80], &device);

        let (output, lengths) = model.forward_with_lengths(input, None);

        assert_eq!(output.dims(), [2, 4, 32]);
        assert_eq!(lengths, vec![4, 4]);
    }
}
