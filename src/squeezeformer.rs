use burn::module::Module;
use burn::tensor::activation::{gelu, sigmoid, silu};
use burn::tensor::{Tensor, backend::Backend};
use burn_nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn_nn::conv::{Conv1d, Conv1dConfig};
use burn_nn::{
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear,
    LinearConfig, PaddingConfig1d,
};

pub mod transcribe;

#[derive(Clone, Copy, Debug)]
pub enum SqueezeformerActivation {
    Gelu,
    Silu,
}

impl SqueezeformerActivation {
    fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Self::Gelu => gelu(input),
            Self::Silu => silu(input),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SqueezeformerFeedForwardConfig {
    pub d_model: usize,
    pub expansion_factor: usize,
    pub dropout: f64,
    pub activation: SqueezeformerActivation,
}

impl SqueezeformerFeedForwardConfig {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            expansion_factor: 4,
            dropout: 0.1,
            activation: SqueezeformerActivation::Silu,
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

    pub fn with_activation(mut self, activation: SqueezeformerActivation) -> Self {
        self.activation = activation;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> SqueezeformerFeedForward<B> {
        let hidden = self.d_model * self.expansion_factor;

        SqueezeformerFeedForward {
            norm: LayerNormConfig::new(self.d_model).init(device),
            linear_in: LinearConfig::new(self.d_model, hidden).init(device),
            linear_out: LinearConfig::new(hidden, self.d_model).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: self.activation,
        }
    }
}

#[derive(Module, Debug)]
pub struct SqueezeformerFeedForward<B: Backend> {
    norm: LayerNorm<B>,
    linear_in: Linear<B>,
    linear_out: Linear<B>,
    dropout: Dropout,
    activation: SqueezeformerActivation,
}

impl<B: Backend> SqueezeformerFeedForward<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let output = self.norm.forward(input);
        let output = self.linear_in.forward(output);
        let output = self.activation.forward(output);
        let output = self.dropout.forward(output);
        let output = self.linear_out.forward(output);
        self.dropout.forward(output)
    }
}

#[derive(Clone, Debug)]
pub struct SqueezeformerConvolutionConfig {
    pub d_model: usize,
    pub kernel_size: usize,
    pub dropout: f64,
}

impl SqueezeformerConvolutionConfig {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            kernel_size: 31,
            dropout: 0.1,
        }
    }

    pub fn with_kernel_size(mut self, kernel_size: usize) -> Self {
        self.kernel_size = kernel_size;
        self
    }

    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> SqueezeformerConvolution<B> {
        SqueezeformerConvolution {
            norm: LayerNormConfig::new(self.d_model).init(device),
            pointwise_in: Conv1dConfig::new(self.d_model, self.d_model * 2, 1).init(device),
            depthwise: Conv1dConfig::new(self.d_model, self.d_model, self.kernel_size)
                .with_groups(self.d_model)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            batch_norm: BatchNormConfig::new(self.d_model).init(device),
            pointwise_out: Conv1dConfig::new(self.d_model, self.d_model, 1).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct SqueezeformerConvolution<B: Backend> {
    norm: LayerNorm<B>,
    pointwise_in: Conv1d<B>,
    depthwise: Conv1d<B>,
    batch_norm: BatchNorm<B>,
    pointwise_out: Conv1d<B>,
    dropout: Dropout,
}

impl<B: Backend> SqueezeformerConvolution<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_model] = input.dims();

        let output = self.norm.forward(input);
        let output = output.swap_dims(1, 2);
        let output = self.pointwise_in.forward(output);
        let mut chunks = output.chunk(2, 1);
        let gate = chunks.remove(1);
        let value = chunks.remove(0);
        let output = value * sigmoid(gate);
        let output = self.depthwise.forward(output);
        let output = self.batch_norm.forward(output);
        let output = silu(output);
        let output = self.pointwise_out.forward(output);
        let output = self.dropout.forward(output);

        output
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, d_model])
    }
}

#[derive(Clone, Debug)]
pub struct SqueezeformerBlockConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub ff_expansion_factor: usize,
    pub conv_kernel_size: usize,
    pub dropout: f64,
    pub activation: SqueezeformerActivation,
}

impl SqueezeformerBlockConfig {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        Self {
            d_model,
            n_heads,
            ff_expansion_factor: 4,
            conv_kernel_size: 31,
            dropout: 0.1,
            activation: SqueezeformerActivation::Silu,
        }
    }

    pub fn with_ff_expansion_factor(mut self, ff_expansion_factor: usize) -> Self {
        self.ff_expansion_factor = ff_expansion_factor;
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

    pub fn with_activation(mut self, activation: SqueezeformerActivation) -> Self {
        self.activation = activation;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> SqueezeformerBlock<B> {
        SqueezeformerBlock {
            ff1: SqueezeformerFeedForwardConfig::new(self.d_model)
                .with_expansion_factor(self.ff_expansion_factor)
                .with_dropout(self.dropout)
                .with_activation(self.activation)
                .init(device),
            attention_norm: LayerNormConfig::new(self.d_model).init(device),
            attention: MultiHeadAttentionConfig::new(self.d_model, self.n_heads)
                .with_dropout(self.dropout)
                .init(device),
            attention_dropout: DropoutConfig::new(self.dropout).init(),
            convolution: SqueezeformerConvolutionConfig::new(self.d_model)
                .with_kernel_size(self.conv_kernel_size)
                .with_dropout(self.dropout)
                .init(device),
            ff2: SqueezeformerFeedForwardConfig::new(self.d_model)
                .with_expansion_factor(self.ff_expansion_factor)
                .with_dropout(self.dropout)
                .with_activation(self.activation)
                .init(device),
            final_norm: LayerNormConfig::new(self.d_model).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct SqueezeformerBlock<B: Backend> {
    ff1: SqueezeformerFeedForward<B>,
    attention_norm: LayerNorm<B>,
    attention: MultiHeadAttention<B>,
    attention_dropout: Dropout,
    convolution: SqueezeformerConvolution<B>,
    ff2: SqueezeformerFeedForward<B>,
    final_norm: LayerNorm<B>,
}

impl<B: Backend> SqueezeformerBlock<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = input.clone();
        let output = input + self.ff1.forward(residual) * 0.5;

        let residual = output.clone();
        let attention_input = self.attention_norm.forward(output);
        let attention_output = self.attention.forward(MhaInput::self_attn(attention_input));
        let output = residual + self.attention_dropout.forward(attention_output.context);

        let residual = output.clone();
        let output = output + self.convolution.forward(residual);

        let residual = output.clone();
        let output = output + self.ff2.forward(residual) * 0.5;
        self.final_norm.forward(output)
    }
}

#[derive(Clone, Debug)]
pub struct SqueezeformerEncoderConfig {
    pub input_dim: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ff_expansion_factor: usize,
    pub conv_kernel_size: usize,
    pub dropout: f64,
    pub activation: SqueezeformerActivation,
}

impl SqueezeformerEncoderConfig {
    pub fn new(input_dim: usize, d_model: usize, n_layers: usize, n_heads: usize) -> Self {
        Self {
            input_dim,
            d_model,
            n_layers,
            n_heads,
            ff_expansion_factor: 4,
            conv_kernel_size: 31,
            dropout: 0.1,
            activation: SqueezeformerActivation::Silu,
        }
    }

    pub fn with_ff_expansion_factor(mut self, ff_expansion_factor: usize) -> Self {
        self.ff_expansion_factor = ff_expansion_factor;
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

    pub fn with_activation(mut self, activation: SqueezeformerActivation) -> Self {
        self.activation = activation;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> SqueezeformerEncoder<B> {
        let blocks = (0..self.n_layers)
            .map(|_| {
                SqueezeformerBlockConfig::new(self.d_model, self.n_heads)
                    .with_ff_expansion_factor(self.ff_expansion_factor)
                    .with_conv_kernel_size(self.conv_kernel_size)
                    .with_dropout(self.dropout)
                    .with_activation(self.activation)
                    .init(device)
            })
            .collect();

        SqueezeformerEncoder {
            input_projection: LinearConfig::new(self.input_dim, self.d_model).init(device),
            input_dropout: DropoutConfig::new(self.dropout).init(),
            blocks,
        }
    }
}

#[derive(Module, Debug)]
pub struct SqueezeformerEncoder<B: Backend> {
    input_projection: Linear<B>,
    input_dropout: Dropout,
    blocks: Vec<SqueezeformerBlock<B>>,
}

impl<B: Backend> SqueezeformerEncoder<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut output = self.input_projection.forward(input);
        output = self.input_dropout.forward(output);

        for block in self.blocks.iter() {
            output = block.forward(output);
        }

        output
    }
}

#[derive(Clone, Debug)]
pub struct SqueezeformerCtcConfig {
    pub encoder: SqueezeformerEncoderConfig,
    pub vocab_size: usize,
}

impl SqueezeformerCtcConfig {
    pub fn new(
        input_dim: usize,
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            encoder: SqueezeformerEncoderConfig::new(input_dim, d_model, n_layers, n_heads),
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

impl<B: Backend> SqueezeformerCtc<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.classifier.forward(self.encoder.forward(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn ctc_model_preserves_batch_and_time_axes() {
        let device = Default::default();
        let model = SqueezeformerCtcConfig::new(80, 16, 2, 4, 32).init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 7, 80], &device);

        let output = model.forward(input);

        assert_eq!(output.dims(), [2, 7, 32]);
    }
}
