use burn::module::Module;
use burn::tensor::activation::{relu, sigmoid, silu, softmax};
use burn::tensor::{Bool, Tensor, TensorData, backend::Backend};
use burn_nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn_nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn_nn::transformer::{TransformerDecoder, TransformerDecoderConfig, TransformerDecoderInput};
use burn_nn::{
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear,
    LinearConfig, PaddingConfig1d, PaddingConfig2d,
};

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
            batch_norm: BatchNormConfig::new(self.dim).init(device),
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
    batch_norm: BatchNorm<B>,
    conv_out: Conv1d<B>,
    conv_dropout: Dropout,
    ff2: FeedForward<B>,
    final_norm: LayerNorm<B>,
}

impl<B: Backend> ConformerBlock<B> {
    fn forward(&self, input: Tensor<B, 3>, key_padding_mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let output = input.clone() + self.ff1.forward(input) * 0.5;

        let attn_input = self.self_attn_norm.forward(output.clone());
        let attn = self
            .self_attn
            .forward(MhaInput::self_attn(attn_input).mask_pad(key_padding_mask));
        let output = output + self.self_attn_dropout.forward(attn.context);

        let [batch_size, seq_len, dim] = output.dims();
        let conv_input = self.conv_norm.forward(output.clone()).swap_dims(1, 2);
        let conv = self.conv_in.forward(conv_input);
        let mut chunks = conv.chunk(2, 1);
        let gate = chunks.remove(1);
        let value = chunks.remove(0);
        let conv = value * sigmoid(gate);
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

impl<B: Backend> ConformerEncoder<B> {
    fn forward(&self, features: Tensor<B, 3>, lengths: Vec<usize>) -> (Tensor<B, 3>, Vec<usize>) {
        let device = features.device();
        let (mut output, lengths) = self.subsampling.forward(features, lengths);
        let mask = padding_mask::<B>(&lengths, output.dims()[1], &device);
        for layer in self.layers.iter() {
            output = layer.forward(output, mask.clone());
        }
        (output, lengths)
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
}

pub struct ParaformerOutput<B: Backend> {
    pub decoder_logits: Tensor<B, 3>,
    pub ctc_log_probs: Tensor<B, 3>,
    pub encoder_lengths: Vec<usize>,
    pub query_lengths: Vec<usize>,
}

impl<B: Backend> ParaformerV2<B> {
    pub fn forward(&self, features: Tensor<B, 3>, lengths: Vec<usize>) -> ParaformerOutput<B> {
        let (encoder_out, encoder_lengths) = self.encoder.forward(features, lengths);
        let ctc_logits = self.ctc_projection.forward(encoder_out.clone());
        let ctc_log_probs = softmax(ctc_logits.clone(), 2).log();
        self.forward_with_posterior_queries(
            encoder_out,
            encoder_lengths,
            softmax(ctc_logits, 2),
            None,
        )
        .with_ctc_log_probs(ctc_log_probs)
    }

    pub fn forward_with_posterior_queries(
        &self,
        encoder_out: Tensor<B, 3>,
        encoder_lengths: Vec<usize>,
        posterior_queries: Tensor<B, 3>,
        query_lengths: Option<Vec<usize>>,
    ) -> ParaformerOutput<B> {
        let device = encoder_out.device();
        let query_lengths = query_lengths.unwrap_or_else(|| encoder_lengths.clone());
        let decoder_in = self.posterior_embed.forward(posterior_queries);
        let memory = match &self.memory_projection {
            Some(projection) => projection.forward(encoder_out),
            None => encoder_out,
        };
        let target_mask = padding_mask::<B>(&query_lengths, decoder_in.dims()[1], &device);
        let memory_mask = padding_mask::<B>(&encoder_lengths, memory.dims()[1], &device);
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
        }
    }
}

impl<B: Backend> ParaformerOutput<B> {
    fn with_ctc_log_probs(mut self, ctc_log_probs: Tensor<B, 3>) -> Self {
        self.ctc_log_probs = ctc_log_probs;
        self
    }
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
}
