use burn::module::Module;
use burn::tensor::activation::{sigmoid, silu};
use burn::tensor::ops::PadMode;
use burn::tensor::{Bool, Tensor, TensorData, backend::Backend};
use burn_nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn_nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn_nn::{
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig1d,
    PaddingConfig2d,
};

const DEFAULT_DOWNSAMPLING: [usize; 6] = [1, 2, 4, 8, 4, 2];
const DEFAULT_ENCODER_DIM: [usize; 6] = [192, 256, 384, 512, 384, 256];
const DEFAULT_NUM_LAYERS: [usize; 6] = [2, 2, 3, 4, 3, 2];
const DEFAULT_NUM_HEADS: [usize; 6] = [4, 4, 4, 8, 4, 4];
const DEFAULT_FEEDFORWARD_DIM: [usize; 6] = [512, 768, 1024, 1536, 1024, 768];
const DEFAULT_CNN_KERNELS: [usize; 6] = [31, 31, 15, 15, 15, 31];

#[derive(Clone, Debug)]
pub struct ZipformerConfig {
    pub input_dim: usize,
    pub output_downsampling_factor: usize,
    pub downsampling_factor: Vec<usize>,
    pub encoder_dim: Vec<usize>,
    pub num_encoder_layers: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub feedforward_dim: Vec<usize>,
    pub cnn_module_kernel: Vec<usize>,
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
            feedforward_dim: DEFAULT_FEEDFORWARD_DIM.to_vec(),
            cnn_module_kernel: DEFAULT_CNN_KERNELS.to_vec(),
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
                feedforward_dim: vec![192, 256, 384, 512, 384, 256],
                ..Self::default()
            }),
            "s" => Some(Self {
                encoder_dim: vec![192, 256, 256, 256, 256, 256],
                num_encoder_layers: vec![2, 2, 2, 2, 2, 2],
                num_heads: vec![4, 4, 4, 8, 4, 4],
                feedforward_dim: vec![512, 768, 768, 768, 768, 768],
                ..Self::default()
            }),
            "m" | "sm" => Some(Self::default()),
            "l" | "ml" => Some(Self {
                encoder_dim: vec![192, 256, 512, 768, 512, 256],
                num_encoder_layers: vec![2, 2, 4, 5, 4, 2],
                num_heads: vec![4, 4, 4, 8, 4, 4],
                feedforward_dim: vec![512, 768, 1536, 2048, 1536, 768],
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

        let stacks = self
            .encoder_dim
            .iter()
            .enumerate()
            .map(|(index, dim)| {
                ZipformerStackConfig {
                    dim: *dim,
                    num_layers: self.num_encoder_layers[index],
                    num_heads: self.num_heads[index],
                    feedforward_dim: self.feedforward_dim[index],
                    conv_kernel_size: self.cnn_module_kernel[index],
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
            output_downsample: PairwiseDownsample::new(self.output_downsampling_factor),
        }
    }
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
            output_norm: LayerNormConfig::new(self.output_dim).init(device),
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
    output_norm: LayerNorm<B>,
}

impl<B: Backend> ConvEmbed<B> {
    fn forward(&self, input: Tensor<B, 3>, lengths: Vec<usize>) -> (Tensor<B, 3>, Vec<usize>) {
        let mut output = input.unsqueeze_dim::<4>(1);
        output = silu(self.conv1.forward(output));
        output = silu(self.conv2.forward(output));
        output = silu(self.conv3.forward(output));

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
        (mask_time(output, &lengths), lengths)
    }
}

#[derive(Clone, Debug)]
struct ZipformerStackConfig {
    dim: usize,
    num_layers: usize,
    num_heads: usize,
    feedforward_dim: usize,
    conv_kernel_size: usize,
    dropout: f64,
    downsample: usize,
}

impl ZipformerStackConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ZipformerStack<B> {
        ZipformerStack {
            downsample: PairwiseDownsample::new(self.downsample),
            upsample: PairwiseUpsample::new(self.downsample),
            blocks: (0..self.num_layers)
                .map(|_| {
                    ZipformerBlockConfig {
                        dim: self.dim,
                        heads: self.num_heads,
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
    downsample: PairwiseDownsample,
    upsample: PairwiseUpsample,
    blocks: Vec<ZipformerBlock<B>>,
}

impl<B: Backend> ZipformerStack<B> {
    fn forward(&self, input: Tensor<B, 3>, lengths: &[usize]) -> Tensor<B, 3> {
        let target_len = input.dims()[1];
        let residual = input.clone();
        let (mut output, down_lengths) = self.downsample.forward(input, lengths);
        for block in self.blocks.iter() {
            output = block.forward(output, &down_lengths);
        }
        output = self.upsample.forward(output, target_len);
        mask_time((residual.clone() + output) * 0.5, lengths)
    }
}

#[derive(Clone, Debug)]
struct ZipformerBlockConfig {
    dim: usize,
    heads: usize,
    feedforward_dim: usize,
    conv_kernel_size: usize,
    dropout: f64,
}

impl ZipformerBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ZipformerBlock<B> {
        ZipformerBlock {
            feed_forward1: FeedForwardConfig::new(
                self.dim,
                (self.feedforward_dim * 3) / 4,
                self.dropout,
            )
            .init(device),
            self_attention1_norm: LayerNormConfig::new(self.dim).init(device),
            self_attention1: MultiHeadAttentionConfig::new(self.dim, self.heads)
                .with_dropout(self.dropout)
                .init(device),
            conv1: ZipformerConvModuleConfig::new(self.dim, self.conv_kernel_size, self.dropout)
                .init(device),
            feed_forward2: FeedForwardConfig::new(self.dim, self.feedforward_dim, self.dropout)
                .init(device),
            mid_norm: LayerNormConfig::new(self.dim).init(device),
            self_attention2_norm: LayerNormConfig::new(self.dim).init(device),
            self_attention2: MultiHeadAttentionConfig::new(self.dim, self.heads)
                .with_dropout(self.dropout)
                .init(device),
            conv2: ZipformerConvModuleConfig::new(self.dim, self.conv_kernel_size, self.dropout)
                .init(device),
            feed_forward3: FeedForwardConfig::new(
                self.dim,
                (self.feedforward_dim * 5) / 4,
                self.dropout,
            )
            .init(device),
            output_norm: LayerNormConfig::new(self.dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
struct ZipformerBlock<B: Backend> {
    feed_forward1: FeedForward<B>,
    self_attention1_norm: LayerNorm<B>,
    self_attention1: MultiHeadAttention<B>,
    conv1: ZipformerConvModule<B>,
    feed_forward2: FeedForward<B>,
    mid_norm: LayerNorm<B>,
    self_attention2_norm: LayerNorm<B>,
    self_attention2: MultiHeadAttention<B>,
    conv2: ZipformerConvModule<B>,
    feed_forward3: FeedForward<B>,
    output_norm: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> ZipformerBlock<B> {
    fn forward(&self, input: Tensor<B, 3>, lengths: &[usize]) -> Tensor<B, 3> {
        let device = input.device();
        let mask = padding_mask::<B>(lengths, input.dims()[1], &device);

        let mut output = input.clone() + self.feed_forward1.forward(input.clone());
        let attn = self.self_attention1.forward(
            MhaInput::self_attn(self.self_attention1_norm.forward(output.clone()))
                .mask_pad(mask.clone()),
        );
        output = mask_time(output + self.dropout.forward(attn.context), lengths);
        output = mask_time(
            output.clone() + self.conv1.forward(output, lengths),
            lengths,
        );
        output = mask_time(output.clone() + self.feed_forward2.forward(output), lengths);
        output = self.mid_norm.forward((input + output) * 0.5);

        let attn = self.self_attention2.forward(
            MhaInput::self_attn(self.self_attention2_norm.forward(output.clone())).mask_pad(mask),
        );
        output = mask_time(output + self.dropout.forward(attn.context), lengths);
        output = mask_time(
            output.clone() + self.conv2.forward(output, lengths),
            lengths,
        );
        output = output.clone() + self.feed_forward3.forward(output);
        self.output_norm.forward(mask_time(output, lengths))
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
            norm: LayerNormConfig::new(self.dim).init(device),
            linear_in: LinearConfig::new(self.dim, self.hidden).init(device),
            linear_out: LinearConfig::new(self.hidden, self.dim).init(device),
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
        self.dropout.forward(self.linear_out.forward(output))
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
            norm: LayerNormConfig::new(self.dim).init(device),
            pointwise_in: Conv1dConfig::new(self.dim, self.dim * 2, 1).init(device),
            depthwise: Conv1dConfig::new(self.dim, self.dim, self.kernel_size)
                .with_groups(self.dim)
                .with_padding(PaddingConfig1d::Explicit(
                    self.kernel_size / 2,
                    self.kernel_size / 2,
                ))
                .init(device),
            pointwise_out: Conv1dConfig::new(self.dim, self.dim, 1).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
struct ZipformerConvModule<B: Backend> {
    norm: LayerNorm<B>,
    pointwise_in: Conv1d<B>,
    depthwise: Conv1d<B>,
    pointwise_out: Conv1d<B>,
    dropout: Dropout,
}

impl<B: Backend> ZipformerConvModule<B> {
    fn forward(&self, input: Tensor<B, 3>, lengths: &[usize]) -> Tensor<B, 3> {
        let [batch_size, seq_len, dim] = input.dims();
        let output = self.norm.forward(input).swap_dims(1, 2);
        let output = self.pointwise_in.forward(output);
        let mut chunks = output.chunk(2, 1);
        let gate = chunks.remove(1);
        let value = chunks.remove(0);
        let output = value * sigmoid(gate);
        let output = silu(self.depthwise.forward(output));
        let output = self.dropout.forward(self.pointwise_out.forward(output));
        let output = output.swap_dims(1, 2).reshape([batch_size, seq_len, dim]);
        mask_time(output, lengths)
    }
}

#[derive(Module, Debug, Clone)]
struct PairwiseDownsample {
    factor: usize,
}

impl PairwiseDownsample {
    fn new(factor: usize) -> Self {
        assert!(factor >= 1 && factor.is_power_of_two());
        Self { factor }
    }

    fn forward<B: Backend>(
        &self,
        input: Tensor<B, 3>,
        lengths: &[usize],
    ) -> (Tensor<B, 3>, Vec<usize>) {
        if self.factor == 1 {
            return (input, lengths.to_vec());
        }
        let [_, seq_len, _] = input.dims();
        let pad = (self.factor - (seq_len % self.factor)) % self.factor;
        let output = if pad > 0 {
            input.pad([(0, pad), (0, 0)], PadMode::Edge)
        } else {
            input
        };
        let [batch_size, padded_len, dim] = output.dims();
        let output = output
            .reshape([batch_size, padded_len / self.factor, self.factor, dim])
            .mean_dim(2)
            .reshape([batch_size, padded_len / self.factor, dim]);
        let lengths = lengths
            .iter()
            .map(|length| ceil_divide(*length, self.factor))
            .collect();
        (output, lengths)
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
    output_downsample: PairwiseDownsample,
}

impl<B: Backend> ZipformerEncoder<B> {
    pub fn forward(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        let (mut output, lengths) = self.conv_embed.forward(features, lengths);
        let mut outputs = Vec::with_capacity(self.stacks.len());
        for (index, stack) in self.stacks.iter().enumerate() {
            output = convert_num_channels(output, self.encoder_dim[index]);
            output = mask_time(output, &lengths);
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
        (mask_time(output, &lengths), lengths)
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

impl<B: Backend> ZipformerCtc<B> {
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
}
