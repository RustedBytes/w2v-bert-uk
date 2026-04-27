use burn::module::Module;
use burn::tensor::activation::gelu;
use burn::tensor::{Bool, Tensor, TensorData, backend::Backend};
use burn_nn::conv::{Conv1d, Conv1dConfig};
use burn_nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};
use burn_nn::{
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig1d,
};

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

    pub fn init<B: Backend>(&self, device: &B::Device) -> Wav2VecBertModel<B> {
        Wav2VecBertModel {
            feature_projection: Wav2VecFeatureProjection {
                layer_norm: LayerNormConfig::new(self.feature_dim).init(device),
                projection: LinearConfig::new(self.feature_dim, self.hidden_size).init(device),
                dropout: DropoutConfig::new(self.dropout).init(),
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
        }
    }
}

#[derive(Module, Debug)]
struct Wav2VecFeatureProjection<B: Backend> {
    layer_norm: LayerNorm<B>,
    projection: Linear<B>,
    dropout: Dropout,
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
}

impl<B: Backend> Wav2VecBertModel<B> {
    pub fn forward(
        &self,
        input_features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        let device = input_features.device();
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
        (
            self.final_norm.forward(mask_time(encoded, &lengths)),
            lengths,
        )
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
        Wav2VecBertCtc {
            encoder: self.encoder.init(device),
            classifier: LinearConfig::new(self.encoder.hidden_size, self.vocab_size).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Wav2VecBertCtc<B: Backend> {
    encoder: Wav2VecBertModel<B>,
    classifier: Linear<B>,
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
}

fn clamp_lengths(lengths: &[usize], max_len: usize) -> Vec<usize> {
    lengths
        .iter()
        .map(|length| (*length).clamp(1, max_len))
        .collect()
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
}
