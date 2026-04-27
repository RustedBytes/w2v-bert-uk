use burn::module::Module;
use burn::tensor::activation::{log_softmax, relu, sigmoid, silu, softmax};
use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};
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
            blank_id: self.resolved_blank_id(),
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

fn masked_cross_entropy<B: Backend>(
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
    let mask = sequence_mask::<B>(&target_lengths, usable_len, &device).float();
    let denom = mask.clone().sum().clamp_min(1.0);
    -(gathered * mask).sum() / denom
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

fn to_i64(values: Vec<usize>) -> Vec<i64> {
    values.into_iter().map(|value| value as i64).collect()
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
}
