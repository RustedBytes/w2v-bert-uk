//! Autodiff registrations for custom ASR CubeCL kernels.
//!
//! These wrappers avoid the straight-through fallback pattern for simple mask
//! kernels by registering real Burn backward steps. The backward math for these
//! kernels is another application of the same mask to the upstream gradient.

use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::ops::PadMode;
use burn::tensor::{Tensor, TensorPrimitive, backend::Backend};
use burn_autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};

pub trait AsrAutodiffKernelBackend: Backend {
    fn relative_shift_raw(input: Tensor<Self, 4>, seq_len: usize) -> Tensor<Self, 4>;
    fn mask_time_raw(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3>;
    fn mask_channel_time_raw(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3>;
    fn residual_add_mask_time_raw(
        residual: Tensor<Self, 3>,
        update: Tensor<Self, 3>,
        lengths: &[usize],
    ) -> Tensor<Self, 3>;
    fn glu_last_dim_raw(input: Tensor<Self, 3>) -> Tensor<Self, 3>;
    fn glu_channel_dim_raw(input: Tensor<Self, 3>) -> Tensor<Self, 3>;
    fn pairwise_downsample_raw(
        input: Tensor<Self, 3>,
        lengths: &[usize],
        weights: Tensor<Self, 1>,
    ) -> Tensor<Self, 3>;
}

#[cfg(feature = "burn-cuda-backend")]
impl<F, I> AsrAutodiffKernelBackend for burn_cuda::Cuda<F, I>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
{
    fn relative_shift_raw(input: Tensor<Self, 4>, seq_len: usize) -> Tensor<Self, 4> {
        crate::cubecl_kernels::relative_shift(input, seq_len)
    }

    fn mask_time_raw(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::cubecl_kernels::mask_time(input, lengths)
    }

    fn mask_channel_time_raw(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::cubecl_kernels::mask_channel_time(input, lengths)
    }

    fn residual_add_mask_time_raw(
        residual: Tensor<Self, 3>,
        update: Tensor<Self, 3>,
        lengths: &[usize],
    ) -> Tensor<Self, 3> {
        let lengths = Tensor::<Self, 1, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(
                lengths
                    .iter()
                    .map(|value| *value as i64)
                    .collect::<Vec<_>>(),
                [lengths.len()],
            ),
            &residual.device(),
        );
        crate::cubecl_kernels::residual_add_mask_time(residual, update, lengths)
    }

    fn glu_last_dim_raw(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        crate::cubecl_kernels::glu_last_dim(input)
    }

    fn glu_channel_dim_raw(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        crate::cubecl_kernels::glu_channel_dim(input)
    }

    fn pairwise_downsample_raw(
        input: Tensor<Self, 3>,
        lengths: &[usize],
        weights: Tensor<Self, 1>,
    ) -> Tensor<Self, 3> {
        let lengths = Tensor::<Self, 1, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(
                lengths
                    .iter()
                    .map(|value| *value as i64)
                    .collect::<Vec<_>>(),
                [lengths.len()],
            ),
            &input.device(),
        );
        crate::cubecl_kernels::pairwise_downsample(input, lengths, weights)
    }
}

#[cfg(feature = "burn-wgpu-backend")]
impl<F, I, BT> AsrAutodiffKernelBackend for burn_wgpu::Wgpu<F, I, BT>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
{
    fn relative_shift_raw(input: Tensor<Self, 4>, seq_len: usize) -> Tensor<Self, 4> {
        crate::cubecl_kernels::relative_shift(input, seq_len)
    }

    fn mask_time_raw(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::cubecl_kernels::mask_time(input, lengths)
    }

    fn mask_channel_time_raw(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3> {
        crate::cubecl_kernels::mask_channel_time(input, lengths)
    }

    fn residual_add_mask_time_raw(
        residual: Tensor<Self, 3>,
        update: Tensor<Self, 3>,
        lengths: &[usize],
    ) -> Tensor<Self, 3> {
        let lengths = Tensor::<Self, 1, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(
                lengths
                    .iter()
                    .map(|value| *value as i64)
                    .collect::<Vec<_>>(),
                [lengths.len()],
            ),
            &residual.device(),
        );
        crate::cubecl_kernels::residual_add_mask_time(residual, update, lengths)
    }

    fn glu_last_dim_raw(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        crate::cubecl_kernels::glu_last_dim(input)
    }

    fn glu_channel_dim_raw(input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        crate::cubecl_kernels::glu_channel_dim(input)
    }

    fn pairwise_downsample_raw(
        input: Tensor<Self, 3>,
        lengths: &[usize],
        weights: Tensor<Self, 1>,
    ) -> Tensor<Self, 3> {
        let lengths = Tensor::<Self, 1, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(
                lengths
                    .iter()
                    .map(|value| *value as i64)
                    .collect::<Vec<_>>(),
                [lengths.len()],
            ),
            &input.device(),
        );
        crate::cubecl_kernels::pairwise_downsample(input, lengths, weights)
    }
}

pub fn relative_shift<B, C>(
    input: Tensor<Autodiff<B, C>, 4>,
    seq_len: usize,
) -> Tensor<Autodiff<B, C>, 4>
where
    B: AsrAutodiffKernelBackend,
    C: CheckpointStrategy,
{
    let input = input.into_primitive().tensor();
    let shape =
        Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(input.primitive.clone())).dims();
    let output = RelativeShiftBackward
        .prepare::<C>([input.node.clone()])
        .compute_bound()
        .stateful();

    let output = match output {
        OpsKind::Tracked(prep) => {
            let raw = relative_shift_primitive::<B>(input.primitive, seq_len);
            prep.finish(
                RelativeShiftState {
                    input_shape: shape,
                    seq_len,
                },
                raw,
            )
        }
        OpsKind::UnTracked(prep) => {
            let raw = relative_shift_primitive::<B>(input.primitive, seq_len);
            prep.finish(raw)
        }
    };
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

pub fn mask_time<B, C>(
    input: Tensor<Autodiff<B, C>, 3>,
    lengths: &[usize],
) -> Tensor<Autodiff<B, C>, 3>
where
    B: AsrAutodiffKernelBackend,
    C: CheckpointStrategy,
{
    let input = input.into_primitive().tensor();
    let output = MaskTimeBackward
        .prepare::<C>([input.node.clone()])
        .compute_bound()
        .stateful();
    let lengths = lengths.to_vec();

    let output = match output {
        OpsKind::Tracked(prep) => {
            let raw = mask_time_primitive::<B>(input.primitive, &lengths);
            prep.finish(lengths, raw)
        }
        OpsKind::UnTracked(prep) => {
            let raw = mask_time_primitive::<B>(input.primitive, &lengths);
            prep.finish(raw)
        }
    };
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

pub fn mask_channel_time<B, C>(
    input: Tensor<Autodiff<B, C>, 3>,
    lengths: &[usize],
) -> Tensor<Autodiff<B, C>, 3>
where
    B: AsrAutodiffKernelBackend,
    C: CheckpointStrategy,
{
    let input = input.into_primitive().tensor();
    let output = MaskChannelTimeBackward
        .prepare::<C>([input.node.clone()])
        .compute_bound()
        .stateful();
    let lengths = lengths.to_vec();

    let output = match output {
        OpsKind::Tracked(prep) => {
            let raw = mask_channel_time_primitive::<B>(input.primitive, &lengths);
            prep.finish(lengths, raw)
        }
        OpsKind::UnTracked(prep) => {
            let raw = mask_channel_time_primitive::<B>(input.primitive, &lengths);
            prep.finish(raw)
        }
    };
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

pub fn residual_add_mask_time<B, C>(
    residual: Tensor<Autodiff<B, C>, 3>,
    update: Tensor<Autodiff<B, C>, 3>,
    lengths: &[usize],
) -> Tensor<Autodiff<B, C>, 3>
where
    B: AsrAutodiffKernelBackend,
    C: CheckpointStrategy,
{
    let residual = residual.into_primitive().tensor();
    let update = update.into_primitive().tensor();
    let output = ResidualAddMaskTimeBackward
        .prepare::<C>([residual.node.clone(), update.node.clone()])
        .compute_bound()
        .stateful();
    let lengths = lengths.to_vec();

    let output = match output {
        OpsKind::Tracked(prep) => {
            let raw = residual_add_mask_time_primitive::<B>(
                residual.primitive,
                update.primitive,
                &lengths,
            );
            prep.finish(lengths, raw)
        }
        OpsKind::UnTracked(prep) => {
            let raw = residual_add_mask_time_primitive::<B>(
                residual.primitive,
                update.primitive,
                &lengths,
            );
            prep.finish(raw)
        }
    };
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

pub fn glu_last_dim<B, C>(input: Tensor<Autodiff<B, C>, 3>) -> Tensor<Autodiff<B, C>, 3>
where
    B: AsrAutodiffKernelBackend,
    C: CheckpointStrategy,
{
    glu_with_layout::<B, C>(input, GluLayout::Last)
}

pub fn glu_channel_dim<B, C>(input: Tensor<Autodiff<B, C>, 3>) -> Tensor<Autodiff<B, C>, 3>
where
    B: AsrAutodiffKernelBackend,
    C: CheckpointStrategy,
{
    glu_with_layout::<B, C>(input, GluLayout::Channel)
}

fn glu_with_layout<B, C>(
    input: Tensor<Autodiff<B, C>, 3>,
    layout: GluLayout,
) -> Tensor<Autodiff<B, C>, 3>
where
    B: AsrAutodiffKernelBackend,
    C: CheckpointStrategy,
{
    let input = input.into_primitive().tensor();
    let output = GluBackward
        .prepare::<C>([input.node.clone()])
        .compute_bound()
        .stateful();

    let output = match output {
        OpsKind::Tracked(prep) => {
            let state = GluState {
                input: input.primitive.clone(),
                layout,
            };
            let raw = glu_primitive::<B>(input.primitive, layout);
            prep.finish(state, raw)
        }
        OpsKind::UnTracked(prep) => {
            let raw = glu_primitive::<B>(input.primitive, layout);
            prep.finish(raw)
        }
    };
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

pub fn pairwise_downsample<B, C>(
    input: Tensor<Autodiff<B, C>, 3>,
    lengths: &[usize],
    weights: Tensor<Autodiff<B, C>, 1>,
) -> (Tensor<Autodiff<B, C>, 3>, Vec<usize>)
where
    B: AsrAutodiffKernelBackend,
    C: CheckpointStrategy,
{
    let output_lengths = downsample_lengths(lengths, 2);
    let input = input.into_primitive().tensor();
    let weights = weights.into_primitive().tensor();
    let output = PairwiseDownsampleBackward
        .prepare::<C>([input.node.clone(), weights.node.clone()])
        .compute_bound()
        .stateful();
    let lengths = lengths.to_vec();

    let output = match output {
        OpsKind::Tracked(prep) => {
            let state = PairwiseDownsampleState {
                input: input.primitive.clone(),
                weights: weights.primitive.clone(),
                lengths: lengths.clone(),
            };
            let raw =
                pairwise_downsample_primitive::<B>(input.primitive, &lengths, weights.primitive);
            prep.finish(state, raw)
        }
        OpsKind::UnTracked(prep) => {
            let raw =
                pairwise_downsample_primitive::<B>(input.primitive, &lengths, weights.primitive);
            prep.finish(raw)
        }
    };
    let output = Tensor::from_primitive(TensorPrimitive::Float(output));
    (mask_time(output, &output_lengths), output_lengths)
}

#[derive(Debug)]
struct MaskTimeBackward;

impl<B> Backward<B, 1> for MaskTimeBackward
where
    B: AsrAutodiffKernelBackend,
{
    type State = Vec<usize>;

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let [node_input] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        if let Some(node) = node_input {
            grads.register::<B>(node.id, mask_time_primitive::<B>(grad, &ops.state));
        }
    }
}

#[derive(Debug)]
struct MaskChannelTimeBackward;

impl<B> Backward<B, 1> for MaskChannelTimeBackward
where
    B: AsrAutodiffKernelBackend,
{
    type State = Vec<usize>;

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let [node_input] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        if let Some(node) = node_input {
            grads.register::<B>(node.id, mask_channel_time_primitive::<B>(grad, &ops.state));
        }
    }
}

#[derive(Debug)]
struct ResidualAddMaskTimeBackward;

impl<B> Backward<B, 2> for ResidualAddMaskTimeBackward
where
    B: AsrAutodiffKernelBackend,
{
    type State = Vec<usize>;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let [node_residual, node_update] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);

        if let Some(node) = node_residual {
            grads.register::<B>(node.id, mask_time_primitive::<B>(grad.clone(), &ops.state));
        }
        if let Some(node) = node_update {
            grads.register::<B>(node.id, mask_time_primitive::<B>(grad, &ops.state));
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum GluLayout {
    Last,
    Channel,
}

#[derive(Clone, Debug)]
struct GluState<B: Backend> {
    input: B::FloatTensorPrimitive,
    layout: GluLayout,
}

#[derive(Debug)]
struct GluBackward;

impl<B> Backward<B, 1> for GluBackward
where
    B: AsrAutodiffKernelBackend,
{
    type State = GluState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let [node_input] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        if let Some(node) = node_input {
            let grad = glu_backward_primitive::<B>(ops.state.input, grad, ops.state.layout);
            grads.register::<B>(node.id, grad);
        }
    }
}

#[derive(Clone, Debug)]
struct RelativeShiftState {
    input_shape: [usize; 4],
    seq_len: usize,
}

#[derive(Debug)]
struct RelativeShiftBackward;

impl<B> Backward<B, 1> for RelativeShiftBackward
where
    B: AsrAutodiffKernelBackend,
{
    type State = RelativeShiftState;

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let [node_input] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        if let Some(node) = node_input {
            let grad = relative_shift_backward_primitive::<B>(
                grad,
                ops.state.input_shape,
                ops.state.seq_len,
            );
            grads.register::<B>(node.id, grad);
        }
    }
}

#[derive(Clone, Debug)]
struct PairwiseDownsampleState<B: Backend> {
    input: B::FloatTensorPrimitive,
    weights: B::FloatTensorPrimitive,
    lengths: Vec<usize>,
}

#[derive(Debug)]
struct PairwiseDownsampleBackward;

impl<B> Backward<B, 2> for PairwiseDownsampleBackward
where
    B: AsrAutodiffKernelBackend,
{
    type State = PairwiseDownsampleState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let [node_input, node_weights] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        let (grad_input, grad_weights) = pairwise_downsample_backward_primitive::<B>(
            ops.state.input,
            ops.state.weights,
            &ops.state.lengths,
            grad,
        );

        if let Some(node) = node_input {
            grads.register::<B>(node.id, grad_input);
        }
        if let Some(node) = node_weights {
            grads.register::<B>(node.id, grad_weights);
        }
    }
}

fn relative_shift_primitive<B>(
    input: B::FloatTensorPrimitive,
    seq_len: usize,
) -> B::FloatTensorPrimitive
where
    B: AsrAutodiffKernelBackend,
{
    B::relative_shift_raw(
        Tensor::from_primitive(TensorPrimitive::Float(input)),
        seq_len,
    )
    .into_primitive()
    .tensor()
}

fn mask_time_primitive<B>(
    input: B::FloatTensorPrimitive,
    lengths: &[usize],
) -> B::FloatTensorPrimitive
where
    B: AsrAutodiffKernelBackend,
{
    B::mask_time_raw(
        Tensor::from_primitive(TensorPrimitive::Float(input)),
        lengths,
    )
    .into_primitive()
    .tensor()
}

fn mask_channel_time_primitive<B>(
    input: B::FloatTensorPrimitive,
    lengths: &[usize],
) -> B::FloatTensorPrimitive
where
    B: AsrAutodiffKernelBackend,
{
    B::mask_channel_time_raw(
        Tensor::from_primitive(TensorPrimitive::Float(input)),
        lengths,
    )
    .into_primitive()
    .tensor()
}

fn residual_add_mask_time_primitive<B>(
    residual: B::FloatTensorPrimitive,
    update: B::FloatTensorPrimitive,
    lengths: &[usize],
) -> B::FloatTensorPrimitive
where
    B: AsrAutodiffKernelBackend,
{
    B::residual_add_mask_time_raw(
        Tensor::from_primitive(TensorPrimitive::Float(residual)),
        Tensor::from_primitive(TensorPrimitive::Float(update)),
        lengths,
    )
    .into_primitive()
    .tensor()
}

fn glu_primitive<B>(input: B::FloatTensorPrimitive, layout: GluLayout) -> B::FloatTensorPrimitive
where
    B: AsrAutodiffKernelBackend,
{
    let input = Tensor::from_primitive(TensorPrimitive::Float(input));
    match layout {
        GluLayout::Last => B::glu_last_dim_raw(input),
        GluLayout::Channel => B::glu_channel_dim_raw(input),
    }
    .into_primitive()
    .tensor()
}

fn glu_backward_primitive<B>(
    input: B::FloatTensorPrimitive,
    grad: B::FloatTensorPrimitive,
    layout: GluLayout,
) -> B::FloatTensorPrimitive
where
    B: AsrAutodiffKernelBackend,
{
    let input: Tensor<B, 3> = Tensor::from_primitive(TensorPrimitive::Float(input));
    let grad: Tensor<B, 3> = Tensor::from_primitive(TensorPrimitive::Float(grad));
    let dim = match layout {
        GluLayout::Last => 2,
        GluLayout::Channel => 1,
    };
    let mut chunks = input.chunk(2, dim);
    let gate = chunks.remove(1);
    let value = chunks.remove(0);
    let sig = sigmoid(gate);
    let grad_value = grad.clone() * sig.clone();
    let grad_gate = grad * value * sig.clone() * (Tensor::ones(sig.shape(), &sig.device()) - sig);
    Tensor::cat(vec![grad_value, grad_gate], dim)
        .into_primitive()
        .tensor()
}

fn relative_shift_backward_primitive<B>(
    grad: B::FloatTensorPrimitive,
    input_shape: [usize; 4],
    seq_len: usize,
) -> B::FloatTensorPrimitive
where
    B: AsrAutodiffKernelBackend,
{
    let grad: Tensor<B, 4> = Tensor::from_primitive(TensorPrimitive::Float(grad));
    let [batch, heads, time, pos_len] = input_shape;
    let device = grad.device();
    let mut output = Tensor::<B, 4>::zeros(input_shape, &device);

    for query in 0..seq_len {
        for key in 0..seq_len {
            let flat = query * pos_len + key;
            let shifted_row = flat / seq_len + 1;
            let shifted_col = flat % seq_len;
            let padded_flat = shifted_row * seq_len + shifted_col;
            let source_query = padded_flat / (pos_len + 1);
            let source_pos = padded_flat % (pos_len + 1);
            if source_query < time && source_pos < pos_len {
                let value =
                    grad.clone()
                        .slice([0..batch, 0..heads, query..query + 1, key..key + 1]);
                output = output.slice_assign(
                    [
                        0..batch,
                        0..heads,
                        source_query..source_query + 1,
                        source_pos..source_pos + 1,
                    ],
                    value,
                );
            }
        }
    }

    output.into_primitive().tensor()
}

fn pairwise_downsample_primitive<B>(
    input: B::FloatTensorPrimitive,
    lengths: &[usize],
    weights: B::FloatTensorPrimitive,
) -> B::FloatTensorPrimitive
where
    B: AsrAutodiffKernelBackend,
{
    B::pairwise_downsample_raw(
        Tensor::from_primitive(TensorPrimitive::Float(input)),
        lengths,
        Tensor::from_primitive(TensorPrimitive::Float(weights)),
    )
    .into_primitive()
    .tensor()
}

fn pairwise_downsample_backward_primitive<B>(
    input: B::FloatTensorPrimitive,
    weights: B::FloatTensorPrimitive,
    lengths: &[usize],
    grad: B::FloatTensorPrimitive,
) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive)
where
    B: AsrAutodiffKernelBackend,
{
    let input: Tensor<B, 3> = Tensor::from_primitive(TensorPrimitive::Float(input));
    let weights: Tensor<B, 1> = Tensor::from_primitive(TensorPrimitive::Float(weights));
    let grad: Tensor<B, 3> = Tensor::from_primitive(TensorPrimitive::Float(grad));
    let [batch, seq_len, channels] = input.dims();
    let output_len = seq_len.div_ceil(2);
    let padded_len = output_len * 2;
    let pad = padded_len - seq_len;
    let device = input.device();

    let window_input = if pad > 0 {
        input.clone().pad([(0, pad), (0, 0)], PadMode::Edge)
    } else {
        input.clone()
    }
    .reshape([batch, output_len, 2, channels]);

    let weights_soft = softmax(weights, 0);
    let mask = padded_sequence_mask::<B>(lengths, seq_len, padded_len, &device)
        .float()
        .reshape([batch, output_len, 2, 1]);
    let masked_weights = weights_soft.clone().reshape([1, 1, 2, 1]) * mask;
    let denom = masked_weights
        .clone()
        .sum_dim(2)
        .reshape([batch, output_len, 1])
        .clamp_min(1.0e-8);
    let alpha = masked_weights / denom.clone().unsqueeze_dim::<4>(2);
    let output = (window_input.clone() * alpha.clone())
        .sum_dim(2)
        .reshape([batch, output_len, channels]);

    let grad_window = grad.clone().unsqueeze_dim::<4>(2) * alpha;
    let grad_padded = grad_window.reshape([batch, padded_len, channels]);
    let grad_input = if pad > 0 {
        let base = grad_padded
            .clone()
            .slice([0..batch, 0..seq_len, 0..channels]);
        let last = base
            .clone()
            .slice([0..batch, seq_len - 1..seq_len, 0..channels])
            + grad_padded.slice([0..batch, seq_len..seq_len + 1, 0..channels]);
        base.slice_assign([0..batch, seq_len - 1..seq_len, 0..channels], last)
    } else {
        grad_padded.slice([0..batch, 0..seq_len, 0..channels])
    };

    let centered = window_input - output.unsqueeze_dim::<4>(2);
    let grad_per_soft_weight = (grad.unsqueeze_dim::<4>(2) * centered
        / denom.unsqueeze_dim::<4>(2))
    .sum_dim(0)
    .sum_dim(1)
    .sum_dim(3)
    .reshape([2]);
    let dot = (grad_per_soft_weight.clone() * weights_soft.clone()).sum();
    let grad_weights = weights_soft.clone() * (grad_per_soft_weight - dot);

    (
        grad_input.into_primitive().tensor(),
        grad_weights.into_primitive().tensor(),
    )
}

fn padded_sequence_mask<B: Backend>(
    lengths: &[usize],
    original_len: usize,
    padded_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, burn::tensor::Bool> {
    let mut values = Vec::with_capacity(lengths.len() * padded_len);
    for length in lengths {
        for index in 0..padded_len {
            let source_index = index.min(original_len.saturating_sub(1));
            values.push(source_index < *length);
        }
    }
    Tensor::from_data(
        burn::tensor::TensorData::new(values, [lengths.len(), padded_len]),
        device,
    )
}

fn downsample_lengths(lengths: &[usize], factor: usize) -> Vec<usize> {
    lengths
        .iter()
        .map(|length| {
            if *length == 0 {
                0
            } else {
                length.div_ceil(factor)
            }
        })
        .collect()
}
