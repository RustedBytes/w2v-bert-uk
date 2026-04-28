//! Autodiff registrations for custom ASR CubeCL kernels.
//!
//! These wrappers avoid the straight-through fallback pattern for simple mask
//! kernels by registering real Burn backward steps. The backward math for these
//! kernels is another application of the same mask to the upstream gradient.

use burn::tensor::{Tensor, TensorPrimitive, backend::Backend};
use burn_autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};

pub trait AsrAutodiffKernelBackend: Backend {
    fn mask_time_raw(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3>;
    fn mask_channel_time_raw(input: Tensor<Self, 3>, lengths: &[usize]) -> Tensor<Self, 3>;
    fn residual_add_mask_time_raw(
        residual: Tensor<Self, 3>,
        update: Tensor<Self, 3>,
        lengths: &[usize],
    ) -> Tensor<Self, 3>;
}

#[cfg(feature = "burn-cuda-backend")]
impl<F, I> AsrAutodiffKernelBackend for burn_cuda::Cuda<F, I>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
{
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
}

#[cfg(feature = "burn-wgpu-backend")]
impl<F, I, BT> AsrAutodiffKernelBackend for burn_wgpu::Wgpu<F, I, BT>
where
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::element::BoolElement,
{
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
