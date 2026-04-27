//! Optional CubeCL kernels for ASR architecture hot spots.
//!
//! The existing Squeezeformer, Zipformer, Paraformer, and W2V-BERT modules are
//! already mostly composed of Burn tensor ops, so Burn/CubeCL provides the heavy
//! convolution, matmul, softmax, and elementwise kernels for CUDA/WGPU backends.
//! This module keeps the custom architecture-specific pieces isolated behind
//! `asr-cubecl-kernels` while Burn's default path remains the portable fallback.

use burn::tensor::{Int as BurnInt, Shape, Tensor as BurnTensor, TensorMetadata, TensorPrimitive};
use burn_cubecl::cubecl::prelude::InputScalar;
use burn_cubecl::{
    CubeBackend, CubeRuntime, FloatElement, IntElement,
    cubecl::{calculate_cube_count_elemwise, prelude::*},
    element::BoolElement,
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};

/// CubeCL-backed tensor type accepted by these custom kernels.
pub type AsrCubeBackend<R, F = f32, I = i32, BT = u8> = CubeBackend<R, F, I, BT>;

/// Architecture-specific custom kernel targets found in the current codebase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsrKernelTarget {
    /// Zipformer's `x * sigmoid(x - 4)` activation.
    ZipformerSwooshL,
    /// Zipformer's `x * sigmoid(x - 1)` activation.
    ZipformerSwooshR,
    /// Squeezeformer/Zipformer relative-position attention shift.
    RelativeShift,
    /// Apply a `[batch, time]` length mask to `[batch, time, channels]`.
    MaskTime,
    /// Apply a `[batch, time]` length mask to `[batch, channels, time]`.
    MaskChannelTime,
}

/// Static analysis of the current ASR architectures and where custom CubeCL
/// kernels add value beyond Burn's built-in kernels.
pub const ASR_KERNEL_TARGETS: &[AsrKernelTarget] = &[
    AsrKernelTarget::ZipformerSwooshL,
    AsrKernelTarget::ZipformerSwooshR,
    AsrKernelTarget::RelativeShift,
    AsrKernelTarget::MaskTime,
    AsrKernelTarget::MaskChannelTime,
];

/// Fused Zipformer Swoosh-L activation for non-fusion CubeCL backends.
pub fn zipformer_swoosh_l<R, F, I, BT, const D: usize>(
    input: BurnTensor<AsrCubeBackend<R, F, I, BT>, D>,
) -> BurnTensor<AsrCubeBackend<R, F, I, BT>, D>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    swoosh_l(input)
}

/// Fused Zipformer Swoosh-R activation for non-fusion CubeCL backends.
pub fn zipformer_swoosh_r<R, F, I, BT, const D: usize>(
    input: BurnTensor<AsrCubeBackend<R, F, I, BT>, D>,
) -> BurnTensor<AsrCubeBackend<R, F, I, BT>, D>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    swoosh_r(input)
}

/// Fused `x * sigmoid(x - 4)` activation.
pub fn swoosh_l<R, F, I, BT, const D: usize>(
    input: BurnTensor<AsrCubeBackend<R, F, I, BT>, D>,
) -> BurnTensor<AsrCubeBackend<R, F, I, BT>, D>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    swoosh_with_offset(input, 4.0)
}

/// Fused `x * sigmoid(x - 1)` activation.
pub fn swoosh_r<R, F, I, BT, const D: usize>(
    input: BurnTensor<AsrCubeBackend<R, F, I, BT>, D>,
) -> BurnTensor<AsrCubeBackend<R, F, I, BT>, D>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    swoosh_with_offset(input, 1.0)
}

/// Relative-position shift used by Squeezeformer and Zipformer attention.
///
/// Input shape is `[batch, heads, seq_len, pos_len]`; output shape is
/// `[batch, heads, seq_len, seq_len]`.
pub fn relative_shift<R, F, I, BT>(
    input: BurnTensor<AsrCubeBackend<R, F, I, BT>, 4>,
    seq_len: usize,
) -> BurnTensor<AsrCubeBackend<R, F, I, BT>, 4>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let primitive = into_float_cube(input);
    let output = relative_shift_cube(primitive, seq_len);
    BurnTensor::from_primitive(TensorPrimitive::Float(output))
}

/// Fused length masking for `[batch, time, channels]` tensors.
///
/// This is the CubeCL fast path for the generic fallback pattern:
/// `input * sequence_mask(lengths).float().unsqueeze_dim(2)`.
pub fn mask_time<R, F, I, BT>(
    input: BurnTensor<AsrCubeBackend<R, F, I, BT>, 3>,
    lengths: &[usize],
) -> BurnTensor<AsrCubeBackend<R, F, I, BT>, 3>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let device = input.device();
    let lengths = lengths_tensor::<R, F, I, BT>(lengths, &device);
    mask_time_with_lengths(input, lengths)
}

/// Fused length masking for `[batch, time, channels]` tensors using an existing
/// device-side lengths tensor.
pub fn mask_time_with_lengths<R, F, I, BT>(
    input: BurnTensor<AsrCubeBackend<R, F, I, BT>, 3>,
    lengths: BurnTensor<AsrCubeBackend<R, F, I, BT>, 1, BurnInt>,
) -> BurnTensor<AsrCubeBackend<R, F, I, BT>, 3>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let primitive = into_float_cube(input);
    let lengths = lengths.into_primitive();
    let output = mask_time_cube(primitive, lengths, MaskLayout::BatchTimeChannels);
    BurnTensor::from_primitive(TensorPrimitive::Float(output))
}

/// Fused length masking for `[batch, channels, time]` tensors.
///
/// This is the CubeCL fast path for the generic fallback pattern:
/// `input * sequence_mask(lengths).float().unsqueeze_dim(1)`.
pub fn mask_channel_time<R, F, I, BT>(
    input: BurnTensor<AsrCubeBackend<R, F, I, BT>, 3>,
    lengths: &[usize],
) -> BurnTensor<AsrCubeBackend<R, F, I, BT>, 3>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let device = input.device();
    let lengths = lengths_tensor::<R, F, I, BT>(lengths, &device);
    mask_channel_time_with_lengths(input, lengths)
}

/// Fused length masking for `[batch, channels, time]` tensors using an existing
/// device-side lengths tensor.
pub fn mask_channel_time_with_lengths<R, F, I, BT>(
    input: BurnTensor<AsrCubeBackend<R, F, I, BT>, 3>,
    lengths: BurnTensor<AsrCubeBackend<R, F, I, BT>, 1, BurnInt>,
) -> BurnTensor<AsrCubeBackend<R, F, I, BT>, 3>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let primitive = into_float_cube(input);
    let lengths = lengths.into_primitive();
    let output = mask_time_cube(primitive, lengths, MaskLayout::BatchChannelsTime);
    BurnTensor::from_primitive(TensorPrimitive::Float(output))
}

fn swoosh_with_offset<R, F, I, BT, const D: usize>(
    input: BurnTensor<AsrCubeBackend<R, F, I, BT>, D>,
    offset: f32,
) -> BurnTensor<AsrCubeBackend<R, F, I, BT>, D>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let primitive = into_float_cube(input);
    let output = swoosh_cube(primitive, offset);
    BurnTensor::from_primitive(TensorPrimitive::Float(output))
}

fn into_float_cube<R, F, I, BT, const D: usize>(
    input: BurnTensor<AsrCubeBackend<R, F, I, BT>, D>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    input.into_primitive().tensor()
}

fn lengths_tensor<R, F, I, BT>(
    lengths: &[usize],
    device: &<AsrCubeBackend<R, F, I, BT> as burn::tensor::backend::Backend>::Device,
) -> BurnTensor<AsrCubeBackend<R, F, I, BT>, 1, BurnInt>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let values = lengths
        .iter()
        .map(|length| i32::try_from(*length).expect("ASR CubeCL length values must fit in i32"))
        .collect::<Vec<_>>();
    BurnTensor::from_ints(values.as_slice(), device)
}

fn swoosh_cube<R: CubeRuntime>(input: CubeTensor<R>, offset: f32) -> CubeTensor<R> {
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        input.shape(),
        input.dtype,
    );
    let num_elems = output.meta.num_elements();
    let cube_dim = CubeDim::new(&output.client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&output.client, num_elems, cube_dim);
    let dtype = output.dtype;

    swoosh_kernel::launch(
        &output.client,
        cube_count,
        cube_dim,
        input.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        InputScalar::new(offset, dtype),
        dtype.into(),
    );

    output
}

fn relative_shift_cube<R: CubeRuntime>(input: CubeTensor<R>, seq_len: usize) -> CubeTensor<R> {
    let shape = input.shape();
    let [batch, heads, input_seq_len, pos_len] = shape.dims();
    assert_eq!(
        input_seq_len, seq_len,
        "relative_shift seq_len must match input dim 2"
    );
    assert!(
        pos_len >= seq_len,
        "relative_shift pos_len must be at least seq_len"
    );

    let output_shape = Shape::new([batch, heads, seq_len, seq_len]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        output_shape,
        input.dtype,
    );
    let num_elems = output.meta.num_elements();
    let cube_dim = CubeDim::new(&output.client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&output.client, num_elems, cube_dim);
    let dtype = output.dtype;

    relative_shift_kernel::launch(
        &output.client,
        cube_count,
        cube_dim,
        input.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        RelativeShiftArgsLaunch::new(batch, heads, seq_len, pos_len),
        dtype.into(),
    );

    output
}

#[derive(Clone, Copy)]
enum MaskLayout {
    BatchTimeChannels,
    BatchChannelsTime,
}

fn mask_time_cube<R: CubeRuntime>(
    input: CubeTensor<R>,
    lengths: CubeTensor<R>,
    layout: MaskLayout,
) -> CubeTensor<R> {
    let shape = input.shape();
    let [batch, dim1, dim2] = shape.dims();
    assert_eq!(
        lengths.shape().dims::<1>()[0],
        batch,
        "mask lengths must match input batch size"
    );

    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        shape,
        input.dtype,
    );
    let num_elems = output.meta.num_elements();
    let cube_dim = CubeDim::new(&output.client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&output.client, num_elems, cube_dim);
    let dtype = output.dtype;
    let int_dtype = lengths.dtype;

    match layout {
        MaskLayout::BatchTimeChannels => mask_time_kernel::launch(
            &output.client,
            cube_count,
            cube_dim,
            input.into_tensor_arg(),
            lengths.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            MaskTimeArgsLaunch::new(batch, dim1, dim2),
            dtype.into(),
            int_dtype.into(),
        ),
        MaskLayout::BatchChannelsTime => mask_channel_time_kernel::launch(
            &output.client,
            cube_count,
            cube_dim,
            input.into_tensor_arg(),
            lengths.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            MaskTimeArgsLaunch::new(batch, dim1, dim2),
            dtype.into(),
            int_dtype.into(),
        ),
    }

    output
}

#[cube(launch)]
fn swoosh_kernel<E: Float>(
    input: &Tensor<E>,
    output: &mut Tensor<E>,
    offset: InputScalar,
    #[define(E)] _dtype: StorageType,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }

    let x = input[pos];
    let shifted = x - offset.get::<E>();
    let sigmoid = E::new(1.0) / (E::new(1.0) + E::exp(-shifted));
    output[pos] = x * sigmoid;
}

#[derive(CubeLaunch, CubeType)]
struct RelativeShiftArgs {
    #[cube(comptime)]
    batch: usize,
    #[cube(comptime)]
    heads: usize,
    #[cube(comptime)]
    seq_len: usize,
    #[cube(comptime)]
    pos_len: usize,
}

#[derive(CubeLaunch, CubeType)]
struct MaskTimeArgs {
    #[cube(comptime)]
    batch: usize,
    #[cube(comptime)]
    dim1: usize,
    #[cube(comptime)]
    dim2: usize,
}

#[cube(launch)]
fn relative_shift_kernel<E: Float>(
    input: &Tensor<E>,
    output: &mut Tensor<E>,
    args: &RelativeShiftArgs,
    #[define(E)] _dtype: StorageType,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }

    let seq_len = comptime![args.seq_len];
    let pos_len = comptime![args.pos_len];
    let heads = comptime![args.heads];
    let batch = comptime![args.batch];

    let j = pos % seq_len;
    let i = (pos / seq_len) % seq_len;
    let head = (pos / (seq_len * seq_len)) % heads;
    let batch_index = pos / (heads * seq_len * seq_len);

    if batch_index >= batch {
        terminate!();
    }

    let padded_offset = seq_len + i * pos_len + j;
    let source_seq = padded_offset / (pos_len + 1);
    let source_pos = padded_offset % (pos_len + 1);

    if source_pos == pos_len {
        output[pos] = E::new(0.0);
    } else {
        let source = ((batch_index * heads + head) * seq_len + source_seq) * pos_len + source_pos;
        output[pos] = input[source];
    }
}

#[cube(launch)]
fn mask_time_kernel<E: Float, I: Int>(
    input: &Tensor<E>,
    lengths: &Tensor<I>,
    output: &mut Tensor<E>,
    args: &MaskTimeArgs,
    #[define(E)] _dtype: StorageType,
    #[define(I)] _int_dtype: StorageType,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }

    let batch = comptime![args.batch];
    let time = comptime![args.dim1];
    let channels = comptime![args.dim2];

    let channel_index = pos % channels;
    let time_index = (pos / channels) % time;
    let batch_index = pos / (time * channels);

    if batch_index >= batch {
        terminate!();
    }

    let length = usize::cast_from(lengths[batch_index]);
    if channel_index < channels && time_index < length {
        output[pos] = input[pos];
    } else {
        output[pos] = E::new(0.0);
    }
}

#[cube(launch)]
fn mask_channel_time_kernel<E: Float, I: Int>(
    input: &Tensor<E>,
    lengths: &Tensor<I>,
    output: &mut Tensor<E>,
    args: &MaskTimeArgs,
    #[define(E)] _dtype: StorageType,
    #[define(I)] _int_dtype: StorageType,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!();
    }

    let batch = comptime![args.batch];
    let channels = comptime![args.dim1];
    let time = comptime![args.dim2];

    let time_index = pos % time;
    let channel_index = (pos / time) % channels;
    let batch_index = pos / (channels * time);

    if batch_index >= batch {
        terminate!();
    }

    let length = usize::cast_from(lengths[batch_index]);
    if channel_index < channels && time_index < length {
        output[pos] = input[pos];
    } else {
        output[pos] = E::new(0.0);
    }
}
