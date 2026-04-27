use anyhow::Result;
use clap::{Parser, ValueEnum};

#[cfg(all(
    feature = "asr-cubecl-kernels",
    any(feature = "burn-wgpu-backend", feature = "burn-cuda-backend")
))]
#[allow(dead_code)]
#[path = "../cubecl_kernels.rs"]
mod cubecl_kernels;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BenchBackend {
    Wgpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum WgpuDeviceKind {
    Default,
    Discrete,
    Integrated,
    Cpu,
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Benchmark custom ASR CubeCL kernels on WGPU or CUDA"
)]
struct Args {
    /// Backend runtime to benchmark.
    #[arg(long, value_enum, default_value_t = BenchBackend::Wgpu)]
    backend: BenchBackend,

    /// Number of measured iterations per kernel.
    #[arg(long, default_value_t = 100)]
    iters: usize,

    /// Number of unmeasured warmup iterations per kernel.
    #[arg(long, default_value_t = 10)]
    warmup: usize,

    /// Batch size.
    #[arg(long, default_value_t = 8)]
    batch: usize,

    /// Number of attention heads.
    #[arg(long, default_value_t = 8)]
    heads: usize,

    /// Sequence length.
    #[arg(long, default_value_t = 512)]
    seq_len: usize,

    /// Channel dimension for 3-D activations.
    #[arg(long, default_value_t = 512)]
    channels: usize,

    /// WGPU device class. Ignored for CUDA.
    #[arg(long, value_enum, default_value_t = WgpuDeviceKind::Default)]
    device: WgpuDeviceKind,

    /// Device index for WGPU discrete/integrated GPU or CUDA device selection.
    #[arg(long, default_value_t = 0)]
    device_index: usize,
}

#[cfg(all(
    feature = "asr-cubecl-kernels",
    any(feature = "burn-wgpu-backend", feature = "burn-cuda-backend")
))]
fn main() -> Result<()> {
    bench::run(Args::parse())
}

#[cfg(not(all(
    feature = "asr-cubecl-kernels",
    any(feature = "burn-wgpu-backend", feature = "burn-cuda-backend")
)))]
fn main() -> Result<()> {
    let _args = Args::parse();
    anyhow::bail!(
        "asr-kernel-bench requires --features asr-cubecl-kernels plus burn-wgpu-backend or burn-cuda-backend"
    )
}

#[cfg(all(
    feature = "asr-cubecl-kernels",
    any(feature = "burn-wgpu-backend", feature = "burn-cuda-backend")
))]
mod bench {
    use super::{Args, BenchBackend};
    use crate::cubecl_kernels;
    use anyhow::{Result, bail};
    use burn::tensor::activation::{sigmoid, softmax};
    use burn::tensor::ops::PadMode;
    use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};
    use std::time::{Duration, Instant};

    struct BenchCase {
        name: &'static str,
        elements: usize,
        standard_elapsed: Duration,
        kernel_elapsed: Duration,
    }

    pub fn run(args: Args) -> Result<()> {
        validate(&args)?;
        match args.backend {
            BenchBackend::Wgpu => run_wgpu(args),
            BenchBackend::Cuda => run_cuda(args),
        }
    }

    fn validate(args: &Args) -> Result<()> {
        if args.iters == 0 {
            bail!("--iters must be greater than zero");
        }
        if args.batch == 0 || args.heads == 0 || args.seq_len == 0 || args.channels == 0 {
            bail!("--batch, --heads, --seq-len, and --channels must be greater than zero");
        }
        Ok(())
    }

    #[cfg(feature = "burn-wgpu-backend")]
    fn run_wgpu(args: Args) -> Result<()> {
        use super::WgpuDeviceKind;
        use burn_wgpu::{Wgpu, WgpuDevice};
        type B = Wgpu<f32, i32, u32>;

        let device = match args.device {
            WgpuDeviceKind::Default => WgpuDevice::DefaultDevice,
            WgpuDeviceKind::Discrete => WgpuDevice::DiscreteGpu(args.device_index),
            WgpuDeviceKind::Integrated => WgpuDevice::IntegratedGpu(args.device_index),
            WgpuDeviceKind::Cpu => WgpuDevice::Cpu,
        };
        run_backend_body!(B, "wgpu", &device, &args)
    }

    #[cfg(not(feature = "burn-wgpu-backend"))]
    fn run_wgpu(_args: Args) -> Result<()> {
        bail!("WGPU benchmarking requires --features burn-wgpu-backend")
    }

    #[cfg(feature = "burn-cuda-backend")]
    fn run_cuda(args: Args) -> Result<()> {
        use burn_cuda::{Cuda, CudaDevice};
        type B = Cuda<f32, i32>;

        let device = CudaDevice {
            index: args.device_index,
        };
        run_backend_body!(B, "cuda", &device, &args)
    }

    #[cfg(not(feature = "burn-cuda-backend"))]
    fn run_cuda(_args: Args) -> Result<()> {
        bail!("CUDA benchmarking requires --features burn-cuda-backend")
    }

    macro_rules! run_backend_body {
        ($backend_type:ty, $backend:expr, $device:expr, $args:expr) => {{
            println!(
                "backend={} device={:?} batch={} heads={} seq_len={} channels={} warmup={} iters={}",
                $backend,
                $device,
                $args.batch,
                $args.heads,
                $args.seq_len,
                $args.channels,
                $args.warmup,
                $args.iters
            );

            let lengths = lengths($args.batch, $args.seq_len);
            let lengths_values = lengths
                .iter()
                .map(|length| i32::try_from(*length).expect("benchmark length must fit in i32"))
                .collect::<Vec<_>>();
            let lengths_tensor =
                Tensor::<$backend_type, 1, Int>::from_ints(lengths_values.as_slice(), $device);
            let mut cases = Vec::new();

            let activation = Tensor::<$backend_type, 3>::ones(
                [$args.batch, $args.seq_len, $args.channels],
                $device,
            );
            cases.push(run_comparison_case::<$backend_type, _, _>(
                "swoosh_l",
                $args.iters,
                $args.warmup,
                $args.batch * $args.seq_len * $args.channels,
                || {
                    let _ = standard_swoosh_l(activation.clone());
                },
                || {
                    let _ = cubecl_kernels::swoosh_l(activation.clone());
                },
                $device,
            )?);
            cases.push(run_comparison_case::<$backend_type, _, _>(
                "swoosh_r",
                $args.iters,
                $args.warmup,
                $args.batch * $args.seq_len * $args.channels,
                || {
                    let _ = standard_swoosh_r(activation.clone());
                },
                || {
                    let _ = cubecl_kernels::swoosh_r(activation.clone());
                },
                $device,
            )?);

            let rel_pos_len = $args.seq_len * 2 - 1;
            let rel = Tensor::<$backend_type, 4>::ones(
                [$args.batch, $args.heads, $args.seq_len, rel_pos_len],
                $device,
            );
            cases.push(run_comparison_case::<$backend_type, _, _>(
                "relative_shift",
                $args.iters,
                $args.warmup,
                $args.batch * $args.heads * $args.seq_len * $args.seq_len,
                || {
                    let _ = standard_relative_shift(rel.clone(), $args.seq_len);
                },
                || {
                    let _ = cubecl_kernels::relative_shift(rel.clone(), $args.seq_len);
                },
                $device,
            )?);

            cases.push(run_comparison_case::<$backend_type, _, _>(
                "mask_time",
                $args.iters,
                $args.warmup,
                $args.batch * $args.seq_len * $args.channels,
                || {
                    let _ = standard_mask_time(activation.clone(), &lengths);
                },
                || {
                    let _ = cubecl_kernels::mask_time(activation.clone(), &lengths);
                },
                $device,
            )?);
            cases.push(run_comparison_case::<$backend_type, _, _>(
                "residual_add_mask_time",
                $args.iters,
                $args.warmup,
                $args.batch * $args.seq_len * $args.channels,
                || {
                    let _ = standard_mask_time(activation.clone() + activation.clone(), &lengths);
                },
                || {
                    let _ = cubecl_kernels::residual_add_mask_time(
                        activation.clone(),
                        activation.clone(),
                        lengths_tensor.clone(),
                    );
                },
                $device,
            )?);

            let channel_time = Tensor::<$backend_type, 3>::ones(
                [$args.batch, $args.channels, $args.seq_len],
                $device,
            );
            cases.push(run_comparison_case::<$backend_type, _, _>(
                "mask_channel_time",
                $args.iters,
                $args.warmup,
                $args.batch * $args.channels * $args.seq_len,
                || {
                    let _ = standard_mask_channel_time(channel_time.clone(), &lengths);
                },
                || {
                    let _ = cubecl_kernels::mask_channel_time(channel_time.clone(), &lengths);
                },
                $device,
            )?);

            let downsample_weights = Tensor::<$backend_type, 1>::from_data(
                TensorData::new(vec![0.25f32, -0.25], [2]),
                $device,
            );
            cases.push(run_comparison_case::<$backend_type, _, _>(
                "pairwise_downsample",
                $args.iters,
                $args.warmup,
                $args.batch * $args.seq_len.div_ceil(2) * $args.channels,
                || {
                    let _ = standard_pairwise_downsample(
                        activation.clone(),
                        &lengths,
                        downsample_weights.clone(),
                    );
                },
                || {
                    let _ = cubecl_kernels::pairwise_downsample(
                        activation.clone(),
                        lengths_tensor.clone(),
                        downsample_weights.clone(),
                    );
                },
                $device,
            )?);

            cases.push(run_comparison_case::<$backend_type, _, _>(
                "sequence_mask",
                $args.iters,
                $args.warmup,
                $args.batch * $args.seq_len,
                || {
                    let _ = standard_sequence_mask::<$backend_type>(&lengths, $args.seq_len, $device);
                },
                || {
                    let _ = cubecl_kernels::sequence_mask_with_lengths(
                        lengths_tensor.clone(),
                        $args.seq_len,
                    );
                },
                $device,
            )?);

            cases.push(run_comparison_case::<$backend_type, _, _>(
                "padding_mask",
                $args.iters,
                $args.warmup,
                $args.batch * $args.seq_len,
                || {
                    let _ = standard_padding_mask::<$backend_type>(&lengths, $args.seq_len, $device);
                },
                || {
                    let _ = cubecl_kernels::padding_mask_with_lengths(
                        lengths_tensor.clone(),
                        $args.seq_len,
                    );
                },
                $device,
            )?);

            cases.push(run_comparison_case::<$backend_type, _, _>(
                "attention_mask",
                $args.iters,
                $args.warmup,
                $args.batch * $args.seq_len * $args.seq_len,
                || {
                    let _ = standard_attention_mask::<$backend_type>(
                        &lengths,
                        $args.seq_len,
                        $device,
                    );
                },
                || {
                    let _ = cubecl_kernels::attention_mask_with_lengths(
                        lengths_tensor.clone(),
                        $args.seq_len,
                        $args.seq_len,
                    );
                },
                $device,
            )?);
            cases.push(run_comparison_case::<$backend_type, _, _>(
                "attention_mask_4d",
                $args.iters,
                $args.warmup,
                $args.batch * $args.heads * $args.seq_len * $args.seq_len,
                || {
                    let _ = standard_attention_mask_4d::<$backend_type>(
                        &lengths,
                        $args.heads,
                        $args.seq_len,
                        $args.seq_len,
                        $device,
                    );
                },
                || {
                    let _ = cubecl_kernels::attention_mask_4d_with_lengths(
                        lengths_tensor.clone(),
                        $args.heads,
                        $args.seq_len,
                        $args.seq_len,
                    );
                },
                $device,
            )?);

            let glu_last = Tensor::<$backend_type, 3>::ones(
                [$args.batch, $args.seq_len, $args.channels * 2],
                $device,
            );
            cases.push(run_comparison_case::<$backend_type, _, _>(
                "glu_last_dim",
                $args.iters,
                $args.warmup,
                $args.batch * $args.seq_len * $args.channels,
                || {
                    let _ = standard_glu_last_dim(glu_last.clone());
                },
                || {
                    let _ = cubecl_kernels::glu_last_dim(glu_last.clone());
                },
                $device,
            )?);

            let glu_channel = Tensor::<$backend_type, 3>::ones(
                [$args.batch, $args.channels * 2, $args.seq_len],
                $device,
            );
            cases.push(run_comparison_case::<$backend_type, _, _>(
                "glu_channel_dim",
                $args.iters,
                $args.warmup,
                $args.batch * $args.channels * $args.seq_len,
                || {
                    let _ = standard_glu_channel_dim(glu_channel.clone());
                },
                || {
                    let _ = cubecl_kernels::glu_channel_dim(glu_channel.clone());
                },
                $device,
            )?);

            print_results(&cases, $args.iters);
            Ok(())
        }};
    }

    use run_backend_body;

    fn run_comparison_case<B, Standard, Kernel>(
        name: &'static str,
        iters: usize,
        warmup: usize,
        elements: usize,
        mut standard: Standard,
        mut kernel: Kernel,
        device: &B::Device,
    ) -> Result<BenchCase>
    where
        B: Backend,
        Standard: FnMut(),
        Kernel: FnMut(),
    {
        for _ in 0..warmup {
            standard();
        }
        B::sync(device)?;
        for _ in 0..warmup {
            kernel();
        }
        B::sync(device)?;

        let standard_start = Instant::now();
        for _ in 0..iters {
            standard();
        }
        B::sync(device)?;
        let standard_elapsed = standard_start.elapsed();

        let kernel_start = Instant::now();
        for _ in 0..iters {
            kernel();
        }
        B::sync(device)?;
        let kernel_elapsed = kernel_start.elapsed();

        Ok(BenchCase {
            name,
            elements,
            standard_elapsed,
            kernel_elapsed,
        })
    }

    fn print_results(cases: &[BenchCase], iters: usize) {
        println!();
        println!(
            "{:<24} {:>14} {:>14} {:>10} {:>14} {:>14}",
            "kernel", "standard_us", "cubecl_us", "speedup", "elems/iter", "cubecl_me/s"
        );
        for case in cases {
            let standard_avg = case.standard_elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
            let kernel_avg = case.kernel_elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
            let speedup = standard_avg / kernel_avg;
            let melems = case.elements as f64 * iters as f64
                / case.kernel_elapsed.as_secs_f64()
                / 1_000_000.0;
            println!(
                "{:<24} {:>14.3} {:>14.3} {:>9.2}x {:>14} {:>14.3}",
                case.name, standard_avg, kernel_avg, speedup, case.elements, melems
            );
        }
    }

    fn standard_swoosh_l<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
        input.clone() * sigmoid(input - 4.0)
    }

    fn standard_swoosh_r<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
        input.clone() * sigmoid(input - 1.0)
    }

    fn standard_relative_shift<B: Backend>(input: Tensor<B, 4>, seq_len: usize) -> Tensor<B, 4> {
        let [batch_size, n_heads, _, pos_len] = input.dims();
        let padded = input.pad([(0, 0), (0, 1)], PadMode::Constant(0.0));
        padded
            .reshape([batch_size, n_heads, pos_len + 1, seq_len])
            .slice_dim(2, 1..pos_len + 1)
            .reshape([batch_size, n_heads, seq_len, pos_len])
            .slice_dim(3, 0..seq_len)
    }

    fn standard_sequence_mask<B: Backend>(
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

    fn standard_padding_mask<B: Backend>(
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

    fn standard_attention_mask<B: Backend>(
        lengths: &[usize],
        max_len: usize,
        device: &B::Device,
    ) -> Tensor<B, 3, Bool> {
        let mask = standard_sequence_mask::<B>(lengths, max_len, device);
        mask.clone()
            .unsqueeze_dim::<3>(1)
            .repeat_dim(1, max_len)
            .bool_and(mask.unsqueeze_dim::<3>(2).repeat_dim(2, max_len))
    }

    fn standard_attention_mask_4d<B: Backend>(
        lengths: &[usize],
        heads: usize,
        query_len: usize,
        key_len: usize,
        device: &B::Device,
    ) -> Tensor<B, 4, Bool> {
        let mut values = Vec::with_capacity(lengths.len() * heads * query_len * key_len);
        for length in lengths {
            for _ in 0..heads {
                for query in 0..query_len {
                    for key in 0..key_len {
                        values.push(query < *length && key < *length);
                    }
                }
            }
        }
        Tensor::from_data(
            TensorData::new(values, [lengths.len(), heads, query_len, key_len]),
            device,
        )
    }

    fn standard_mask_time<B: Backend>(input: Tensor<B, 3>, lengths: &[usize]) -> Tensor<B, 3> {
        let [_, seq_len, _] = input.dims();
        let mask = standard_sequence_mask::<B>(lengths, seq_len, &input.device())
            .float()
            .unsqueeze_dim::<3>(2);
        input * mask
    }

    fn standard_mask_channel_time<B: Backend>(
        input: Tensor<B, 3>,
        lengths: &[usize],
    ) -> Tensor<B, 3> {
        let [_, _, seq_len] = input.dims();
        let mask = standard_sequence_mask::<B>(lengths, seq_len, &input.device())
            .float()
            .unsqueeze_dim::<3>(1);
        input * mask
    }

    fn standard_pairwise_downsample<B: Backend>(
        input: Tensor<B, 3>,
        lengths: &[usize],
        weights: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, channels] = input.dims();
        let output_len = seq_len.div_ceil(2);
        let padded_len = output_len * 2;
        let pad = padded_len - seq_len;
        let output = if pad > 0 {
            input.pad([(0, pad), (0, 0)], PadMode::Edge)
        } else {
            input
        };
        let window = output.reshape([batch_size, output_len, 2, channels]);
        let weights = softmax(weights, 0).reshape([1, 1, 2, 1]);
        let mask =
            standard_padded_sequence_mask::<B>(lengths, seq_len, padded_len, &window.device())
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
            .reshape([batch_size, output_len, channels])
            / denom;
        standard_mask_time(
            output,
            &lengths
                .iter()
                .map(|length| length.div_ceil(2))
                .collect::<Vec<_>>(),
        )
    }

    fn standard_glu_last_dim<B: Backend>(input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut chunks = input.chunk(2, 2);
        let gate = chunks.remove(1);
        let value = chunks.remove(0);
        value * sigmoid(gate)
    }

    fn standard_glu_channel_dim<B: Backend>(input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut chunks = input.chunk(2, 1);
        let gate = chunks.remove(1);
        let value = chunks.remove(0);
        value * sigmoid(gate)
    }

    fn lengths(batch: usize, seq_len: usize) -> Vec<usize> {
        (0..batch)
            .map(|index| {
                let trim = index % seq_len.max(1);
                seq_len.saturating_sub(trim).max(1)
            })
            .collect()
    }

    fn standard_padded_sequence_mask<B: Backend>(
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
}
