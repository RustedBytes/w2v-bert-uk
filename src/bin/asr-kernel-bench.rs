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
    use burn::tensor::{Int, Tensor, backend::Backend};
    use std::time::{Duration, Instant};

    struct BenchCase {
        name: &'static str,
        elements: usize,
        elapsed: Duration,
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
            cases.push(run_case::<$backend_type, _>(
                "swoosh_l",
                $args.iters,
                $args.warmup,
                $args.batch * $args.seq_len * $args.channels,
                || {
                    let _ = cubecl_kernels::swoosh_l(activation.clone());
                },
                $device,
            )?);
            cases.push(run_case::<$backend_type, _>(
                "swoosh_r",
                $args.iters,
                $args.warmup,
                $args.batch * $args.seq_len * $args.channels,
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
            cases.push(run_case::<$backend_type, _>(
                "relative_shift",
                $args.iters,
                $args.warmup,
                $args.batch * $args.heads * $args.seq_len * $args.seq_len,
                || {
                    let _ = cubecl_kernels::relative_shift(rel.clone(), $args.seq_len);
                },
                $device,
            )?);

            cases.push(run_case::<$backend_type, _>(
                "mask_time",
                $args.iters,
                $args.warmup,
                $args.batch * $args.seq_len * $args.channels,
                || {
                    let _ = cubecl_kernels::mask_time(activation.clone(), &lengths);
                },
                $device,
            )?);

            let channel_time = Tensor::<$backend_type, 3>::ones(
                [$args.batch, $args.channels, $args.seq_len],
                $device,
            );
            cases.push(run_case::<$backend_type, _>(
                "mask_channel_time",
                $args.iters,
                $args.warmup,
                $args.batch * $args.channels * $args.seq_len,
                || {
                    let _ = cubecl_kernels::mask_channel_time(channel_time.clone(), &lengths);
                },
                $device,
            )?);

            cases.push(run_case::<$backend_type, _>(
                "sequence_mask",
                $args.iters,
                $args.warmup,
                $args.batch * $args.seq_len,
                || {
                    let _ = cubecl_kernels::sequence_mask_with_lengths(
                        lengths_tensor.clone(),
                        $args.seq_len,
                    );
                },
                $device,
            )?);

            cases.push(run_case::<$backend_type, _>(
                "padding_mask",
                $args.iters,
                $args.warmup,
                $args.batch * $args.seq_len,
                || {
                    let _ = cubecl_kernels::padding_mask_with_lengths(
                        lengths_tensor.clone(),
                        $args.seq_len,
                    );
                },
                $device,
            )?);

            cases.push(run_case::<$backend_type, _>(
                "attention_mask",
                $args.iters,
                $args.warmup,
                $args.batch * $args.seq_len * $args.seq_len,
                || {
                    let _ = cubecl_kernels::attention_mask_with_lengths(
                        lengths_tensor.clone(),
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
            cases.push(run_case::<$backend_type, _>(
                "glu_last_dim",
                $args.iters,
                $args.warmup,
                $args.batch * $args.seq_len * $args.channels,
                || {
                    let _ = cubecl_kernels::glu_last_dim(glu_last.clone());
                },
                $device,
            )?);

            let glu_channel = Tensor::<$backend_type, 3>::ones(
                [$args.batch, $args.channels * 2, $args.seq_len],
                $device,
            );
            cases.push(run_case::<$backend_type, _>(
                "glu_channel_dim",
                $args.iters,
                $args.warmup,
                $args.batch * $args.channels * $args.seq_len,
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

    fn run_case<B, F>(
        name: &'static str,
        iters: usize,
        warmup: usize,
        elements: usize,
        mut f: F,
        device: &B::Device,
    ) -> Result<BenchCase>
    where
        B: Backend,
        F: FnMut(),
    {
        for _ in 0..warmup {
            f();
        }
        B::sync(device)?;

        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        B::sync(device)?;

        Ok(BenchCase {
            name,
            elements,
            elapsed: start.elapsed(),
        })
    }

    fn print_results(cases: &[BenchCase], iters: usize) {
        println!();
        println!(
            "{:<24} {:>14} {:>14} {:>14}",
            "kernel", "avg_us", "elems/iter", "melems/s"
        );
        for case in cases {
            let avg = case.elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
            let melems =
                case.elements as f64 * iters as f64 / case.elapsed.as_secs_f64() / 1_000_000.0;
            println!(
                "{:<24} {:>14.3} {:>14} {:>14.3}",
                case.name, avg, case.elements, melems
            );
        }
    }

    fn lengths(batch: usize, seq_len: usize) -> Vec<usize> {
        (0..batch)
            .map(|index| {
                let trim = index % seq_len.max(1);
                seq_len.saturating_sub(trim).max(1)
            })
            .collect()
    }
}
