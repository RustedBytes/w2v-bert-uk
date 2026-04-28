use anyhow::{Result, bail};
use burn::tensor::{Distribution, Tensor, backend::AutodiffBackend, backend::Backend};
use clap::{Parser, ValueEnum};
use std::time::{Duration, Instant};
use w2v_bert_uk::squeezeformer::{SqueezeformerCtcConfig, SqueezeformerEncoderConfig};
use w2v_bert_uk::wav2vec::{Wav2VecBertConfig, Wav2VecBertCtcConfig};
use w2v_bert_uk::zipformer::{ZipformerConfig, ZipformerCtcConfig, ZipformerKernelBackend};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BenchBackend {
    Cpu,
    Cuda,
    Wgpu,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BenchArchitecture {
    Squeezeformer,
    Zipformer,
    W2vBert,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BenchMode {
    Forward,
    ForwardBackward,
    Both,
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Benchmark full ASR model forward and train-surrogate steps"
)]
struct Args {
    /// Backend runtime to benchmark.
    #[arg(long, value_enum, default_value_t = BenchBackend::Cpu)]
    backend: BenchBackend,

    /// Architecture to benchmark.
    #[arg(long, value_enum, default_value_t = BenchArchitecture::Zipformer)]
    architecture: BenchArchitecture,

    /// Model variant where supported.
    #[arg(long, default_value = "xs")]
    variant: String,

    /// Benchmark mode.
    #[arg(long, value_enum, default_value_t = BenchMode::Both)]
    mode: BenchMode,

    /// Number of measured iterations.
    #[arg(long, default_value_t = 10)]
    iters: usize,

    /// Number of unmeasured warmup iterations.
    #[arg(long, default_value_t = 2)]
    warmup: usize,

    /// Batch size.
    #[arg(long, default_value_t = 2)]
    batch: usize,

    /// Input frames before architecture subsampling.
    #[arg(long, default_value_t = 128)]
    frames: usize,

    /// Input feature dimension. Zero selects the architecture default.
    #[arg(long, default_value_t = 0)]
    input_dim: usize,

    /// Vocabulary size.
    #[arg(long, default_value_t = 128)]
    vocab_size: usize,

    /// Hidden/model dimension for custom Squeezeformer or W2V-BERT configs.
    #[arg(long, default_value_t = 256)]
    d_model: usize,

    /// Number of layers for custom Squeezeformer or W2V-BERT configs.
    #[arg(long, default_value_t = 2)]
    num_layers: usize,

    /// Number of attention heads for custom Squeezeformer or W2V-BERT configs.
    #[arg(long, default_value_t = 4)]
    num_heads: usize,

    /// CUDA/WGPU device index.
    #[arg(long, default_value_t = 0)]
    device_index: usize,
}

#[derive(Clone, Debug)]
struct BenchResult {
    name: &'static str,
    avg: Duration,
    output_shape: [usize; 3],
    output_lengths: Vec<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate(&args)?;
    match args.backend {
        BenchBackend::Cpu => run_cpu(args),
        BenchBackend::Cuda => run_cuda(args),
        BenchBackend::Wgpu => run_wgpu(args),
    }
}

fn validate(args: &Args) -> Result<()> {
    if args.iters == 0 {
        bail!("--iters must be greater than zero");
    }
    if args.batch == 0 || args.frames == 0 || args.vocab_size == 0 {
        bail!("--batch, --frames, and --vocab-size must be greater than zero");
    }
    Ok(())
}

fn run_cpu(args: Args) -> Result<()> {
    type B = burn_ndarray::NdArray<f32>;
    let device = Default::default();
    run_backend_body::<B>(&args, "cpu", &device)
}

#[cfg(feature = "burn-cuda-backend")]
fn run_cuda(args: Args) -> Result<()> {
    type B = burn_cuda::Cuda<f32>;
    let device = burn_cuda::CudaDevice {
        index: args.device_index,
    };
    run_backend_body::<B>(&args, "cuda", &device)
}

#[cfg(not(feature = "burn-cuda-backend"))]
fn run_cuda(_args: Args) -> Result<()> {
    bail!("CUDA model benchmarking requires --features burn-cuda-backend")
}

#[cfg(feature = "burn-wgpu-backend")]
fn run_wgpu(args: Args) -> Result<()> {
    type B = burn_wgpu::Wgpu<f32>;
    let device = burn_wgpu::WgpuDevice::DiscreteGpu(args.device_index);
    run_backend_body::<B>(&args, "wgpu", &device)
}

#[cfg(not(feature = "burn-wgpu-backend"))]
fn run_wgpu(_args: Args) -> Result<()> {
    bail!("WGPU model benchmarking requires --features burn-wgpu-backend")
}

fn run_backend_body<B>(args: &Args, backend_name: &str, device: &B::Device) -> Result<()>
where
    B: Backend + ZipformerKernelBackend,
    burn_autodiff::Autodiff<B>: Backend<Device = B::Device> + ZipformerKernelBackend,
{
    let input_dim = resolved_input_dim(args);
    let lengths = input_lengths(args.batch, args.frames);
    println!(
        "backend={} device={:?} architecture={:?} variant={} batch={} frames={} input_dim={} vocab={} warmup={} iters={}",
        backend_name,
        device,
        args.architecture,
        args.variant,
        args.batch,
        args.frames,
        input_dim,
        args.vocab_size,
        args.warmup,
        args.iters
    );

    let mut results = Vec::new();
    if matches!(args.mode, BenchMode::Forward | BenchMode::Both) {
        results.push(match args.architecture {
            BenchArchitecture::Squeezeformer => bench_forward::<B, _>(
                args,
                device,
                "forward",
                build_squeezeformer::<B>(args, input_dim, device),
                input_dim,
                &lengths,
                |model, input, lengths| model.forward_with_lengths(input, Some(lengths)),
            )?,
            BenchArchitecture::Zipformer => bench_forward::<B, _>(
                args,
                device,
                "forward",
                build_zipformer::<B>(args, input_dim, device),
                input_dim,
                &lengths,
                |model, input, lengths| model.forward_with_lengths(input, lengths),
            )?,
            BenchArchitecture::W2vBert => bench_forward::<B, _>(
                args,
                device,
                "forward",
                build_w2v_bert::<B>(args, input_dim, device),
                input_dim,
                &lengths,
                |model, input, lengths| model.forward_with_lengths(input, lengths),
            )?,
        });
    }

    if matches!(args.mode, BenchMode::ForwardBackward | BenchMode::Both) {
        type AD<Inner> = burn_autodiff::Autodiff<Inner>;
        results.push(match args.architecture {
            BenchArchitecture::Squeezeformer => bench_forward_backward::<B, _>(
                args,
                device,
                "forward_backward",
                build_squeezeformer::<AD<B>>(args, input_dim, device),
                input_dim,
                &lengths,
                |model, input, lengths| model.forward_with_lengths(input, Some(lengths)),
            )?,
            BenchArchitecture::Zipformer => bench_forward_backward::<B, _>(
                args,
                device,
                "forward_backward",
                build_zipformer::<AD<B>>(args, input_dim, device),
                input_dim,
                &lengths,
                |model, input, lengths| model.forward_with_lengths(input, lengths),
            )?,
            BenchArchitecture::W2vBert => bench_forward_backward::<B, _>(
                args,
                device,
                "forward_backward",
                build_w2v_bert::<AD<B>>(args, input_dim, device),
                input_dim,
                &lengths,
                |model, input, lengths| model.forward_with_lengths(input, lengths),
            )?,
        });
    }

    print_results(&results);
    Ok(())
}

fn bench_forward<B, M>(
    args: &Args,
    device: &B::Device,
    name: &'static str,
    model: M,
    input_dim: usize,
    lengths: &[usize],
    mut forward: impl FnMut(&M, Tensor<B, 3>, Vec<usize>) -> (Tensor<B, 3>, Vec<usize>),
) -> Result<BenchResult>
where
    B: Backend,
{
    for _ in 0..args.warmup {
        let input = random_input::<B>(args, input_dim, device);
        let _ = forward(&model, input, lengths.to_vec());
    }
    B::sync(device)?;

    let start = Instant::now();
    let mut output_shape = [0, 0, 0];
    let mut output_lengths = Vec::new();
    for _ in 0..args.iters {
        let input = random_input::<B>(args, input_dim, device);
        let (output, lengths) = forward(&model, input, lengths.to_vec());
        output_shape = output.dims();
        output_lengths = lengths;
    }
    B::sync(device)?;

    Ok(BenchResult {
        name,
        avg: start.elapsed() / args.iters as u32,
        output_shape,
        output_lengths,
    })
}

fn bench_forward_backward<Inner, M>(
    args: &Args,
    device: &Inner::Device,
    name: &'static str,
    model: M,
    input_dim: usize,
    lengths: &[usize],
    mut forward: impl FnMut(
        &M,
        Tensor<burn_autodiff::Autodiff<Inner>, 3>,
        Vec<usize>,
    ) -> (Tensor<burn_autodiff::Autodiff<Inner>, 3>, Vec<usize>),
) -> Result<BenchResult>
where
    Inner: Backend,
    burn_autodiff::Autodiff<Inner>:
        AutodiffBackend<Device = Inner::Device> + ZipformerKernelBackend,
{
    type AD<B> = burn_autodiff::Autodiff<B>;

    for _ in 0..args.warmup {
        let input = random_input::<AD<Inner>>(args, input_dim, device);
        let (output, _) = forward(&model, input, lengths.to_vec());
        let _ = output.mean().backward();
    }
    <AD<Inner> as Backend>::sync(device)?;

    let start = Instant::now();
    let mut output_shape = [0, 0, 0];
    let mut output_lengths = Vec::new();
    for _ in 0..args.iters {
        let input = random_input::<AD<Inner>>(args, input_dim, device);
        let (output, lengths) = forward(&model, input, lengths.to_vec());
        output_shape = output.dims();
        output_lengths = lengths;
        let _ = output.mean().backward();
    }
    <AD<Inner> as Backend>::sync(device)?;

    Ok(BenchResult {
        name,
        avg: start.elapsed() / args.iters as u32,
        output_shape,
        output_lengths,
    })
}

fn build_squeezeformer<B: Backend>(
    args: &Args,
    input_dim: usize,
    device: &B::Device,
) -> w2v_bert_uk::squeezeformer::SqueezeformerCtc<B> {
    let encoder = SqueezeformerEncoderConfig::variant(&args.variant).unwrap_or_else(|| {
        SqueezeformerEncoderConfig::new(input_dim, args.d_model, args.num_layers, args.num_heads)
            .with_time_indices(Vec::new(), Vec::new())
    });
    SqueezeformerCtcConfig {
        encoder,
        vocab_size: args.vocab_size,
    }
    .init(device)
}

fn build_zipformer<B: Backend>(
    args: &Args,
    input_dim: usize,
    device: &B::Device,
) -> w2v_bert_uk::zipformer::ZipformerCtc<B> {
    let mut encoder =
        ZipformerConfig::variant(&args.variant).unwrap_or_else(|| ZipformerConfig::new(input_dim));
    encoder.input_dim = input_dim;
    ZipformerCtcConfig {
        encoder,
        vocab_size: args.vocab_size,
    }
    .init(device)
}

fn build_w2v_bert<B: Backend>(
    args: &Args,
    input_dim: usize,
    device: &B::Device,
) -> w2v_bert_uk::wav2vec::Wav2VecBertCtc<B> {
    let heads = args.num_heads.max(1);
    let mut encoder = Wav2VecBertConfig::new(input_dim, args.d_model)
        .with_layers(args.num_layers)
        .with_dropout(0.0);
    encoder.num_attention_heads = heads;
    encoder.intermediate_size = args.d_model * 4;
    Wav2VecBertCtcConfig {
        encoder,
        vocab_size: args.vocab_size,
    }
    .init(device)
}

fn random_input<B: Backend>(args: &Args, input_dim: usize, device: &B::Device) -> Tensor<B, 3> {
    Tensor::random(
        [args.batch, args.frames, input_dim],
        Distribution::Normal(0.0, 1.0),
        device,
    )
}

fn input_lengths(batch: usize, frames: usize) -> Vec<usize> {
    (0..batch)
        .map(|index| frames.saturating_sub(index % frames.max(1)).max(1))
        .collect()
}

fn resolved_input_dim(args: &Args) -> usize {
    if args.input_dim > 0 {
        return args.input_dim;
    }
    match args.architecture {
        BenchArchitecture::W2vBert => 160,
        BenchArchitecture::Squeezeformer | BenchArchitecture::Zipformer => 80,
    }
}

fn print_results(results: &[BenchResult]) {
    println!();
    println!(
        "{:<18} {:>12} {:>18} {:>18}",
        "step", "avg_ms", "output_shape", "lengths"
    );
    for result in results {
        println!(
            "{:<18} {:>12.3} {:>18?} {:>18?}",
            result.name,
            result.avg.as_secs_f64() * 1_000.0,
            result.output_shape,
            result.output_lengths
        );
    }
}
