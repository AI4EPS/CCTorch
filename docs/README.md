# CCTorch: Cross-Correlation using PyTorch

![](assets/diagram.png)

CCTorch runs seismic cross-correlation on the GPU. It is built on PyTorch, so the
same code runs on a laptop GPU, on several GPUs through `torchrun`, or across
nodes under Slurm.

## Install

From PyPI:

```bash
pip install CCTorch
```

Or from source, which is what you want if you plan to run `run.py`:

```bash
git clone https://github.com/AI4EPS/CCTorch.git
cd CCTorch
pip install -e .
```

## Quick start

A run is driven by `run.py`. At minimum it needs a list of data files and a
place to put results:

```bash
python run.py --data_list1 data_list.txt \
              --data_path1 data \
              --result_path results
```

Ambient noise, correlating in the STFT domain with a 30 s maximum lag:

```bash
python run.py --data_list1 noise_list.txt \
              --data_path1 noise_data \
              --sampling_rate 50 \
              --maxlag 30 \
              --domain stft
```

Pass `--pair_list` instead of a second data list when you already know which
pairs to correlate, and `--auto_xcorr` for auto-correlation.

## Several GPUs

```bash
torchrun --standalone --nproc_per_node=8 run.py --data_list1 data_list.txt --data_path1 data
```

## Several nodes

```bash
sbatch --nodes=2 --ntasks=2 sbatch_run.sh
```

## Common options

| Option | Default | Meaning |
|---|---|---|
| `--data_list1`, `--data_list2` | — | file lists to correlate |
| `--data_path1`, `--data_path2` | — | where those files live |
| `--pair_list` | — | explicit pairs, instead of a second list |
| `--auto_xcorr` | off | correlate a dataset with itself |
| `--domain` | `time` | `time`, `frequency`, or `stft` |
| `--maxlag` | `0.5` | maximum lag, in seconds |
| `--dt` | `0.01` | sampling interval, in seconds |
| `--sampling_rate` | — | alternative to `--dt` |
| `--result_path` | `./results` | output directory |
| `--batch_size`, `--workers` | — | throughput tuning |
| `--device` | — | `cuda` or `cpu` |

`python run.py --help` lists the rest, including DAS-specific channel selection
(`--min_channel`, `--max_channel`, `--delta_channel`) and reduction options
(`--reduce_t`, `--reduce_c`).

## Examples

Complete, runnable workflows live in
[`examples/`](https://github.com/AI4EPS/CCTorch/tree/main/examples) — the
California ambient-noise set covers downsampling, building file lists, and
submitting multi-node jobs.

## API reference

See [Reference](reference.md).
