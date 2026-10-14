# Week 6: Energy-based Models

## Overview

Generative modeling learns a probability distribution from examples. Energy-based models represent distributions through energy functions, assigning lower energy to configurations that are more compatible with the learned model. This week studies restricted Boltzmann machines, classical Hopfield networks, and modern Hopfield retrieval.

## Environment Setup

The project uses an isolated `uv` environment. Python 3.11 or newer is required; the supplied lock file records the resolved environment. Run commands from this directory:

```bash
uv sync
uv run python -c "import jax; print(jax.devices())"
```

JAX uses the CPU backend by default. The device check should report a `CpuDevice`. Run all scripts through `uv run` for reproducible dependency selection.

## Directory Layout

- `scripts/` contains reproducible dataset, training, evaluation, and plotting scripts.
- `evidence/` contains figures used to inspect and present the experiments.
- `artifacts/` contains raw numerical outputs and generated experiment files.
- `inputs/` contains downloaded datasets and is excluded from version control.
- `rbm.toml` contains the fixed RBM training configuration.

Raw downloads and generated artifacts should not be committed. Keep them in `inputs/` and `artifacts/`; keep source code, documentation, and selected figures under version control as appropriate.

## Part 1: Understand the Datasets

The dataset readers preserve the original pixel values and native image dimensions. MNIST is loaded from gzip-compressed IDX files, and CIFAR-10 is loaded from the authors' binary batch archive.

Download and inspect the datasets with:

```bash
uv run python scripts/download_data.py
uv run python scripts/plot_datasets.py
```

The expected dataset sizes are 60,000 training and 10,000 test images for MNIST, and 50,000 training and 10,000 test images for CIFAR-10. The scripts write distribution figures to `evidence/` and the numerical summary to `artifacts/part1_dataset_summary.json`.

The actual download URLs are:

- MNIST CVDF mirror:
  - `https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz`
  - `https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz`
  - `https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz`
  - `https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz`
- CIFAR-10 authors' binary archive:
  - `https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz`
  - fallback mirror: `https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-binary.tar.gz`

## Part 2: Restricted Boltzmann Machine

The RBM uses thresholded MNIST pixels as 784 binary visible units and 128 hidden units. Its energy, conditional probabilities, visible free energy, block Gibbs sampler, and contrastive divergence objectives are implemented with JAX. Both CD-1 and CD-10 are evaluated with deterministic initialization, shuffling, Gibbs sampling, and evaluation seeds.

The fixed configuration uses 20 epochs, batches of 100, SGD with learning rate 0.1, zero momentum, zero weight decay, weight standard deviation 0.01, and visible biases initialized from clipped training pixel frequencies.

Run training and its verification plot with:

```bash
uv run python scripts/rbm.py
uv run python scripts/plot_rbm_training.py
```

The scripts write `artifacts/rbm_training.json` and the saved model parameters, followed by `evidence/rbm_learning_curve.png`. Generation and completion use the frozen trained parameters:

```bash
uv run python scripts/rbm_generation.py
uv run python scripts/rbm_completion.py
```

### Reconstruction vs Free-running Generation

Reconstruction evaluates how the trained RBM responds when an observed image is provided as the starting point. A held-out MNIST image is initialized as the visible state, and Gibbs sampling is used to measure how well the model recovers or preserves the input. This evaluates conditional recovery behavior and does not demonstrate that the model can generate new samples.

Free-running generation evaluates whether the learned energy model captures the data distribution. The Gibbs chain starts from random binary configurations without providing an image or label. The model must develop structured patterns through sampling alone. Therefore, successful reconstruction does not necessarily imply successful generation.

### Image Completion as Inference

Image completion is an inference task with fixed model parameters. The observed pixels are clamped while the missing pixels are sampled from the conditional distribution learned by the RBM.

## Part 3: Classical Hopfield Network

The classical Hopfield experiment generates random plus-or-minus-one patterns, constructs the Hebbian coupling matrix with a zero diagonal, and performs zero-temperature asynchronous recall with a random spin order. It measures unstable bits, final overlap, and recall success over pattern loads from 0.05 to 0.25 for network sizes 100, 200, 400, and 800.

Run the experiment and create its verification plots with:

```bash
uv run python scripts/hopfield.py
uv run python scripts/plot_hopfield.py
```

The numerical results are stored in `artifacts/hopfield_results.json` and `artifacts/hopfield_summary.json`. The figures are `evidence/hopfield_unstable_bits.png` and `evidence/hopfield_capacity.png`.

## Part 4: Modern Hopfield Network

The modern Hopfield experiment stores the first `P` binarized MNIST training images in spin representation. One update is

```text
x_new = X @ softmax(beta * X.T @ x)
```

This is equivalent to a single-query attention operation with query `x`, keys `X.T`, and values `X.T`. Retrieval is tested after replacing the lower half of a query with random spins. A successful retrieval has overlap at least 0.95 and identifies the target as the nearest stored pattern.

Run the numerical experiment and final verification plots with:

```bash
uv run python scripts/modern_hopfield.py
uv run python scripts/plot_modern_hopfield.py
```

The numerical outputs are `artifacts/attention_equivalence.json` and `artifacts/modern_hopfield_results.json`. The verification figures are `evidence/attention_weights.png`, `evidence/modern_retrieval_examples.png`, and `evidence/retrieval_success.png`.

## Reproducibility and Version Control

The experiments use explicit deterministic seeds and fixed stored data order. Run scripts from `week6/` through `uv run`. Raw downloads, model parameter dumps, caches, and generated numerical artifacts are excluded from version control. The evidence directory remains trackable for selected figures and review.
