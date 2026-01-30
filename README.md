# distribution-coder

**Optimal arithmetic coding over symbol probability distributions.**

`distribution-coder` is a high-performance Python library for compressing sequences of symbols based on step-wise probability distributions. It is designed specifically for **Neural Data Compression** tasks, such as compressing the output of Large Language Models (LLMs), autoregressive transformers, or any next-token prediction system.

It is implemented in **Rust** using PyO3, offering **zero-copy** and **zero-allocation** operations for maximum throughput and low latency.

## Features

* **Precision:** Uses 32-bit frequency precision (backed by 128-bit integer arithmetic) to capture probabilities without underflow, achieving theoretical entropy limits.
* **Zero-Copy Dispatcher:** Natively handles `float32`, `float64`, `float16`, and `bfloat16` arrays. It reads memory directly from Numpy/PyTorch without casting or copying.
* **Framework Agnostic:** Seamlessly accepts input from **PyTorch**, **NumPy**, **JAX**, **TensorFlow**, or standard Python lists.
* **Cache-Friendly:** Uses a streaming, two-pass algorithm that never allocates heap memory for probability tables, preventing cache thrashing during long sequence generation.
* **Cross-Platform Determinism:** Guarantees bit-exact reconstruction across different hardware architectures (x86, ARM, etc.) by avoiding hardware-specific floating-point intrinsics.

## Installation

```bash
pip install distribution-coder

```

## Quick Start

### Basic Usage

The standard workflow involves an "Encoder" loop and a "Decoder" loop. Both must generate/receive the **exact same** probability distributions in the same order.

```python
import numpy as np
from distribution_coder import DistributionCoder

# --- 1. Encoding ---
encoder = DistributionCoder()

# Mock probability distributions for a sequence of 3 steps
# (In reality, these come from your Neural Network)
step1_probs = [0.1, 0.7, 0.2] # Symbol 1 is likely
step2_probs = [0.8, 0.1, 0.1] # Symbol 0 is likely
step3_probs = [0.05, 0.05, 0.9] # Symbol 2 is likely

# The actual symbols that occurred
symbols = [1, 0, 2]

# Step-wise encoding
encoder.encode_step(step1_probs, symbols[0])
encoder.encode_step(step2_probs, symbols[1])
encoder.encode_step(step3_probs, symbols[2])

# Get compressed bytes
compressed_data = encoder.finish_encoding()
print(f"Compressed size: {len(compressed_data)} bytes")

# --- 2. Decoding ---
decoder = DistributionCoder()
decoder.start_decoding(compressed_data)

# Step-wise decoding
# We feed the SAME distributions and ask for the symbol back
decoded_sym1 = decoder.decode_step(step1_probs)
decoded_sym2 = decoder.decode_step(step2_probs)
decoded_sym3 = decoder.decode_step(step3_probs)

assert [decoded_sym1, decoded_sym2, decoded_sym3] == symbols
print("Successfully decoded sequence!")

```

## Advanced Usage

### Working with PyTorch & Mixed Precision

`distribution-coder` is optimized for modern Deep Learning workflows. It detects Tensor types and reads their underlying memory directly.

**Supported Data Types:**

* `float32` (Standard)
* `float16` (Half Precision - Zero Copy)
* `bfloat16` (Brain Floating Point - Zero Copy)
* `float64` (Double Precision)

```python
import torch
from distribution_coder import DistributionCoder

coder = DistributionCoder()

# 1. PyTorch Tensor (CPU)
# Zero-copy access. No need to convert to numpy.
probs_fp32 = torch.softmax(torch.randn(100), dim=0)
coder.encode_step(probs_fp32, 5)

# 2. BFloat16 (TPU/Newer GPU format)
# Handled natively in Rust without casting to float32.
probs_bf16 = probs_fp32.to(torch.bfloat16)
coder.encode_step(probs_bf16, 10)

# 3. GPU Tensors
# Automatically moves to CPU for processing (copy required)
if torch.cuda.is_available():
    probs_gpu = torch.randn(100).cuda()
    coder.encode_step(probs_gpu, 2)

```

### Minimizing Latency

For the lowest possible latency (e.g., real-time voice applications), ensure your probability arrays are:

1. **Contiguous:** `np.ascontiguousarray(probs)` or `tensor.contiguous()`.
2. **Native Types:** Use `float32`, `float16`, or `bfloat16`. Python lists will trigger a fast C-level conversion, but native arrays are faster.

## Performance Architecture

Traditional Arithmetic Coders often allocate a Cumulative Distribution Function (CDF) array for every token. For a vocabulary of 50,000 tokens, this means allocating, writing, and freeing ~200KB of memory *per step*.

`distribution-coder` solves this bottleneck:

1. **Streaming Calculation:** It uses a two-pass algorithm (Analysis Pass + Search Pass) that iterates over the probability array in L1 cache without allocating heap memory.
2. **Integer Math:** Probabilities are quantized to 32-bit integers summing to . Intermediate calculations use `u128` to prevent overflow, allowing for "sharp" probabilities (high confidence) to use minimal bits.

## API Reference

### `DistributionCoder`

#### `__init__()`

Creates a new coder instance with a fresh state.

#### `encode_step(distribution, symbol: int)`

Encodes a single symbol based on the provided probability distribution.

* **distribution**: `Union[list, np.ndarray, torch.Tensor, jax.Array]`. The probability distribution. Sum does not strictly need to be 1.0 (it will be normalized), but it should be close.
* **symbol**: `int`. The index of the symbol to encode (`0 <= symbol < len(distribution)`).

#### `finish_encoding() -> bytes`

Finalizes the arithmetic coding process, flushes the internal bit buffer, and returns the compressed byte sequence.

#### `start_decoding(input_bytes: bytes)`

Resets the state and loads a compressed byte sequence for decoding.

* **input_bytes**: The bytes object returned by `finish_encoding()`.

#### `decode_step(distribution) -> int`

Decodes the next symbol from the stream based on the provided probability distribution.

* **distribution**: Must be identical to the distribution used at this step during encoding.
* **Returns**: The decoded symbol index.