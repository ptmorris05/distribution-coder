import numpy as np

from ._distribution_coder import ArithmeticCoder as _RustCoder

__all__ = ["DistributionCoder"]


class DistributionCoder:
    """
    A wrapper around the Rust Arithmetic Coder that handles
    automatic type conversion (PyTorch, Lists, etc.) efficiently.
    """

    def __init__(self):
        self._coder = _RustCoder()

    def start_decoding(self, input_bytes: bytes):
        """Prepares the decoder with a compressed byte stream."""
        self._coder.start_decoding(input_bytes)

    def finish_encoding(self) -> bytes:
        """Flushes the encoder and returns the compressed bytes."""
        return self._coder.finish_encoding()

    def encode_step(self, distribution, symbol: int):
        """
        Encode a single symbol based on the provided probability distribution.

        Args:
            distribution: A list, numpy array, or PyTorch/JAX tensor.
            symbol: The integer index of the symbol to encode.
        """
        # We pass the raw object to Rust.
        # Rust checks if it's a known numpy type (f32, f64, f16, bf16).
        # If not (e.g. PyTorch Tensor, List), we normalize to f32 here.
        dist = self._normalize_input(distribution)
        self._coder.encode_step(dist, symbol)

    def decode_step(self, distribution) -> int:
        """
        Decode a single symbol based on the provided probability distribution.

        Args:
            distribution: A list, numpy array, or PyTorch/JAX tensor.
        Returns:
            The decoded symbol index.
        """
        dist = self._normalize_input(distribution)
        return self._coder.decode_step(dist)

    def _normalize_input(self, dist):
        """
        Normalization logic:
        1. Handle PyTorch/JAX/TensorFlow tensors without importing them.
        2. Handle Lists/Tuples -> np.array.
        3. If it's already a Numpy array, pass it through raw (Rust will dispatch).
        """
        # Duck-type check for Tensor-like objects
        if hasattr(dist, "numpy"):
            # If on GPU, we must move to CPU first.
            if hasattr(dist, "device") and hasattr(dist, "cpu"):
                # Rough check if it's a torch tensor on CUDA/MPS
                if str(dist.device) != "cpu":
                    dist = dist.cpu()

            # This returns a numpy array
            return dist.numpy()

        # If it's already a numpy array, return as is (Zero-Copy pass-through).
        if isinstance(dist, np.ndarray):
            return dist

        # Fallback for Lists/Tuples: Convert to f32 numpy array.
        # FIX: Removed 'copy=False' because Lists MUST be copied to become Arrays.
        return np.array(dist, dtype=np.float32)
