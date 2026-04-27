"""
Shared custom Keras layers for the ML pipeline.

This module provides a single source of truth for custom layers used by
both the Transformer trainer and the HybridPredictor at load time.
"""

import numpy as np
import tensorflow as tf


class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    """
    Sinusoidal Positional Encoding (Vaswani et al., 2017).

    Adds fixed sinusoidal position information to input embeddings so the
    Transformer can exploit the sequential order of time steps.

    For a sequence of length T with embedding dimension d:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    The encoding is pre-computed for `max_len` positions and sliced at
    call time, so it works with variable-length inputs up to `max_len`.
    """

    def __init__(self, max_len: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        d_model = int(input_shape[-1])
        pe = np.zeros((self.max_len, d_model))
        position = np.arange(0, self.max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[:d_model // 2])
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len})
        return config
