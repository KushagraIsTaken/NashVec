"""
Game-theoretic autoencoder architecture and training.

This module implements the core game-theoretic autoencoder model that balances
reconstruction quality with retrieval performance through a zero-sum game
between an encoder and retriever.
"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, losses
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def build_encoder_decoder(input_dim: int, latent_dim: int) -> Tuple[Model, Model]:
    """
    Build encoder and decoder models for the autoencoder.

    Creates separate functional encoder and decoder models that can be used
    independently or as part of the game-theoretic autoencoder.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input embeddings (e.g., 384 for MiniLM).
    latent_dim : int
        Dimensionality of latent space for compression.

    Returns
    -------
    Tuple[Model, Model]
        Tuple of (encoder, decoder) Keras models.

    Examples
    --------
    >>> encoder, decoder = build_encoder_decoder(input_dim=384, latent_dim=128)
    >>> print(f"Encoder output shape: {encoder.output_shape}")
    >>> print(f"Decoder output shape: {decoder.output_shape}")
    """
    logger.info(f"Building encoder-decoder architecture: {input_dim} -> {latent_dim}")

    # Encoder
    input_layer = Input(shape=(input_dim,), name="encoder_input")
    encoded = layers.Dense(256, activation='relu')(input_layer)
    encoded = layers.Dense(latent_dim, activation='relu', name="latent_space")(encoded)
    encoder = Model(input_layer, encoded, name="encoder")

    # Decoder
    latent_input = Input(shape=(latent_dim,), name="decoder_input")
    decoded = layers.Dense(256, activation='relu')(latent_input)
    decoded = layers.Dense(input_dim, activation='sigmoid', name="reconstruction")(decoded)
    decoder = Model(latent_input, decoded, name="decoder")

    logger.debug("Encoder-decoder architecture built successfully")
    return encoder, decoder


class GameTheoreticAutoencoder(Model):
    """
    Game-theoretic autoencoder model implementing a zero-sum game.

    This model balances two competing objectives:
    1. Reconstruction loss: Preserve information in latent space
    2. Retrieval loss: Maintain semantic relationships for efficient search

    The model uses a triplet loss in the latent space to ensure that similar
    items are close together, while simultaneously minimizing reconstruction error.

    Parameters
    ----------
    encoder : tf.keras.Model
        The encoder network.
    decoder : tf.keras.Model
        The decoder network.
    lambda_retrieval : float, optional
        Weight for the retrieval loss relative to reconstruction loss.
        Default is 0.5.
    margin : float, optional
        Margin for the triplet loss. Default is 0.2.

    Attributes
    ----------
    encoder : tf.keras.Model
        The encoder network.
    decoder : tf.keras.Model
        The decoder network.
    lambda_retrieval : float
        Weight for retrieval loss.
    margin : float
        Triplet loss margin.

    Examples
    --------
    >>> encoder, decoder = build_encoder_decoder(384, 128)
    >>> model = GameTheoreticAutoencoder(encoder, decoder, lambda_retrieval=0.5)
    >>> model.compile(optimizer='adam', ae_loss_fn=losses.MeanSquaredError())
    >>> model.fit([anchors, positives, negatives], epochs=10)
    """

    def __init__(self, encoder: Model, decoder: Model, 
                 lambda_retrieval: float = 0.5, margin: float = 0.2):
        """
        Initialize the game-theoretic autoencoder.

        Parameters
        ----------
        encoder : tf.keras.Model
            The encoder network.
        decoder : tf.keras.Model
            The decoder network.
        lambda_retrieval : float, optional
            Weight for the retrieval loss. Default is 0.5.
        margin : float, optional
            Margin for the triplet loss. Default is 0.2.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lambda_retrieval = lambda_retrieval
        self.margin = margin

        # Metrics to track during training
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.ae_loss_tracker = tf.keras.metrics.Mean(name="ae_loss")
        self.retrieval_loss_tracker = tf.keras.metrics.Mean(name="retrieval_loss")

        logger.info(f"Game-theoretic autoencoder initialized with "
                   f"lambda={lambda_retrieval}, margin={margin}")

    def compile(self, optimizer, ae_loss_fn):
        """
        Compile the model with optimizer and reconstruction loss.

        Parameters
        ----------
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer for training.
        ae_loss_fn : callable
            Loss function for reconstruction loss.
        """
        super().compile(optimizer=optimizer)
        self.ae_loss_fn = ae_loss_fn
        logger.info("Model compiled successfully")

    @property
    def metrics(self):
        """Return list of metrics to be reset at the start of each epoch."""
        return [self.total_loss_tracker, self.ae_loss_tracker, self.retrieval_loss_tracker]

    def _triplet_loss(self, anchor_latent, positive_latent, negative_latent):
        """
        Calculate triplet loss in the latent space.

        Triplet loss ensures that:
        - Distance(anchor, positive) < Distance(anchor, negative) + margin

        Parameters
        ----------
        anchor_latent : tf.Tensor
            Latent representation of anchor items.
        positive_latent : tf.Tensor
            Latent representation of positive items (nearest neighbors).
        negative_latent : tf.Tensor
            Latent representation of negative items (random samples).

        Returns
        -------
        tf.Tensor
            Scalar triplet loss.
        """
        # Using squared Euclidean distance
        pos_dist = tf.reduce_sum(tf.square(anchor_latent - positive_latent), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor_latent - negative_latent), axis=-1)

        loss = tf.maximum(0.0, pos_dist - neg_dist + self.margin)
        return tf.reduce_mean(loss)

    def train_step(self, data):
        """
        Custom training step implementing the game-theoretic loss.

        Parameters
        ----------
        data : tuple
            Tuple of (anchors, positives, negatives) tensors.

        Returns
        -------
        dict
            Dictionary of metric values.
        """
        # Unpack the triplet data
        anchors, positives, negatives = data[0]

        with tf.GradientTape() as tape:
            # Forward pass
            anchor_latent = self.encoder(anchors, training=True)
            positive_latent = self.encoder(positives, training=True)
            negative_latent = self.encoder(negatives, training=True)

            # Reconstruction
            anchor_reconstructed = self.decoder(anchor_latent, training=True)

            # Calculate losses (the "game")
            ae_loss = self.ae_loss_fn(anchors, anchor_reconstructed)
            retrieval_loss = self._triplet_loss(anchor_latent, positive_latent, negative_latent)
            total_loss = ae_loss + self.lambda_retrieval * retrieval_loss

        # Backward pass
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.ae_loss_tracker.update_state(ae_loss)
        self.retrieval_loss_tracker.update_state(retrieval_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "ae_loss": self.ae_loss_tracker.result(),
            "retrieval_loss": self.retrieval_loss_tracker.result(),
        }

    def call(self, x):
        """
        Forward pass for inference.

        Parameters
        ----------
        x : tf.Tensor
            Input data.

        Returns
        -------
        tf.Tensor
            Reconstructed output.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

