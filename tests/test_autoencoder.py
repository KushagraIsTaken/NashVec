"""
Tests for the game-theoretic autoencoder module.
"""

import numpy as np
import tensorflow as tf
import pytest
from nashvec.autoencoder import GameTheoreticAutoencoder, build_encoder_decoder


@pytest.fixture
def sample_data():
    """Generate sample embedding data for testing."""
    return np.random.rand(100, 384).astype(np.float32)


@pytest.fixture
def triplet_data(sample_data):
    """Generate triplet data for testing."""
    n = len(sample_data)
    anchors = sample_data[:50]
    positives = sample_data[50:100]
    negatives = sample_data[25:75]
    return anchors, positives, negatives


def test_build_encoder_decoder():
    """Test encoder-decoder architecture construction."""
    encoder, decoder = build_encoder_decoder(input_dim=384, latent_dim=128)
    
    assert encoder.input_shape == (None, 384)
    assert encoder.output_shape == (None, 128)
    assert decoder.input_shape == (None, 128)
    assert decoder.output_shape == (None, 384)


def test_game_theoretic_autoencoder_creation():
    """Test game-theoretic autoencoder model creation."""
    encoder, decoder = build_encoder_decoder(input_dim=384, latent_dim=128)
    model = GameTheoreticAutoencoder(encoder, decoder, lambda_retrieval=0.5, margin=0.2)
    
    assert model.encoder is not None
    assert model.decoder is not None
    assert model.lambda_retrieval == 0.5
    assert model.margin == 0.2


def test_model_compile():
    """Test model compilation."""
    encoder, decoder = build_encoder_decoder(input_dim=384, latent_dim=128)
    model = GameTheoreticAutoencoder(encoder, decoder)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        ae_loss_fn=tf.keras.losses.MeanSquaredError()
    )
    
    assert model.optimizer is not None
    assert model.ae_loss_fn is not None


def test_forward_pass(triplet_data):
    """Test forward pass through the model."""
    encoder, decoder = build_encoder_decoder(input_dim=384, latent_dim=128)
    model = GameTheoreticAutoencoder(encoder, decoder)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        ae_loss_fn=tf.keras.losses.MeanSquaredError()
    )
    
    anchors, positives, negatives = triplet_data
    output = model(anchors[:10])
    
    assert output.shape == (10, 384)


def test_triplet_loss():
    """Test triplet loss calculation."""
    encoder, decoder = build_encoder_decoder(input_dim=384, latent_dim=128)
    model = GameTheoreticAutoencoder(encoder, decoder, margin=0.2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        ae_loss_fn=tf.keras.losses.MeanSquaredError()
    )
    
    # Create dummy latent representations
    anchor_latent = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    positive_latent = tf.constant([[1.1, 0.1], [0.1, 1.1]])
    negative_latent = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    
    loss = model._triplet_loss(anchor_latent, positive_latent, negative_latent)
    
    assert loss.shape == ()
    assert tf.reduce_all(loss >= 0)


def test_training_step(triplet_data):
    """Test custom training step."""
    encoder, decoder = build_encoder_decoder(input_dim=384, latent_dim=128)
    model = GameTheoreticAutoencoder(encoder, decoder)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        ae_loss_fn=tf.keras.losses.MeanSquaredError()
    )
    
    anchors, positives, negatives = triplet_data
    result = model.train_step(([anchors[:10], positives[:10], negatives[:10]],))
    
    assert 'loss' in result
    assert 'ae_loss' in result
    assert 'retrieval_loss' in result
    assert result['loss'] >= 0

