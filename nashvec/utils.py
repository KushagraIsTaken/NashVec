"""
Configuration and helper utilities for NashVec.

This module provides configuration management, logging setup, and various
utility functions for the NashVec package.
"""

import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class NashVecConfig:
    """
    Configuration class for NashVec.

    Stores all configuration parameters for training and inference.

    Parameters
    ----------
    model_dir : str, optional
        Directory to save/load models. Default is ~/.nashvec/models/.
    transformer_model : str, optional
        Sentence-BERT model name. Default is "all-MiniLM-L6-v2".
    input_dim : int, optional
        Input embedding dimension. Default is 384.
    latent_dim : int, optional
        Latent space dimension. Default is 128.
    epochs : int, optional
        Training epochs. Default is 10.
    batch_size : int, optional
        Batch size for training. Default is 32.
    lambda_retrieval : float, optional
        Weight for retrieval loss. Default is 0.5.
    margin : float, optional
        Triplet loss margin. Default is 0.2.
    ef_construction : int, optional
        HNSW construction parameter. Default is 50.
    M : int, optional
        HNSW number of bi-directional links. Default is 16.
    ef : int, optional
        HNSW search parameter. Default is 50.
    top_n : int, optional
        Default number of search results. Default is 5.
    candidate_multiplier : int, optional
        Multiplier for candidate retrieval. Default is 3.

    Examples
    --------
    >>> config = NashVecConfig(latent_dim=64, epochs=20)
    >>> print(config.model_dir)
    /home/user/.nashvec/models/
    """

    # Directories
    model_dir: str = os.path.join(str(Path.home()), ".nashvec", "models")
    
    # Model parameters
    transformer_model: str = "all-MiniLM-L6-v2"
    input_dim: int = 384
    latent_dim: int = 128
    
    # Training parameters
    epochs: int = 10
    batch_size: int = 32
    lambda_retrieval: float = 0.5
    margin: float = 0.2
    
    # HNSW parameters
    ef_construction: int = 50
    M: int = 16
    ef: int = 50
    
    # Search parameters
    top_n: int = 5
    candidate_multiplier: int = 3
    
    def __post_init__(self):
        """Ensure model directory exists."""
        os.makedirs(self.model_dir, exist_ok=True)


def setup_logging(level: int = logging.INFO, 
                  log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for NashVec.

    Parameters
    ----------
    level : int, optional
        Logging level. Default is logging.INFO.
    log_file : str, optional
        File path for logging. If None, only console logging.
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger.info(f"Logging configured (level={level})")


def get_model_path(config: NashVecConfig, model_name: str = "model") -> str:
    """
    Get the full path to a saved model.

    Parameters
    ----------
    config : NashVecConfig
        Configuration object.
    model_name : str, optional
        Name of the model. Default is "model".

    Returns
    -------
    str
        Full path to the model file.
    """
    return os.path.join(config.model_dir, f"{model_name}.h5")


def save_model(model, config: NashVecConfig, model_name: str = "model") -> str:
    """
    Save a trained model to disk.

    Parameters
    ----------
    model : tf.keras.Model
        Model to save.
    config : NashVecConfig
        Configuration object.
    model_name : str, optional
        Name for the model file. Default is "model".

    Returns
    -------
    str
        Path where the model was saved.
    """
    model_path = get_model_path(config, model_name)
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    return model_path


def load_model(config: NashVecConfig, model_name: str = "model"):
    """
    Load a saved model from disk.

    Parameters
    ----------
    config : NashVecConfig
        Configuration object.
    model_name : str, optional
        Name of the model to load. Default is "model".

    Returns
    -------
    tf.keras.Model
        Loaded model.
    """
    import tensorflow as tf
    
    model_path = get_model_path(config, model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


def format_results(results: list, max_length: int = 100) -> str:
    """
    Format search results for display.

    Parameters
    ----------
    results : list
        List of (sentence, score) tuples.
    max_length : int, optional
        Maximum length for sentence display. Default is 100.

    Returns
    -------
    str
        Formatted results string.
    """
    lines = []
    for i, (sentence, score) in enumerate(results, 1):
        truncated = sentence[:max_length] + "..." if len(sentence) > max_length else sentence
        lines.append(f"  {i}. Score: {score:.4f}\n     Text: {truncated}")
    return "\n".join(lines)


def print_evaluation_metrics(results: dict, system_name: str = "System") -> None:
    """
    Print formatted evaluation metrics.

    Parameters
    ----------
    results : dict
        Dictionary containing metrics (query_time, avg_similarity, utility).
    system_name : str, optional
        Name of the system. Default is "System".
    """
    print(f"\n{system_name} Metrics:")
    print(f"Query Time: {results['query_time']:.4f} seconds")
    print(f"Average Similarity: {results['avg_similarity']:.4f}")
    print(f"Utility: {results['utility']:.4f}")
    print("Top Results:")
    print(format_results(results['results']))

