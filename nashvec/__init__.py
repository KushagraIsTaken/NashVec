"""
NashVec: Optimization of Latent-Space Compression using Game-Theoretic Techniques
for Transformer-Based Vector Search

This package provides game-theoretic autoencoders for efficient vector search
and retrieval using transformer embeddings.
"""

__version__ = "0.1.0"
__author__ = "Kushagra Agrawal, Nisharg Nargund, Oishani Banerjee"

from .data import load_alpaca_data
from .embedding import SentenceEmbedder
from .autoencoder import GameTheoreticAutoencoder, build_encoder_decoder
from .index import HybridAutoencoderVectorStore, FAISSVectorStore
from .search import HybridSearcher, compute_utility
from .utils import NashVecConfig, setup_logging

__all__ = [
    "load_alpaca_data",
    "SentenceEmbedder",
    "GameTheoreticAutoencoder",
    "build_encoder_decoder",
    "HybridAutoencoderVectorStore",
    "FAISSVectorStore",
    "HybridSearcher",
    "compute_utility",
    "NashVecConfig",
    "setup_logging",
    "__version__",
    "__author__",
]
