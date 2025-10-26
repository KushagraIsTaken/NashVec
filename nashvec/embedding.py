"""
Transformer embedding generation using Sentence-BERT.

This module provides utilities for generating embeddings from text using
pre-trained transformer models.
"""

import logging
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceEmbedder:
    """
    Wrapper for SentenceTransformer for generating embeddings from text.

    This class provides a simplified interface for generating dense vector
    embeddings from sentences using pre-trained transformer models.

    Parameters
    ----------
    model_name : str, optional
        Name of the Sentence-BERT model to use. Default is "all-MiniLM-L6-v2".
        For a list of available models, see:
        https://www.sbert.net/docs/pretrained_models.html

    Attributes
    ----------
    model : SentenceTransformer
        The underlying SentenceTransformer model.
    dimension : int
        Dimensionality of the embeddings produced by the model.

    Examples
    --------
    >>> embedder = SentenceEmbedder()
    >>> embeddings = embedder.encode(["Hello world", "How are you?"])
    >>> print(embeddings.shape)
    (2, 384)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the SentenceEmbedder.

        Parameters
        ----------
        model_name : str, optional
            Name of the Sentence-BERT model to use.
            Default is "all-MiniLM-L6-v2".
        """
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Generate embeddings for the given text(s).

        Parameters
        ----------
        texts : str or List[str]
            Single text string or list of text strings to encode.
        **kwargs
            Additional arguments to pass to the SentenceTransformer encode method.

        Returns
        -------
        np.ndarray
            Embeddings array of shape (n_texts, embedding_dim).

        Examples
        --------
        >>> embedder = SentenceEmbedder()
        >>> # Single text
        >>> embedding = embedder.encode("Hello world")
        >>> print(embedding.shape)
        (384,)
        >>> # Multiple texts
        >>> embeddings = embedder.encode(["Hello", "World"])
        >>> print(embeddings.shape)
        (2, 384)
        """
        if isinstance(texts, str):
            texts = [texts]

        logger.debug(f"Encoding {len(texts)} text(s)")
        
        try:
            embeddings = self.model.encode(texts, **kwargs)
            return np.array(embeddings)
        
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise

    def encode_batch(self, 
                    texts: List[str], 
                    batch_size: int = 32,
                    show_progress: bool = True,
                    **kwargs) -> np.ndarray:
        """
        Generate embeddings in batches for large datasets.

        Parameters
        ----------
        texts : List[str]
            List of text strings to encode.
        batch_size : int, optional
            Number of texts to encode at once. Default is 32.
        show_progress : bool, optional
            Whether to show a progress bar. Default is True.
        **kwargs
            Additional arguments to pass to the SentenceTransformer encode method.

        Returns
        -------
        np.ndarray
            Embeddings array of shape (n_texts, embedding_dim).

        Examples
        --------
        >>> embedder = SentenceEmbedder()
        >>> texts = ["Text " + str(i) for i in range(1000)]
        >>> embeddings = embedder.encode_batch(texts, batch_size=64)
        >>> print(embeddings.shape)
        (1000, 384)
        """
        logger.info(f"Encoding {len(texts)} texts in batches of {batch_size}")

        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                show_progress_bar=show_progress,
                **kwargs
            )
            return np.array(embeddings)
        
        except Exception as e:
            logger.error(f"Error batch encoding: {str(e)}")
            raise

