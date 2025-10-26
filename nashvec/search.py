"""
Hybrid search, re-ranking, and utility computation utilities.

This module provides search utilities and evaluation functions for NashVec.
"""

import logging
import time
from typing import List, Tuple, Optional
from .index import HybridAutoencoderVectorStore, FAISSVectorStore
from .data import load_alpaca_data

logger = logging.getLogger(__name__)


def compute_utility(accuracy: float, query_time: float, alpha: float = 1.0, beta: float = 1.0) -> float:
    """
    Compute the utility score for a retrieval system.

    Utility measures the trade-off between accuracy (relevance) and query time.
    Higher scores indicate better performance.

    Parameters
    ----------
    accuracy : float
        Average similarity score or accuracy metric (0-1).
    query_time : float
        Query time in seconds.
    alpha : float, optional
        Weight for accuracy. Default is 1.0.
    beta : float, optional
        Weight for query time. Default is 1.0.

    Returns
    -------
    float
        Utility score = alpha * accuracy - beta * query_time

    Examples
    --------
    >>> util = compute_utility(accuracy=0.95, query_time=0.01, alpha=1.0, beta=1.0)
    >>> print(f"Utility: {util:.4f}")
    """
    return alpha * accuracy - beta * query_time


class HybridSearcher:
    """
    High-level interface for hybrid search operations.

    This class provides a unified interface for training and querying
    both FAISS and hybrid (autoencoder + HNSW) vector stores.

    Parameters
    ----------
    use_hybrid : bool, optional
        Whether to use hybrid (compressed) or baseline FAISS search.
        Default is True.

    Attributes
    ----------
    store : HybridAutoencoderVectorStore or FAISSVectorStore
        The underlying vector store.
    use_hybrid : bool
        Whether hybrid mode is enabled.

    Examples
    --------
    >>> searcher = HybridSearcher(use_hybrid=True)
    >>> searcher.load_and_train(limit=500, epochs=10)
    >>> results = searcher.search("query text", top_n=5)
    """

    def __init__(self, use_hybrid: bool = True):
        """Initialize the hybrid searcher."""
        self.use_hybrid = use_hybrid
        
        if use_hybrid:
            self.store = HybridAutoencoderVectorStore()
        else:
            self.store = FAISSVectorStore()
        
        logger.info(f"HybridSearcher initialized (use_hybrid={use_hybrid})")

    def load_and_train(self, 
                      limit: int = 500, 
                      epochs: int = 10,
                      batch_size: int = 32,
                      lambda_retrieval: float = 0.5,
                      margin: float = 0.2) -> None:
        """
        Load dataset, add to store, and train if using hybrid mode.

        Parameters
        ----------
        limit : int, optional
            Number of samples to load. Default is 500.
        epochs : int, optional
            Training epochs for autoencoder. Default is 10.
        batch_size : int, optional
            Training batch size. Default is 32.
        lambda_retrieval : float, optional
            Weight for retrieval loss. Default is 0.5.
        margin : float, optional
            Triplet loss margin. Default is 0.2.
        """
        logger.info(f"Loading dataset (limit={limit})...")
        alpaca_data = load_alpaca_data(limit=limit)
        
        for sentence in alpaca_data:
            self.store.add(sentence)
        
        logger.info(f"Added {len(alpaca_data)} sentences to store")
        
        if self.use_hybrid:
            logger.info("Training autoencoder...")
            self.store.train_autoencoder(
                epochs=epochs,
                batch_size=batch_size,
                lambda_retrieval=lambda_retrieval,
                margin=margin
            )
            
            logger.info("Building HNSW index...")
            self.store.build_index()
        
        logger.info("Training complete")

    def search(self, query: str, top_n: int = 5, candidate_multiplier: int = 3) -> List[Tuple[str, float]]:
        """
        Search for similar sentences.

        Parameters
        ----------
        query : str
            Query string.
        top_n : int, optional
            Number of results. Default is 5.
        candidate_multiplier : int, optional
            Multiplier for candidate retrieval (hybrid only). Default is 3.

        Returns
        -------
        List[Tuple[str, float]]
            List of (sentence, score) tuples.
        """
        if self.use_hybrid:
            return self.store.search(query, top_n=top_n, candidate_multiplier=candidate_multiplier)
        else:
            return self.store.search(query, top_n=top_n)

    def evaluate(self, 
                 query: str, 
                 top_n: int = 5,
                 alpha: float = 1.0,
                 beta: float = 1.0) -> dict:
        """
        Evaluate search performance with utility metrics.

        Parameters
        ----------
        query : str
            Query string.
        top_n : int, optional
            Number of results. Default is 5.
        alpha : float, optional
            Weight for accuracy. Default is 1.0.
        beta : float, optional
            Weight for query time. Default is 1.0.

        Returns
        -------
        dict
            Dictionary containing:
            - query_time: float
            - avg_similarity: float
            - utility: float
            - results: List[Tuple[str, float]]
        """
        start_time = time.time()
        results = self.search(query, top_n=top_n)
        query_time = time.time() - start_time
        
        avg_sim = sum(score for (_, score) in results) / len(results) if results else 0.0
        utility = compute_utility(avg_sim, query_time, alpha=alpha, beta=beta)
        
        return {
            "query_time": query_time,
            "avg_similarity": avg_sim,
            "utility": utility,
            "results": results
        }


def benchmark_search(queries: List[str], 
                     use_hybrid: bool = True,
                     limit: int = 500,
                     epochs: int = 10,
                     top_n: int = 5) -> dict:
    """
    Benchmark search performance on multiple queries.

    Parameters
    ----------
    queries : List[str]
        List of query strings.
    use_hybrid : bool, optional
        Whether to use hybrid search. Default is True.
    limit : int, optional
        Number of training samples. Default is 500.
    epochs : int, optional
        Training epochs. Default is 10.
    top_n : int, optional
        Number of results per query. Default is 5.

    Returns
    -------
    dict
        Dictionary containing:
        - avg_query_time: float
        - avg_similarity: float
        - avg_utility: float
        - all_results: List[List[Tuple[str, float]]]
    """
    logger.info(f"Benchmarking with {len(queries)} queries")
    
    searcher = HybridSearcher(use_hybrid=use_hybrid)
    searcher.load_and_train(limit=limit, epochs=epochs)
    
    all_results = []
    query_times = []
    similarities = []
    utilities = []
    
    for query in queries:
        result = searcher.evaluate(query, top_n=top_n)
        all_results.append(result['results'])
        query_times.append(result['query_time'])
        similarities.append(result['avg_similarity'])
        utilities.append(result['utility'])
    
    return {
        "avg_query_time": sum(query_times) / len(query_times),
        "avg_similarity": sum(similarities) / len(similarities),
        "avg_utility": sum(utilities) / len(utilities),
        "all_results": all_results
    }

