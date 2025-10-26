"""
Tests for search and retrieval modules.
"""

import numpy as np
import pytest
from nashvec.search import compute_utility, HybridSearcher
from nashvec.index import FAISSVectorStore, HybridAutoencoderVectorStore


def test_compute_utility():
    """Test utility computation."""
    # High accuracy, low query time -> high utility
    util1 = compute_utility(accuracy=0.95, query_time=0.01, alpha=1.0, beta=1.0)
    
    # Low accuracy, high query time -> low utility
    util2 = compute_utility(accuracy=0.5, query_time=0.5, alpha=1.0, beta=1.0)
    
    assert util1 > util2
    assert util1 > 0
    assert util2 < 0


def test_faiss_vector_store():
    """Test FAISS vector store basic operations."""
    store = FAISSVectorStore()
    
    # Add some test sentences
    store.add("Hello world")
    store.add("How are you?")
    store.add("Python programming")
    
    assert len(store.sentences) == 3
    
    # Search
    results = store.search("Hi there", top_n=2)
    
    assert len(results) == 2
    assert isinstance(results[0], tuple)
    assert isinstance(results[0][0], str)  # sentence
    assert isinstance(results[0][1], float)  # score


def test_hybrid_vector_store_add():
    """Test hybrid vector store add operation."""
    store = HybridAutoencoderVectorStore()
    
    # Add some test sentences
    store.add("Hello world")
    store.add("How are you?")
    store.add("Python programming")
    
    assert len(store.sentences) == 3
    assert len(store.embeddings) == 3


@pytest.mark.slow
def test_hybrid_vector_store_training():
    """Test hybrid vector store training (slow test)."""
    store = HybridAutoencoderVectorStore(input_dim=384, latent_dim=64)
    
    # Add minimal data
    for i in range(50):
        store.add(f"Sentence {i} for testing purposes")
    
    # Train with minimal epochs
    store.train_autoencoder(epochs=2, batch_size=16)
    
    assert store.encoder is not None
    assert store.compressed_embeddings is not None
    assert store.compressed_embeddings.shape[1] == 64


@pytest.mark.slow
def test_hybrid_vector_store_full_pipeline():
    """Test full hybrid pipeline (slow test)."""
    store = HybridAutoencoderVectorStore(input_dim=384, latent_dim=64)
    
    # Add minimal data
    for i in range(50):
        store.add(f"Sentence {i} for testing purposes with more text")
    
    # Train
    store.train_autoencoder(epochs=2, batch_size=16)
    
    # Build index
    store.build_index()
    
    # Search
    results = store.search("testing", top_n=3)
    
    assert len(results) == 3
    assert isinstance(results[0], tuple)


def test_hybrid_searcher_initialization():
    """Test hybrid searcher initialization."""
    searcher_hybrid = HybridSearcher(use_hybrid=True)
    searcher_faiss = HybridSearcher(use_hybrid=False)
    
    assert searcher_hybrid.use_hybrid == True
    assert searcher_faiss.use_hybrid == False


def test_utility_edge_cases():
    """Test utility computation edge cases."""
    # Zero accuracy
    util = compute_utility(accuracy=0.0, query_time=0.01)
    assert util < 0
    
    # Zero query time
    util = compute_utility(accuracy=0.9, query_time=0.0)
    assert util > 0
    
    # Negative beta (penalizing accuracy)
    util = compute_utility(accuracy=0.5, query_time=0.1, alpha=1.0, beta=-1.0)
    assert util > 0


def test_search_top_n_parameter():
    """Test that top_n parameter is respected."""
    store = FAISSVectorStore()
    
    for i in range(20):
        store.add(f"Sentence number {i}")
    
    results = store.search("query", top_n=5)
    assert len(results) == 5
    
    results = store.search("query", top_n=10)
    assert len(results) == 10

