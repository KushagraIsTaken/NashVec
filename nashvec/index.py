"""
Vector store implementations using FAISS and HNSW indexing.

This module provides efficient indexing and retrieval systems for both
compressed latent embeddings and raw transformer embeddings.
"""

import logging
import numpy as np
import faiss
import hnswlib
import tensorflow as tf
from typing import List, Tuple, Optional
from .embedding import SentenceEmbedder

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.

    Uses FAISS (Facebook AI Similarity Search) for fast approximate nearest
    neighbor search on high-dimensional embeddings.

    Parameters
    ----------
    transformer_model : str, optional
        Sentence-BERT model to use for embeddings. Default is "all-MiniLM-L6-v2".
    dim : int, optional
        Dimensionality of embeddings. Default is 384.

    Attributes
    ----------
    model : SentenceEmbedder
        Embedding model.
    index : faiss.Index
        FAISS index for similarity search.
    sentences : List[str]
        List of indexed sentences.
    embeddings : List[np.ndarray]
        List of embeddings.

    Examples
    --------
    >>> store = FAISSVectorStore()
    >>> store.add("Hello world")
    >>> store.add("How are you?")
    >>> results = store.search("Hi there", top_n=2)
    """

    def __init__(self, transformer_model: str = "all-MiniLM-L6-v2", dim: int = 384):
        """Initialize the FAISS vector store."""
        self.model = SentenceEmbedder(transformer_model)
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner Product for cosine similarity
        self.sentences = []
        self.embeddings = []
        logger.info("FAISSVectorStore initialized")

    def add(self, sentence: str) -> None:
        """
        Add a sentence to the index.

        Parameters
        ----------
        sentence : str
            Sentence to add to the index.
        """
        self.sentences.append(sentence)
        emb = self.model.encode(sentence)
        self.embeddings.append(emb)
        
        # Normalize and add to index
        vec = np.array([emb], dtype=np.float32)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        logger.debug(f"Added sentence to FAISS index (total: {len(self.sentences)})")

    def search(self, query: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar sentences.

        Parameters
        ----------
        query : str
            Query string.
        top_n : int, optional
            Number of results to return. Default is 5.

        Returns
        -------
        List[Tuple[str, float]]
            List of (sentence, similarity_score) tuples.
        """
        query_emb = self.model.encode(query)
        qvec = np.array([query_emb], dtype=np.float32)
        faiss.normalize_L2(qvec)
        
        distances, indices = self.index.search(qvec, top_n)
        
        results = []
        for i in range(top_n):
            idx = indices[0][i]
            if idx != -1:
                sim = distances[0][i]
                results.append((self.sentences[idx], float(sim)))
        
        return results


class HybridAutoencoderVectorStore:
    """
    Hybrid vector store using game-theoretic autoencoder compression and HNSW.

    Combines the benefits of compression (from the game-theoretic autoencoder)
    with the speed of HNSW (Hierarchical NSW) indexing for efficient retrieval.

    Parameters
    ----------
    transformer_model : str, optional
        Sentence-BERT model name. Default is "all-MiniLM-L6-v2".
    input_dim : int, optional
        Input embedding dimension. Default is 384.
    latent_dim : int, optional
        Compressed embedding dimension. Default is 128.
    space : str, optional
        HNSW distance metric. Default is 'cosine'.

    Attributes
    ----------
    model : SentenceEmbedder
        Embedding model.
    sentences : List[str]
        List of indexed sentences.
    embeddings : List[np.ndarray]
        List of raw embeddings.
    compressed_embeddings : np.ndarray
        Compressed embeddings after autoencoder training.
    encoder : tf.keras.Model
        Trained encoder model.
    game_ae : GameTheoreticAutoencoder
        Full game-theoretic autoencoder model.
    hnsw_index : hnswlib.Index
        HNSW index for fast retrieval.

    Examples
    --------
    >>> store = HybridAutoencoderVectorStore()
    >>> for text in texts:
    ...     store.add(text)
    >>> store.train_autoencoder(epochs=10)
    >>> store.build_index()
    >>> results = store.search("query text", top_n=5)
    """

    def __init__(self, 
                 transformer_model: str = "all-MiniLM-L6-v2",
                 input_dim: int = 384,
                 latent_dim: int = 128,
                 space: str = 'cosine'):
        """Initialize the hybrid vector store."""
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.space = space
        self.model = SentenceEmbedder(transformer_model)
        self.sentences = []
        self.embeddings = []
        self.compressed_embeddings = None
        self.encoder = None
        self.game_ae = None
        self.hnsw_index = None
        logger.info("HybridAutoencoderVectorStore initialized")

    def add(self, sentence: str) -> None:
        """
        Add a sentence to the store.

        Parameters
        ----------
        sentence : str
            Sentence to add.
        """
        self.sentences.append(sentence)
        embedding = self.model.encode(sentence)
        self.embeddings.append(embedding)
        logger.debug(f"Added sentence (total: {len(self.sentences)})")

    def _generate_triplets(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate training triplets for the game-theoretic loss.

        Creates (anchor, positive, negative) triplets where:
        - Positive: nearest neighbor in original space
        - Negative: random sample

        Parameters
        ----------
        X : np.ndarray
            Embeddings array.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (anchors, positives, negatives) arrays.
        """
        logger.info("Generating training triplets...")
        n = X.shape[0]

        # Use FAISS to find nearest neighbors
        index = faiss.IndexFlatL2(self.input_dim)
        index.add(X.astype(np.float32))
        distances, indices = index.search(X.astype(np.float32), k=2)

        anchors, positives, negatives = [], [], []

        for i in range(n):
            anchors.append(X[i])
            pos_idx = indices[i][1]  # Index 0 is the vector itself
            positives.append(X[pos_idx])

            # Negative is a random sample (not anchor or positive)
            neg_idx = i
            while neg_idx == i or neg_idx == pos_idx:
                neg_idx = np.random.randint(0, n)
            negatives.append(X[neg_idx])

        logger.info(f"Generated {len(anchors)} triplets")
        return (np.array(anchors), np.array(positives), np.array(negatives))

    def train_autoencoder(self, 
                         epochs: int = 10, 
                         batch_size: int = 32,
                         lambda_retrieval: float = 0.5,
                         margin: float = 0.2) -> None:
        """
        Train the game-theoretic autoencoder.

        Parameters
        ----------
        epochs : int, optional
            Number of training epochs. Default is 10.
        batch_size : int, optional
            Batch size for training. Default is 32.
        lambda_retrieval : float, optional
            Weight for retrieval loss. Default is 0.5.
        margin : float, optional
            Triplet loss margin. Default is 0.2.
        """
        from .autoencoder import GameTheoreticAutoencoder, build_encoder_decoder
        
        X = np.array(self.embeddings)
        
        # Generate triplets
        anchors, positives, negatives = self._generate_triplets(X)
        
        # Build model components
        encoder, decoder = build_encoder_decoder(self.input_dim, self.latent_dim)
        
        # Create game-theoretic model
        self.game_ae = GameTheoreticAutoencoder(
            encoder,
            decoder,
            lambda_retrieval=lambda_retrieval,
            margin=margin
        )
        
        # Compile
        self.game_ae.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            ae_loss_fn=tf.keras.losses.MeanSquaredError()
        )
        
        # Train
        logger.info(f"Training autoencoder for {epochs} epochs...")
        self.game_ae.fit(
            [anchors, positives, negatives],
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Store encoder and compress embeddings
        self.encoder = self.game_ae.encoder
        self.compressed_embeddings = self.encoder.predict(X)
        logger.info("Autoencoder training complete")

    def build_index(self, ef_construction: int = 50, M: int = 16, ef: int = 50) -> None:
        """
        Build HNSW index on compressed embeddings.

        Parameters
        ----------
        ef_construction : int, optional
            Construction-time parameter. Default is 50.
        M : int, optional
            Number of bi-directional links. Default is 16.
        ef : int, optional
            Search-time parameter. Default is 50.
        """
        if self.compressed_embeddings is None:
            raise ValueError("Compressed embeddings not found. Train autoencoder first.")

        logger.info("Building HNSW index...")
        num_elements = len(self.compressed_embeddings)
        
        self.hnsw_index = hnswlib.Index(space=self.space, dim=self.latent_dim)
        self.hnsw_index.init_index(
            max_elements=num_elements,
            ef_construction=ef_construction,
            M=M
        )
        self.hnsw_index.add_items(np.array(self.compressed_embeddings))
        self.hnsw_index.set_ef(ef)
        logger.info("HNSW index built successfully")

    def search(self, query: str, top_n: int = 5, candidate_multiplier: int = 3) -> List[Tuple[str, float]]:
        """
        Hybrid search with re-ranking.

        Performs approximate search in compressed space, then re-ranks
        candidates in original space.

        Parameters
        ----------
        query : str
            Query string.
        top_n : int, optional
            Number of final results. Default is 5.
        candidate_multiplier : int, optional
            Multiplier for candidate retrieval. Default is 3.

        Returns
        -------
        List[Tuple[str, float]]
            List of (sentence, similarity_score) tuples.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        if self.encoder is None or self.hnsw_index is None:
            raise ValueError("Model not trained. Call train_autoencoder() and build_index() first.")

        # Encode query
        query_embedding = self.model.encode(query)
        query_compressed = self.encoder.predict(np.array([query_embedding]))[0]

        # Search in compressed space
        candidate_k = top_n * candidate_multiplier
        labels, _ = self.hnsw_index.knn_query(query_compressed, k=candidate_k)

        # Re-rank in original space
        candidate_embs = np.array([self.compressed_embeddings[idx] for idx in labels[0]])
        sim_scores = cosine_similarity([query_compressed], candidate_embs)[0]

        ranked_candidates = sorted(
            zip(labels[0], sim_scores), 
            key=lambda x: x[1], 
            reverse=True
        )

        results = [(self.sentences[idx], float(score)) 
                   for idx, score in ranked_candidates[:top_n]]
        return results

