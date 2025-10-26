# NashVec

**Optimization of Latent-Space Compression using Game-Theoretic Techniques for Transformer-Based Vector Search**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

NashVec is a game-theoretic approach to optimizing vector search by balancing information preservation and retrieval efficiency. By formulating vector compression as a zero-sum game between an encoder (compressor) and retriever, NashVec achieves superior search performance with reduced storage requirements.

### Key Features

- 🔬 **Game-Theoretic Optimization**: Balances reconstruction quality with retrieval performance using a Nash equilibrium approach
- 🎯 **Hybrid Search**: Combines compressed latent representations with HNSW indexing for fast, accurate retrieval
- 📊 **Dual Backend Support**: FAISS for baseline comparison and custom HNSW for hybrid search
- 🚀 **Production-Ready**: Modular design with CLI tools, comprehensive tests, and extensive documentation
- 🔧 **Customizable**: Adjustable loss weights, dimensions, and hyperparameters for different use cases

## Installation

### Requirements

- Python 3.8 or higher
- pip

### Install from PyPI

```bash
pip install nashvec
```

### Install from Source

```bash
git clone https://github.com/kushagraagrawal/NashVec.git
cd NashVec
pip install -e .
```

### Install with Test Dependencies

```bash
pip install "nashvec[dev]"
```

## Quick Start

### Using the Python API

```python
from nashvec import HybridSearcher

# Initialize hybrid searcher with game-theoretic compression
searcher = HybridSearcher(use_hybrid=True)

# Load data and train
searcher.load_and_train(limit=500, epochs=10)

# Search
results = searcher.search("Explain the process of photosynthesis", top_n=5)
for sentence, score in results:
    print(f"Score: {score:.4f}\n{sentence}\n")
```

### Using the CLI

```bash
# Train a model
nashvec-train --epochs 10 --batch-size 32

# Query the model
nashvec-query "Explain the process of photosynthesis" --top-n 5

# Benchmark performance
nashvec-benchmark --limit 500 --epochs 10
```

## Architecture

### Game-Theoretic Framework

NashVec implements a two-player game:

1. **Encoder (Compressor)**: Minimizes reconstruction error to preserve information
2. **Retriever**: Maximizes search efficiency by clustering similar items in latent space

The game-theoretic loss function:

```
L_total = L_reconstruction + λ × L_triplet
```

Where:
- `L_reconstruction`: MSE between original and reconstructed embeddings
- `L_triplet`: Triplet loss ensuring similar items are close in latent space
- `λ`: Balance parameter (typically 0.5)

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Transformer Embeddings                   │
│                   (Sentence-BERT, 384-dim)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Game-Theoretic Autoencoder                      │
│  ┌──────────────┐      ┌──────────────┐                      │
│  │   Encoder    │─────▶│   Latent     │                      │
│  │  (compress)  │      │   Space      │                      │
│  └──────────────┘      │  (128-dim)   │                      │
│                        └──────┬───────┘                      │
│                         │     │     │                        │
│                         │     │     ▼                        │
│                         │     │   Decoder                    │
│                         │     │   (reconstruct)              │
│                         ▼     ▼     ▼                        │
│                    Triplet Loss + MSE Loss                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  HNSW Index (Fast Retrieval)                 │
│                         +                                    │
│               Re-ranking (Original Space)                     │
└─────────────────────────────────────────────────────────────┘
```

## Documentation

### Core Modules

#### `nashvec.data`
Dataset loading and preprocessing:
```python
from nashvec.data import load_alpaca_data, load_custom_dataset

# Load Alpaca dataset
instructions = load_alpaca_data(limit=500)

# Load custom dataset
texts = load_custom_dataset("data.csv", text_column="text")
```

#### `nashvec.embedding`
Sentence embedding generation:
```python
from nashvec.embedding import SentenceEmbedder

embedder = SentenceEmbedder(model_name="all-MiniLM-L6-v2")
embeddings = embedder.encode_batch(texts, batch_size=32)
```

#### `nashvec.autoencoder`
Game-theoretic autoencoder:
```python
from nashvec.autoencoder import GameTheoreticAutoencoder, build_encoder_decoder

encoder, decoder = build_encoder_decoder(input_dim=384, latent_dim=128)
model = GameTheoreticAutoencoder(encoder, decoder, lambda_retrieval=0.5)
model.compile(optimizer='adam', ae_loss_fn='mse')
```

#### `nashvec.search`
High-level search interface:
```python
from nashvec.search import HybridSearcher, compute_utility

searcher = HybridSearcher(use_hybrid=True)
results = searcher.search("query text", top_n=5)
utility = compute_utility(accuracy=0.95, query_time=0.01)
```

### Configuration

```python
from nashvec.utils import NashVecConfig

config = NashVecConfig(
    latent_dim=128,
    epochs=10,
    lambda_retrieval=0.5,
    margin=0.2
)
```

## Examples

### Example 1: Basic Search

```python
from nashvec import HybridSearcher

# Create searcher
searcher = HybridSearcher(use_hybrid=True)

# Train on dataset
searcher.load_and_train(limit=500, epochs=10)

# Query
results = searcher.search("How does photosynthesis work?", top_n=3)
for i, (text, score) in enumerate(results, 1):
    print(f"{i}. [{score:.4f}] {text}")
```

### Example 2: Evaluation with Metrics

```python
from nashvec import HybridSearcher

searcher = HybridSearcher(use_hybrid=True)
searcher.load_and_train(limit=500, epochs=10)

# Get evaluation metrics
metrics = searcher.evaluate("query text", top_n=5)
print(f"Query Time: {metrics['query_time']:.4f}s")
print(f"Avg Similarity: {metrics['avg_similarity']:.4f}")
print(f"Utility: {metrics['utility']:.4f}")
```

### Example 3: Comparison Study

```python
from nashvec import HybridSearcher

queries = [
    "Explain machine learning",
    "What is Python?",
    "Describe neural networks"
]

# Test hybrid system
hybrid_searcher = HybridSearcher(use_hybrid=True)
hybrid_searcher.load_and_train(limit=500, epochs=10)

# Test baseline
faiss_searcher = HybridSearcher(use_hybrid=False)
faiss_searcher.load_and_train(limit=500)

for query in queries:
    hybrid_results = hybrid_searcher.evaluate(query)
    faiss_results = faiss_searcher.evaluate(query)
    
    if hybrid_results['utility'] > faiss_results['utility']:
        print(f"✓ Hybrid wins for '{query}'")
    else:
        print(f"✓ Baseline wins for '{query}'")
```

## Running Examples

```bash
# Run the demo
python examples/demo_search.py

# Run tests
pytest tests/

# Run specific test
pytest tests/test_autoencoder.py
```

## API Reference

See [full API documentation](docs/api.md) for detailed information about all modules, classes, and functions.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use NashVec in your research, please cite:

```bibtex
@misc{agrawal2025optimizationlatentspacecompressionusing,
      title={Optimization of Latent-Space Compression using Game-Theoretic Techniques for Transformer-Based Vector Search}, 
      author={Kushagra Agrawal and Nisharg Nargund and Oishani Banerjee},
      year={2025},
      eprint={2508.18877},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2508.18877}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Sentence Transformers for embedding generation
- Facebook AI Research for FAISS
- Yury Malkov for HNSW algorithm
- Hugging Face for datasets

## Support

For issues, questions, or contributions:
- GitHub: [https://github.com/kushagraagrawal/NashVec](https://github.com/kushagraagrawal/NashVec)
- PyPI: [https://pypi.org/project/nashvec/](https://pypi.org/project/nashvec/)

---

**NashVec** - Game-Theoretic Vector Search for the Modern Era 🔬🚀