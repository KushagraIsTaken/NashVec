# NashVec Quick Start Guide

## Installation (3 Steps)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Package

```bash
pip install -e .
```

### 3. Verify Installation

```bash
python -c "import nashvec; print('NashVec installed successfully!')"
```

## Basic Usage

### Python API

```python
from nashvec import HybridSearcher

# Initialize the searcher
searcher = HybridSearcher(use_hybrid=True)

# Load data and train
searcher.load_and_train(limit=500, epochs=10)

# Search
results = searcher.search("Explain photosynthesis", top_n=5)

# Display results
for i, (text, score) in enumerate(results, 1):
    print(f"{i}. [{score:.4f}] {text[:80]}...")
```

### Command Line

```bash
# Train a model
nashvec-train --epochs 10

# Query
nashvec-query "Explain photosynthesis"

# Benchmark
nashvec-benchmark
```

## Run the Demo

```bash
python examples/demo_search.py
```

## Run Tests

```bash
# All tests
pytest tests/

# Specific test
pytest tests/test_search.py -v
```

## What You Get

✅ **Game-theoretic autoencoder** for optimal compression  
✅ **FAISS baseline** for comparison  
✅ **HNSW indexing** for fast retrieval  
✅ **Hybrid search** with re-ranking  
✅ **CLI tools** for easy use  
✅ **Comprehensive tests** for reliability  
✅ **Full documentation** for research use

## Next Steps

1. Read `README.md` for detailed documentation
2. Check `examples/demo_search.py` for examples
3. See `PRODUCTION_README.md` for implementation details

**Happy searching!** 🔍

