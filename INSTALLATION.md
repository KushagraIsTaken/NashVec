# NashVec Installation Guide

## Quick Start

### 1. Install from Source

```bash
# Navigate to the project directory
cd NashVec

# Install NashVec with dependencies
pip install -e .
```

### 2. Run Verification

```bash
# Run the verification script
python verify_installation.py
```

### 3. Test the Package

```bash
# Run the test suite
pytest tests/

# Run a specific test
pytest tests/test_autoencoder.py -v
```

### 4. Try the Demo

```bash
# Run the example demo
python examples/demo_search.py
```

## Installation Steps

### Step 1: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

### Step 2: Verify Installation

```bash
python -c "import nashvec; print(nashvec.__version__)"
```

Expected output:
```
0.1.0
```

### Step 3: Run Tests

```bash
pytest tests/ -v
```

### Step 4: Try the CLI

```bash
# Show help
nashvec-train --help
nashvec-query --help
nashvec-benchmark --help
```

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'datasets'

**Solution:** Install the datasets package:
```bash
pip install datasets
```

### Issue: ModuleNotFoundError: No module named 'faiss'

**Solution:** Install FAISS:
```bash
pip install faiss-cpu
```

### Issue: ModuleNotFoundError: No module named 'hnswlib'

**Solution:** Install HNSW library:
```bash
pip install hnswlib
```

### Issue: TensorFlow not found

**Solution:** Install TensorFlow:
```bash
pip install tensorflow
```

## Testing

### Run All Tests

```bash
pytest tests/
```

### Run with Coverage

```bash
pytest --cov=nashvec tests/
```

### Run Specific Test Categories

```bash
# Skip slow tests
pytest -m "not slow" tests/

# Run only fast tests
pytest -m "not slow" tests/
```

## Development Setup

For development, install with dev dependencies:

```bash
pip install -e ".[dev]"
```

This includes:
- pytest
- pytest-cov
- black
- flake8
- mypy

## Next Steps

1. Read the [README.md](README.md) for usage examples
2. Check [examples/demo_search.py](examples/demo_search.py) for a working example
3. Review the [API documentation](docs/api.md) (if available)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the error messages carefully
3. Open an issue on GitHub with:
   - Your Python version
   - Operating system
   - Full error traceback
   - Steps to reproduce

