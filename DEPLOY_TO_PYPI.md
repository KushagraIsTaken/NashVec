# Deploying NashVec to PyPI

## Prerequisites

1. **Create accounts**:
   - PyPI account: https://pypi.org/account/register/
   - TestPyPI account: https://test.pypi.org/account/register/

2. **Install build tools**:
```bash
pip install --upgrade build twine
```

## Step-by-Step Deployment

### 1. Test Locally

```bash
# Install the package locally
pip install -e .

# Run tests
pytest tests/

# Verify CLI tools work
nashvec-train --help
```

### 2. Build Distribution

```bash
# Clean previous builds
python -m build

# This creates:
# - dist/nashvec-0.1.0.tar.gz
# - dist/nashvec-0.1.0-py3-none-any.whl
```

### 3. Test on TestPyPI First

```bash
# Upload to TestPyPI
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ nashvec
```

### 4. Deploy to PyPI

```bash
# Upload to real PyPI
twine upload dist/*
```

You'll be prompted for your PyPI username and password.

### 5. Verify Installation

```bash
# Uninstall local version
pip uninstall nashvec

# Install from PyPI
pip install nashvec

# Test
python -c "import nashvec; print(nashvec.__version__)"
```

## After Deployment

### Update GitHub Repository

If you have a GitHub repo:
```bash
# Add files
git add .

# Commit
git commit -m "Release v0.1.0 - NashVec on PyPI"

# Push
git push origin main

# Create release tag
git tag v0.1.0
git push origin v0.1.0
```

### Update README

Users will now install with:
```bash
pip install nashvec
```

Update your README.md to reflect this.

## Updating the Package

For future releases:

1. **Update version** in:
   - `nashvec/__init__.py`
   - `setup.py`
   - `pyproject.toml`

2. **Rebuild**:
```bash
python -m build
```

3. **Upload**:
```bash
twine upload dist/*
```

## Important Notes

- **Version**: Make sure to increment the version number for each release
- **Test First**: Always test on TestPyPI before deploying to PyPI
- **Dependencies**: Ensure all dependencies are correctly listed in `requirements.txt`
- **Documentation**: README.md will appear on PyPI as the package description
- **License**: Must include LICENSE file for PyPI
- **No Secrets**: Never commit API keys or passwords to git

## Troubleshooting

### "ModuleNotFoundError" during build
- Make sure all dependencies are in `requirements.txt`
- Check that `__init__.py` imports are correct

### "File already exists" on PyPI
- Bump the version number in all files
- Run build again

### CLI tools not working after pip install
- Verify entry_points in setup.py are correct
- Check that the package installed in the right location

## Security

- Use `twine upload` (not `setup.py upload`) - it's more secure
- Consider using API tokens instead of passwords
- Sign your packages for extra security (optional)

## Environment Variables

Set these for automated deployments:
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-your-api-token
```

