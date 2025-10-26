"""
Setup configuration for NashVec package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    install_requires = [
        "faiss-cpu>=1.12.0",
        "datasets>=2.0.0",
        "hnswlib>=0.8.0",
        "sentence-transformers>=2.0.0",
        "tensorflow>=2.10.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.25.0",
        "pandas>=2.0.0",
    ]

setup(
    name="nashvec",
    version="0.1.0",
    author="Kushagra Agrawal, Nisharg Nargund, Oishani Banerjee",
    author_email="kushagraagrawal@ieee.org",
    description="Optimization of Latent-Space Compression using Game-Theoretic Techniques for Transformer-Based Vector Search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kushagraagrawal/NashVec",
    packages=find_packages(exclude=["tests", "examples", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nashvec-train=nashvec.cli:train",
            "nashvec-query=nashvec.cli:query",
            "nashvec-benchmark=nashvec.cli:benchmark",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

