"""
Dataset loading and preprocessing utilities.

This module provides functions to load and preprocess datasets for training
and evaluation of the NashVec model.
"""

import logging
from typing import List, Optional
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


def load_alpaca_data(limit: Optional[int] = 500) -> List[str]:
    """
    Load the Alpaca instruction dataset for training and evaluation.

    The Alpaca dataset contains instruction-output pairs designed for training
    language models. This function extracts the instruction field.

    Parameters
    ----------
    limit : int, optional
        Maximum number of samples to load from the dataset.
        Default is 500.

    Returns
    -------
    List[str]
        List of instruction strings from the Alpaca dataset.

    Examples
    --------
    >>> instructions = load_alpaca_data(limit=100)
    >>> print(f"Loaded {len(instructions)} instructions")
    """
    try:
        logger.info(f"Loading Alpaca dataset (limit={limit})...")
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        instructions = dataset["instruction"]
        logger.info(f"Successfully loaded {len(instructions)} instructions")
        return instructions
    
    except Exception as e:
        logger.error(f"Error loading Alpaca dataset: {str(e)}")
        raise


def load_custom_dataset(file_path: str, text_column: str = "text") -> List[str]:
    """
    Load a custom dataset from a file.

    Supports CSV, JSON, and text files.

    Parameters
    ----------
    file_path : str
        Path to the dataset file.
    text_column : str, optional
        Name of the column containing text data (for CSV/JSON).
        Default is "text".

    Returns
    -------
    List[str]
        List of text strings from the dataset.
    """
    logger.info(f"Loading custom dataset from {file_path}...")
    
    try:
        if file_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(file_path)
            texts = df[text_column].tolist()
        
        elif file_path.endswith('.json'):
            import pandas as pd
            df = pd.read_json(file_path)
            texts = df[text_column].tolist()
        
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        logger.info(f"Successfully loaded {len(texts)} texts")
        return texts
    
    except Exception as e:
        logger.error(f"Error loading custom dataset: {str(e)}")
        raise


def preprocess_text(texts: List[str], 
                   min_length: int = 10, 
                   max_length: Optional[int] = None) -> List[str]:
    """
    Preprocess text data by filtering and cleaning.

    Parameters
    ----------
    texts : List[str]
        List of text strings to preprocess.
    min_length : int, optional
        Minimum length of text to keep. Default is 10.
    max_length : int, optional
        Maximum length of text to keep. If None, no maximum.

    Returns
    -------
    List[str]
        List of preprocessed text strings.
    """
    logger.info("Preprocessing texts...")
    
    processed = []
    for text in texts:
        # Strip whitespace
        text = text.strip()
        
        # Apply length filters
        if len(text) < min_length:
            continue
        
        if max_length and len(text) > max_length:
            text = text[:max_length]
        
        processed.append(text)
    
    logger.info(f"Filtered {len(texts)} texts down to {len(processed)} texts")
    return processed

