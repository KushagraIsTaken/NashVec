"""
Command-line interface for NashVec.

This module provides CLI commands for training, querying, and benchmarking
NashVec models.
"""

import argparse
import logging
import sys
from pathlib import Path
from .search import HybridSearcher
from .utils import setup_logging, print_evaluation_metrics, NashVecConfig

logger = logging.getLogger(__name__)


def train(args):
    """Train a NashVec model."""
    setup_logging(level=logging.INFO)
    
    logger.info("Starting NashVec training...")
    logger.info(f"Limit: {args.limit}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Lambda retrieval: {args.lambda_retrieval}")
    
    searcher = HybridSearcher(use_hybrid=not args.no_hybrid)
    searcher.load_and_train(
        limit=args.limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_retrieval=args.lambda_retrieval,
        margin=args.margin
    )
    
    logger.info("Training complete!")
    return 0


def query(args):
    """Query a NashVec model."""
    setup_logging(level=logging.INFO)
    
    logger.info(f"Querying with: '{args.query}'")
    logger.info(f"Limit: {args.limit}")
    logger.info(f"Top N: {args.top_n}")
    
    searcher = HybridSearcher(use_hybrid=not args.no_hybrid)
    searcher.load_and_train(limit=args.limit, epochs=args.epochs)
    
    results = searcher.evaluate(args.query, top_n=args.top_n)
    
    system_name = "Hybrid (Game-Theoretic)" if not args.no_hybrid else "FAISS"
    print_evaluation_metrics(results, system_name=system_name)
    
    return 0


def benchmark(args):
    """Benchmark NashVec performance."""
    setup_logging(level=logging.INFO)
    
    queries = [
        "Explain the process of photosynthesis",
        "How does a neural network learn?",
        "What is the capital of France?",
        "Describe the structure of DNA",
        "How does Python handle memory management?",
    ]
    
    logger.info(f"Running benchmark with {len(queries)} queries")
    
    for use_hybrid in [False, True]:
        system_name = "FAISS (Baseline)" if not use_hybrid else "Hybrid (Game-Theoretic)"
        logger.info(f"\nBenchmarking {system_name}...")
        
        searcher = HybridSearcher(use_hybrid=use_hybrid)
        searcher.load_and_train(limit=args.limit, epochs=args.epochs)
        
        all_metrics = []
        for query_text in queries:
            result = searcher.evaluate(query_text, top_n=args.top_n)
            all_metrics.append(result)
        
        avg_query_time = sum(m['query_time'] for m in all_metrics) / len(all_metrics)
        avg_similarity = sum(m['avg_similarity'] for m in all_metrics) / len(all_metrics)
        avg_utility = sum(m['utility'] for m in all_metrics) / len(all_metrics)
        
        print(f"\n{system_name} Results:")
        print(f"  Average Query Time: {avg_query_time:.4f} seconds")
        print(f"  Average Similarity: {avg_similarity:.4f}")
        print(f"  Average Utility: {avg_utility:.4f}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NashVec: Game-Theoretic Vector Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nashvec-train --epochs 10 --batch-size 32
  nashvec-query "Explain the process of photosynthesis"
  nashvec-benchmark --limit 500
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a NashVec model")
    train_parser.add_argument("--limit", type=int, default=500, help="Number of samples")
    train_parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--lambda-retrieval", type=float, default=0.5, help="Lambda for retrieval loss")
    train_parser.add_argument("--margin", type=float, default=0.2, help="Triplet loss margin")
    train_parser.add_argument("--no-hybrid", action="store_true", help="Use baseline FAISS instead")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query a NashVec model")
    query_parser.add_argument("query", help="Query string")
    query_parser.add_argument("--limit", type=int, default=500, help="Number of samples")
    query_parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    query_parser.add_argument("--top-n", type=int, default=5, help="Number of results")
    query_parser.add_argument("--no-hybrid", action="store_true", help="Use baseline FAISS instead")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark NashVec performance")
    bench_parser.add_argument("--limit", type=int, default=500, help="Number of samples")
    bench_parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    bench_parser.add_argument("--top-n", type=int, default=5, help="Number of results per query")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate function
    if args.command == "train":
        return train(args)
    elif args.command == "query":
        return query(args)
    elif args.command == "benchmark":
        return benchmark(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

