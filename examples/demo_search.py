"""
Example: Demonstration of NashVec hybrid search.

This script demonstrates how to use NashVec for efficient vector search
with game-theoretic compression.
"""

import logging
from nashvec import HybridSearcher, compute_utility, setup_logging

# Set up logging
setup_logging(level=logging.INFO)

def main():
    """Main demonstration function."""
    print("=" * 60)
    print("NashVec: Game-Theoretic Vector Search Demo")
    print("=" * 60)
    
    # Configuration
    limit = 500  # Number of samples to use
    epochs = 10  # Training epochs
    query = "Explain the process of photosynthesis."
    
    # Build and train hybrid searcher
    print("\n1. Building Hybrid (Game-Theoretic) System...")
    print("-" * 60)
    hybrid_searcher = HybridSearcher(use_hybrid=True)
    hybrid_searcher.load_and_train(limit=limit, epochs=epochs)
    
    # Build baseline FAISS searcher
    print("\n2. Building Baseline FAISS System...")
    print("-" * 60)
    faiss_searcher = HybridSearcher(use_hybrid=False)
    faiss_searcher.load_and_train(limit=limit)
    
    # Evaluate both systems
    print(f"\n3. Evaluating with query: '{query}'")
    print("-" * 60)
    
    hybrid_results = hybrid_searcher.evaluate(query, top_n=5)
    faiss_results = faiss_searcher.evaluate(query, top_n=5)
    
    # Display results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\nHybrid (Game-Theoretic) System:")
    print(f"Query Time: {hybrid_results['query_time']:.4f} seconds")
    print(f"Average Similarity: {hybrid_results['avg_similarity']:.4f}")
    print(f"Utility: {hybrid_results['utility']:.4f}")
    print("\nTop Results:")
    for i, (sentence, score) in enumerate(hybrid_results['results'], 1):
        print(f"  {i}. Score: {score:.4f}")
        print(f"     {sentence[:100]}{'...' if len(sentence) > 100 else ''}")
    
    print("\nBaseline FAISS System:")
    print(f"Query Time: {faiss_results['query_time']:.4f} seconds")
    print(f"Average Similarity: {faiss_results['avg_similarity']:.4f}")
    print(f"Utility: {faiss_results['utility']:.4f}")
    print("\nTop Results:")
    for i, (sentence, score) in enumerate(faiss_results['results'], 1):
        print(f"  {i}. Score: {score:.4f}")
        print(f"     {sentence[:100]}{'...' if len(sentence) > 100 else ''}")
    
    # Game outcome
    print("\n" + "=" * 60)
    print("GAME-THEORETIC OUTCOME")
    print("=" * 60)
    
    if hybrid_results['utility'] > faiss_results['utility']:
        print("\n✓ Hybrid (Game-Theoretic) System WINS!")
        print(f"  Higher utility: {hybrid_results['utility']:.4f} > {faiss_results['utility']:.4f}")
    elif faiss_results['utility'] > hybrid_results['utility']:
        print("\n✓ Baseline FAISS System WINS!")
        print(f"  Higher utility: {faiss_results['utility']:.4f} > {hybrid_results['utility']:.4f}")
    else:
        print("\n✓ It's a TIE!")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

