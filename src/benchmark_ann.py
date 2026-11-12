import json
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ann_index import ApproximateNearestNeighborIndex

VOCAB_DIR = BASE_DIR / "models" / "serving_vocabularies"
ARTICLE_VOCAB_PATH = VOCAB_DIR / "article_id_vocabulary.json"
CUSTOMER_VOCAB_PATH = VOCAB_DIR / "customer_id_vocabulary.json"

EMBEDDING_DIM = 32
NUM_TEST_QUERIES = 200
TOP_K = 10
N_CLUSTERS = 64
N_PROBE = 8


def main():
    print("=" * 60)
    print("  H&M Recommendation Engine: ANN Candidate Retrieval Benchmark")
    print("=" * 60)

    # 1. Load article IDs
    if not ARTICLE_VOCAB_PATH.exists():
        print(f"Warning: {ARTICLE_VOCAB_PATH} not found. Creating synthetic benchmark vocabulary.")
        article_ids = [f"article_{i:06d}" for i in range(5000)]
    else:
        with open(ARTICLE_VOCAB_PATH, "r", encoding="utf-8") as f:
            article_ids = json.load(f)

    # Subsample or use articles for benchmarking (e.g. up to 10,000 articles for fast verification)
    sample_size = min(len(article_ids), 10000)
    benchmark_articles = article_ids[:sample_size]
    print(f"Candidate article pool: {len(benchmark_articles):,} items (dim={EMBEDDING_DIM})")

    # 2. Simulate or load candidate embeddings
    rng = np.random.default_rng(42)
    candidate_embeddings = rng.standard_normal((len(benchmark_articles), EMBEDDING_DIM)).astype(np.float32)

    # 3. Build ANN Index
    print(f"\nBuilding ANN Inverted File Index (n_clusters={N_CLUSTERS})...")
    build_start = time.perf_counter()
    ann_index = ApproximateNearestNeighborIndex(n_clusters=N_CLUSTERS, random_state=42)
    ann_index.build(candidate_embeddings, benchmark_articles)
    build_time = time.perf_counter() - build_start
    print(f"Index built in {build_time:.3f} seconds.")

    # 4. Generate test query vectors
    test_queries = rng.standard_normal((NUM_TEST_QUERIES, EMBEDDING_DIM)).astype(np.float32)

    # 5. Benchmark Brute Force Search
    print(f"\nRunning {NUM_TEST_QUERIES} queries: Exact Brute-Force Dot-Product...")
    exact_start = time.perf_counter()
    exact_results = []
    for q in test_queries:
        exact_results.append(ann_index.exact_query(q, top_k=TOP_K))
    exact_total_time = time.perf_counter() - exact_start
    exact_avg_ms = (exact_total_time / NUM_TEST_QUERIES) * 1000
    exact_qps = NUM_TEST_QUERIES / exact_total_time

    # 6. Benchmark ANN Index Search
    print(f"Running {NUM_TEST_QUERIES} queries: ANN Index (n_probe={N_PROBE})...")
    ann_start = time.perf_counter()
    ann_results = []
    for q in test_queries:
        ann_results.append(ann_index.query(q, top_k=TOP_K, n_probe=N_PROBE))
    ann_total_time = time.perf_counter() - ann_start
    ann_avg_ms = (ann_total_time / NUM_TEST_QUERIES) * 1000
    ann_qps = NUM_TEST_QUERIES / ann_total_time

    # 7. Compute Recall@K
    recalls = []
    for exact_res, ann_res in zip(exact_results, ann_results):
        exact_set = {item_id for item_id, _ in exact_res}
        ann_set = {item_id for item_id, _ in ann_res}
        if exact_set:
            recalls.append(len(exact_set.intersection(ann_set)) / len(exact_set))

    mean_recall = np.mean(recalls) * 100.0 if recalls else 0.0
    speedup = exact_avg_ms / ann_avg_ms if ann_avg_ms > 0 else 1.0

    print("\n" + "=" * 60)
    print("  ANN PERFORMANCE & RECALL BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Candidate Space:          {len(benchmark_articles):,} items")
    print(f"Test Queries Evaluated:   {NUM_TEST_QUERIES}")
    print(f"Top-K Items Requested:    {TOP_K}")
    print(f"Clusters / Probed:        {N_CLUSTERS} / {N_PROBE}")
    print("-" * 60)
    print(f"Exact Brute-Force:        {exact_avg_ms:.3f} ms/query ({exact_qps:.1f} QPS)")
    print(f"ANN Index Retrieval:      {ann_avg_ms:.3f} ms/query ({ann_qps:.1f} QPS)")
    print(f"Latency Speedup:          {speedup:.2f}x faster")
    print(f"Candidate Recall@{TOP_K}:      {mean_recall:.2f}% overlap with exact top-{TOP_K}")
    print("=" * 60)

    # 8. Test persistence
    output_index_path = BASE_DIR / "models" / "ann_index.npz"
    ann_index.save(output_index_path)
    print(f"Saved ANN index archive to: {output_index_path}")
    loaded_index = ApproximateNearestNeighborIndex.load(output_index_path)
    print(f"Successfully reloaded index with {len(loaded_index.item_ids):,} items.")


if __name__ == "__main__":
    main()
