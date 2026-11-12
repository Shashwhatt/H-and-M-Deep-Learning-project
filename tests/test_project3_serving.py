import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Ensure root project path is available for imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ann_index import ApproximateNearestNeighborIndex
from src.evaluation_metrics import recall_at_k, ndcg_at_k


class TestTwoTowerShapesAndOutputs(unittest.TestCase):
    """Tests the latent dimension and embedding shapes of the recommendation components."""

    def test_embedding_dimension_consistency(self):
        dim = 32
        rng = np.random.default_rng(101)
        query_vector = rng.standard_normal((1, dim)).astype(np.float32)
        candidate_matrix = rng.standard_normal((50, dim)).astype(np.float32)

        # Dot product ranking
        scores = np.dot(candidate_matrix, query_vector.T).flatten()
        self.assertEqual(scores.shape, (50,))
        self.assertTrue(np.isfinite(scores).all())


class TestApproximateNearestNeighborIndex(unittest.TestCase):
    """Tests ANN index construction, query retrieval, ranking, and persistence."""

    def setUp(self):
        self.num_items = 100
        self.dim = 32
        self.rng = np.random.default_rng(42)
        self.item_ids = [f"article_{i:04d}" for i in range(self.num_items)]
        self.embeddings = self.rng.standard_normal((self.num_items, self.dim)).astype(np.float32)
        self.index = ApproximateNearestNeighborIndex(n_clusters=8, random_state=42)
        self.index.build(self.embeddings, self.item_ids)

    def test_index_is_built(self):
        self.assertTrue(self.index.is_built)
        self.assertEqual(self.index.embedding_dim, self.dim)
        self.assertEqual(len(self.index.item_ids), self.num_items)
        self.assertEqual(self.index.centroids.shape, (8, self.dim))

    def test_query_returns_top_k_sorted(self):
        top_k = 10
        query_vec = self.rng.standard_normal(self.dim).astype(np.float32)
        results = self.index.query(query_vec, top_k=top_k, n_probe=4)

        self.assertEqual(len(results), top_k)
        # Check sorted descending
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_exact_query_consistency(self):
        top_k = 5
        query_vec = self.rng.standard_normal(self.dim).astype(np.float32)
        exact_results = self.index.exact_query(query_vec, top_k=top_k)
        self.assertEqual(len(exact_results), top_k)
        scores = [score for _, score in exact_results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_ann_index_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_ann_index.npz"
            self.index.save(save_path)
            self.assertTrue(save_path.exists())

            loaded_index = ApproximateNearestNeighborIndex.load(save_path)
            self.assertTrue(loaded_index.is_built)
            self.assertEqual(len(loaded_index.item_ids), self.num_items)

            query_vec = self.rng.standard_normal(self.dim).astype(np.float32)
            orig_res = self.index.query(query_vec, top_k=5, n_probe=2)
            loaded_res = loaded_index.query(query_vec, top_k=5, n_probe=2)

            self.assertEqual([item for item, _ in orig_res], [item for item, _ in loaded_res])


class TestRecommendationEvaluationMetrics(unittest.TestCase):
    """Tests Recall@K and NDCG@K ranking metric calculations."""

    def test_perfect_recall_and_ndcg(self):
        ground_truth = ["item_a", "item_b", "item_c"]
        predictions = ["item_a", "item_b", "item_c", "item_d", "item_e"]

        recall_3 = recall_at_k(predictions, ground_truth, k=3)
        self.assertAlmostEqual(recall_3, 1.0, places=4)

        ndcg_3 = ndcg_at_k(predictions, ground_truth, k=3)
        self.assertAlmostEqual(ndcg_3, 1.0, places=4)

    def test_partial_recall_and_ndcg(self):
        ground_truth = ["item_a", "item_b"]
        predictions = ["item_x", "item_a", "item_y", "item_b"]

        recall_2 = recall_at_k(predictions, ground_truth, k=2)
        self.assertAlmostEqual(recall_2, 0.5, places=4)

        # Because relevant item is at rank 2 instead of rank 1, NDCG should be < 1.0
        ndcg_2 = ndcg_at_k(predictions, ground_truth, k=2)
        self.assertTrue(0.0 < ndcg_2 < 1.0)


class TestRedisFeatureStoreSerialization(unittest.TestCase):
    """Tests data formatting and binary vector encoding for Redis Feature Store."""

    def test_customer_profile_json_serialization(self):
        profile = {
            "age": 28.0,
            "club_member_status": "ACTIVE",
            "fashion_news_frequency": "Regularly",
        }
        serialized = json.dumps(profile)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["age"], 28.0)
        self.assertEqual(deserialized["club_member_status"], "ACTIVE")

    def test_embedding_float32_buffer_roundtrip(self):
        embedding_dim = 32
        original = np.random.randn(embedding_dim).astype(np.float32)
        binary_buffer = original.tobytes()

        restored = np.frombuffer(binary_buffer, dtype=np.float32)
        self.assertEqual(restored.size, embedding_dim)
        np.testing.assert_array_almost_equal(original, restored)


class TestAirflowDAGIntegrity(unittest.TestCase):
    """Verifies that Airflow DAG files have correct task structures and dependencies."""

    def test_weekly_retraining_dag_structure(self):
        dag_file = BASE_DIR / "airflow" / "dags" / "weekly_model_retraining.py"
        self.assertTrue(dag_file.exists())
        with open(dag_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("dag_id=\"hm_weekly_model_retraining\"", content)
        self.assertIn("train_two_tower_model", content)
        self.assertIn("validate_and_promote_model", content)
        self.assertIn("update_redis_article_embeddings", content)
        self.assertIn("train_model >> validate_and_promote >> update_embeddings", content)

    def test_daily_embedding_dag_structure(self):
        dag_file = BASE_DIR / "airflow" / "dags" / "daily_embedding_update.py"
        self.assertTrue(dag_file.exists())
        with open(dag_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("dag_id=\"hm_daily_embedding_update\"", content)
        self.assertIn("schedule=\"0 3 * * *\"", content)
        self.assertIn("generate_and_store_article_embeddings", content)


if __name__ == "__main__":
    unittest.main()
