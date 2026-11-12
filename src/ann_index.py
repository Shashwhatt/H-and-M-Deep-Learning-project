import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class ApproximateNearestNeighborIndex:
    """
    Approximate Nearest Neighbor (ANN) index for fast candidate retrieval
    in recommendation systems.

    Implements an Inverted File (IVF) structure with spherical cosine clustering
    to partition item embedding spaces into clusters (Voronoi cells).

    During inference, instead of comparing a query against all N candidates,
    the index evaluates only the top `n_probe` centroids and searches their
    corresponding candidate partitions, achieving sub-linear O((n_probe/C) * N)
    retrieval time while preserving high Recall@K.
    """

    def __init__(self, n_clusters: int = 32, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.centroids: Optional[np.ndarray] = None
        self.inverted_lists: Dict[int, List[int]] = {}
        self.item_ids: List[str] = []
        self.normalized_embeddings: Optional[np.ndarray] = None
        self.embedding_dim: int = 0
        self.is_built: bool = False

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2 normalizes vectors for efficient cosine similarity via dot product."""
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (vectors / norms).astype(np.float32)

    def build(self, embeddings: np.ndarray, item_ids: List[str], max_iter: int = 25) -> "ApproximateNearestNeighborIndex":
        """
        Builds the ANN IVF index from item embeddings and corresponding item IDs.

        Args:
            embeddings: Float32 array of shape (N, D).
            item_ids: List of item identifier strings of length N.
            max_iter: Maximum iterations for spherical k-means clustering.
        """
        if len(embeddings) != len(item_ids):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(item_ids)} item IDs."
            )

        n_samples, dim = embeddings.shape
        self.embedding_dim = dim
        self.item_ids = list(item_ids)

        # Normalize item embeddings for cosine similarity
        self.normalized_embeddings = self._normalize(embeddings)

        # Adjust clusters if dataset is small
        actual_clusters = min(self.n_clusters, n_samples)
        self.n_clusters = max(1, actual_clusters)

        # Initialize centroids using k-means++ style initialization
        rng = np.random.default_rng(self.random_state)
        init_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = self.normalized_embeddings[init_indices].copy()

        # Spherical k-means clustering loop
        for _ in range(max_iter):
            # Assign each vector to closest centroid (max dot product)
            sims = np.dot(self.normalized_embeddings, centroids.T)
            labels = np.argmax(sims, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for c in range(self.n_clusters):
                mask = (labels == c)
                if np.any(mask):
                    cluster_mean = self.normalized_embeddings[mask].mean(axis=0)
                    norm = np.linalg.norm(cluster_mean)
                    new_centroids[c] = cluster_mean / norm if norm > 0 else centroids[c]
                else:
                    # Re-seed empty cluster from random sample
                    rand_idx = rng.integers(0, n_samples)
                    new_centroids[c] = self.normalized_embeddings[rand_idx]

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < 1e-4:
                break

        self.centroids = centroids

        # Build inverted lists mapping cluster_id -> list of sample indices
        sims = np.dot(self.normalized_embeddings, self.centroids.T)
        final_labels = np.argmax(sims, axis=1)

        self.inverted_lists = {c: [] for c in range(self.n_clusters)}
        for idx, cluster_id in enumerate(final_labels):
            self.inverted_lists[cluster_id].append(idx)

        self.is_built = True
        return self

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        n_probe: int = 4
    ) -> List[Tuple[str, float]]:
        """
        Retrieves top_k nearest candidates for a single query embedding vector.

        Args:
            query_embedding: Float vector of shape (D,) or (1, D).
            top_k: Number of nearest items to return.
            n_probe: Number of centroid clusters to inspect (speed vs recall tradeoff).

        Returns:
            List of (item_id, similarity_score) sorted descending by score.
        """
        if not self.is_built:
            raise RuntimeError("ANN index has not been built yet. Call build() or load().")

        query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query_vec.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query dimension {query_vec.shape[0]} does not match index dimension {self.embedding_dim}."
            )

        # Normalize query vector
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        # Find closest centroids (n_probe clusters)
        centroid_sims = np.dot(self.centroids, query_vec)
        probe_count = min(max(1, n_probe), self.n_clusters)
        probe_clusters = np.argpartition(centroid_sims, -probe_count)[-probe_count:]
        probe_clusters = probe_clusters[np.argsort(centroid_sims[probe_clusters])[::-1]]

        # Gather candidate indices from probed clusters
        candidate_indices = []
        for c in probe_clusters:
            candidate_indices.extend(self.inverted_lists.get(int(c), []))

        if not candidate_indices:
            # Fallback to all items if probed clusters were empty
            candidate_indices = list(range(len(self.item_ids)))

        candidate_indices = np.array(candidate_indices, dtype=np.int32)
        candidate_vecs = self.normalized_embeddings[candidate_indices]

        # Compute cosine similarities for candidates
        scores = np.dot(candidate_vecs, query_vec)

        # Select top-k
        k = min(top_k, len(candidate_indices))
        if k <= 0:
            return []

        top_local_idx = np.argpartition(scores, -k)[-k:]
        top_local_idx = top_local_idx[np.argsort(scores[top_local_idx])[::-1]]

        return [
            (self.item_ids[candidate_indices[i]], float(scores[i]))
            for i in top_local_idx
        ]

    def exact_query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Computes exact brute-force top-k cosine similarity (used as ground truth baseline)."""
        if not self.is_built:
            raise RuntimeError("ANN index has not been built yet.")

        query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        scores = np.dot(self.normalized_embeddings, query_vec)
        k = min(top_k, len(self.item_ids))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            (self.item_ids[i], float(scores[i]))
            for i in top_indices
        ]

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves the index to an .npz archive."""
        if not self.is_built:
            raise RuntimeError("Cannot save an unbuilt ANN index.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "embedding_dim": self.embedding_dim,
            "item_ids": self.item_ids,
        }

        # Pack inverted lists
        inv_list_keys = []
        inv_list_vals = []
        for c, indices in self.inverted_lists.items():
            inv_list_keys.append(int(c))
            inv_list_vals.append(np.array(indices, dtype=np.int32))

        np.savez_compressed(
            filepath,
            centroids=self.centroids,
            normalized_embeddings=self.normalized_embeddings,
            metadata=json.dumps(meta),
            inv_keys=np.array(inv_list_keys, dtype=np.int32),
            **{f"inv_val_{k}": v for k, v in zip(inv_list_keys, inv_list_vals)}
        )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ApproximateNearestNeighborIndex":
        """Loads a saved ANN index from an .npz archive."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"ANN index file not found: {filepath}")

        with np.load(filepath, allow_pickle=True) as data:
            meta = json.loads(str(data["metadata"]))
            index = cls(
                n_clusters=meta["n_clusters"],
                random_state=meta["random_state"]
            )
            index.embedding_dim = meta["embedding_dim"]
            index.item_ids = meta["item_ids"]
            index.centroids = data["centroids"]
            index.normalized_embeddings = data["normalized_embeddings"]

            inv_keys = data["inv_keys"]
            index.inverted_lists = {}
            for k in inv_keys:
                key_int = int(k)
                index.inverted_lists[key_int] = list(data[f"inv_val_{key_int}"])

            index.is_built = True
            return index
