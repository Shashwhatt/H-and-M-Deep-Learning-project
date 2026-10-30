
import math


def recall_at_k(recommended_items, relevant_items, k=5):
    """Fraction of relevant items found in the top k recommendations."""
    relevant_items = set(relevant_items)

    if not relevant_items:
        return 0.0

    top_k = recommended_items[:k]
    hits = len(set(top_k) & relevant_items)

    return hits / len(relevant_items)


def ndcg_at_k(recommended_items, relevant_items, k=5):
    """Measures whether relevant items appear near the top."""
    relevant_items = set(relevant_items)
    top_k = recommended_items[:k]

    dcg = sum(
        1 / math.log2(rank + 2)
        for rank, item in enumerate(top_k)
        if item in relevant_items
    )

    ideal_hits = min(len(relevant_items), k)
    ideal_dcg = sum(
        1 / math.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    return dcg / ideal_dcg if ideal_dcg else 0.0


if __name__ == "__main__":
    recommendations = ["A", "X", "B", "Y", "Z"]
    relevant = {"A", "B", "C"}

    print("Example Recall@5:", recall_at_k(recommendations, relevant, 5))
    print("Example NDCG@5:", ndcg_at_k(recommendations, relevant, 5))
