"""Retrieval metrics, computed as trec_eval (and so pytrec_eval, BEIR and MTEB) computes them."""

import math
from collections.abc import Mapping, Sequence

K_VALUES = (1, 3, 5, 10, 20, 100, 1000)


def rank(scores: Mapping[str, float]) -> list[str]:
    """Order retrieved documents as trec_eval does.

    Args:
        scores: Each retrieved document's score.

    Returns:
        Document ids by score, highest first; ties are broken by id, in reverse string order.
    """
    by_id = sorted(scores, reverse=True)
    return sorted(by_id, key=lambda doc: scores[doc], reverse=True)  # stable: keeps id order


def ndcg_at_k(ranking: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    """Normalised discounted cumulative gain of the first `k` documents.

    Args:
        ranking: Retrieved document ids, in rank order.
        judgments: Relevance grade of each judged document; unjudged documents count as 0.
        k: Cutoff.

    Returns:
        The gain of the ranking over that of the ideal ranking, with each document's gain its
        grade, discounted by log2(rank + 1); 0 if no judged document is relevant.
    """
    gains = [max(judgments.get(doc, 0), 0) for doc in ranking[:k]]
    ideal = sorted((grade for grade in judgments.values() if grade > 0), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    return _dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(ranking: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    """Fraction of the relevant documents (grade at least 1) among the first `k`.

    Args:
        ranking: Retrieved document ids, in rank order.
        judgments: Relevance grade of each judged document.
        k: Cutoff.

    Returns:
        The recall, or 0 if no document is relevant.
    """
    relevant = {doc for doc, grade in judgments.items() if grade >= 1}
    return len(relevant.intersection(ranking[:k])) / len(relevant) if relevant else 0.0


def per_query(
    run: Mapping[str, Mapping[str, float]],
    qrels: Mapping[str, Mapping[str, int]],
    k_values: Sequence[int] = K_VALUES,
) -> dict[str, dict[str, float]]:
    """nDCG@k and Recall@k of each query with both retrieved documents and judgments.

    Args:
        run: Each query's retrieved documents and their scores.
        qrels: Each query's relevance judgments.
        k_values: Cutoffs.

    Returns:
        For each such query, its metrics, named `"ndcg@k"` and `"recall@k"`.
    """
    metrics = {}
    for query in run.keys() & qrels.keys():
        ranking = rank(run[query])
        metrics[query] = {
            **{f"ndcg@{k}": ndcg_at_k(ranking, qrels[query], k) for k in k_values},
            **{f"recall@{k}": recall_at_k(ranking, qrels[query], k) for k in k_values},
        }
    return metrics


def evaluate(
    run: Mapping[str, Mapping[str, float]],
    qrels: Mapping[str, Mapping[str, int]],
    k_values: Sequence[int] = K_VALUES,
) -> dict[str, float]:
    """Mean nDCG@k and Recall@k over the queries with both results and judgments, as MTEB.

    Args:
        run: Each query's retrieved documents and their scores.
        qrels: Each query's relevance judgments.
        k_values: Cutoffs.

    Returns:
        Each metric's mean, unrounded.

    Raises:
        ValueError: If no query has both results and judgments.
    """
    metrics = per_query(run, qrels, k_values)
    if not metrics:
        raise ValueError("no query has both results and judgments")
    names = next(iter(metrics.values())).keys()
    return {name: sum(m[name] for m in metrics.values()) / len(metrics) for name in names}


def _dcg(gains: Sequence[float]) -> float:
    return sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
