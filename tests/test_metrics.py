"""Our retrieval metrics against pytrec_eval, the reference used by BEIR and MTEB."""

import random

import pytest
import pytrec_eval

from biomed_retrieval_jax.metrics import K_VALUES, evaluate, per_query

# Largest difference allowed between our metrics and pytrec_eval's, both in float64. Measured:
# 2.2e-16 at most (one rounding step); breaking ties in the wrong order gives ~5e-3.
TOLERANCE = 1e-12


def random_case(seed):
    """Graded judgments, coarse scores (so many ties) and numeric-looking ids, with queries
    whose judgments are all zero and queries only in the run or only in the judgments."""
    rng = random.Random(seed)
    docs = [str(i) for i in range(60)]
    run, qrels = {}, {}
    for q in range(40):
        query = str(q)
        if q % 10 != 9:  # every tenth query has no judgments
            judged = rng.sample(docs, rng.randint(1, 15))
            grades = [0] if q % 10 == 7 else [0, 1, 1, 2]  # some queries: all zero
            qrels[query] = {doc: rng.choice(grades) for doc in judged}
        if q % 10 != 8:  # every tenth query has no results
            retrieved = rng.sample(docs, rng.randint(1, 50))
            run[query] = {doc: round(rng.random(), 1) for doc in retrieved}
    return run, qrels


def pytrec(run, qrels):
    cutoffs = ",".join(map(str, K_VALUES))
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{cutoffs}", f"recall.{cutoffs}"})
    return {
        query: {
            **{f"ndcg@{k}": scores[f"ndcg_cut_{k}"] for k in K_VALUES},
            **{f"recall@{k}": scores[f"recall_{k}"] for k in K_VALUES},
        }
        for query, scores in evaluator.evaluate(run).items()
    }


@pytest.mark.parametrize("seed", range(5))
def test_per_query_metrics_match_pytrec_eval(seed):
    run, qrels = random_case(seed)
    ours, reference = per_query(run, qrels), pytrec(run, qrels)
    assert ours.keys() == reference.keys()
    worst = max(abs(ours[q][m] - reference[q][m]) for q in reference for m in reference[q])
    print(f"seed {seed}: {len(reference)} queries, largest difference {worst:.1e}")
    assert worst <= TOLERANCE


def test_means_match_mteb_averaging():
    run, qrels = random_case(0)
    reference = pytrec(run, qrels)
    ours = evaluate(run, qrels)
    for name in ours:
        expected = sum(scores[name] for scores in reference.values()) / len(reference)
        assert abs(ours[name] - expected) <= TOLERANCE


def test_evaluate_rejects_disjoint_queries():
    with pytest.raises(ValueError, match="no query"):
        evaluate({"a": {"d": 1.0}}, {"b": {"d": 1}})
