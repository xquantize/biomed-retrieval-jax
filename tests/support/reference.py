"""Capture reference tokenization and embeddings from Transformers, in float32 and float64.

Each input is embedded alone, without padding, and pooled (and normalised) as stated in
`POOLING` and `NORMALIZE`, which are written out here rather than read from the checkpoint so
that the reference does not share the code under test. For checkpoints with a
sentence-transformers pipeline, that pipeline's own float32 embeddings are captured too, to
confirm the stated pooling.
"""

import numpy as np

REPEATED = "persistent fatigue, joint pain and an intermittent low-grade fever"
TEXTS = [
    "diabetes treatment",
    "Which drugs lower blood pressure in older adults with chronic kidney disease?",
    "Hello!",
    "Symptoms: " + ", ".join([REPEATED] * 10),  # over 64 tokens
]
PAIRS = [
    (
        "Vitamin D supplementation and bone density in older adults",
        "We randomised 400 adults aged over 65 to vitamin D or placebo for two years. "
        "Bone mineral density at the hip did not differ between the groups.",
    ),
    ("Short title", "Short abstract."),
    ("A long abstract", " ".join([REPEATED] * 80)),  # over 512 tokens: abstract truncated
    (" ".join([REPEATED] * 25), " ".join([REPEATED] * 80)),  # both long: both truncated
]

# Pooling of each encoder, from its model card or sentence-transformers configuration.
POOLING = {
    "medcpt-query": "cls",
    "medcpt-article": "cls",
    "pubmedbert-embeddings": "mean",
    "all-minilm-l6-v2": "mean",
}
# Encoders whose embeddings are scaled to unit length.
NORMALIZE = {"all-minilm-l6-v2"}
# Encoders whose checkpoints include a sentence-transformers pipeline.
SENTENCE_TRANSFORMERS = {"pubmedbert-embeddings", "all-minilm-l6-v2"}


def inputs(name: str) -> list:
    """Test inputs for an encoder: (title, abstract) pairs for MedCPT's article encoder,
    texts for the others (with each pair joined by a space, to include long inputs)."""
    if name == "medcpt-article":
        return list(PAIRS)
    if name == "medcpt-query":
        return list(TEXTS)
    return TEXTS + [f"{title} {abstract}" for title, abstract in PAIRS]


def capture(name: str) -> dict[str, np.ndarray]:
    """Transformers' tokenization and pooled embeddings of `inputs(name)`.

    Embeddings are computed with each input alone, in float32 and in float64.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    from biomed_retrieval_jax.encoders import ENCODERS

    spec = ENCODERS[name]
    xs = inputs(name)
    hf = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
    tokens = [
        hf(*(x if isinstance(x, tuple) else (x,)), truncation=True, max_length=spec.max_length)
        for x in xs
    ]
    out = {
        "token_ids": np.concatenate([t["input_ids"] for t in tokens]),
        "token_types": np.concatenate([t["token_type_ids"] for t in tokens]),
        "lengths": np.array([len(t["input_ids"]) for t in tokens]),
    }
    for dtype in (torch.float32, torch.float64):
        model = AutoModel.from_pretrained(
            spec.repo, revision=spec.revision, dtype=dtype, attn_implementation="eager"
        )
        rows = []
        for t in tokens:
            with torch.no_grad():
                hidden = model.eval()(**{k: torch.tensor([v]) for k, v in t.items()})
            hidden = hidden.last_hidden_state[0]
            pooled = hidden[0] if POOLING[name] == "cls" else hidden.mean(0)
            if name in NORMALIZE:
                pooled = torch.nn.functional.normalize(pooled, dim=0)
            rows.append(pooled.numpy())
        out[f"embeddings_{str(dtype).removeprefix('torch.')}"] = np.stack(rows)
    if name in SENTENCE_TRANSFORMERS:
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(spec.repo, device="cpu", revision=spec.revision)
        out["sentence_transformers_float32"] = st.to(torch.float32).encode(xs)
    return out
