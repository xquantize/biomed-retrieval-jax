"""Pretrained checkpoints used by the test suite.

Standard library only: CI imports this module with a bare Python to list its jobs.
"""

# Pretrained encoders: test id -> Hugging Face repository. The ids are the keys of the
# package's `ENCODERS`, and a fast test checks that the two agree.
ENCODER_CHECKPOINTS: dict[str, str] = {
    "medcpt-query": "ncbi/MedCPT-Query-Encoder",  # [CLS] pooling, queries up to 64 tokens
    "medcpt-article": "ncbi/MedCPT-Article-Encoder",  # [CLS] pooling, (title, abstract) pairs
    "pubmedbert-embeddings": "NeuML/pubmedbert-base-embeddings",  # mean pooling
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",  # mean pooling, normalised
}

# Every registered checkpoint: used to validate --checkpoint and to create CI jobs.
ALL_CHECKPOINTS = {**ENCODER_CHECKPOINTS}
