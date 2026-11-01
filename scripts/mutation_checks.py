"""Check that the encoder tests catch deliberately introduced bugs, and how small a bug.

Each mutation is checked as `tests/test_encoders.py` would: the float32 test requires an
error (against float64) within twice PyTorch float32's, and the float64 test a relative error
below `FLOAT64_RELATIVE_TOLERANCE`. Uses the references cached in `reference/` by the
checkpoint tests, so run those first.

Mutations:

- a wrong LayerNorm epsilon (BERT's is 1e-12), as from misreading `config.json`: swept from
  1e-10 to 1e-4 to find the smallest change each test detects;
- token types dropped (all zero), which matters for (title, abstract) pairs.

Run with `uv run python scripts/mutation_checks.py`.
"""

import json
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from eqx_zoo import Encoder
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from biomed_retrieval_jax.encoders import ENCODERS, TextEncoder

jax.config.update("jax_default_matmul_precision", "highest")
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from support.reference import POOLING, inputs  # noqa: E402
from test_encoders import FLOAT32_ERROR_RATIO, FLOAT64_RELATIVE_TOLERANCE  # noqa: E402

NAMES = ("medcpt-query", "medcpt-article", "pubmedbert-embeddings")
EPSILONS = (1e-10, 1e-8, 1e-6, 1e-4)


def _encoder(name, eps, directory, dtype):
    """The encoder, unchanged if `eps` is None, else with LayerNorm epsilon `eps`."""
    spec = ENCODERS[name]
    if eps is None:
        return TextEncoder.from_spec(spec, dtype=dtype)
    path = directory / f"{name}-{eps:g}"
    if not path.exists():
        path.mkdir()
        source = Path(snapshot_download(spec.repo, allow_patterns=["config.json", "*.safetensors"]))
        config = json.loads((source / "config.json").read_text())
        config["layer_norm_eps"] = eps
        (path / "config.json").write_text(json.dumps(config))
        (path / "model.safetensors").symlink_to((source / "model.safetensors").resolve())
    encoder = Encoder.from_pretrained(path, dtype=dtype, pooling=POOLING[name])
    return TextEncoder(encoder, Tokenizer.from_pretrained(spec.repo), spec.max_length)


def _max_error(value, reference):
    return float(np.abs(value - reference).max())


def main():
    """Print, for each mutation, whether the float32 and float64 tests would catch it."""
    references = {name: dict(np.load(ROOT / "reference" / f"{name}.npz")) for name in NAMES}
    mutations = [("none", None)] + [(f"eps {eps:g}", eps) for eps in EPSILONS]
    float32, float64 = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for name in NAMES:
            exact = references[name]["embeddings_float64"]
            for label, eps in mutations:
                encoder = _encoder(name, eps, directory, jnp.float32)
                float32[name, label] = _max_error(encoder.embed(inputs(name), batch_size=3), exact)

        original = TextEncoder._pad

        def without_types(self, encodings, length):
            ids, types, mask = original(self, encodings, length)
            return ids, np.zeros_like(types), mask

        TextEncoder._pad = without_types
        try:
            encoder = TextEncoder.from_spec(ENCODERS["medcpt-article"])
            exact = references["medcpt-article"]["embeddings_float64"]
            float32["medcpt-article", "no token types"] = _max_error(
                encoder.embed(inputs("medcpt-article"), batch_size=3), exact
            )
        finally:
            TextEncoder._pad = original

        # Float64 mode is global, so it is switched on only after every float32 measurement.
        jax.config.update("jax_enable_x64", True)
        for name in NAMES:
            exact = references[name]["embeddings_float64"]
            for label, eps in mutations:
                encoder = _encoder(name, eps, directory, jnp.float64)
                error = _max_error(encoder.embed(inputs(name), batch_size=3), exact)
                float64[name, label] = error / float(np.abs(exact).max())

    print(
        f"\n{'encoder':22s} {'mutation':15s} {'float32 error':>13s} {'bound':>9s}  caught"
        f"  {'float64 rel.':>12s} {'bound':>9s}  caught"
    )
    for (name, label), error in float32.items():
        exact = references[name]["embeddings_float64"]
        bound = FLOAT32_ERROR_RATIO * _max_error(references[name]["embeddings_float32"], exact)
        line = f"{name:22s} {label:15s} {error:13.2e} {bound:9.2e}  {str(error > bound):6s}"
        if (name, label) in float64:
            rel = float64[name, label]
            line += f"  {rel:12.2e} {FLOAT64_RELATIVE_TOLERANCE:9.0e}  "
            line += str(rel >= FLOAT64_RELATIVE_TOLERANCE)
        print(line)


if __name__ == "__main__":
    main()
