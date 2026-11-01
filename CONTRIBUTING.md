# Contributing to biomed-retrieval-jax

Thanks for your interest in contributing. biomed-retrieval-jax is a semantic search over
biomedical literature built on [eqx-zoo](https://github.com/xquantize/eqx-zoo), whose models and
retrieval results are measured rather than claimed, so every change comes with tests or
measurements that prove it. It is a research demo, not for clinical use.

## Development setup

```bash
git clone https://github.com/xquantize/biomed-retrieval-jax && cd biomed-retrieval-jax
uv sync
uv run pytest                          # fast: unit tests, no downloads
uv run pytest --checkpoint NAME        # one pretrained checkpoint, as CI runs them
uv run ruff format && uv run ruff check
```

Reference outputs are captured from the PyTorch implementation on first use and cached in
`reference/`; add `--regenerate-reference` to refresh them.

Checkpoint tests download and load real models, so plain `uv run pytest` leaves them out, and
every run prints its peak memory. For most changes the fast tests are enough locally: CI runs
every checkpoint on each pull request.

## Conventions

- Google-style docstrings on all public modules, classes and functions, with jaxtyping shape
  annotations. Float32 by default.
- Every tolerance and threshold comes from a measurement, ideally against a yardstick such as
  how much the reference implementation disagrees with itself, and is validated by
  deliberately introducing a bug that the test must catch. Note the platform it was measured
  on: CI runs on x86-64 Linux.
- Reference PyTorch models are loaded directly in float32.
- `tests/support/registry.py` uses only the standard library: CI imports it with a bare Python
  to create one job per checkpoint.
- The README uses absolute URLs for images and links, because PyPI renders it without the
  repository.

## Data and licences

- Public research data only. Never add patient data or other personal health information.
- Before adding a model or dataset, check its licence and record it, with a link to the source,
  in the pull request.

## Adding a checkpoint

Add it to `ENCODER_CHECKPOINTS` in `tests/support/registry.py`, with a comment noting its
pooling, and run the full test suite.

## Pull requests

- Keep each pull request to one change, with tests, and keep diffs minimal.
- Pull requests that change retrieval results should include before/after metrics on the same
  benchmark.

## Versioning

biomed-retrieval-jax follows semantic versioning, adapted for releases before 1.0.

- **Public API:** the names importable from `biomed_retrieval_jax`. Everything else, including
  modules whose names start with `_`, is internal and may change in any release.
- **Patch releases (`0.x.Z`)** contain backwards-compatible changes: new features, fixes and
  documentation. Most releases are patch releases.
- **Minor releases (`0.X.0`)** are reserved for changes that break the public API. Their release
  notes list every break and how to migrate.

## Releasing (maintainers)

1. From an up-to-date `main`: `git switch -c release/vX.Y.Z`, then `uv version --bump patch`
   (or `minor` for a breaking change; see Versioning), and update `version` and
   `date-released` in `CITATION.cff`.
2. Commit, push, open a pull request, wait for CI, and squash-merge.
3. `gh release create vX.Y.Z --target main --title vX.Y.Z --notes-file <notes>`. The Release
   workflow checks that the tag matches the package version, then publishes to PyPI.
4. Wait a few seconds for the run to register, then find it with
   `gh run list --workflow release.yml` and follow it with `gh run watch <id>`.
5. Verify the published package:
   `uv run --isolated --no-project --refresh --with "biomed-retrieval-jax==X.Y.Z" python -c "import biomed_retrieval_jax; print(biomed_retrieval_jax.__version__)"`
