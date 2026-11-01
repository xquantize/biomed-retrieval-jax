"""Pretrained checkpoints used by the test suite.

Standard library only: CI imports this module with a bare Python to list its jobs.
"""

# Pretrained encoders: test id -> Hugging Face repository. Added once each checkpoint's
# licence, architecture and pooling have been checked.
ENCODER_CHECKPOINTS: dict[str, str] = {}

# Every registered checkpoint: used to validate --checkpoint and to create CI jobs.
ALL_CHECKPOINTS = {**ENCODER_CHECKPOINTS}
