"""Shared test setup.

No network access and no real credential is required by anything under
``tests/`` except ``test_orcarouter_live.py``, which is the only file allowed
to see a real ``ORCAROUTER_API_KEY``. Everything else uses fake values so a
test can never mint a real key, hit the public service, or write a real secret
to disk.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Capture the real key (if the runner provided one) and then take it out of the
# environment, so a stray test cannot accidentally read it.
REAL_API_KEY = os.environ.pop("ORCAROUTER_API_KEY", "")

# Likewise for the origin overrides: a test must never inherit a real one.
for _var in ("ORCA_BASE_URL", "ORCA_AUTH_BASE_URL", "ORCA_API_BASE_URL"):
    os.environ.pop(_var, None)

import pytest  # noqa: E402


@pytest.fixture()
def live_api_key() -> str:
    """The real key, for the live check only."""
    if not REAL_API_KEY:
        pytest.skip("ORCAROUTER_API_KEY is not set")
    return REAL_API_KEY
