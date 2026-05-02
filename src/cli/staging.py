"""Local staging directories for transient CLI artifacts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGING_ROOT = PROJECT_ROOT / "artifacts" / "staging"


@contextmanager
def temporary_staging_directory(*, prefix: str) -> Iterator[Path]:
    """Create a temporary artifact staging directory under the repo workspace."""
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=prefix, dir=STAGING_ROOT) as staging_root:
        yield Path(staging_root)
