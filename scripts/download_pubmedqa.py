"""Download the PubMedQA expert-labelled set (PQA-L) from the official repository.

Example:
    python scripts/download_pubmedqa.py
"""

from __future__ import annotations

import argparse
import sys

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.evaluation.dataset import PUBMEDQA_REPO, download_pubmedqa, load_pubmedqa


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    try:
        settings = get_settings()
        target = settings.data_dir / "eval" / "pubmedqa"
        manifest = download_pubmedqa(target, force=args.force)
        print(f"Source: {PUBMEDQA_REPO} (downloaded {manifest['downloaded_at']})")
        for name, info in manifest["files"].items():
            print(f"  {name}: {info['bytes']} bytes, sha256 {info['sha256'][:16]}...")
        for split in ("dev", "validation", "test"):
            examples, skipped = load_pubmedqa(target, split, seed=settings.seed)
            print(f"  split {split}: {len(examples)} examples ({skipped} malformed rows skipped)")
    except AppError as exc:
        print(f"Error: {exc.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
