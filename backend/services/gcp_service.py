"""Optional Google Cloud Storage integration for model checkpoints.

All public functions are safe to call unconditionally:
  - When GCP_PROJECT_ID / GCP_BUCKET_NAME are unset, every call is a no-op
    that returns False (or None for downloads).
  - When `google-cloud-storage` isn't installed, same — a single warning
    is emitted on first use, then the module silently no-ops.

This keeps `pip install -r requirements.txt` light for local dev — users
who actually deploy to GCP add `google-cloud-storage` themselves (it is
listed in `gcp/requirements-gcp.txt`).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

from environment import config

_warned = False
_client = None
_bucket = None


def _gcp_enabled() -> bool:
    return bool(config.GCP_PROJECT_ID) and bool(config.GCP_BUCKET_NAME)


def _get_bucket():
    global _client, _bucket, _warned
    if _bucket is not None:
        return _bucket
    if not _gcp_enabled():
        return None
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        if not _warned:
            warnings.warn(
                "google-cloud-storage not installed — GCS checkpoint upload "
                "disabled. `pip install google-cloud-storage` to enable."
            )
            _warned = True
        return None
    try:
        _client = storage.Client(project=config.GCP_PROJECT_ID)
        _bucket = _client.bucket(config.GCP_BUCKET_NAME)
    except Exception as e:
        if not _warned:
            warnings.warn(f"GCS client init failed ({e}) — uploads disabled.")
            _warned = True
        return None
    return _bucket


def upload_checkpoint(local_path: str | Path, remote_key: str) -> bool:
    """Upload a single file to GCS. Returns True on success, False otherwise."""
    bucket = _get_bucket()
    if bucket is None:
        return False
    p = Path(local_path)
    if not p.exists():
        return False
    try:
        blob = bucket.blob(remote_key)
        blob.upload_from_filename(str(p))
        return True
    except Exception as e:
        warnings.warn(f"GCS upload failed for {remote_key}: {e}")
        return False


def download_checkpoint(remote_key: str, local_path: str | Path) -> Optional[str]:
    """Download a GCS object to disk. Returns local path on success, None otherwise."""
    bucket = _get_bucket()
    if bucket is None:
        return None
    try:
        blob = bucket.blob(remote_key)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        return str(local_path)
    except Exception as e:
        warnings.warn(f"GCS download failed for {remote_key}: {e}")
        return None


def upload_run_checkpoints(run_id: str, local_dir: str | Path) -> int:
    """Upload every *.pt under `local_dir` to gs://<bucket>/runs/<run_id>/.
    Returns count of successfully uploaded files.
    """
    if not _gcp_enabled():
        return 0
    d = Path(local_dir)
    if not d.exists():
        return 0
    n = 0
    for f in d.glob("*.pt"):
        if upload_checkpoint(f, f"runs/{run_id}/{f.name}"):
            n += 1
    return n
