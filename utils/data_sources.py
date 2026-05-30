"""Resolve where raw research inputs live (local files vs. remote object storage).

Research scripts read large source datasets (the tick parquet, the economic-events
parquet) from local ``input-data/`` by default. When the same scripts run on a
GitHub Actions runner the data lives in a Cloudflare R2 bucket instead, and is read
in place via polars range reads (``pl.scan_parquet`` over an ``s3://`` URL only
fetches the parquet row groups a predicate touches -- it never downloads the file).

The location is selected purely by environment variables, so nothing changes when
running locally (env unset -> local paths, ``storage_options`` -> ``None``):

- ``TICK_DATA_URL``      -- override for ``merged_nq_ticks.parquet`` (e.g. ``s3://bucket/source/merged_nq_ticks.parquet``)
- ``ECON_EVENTS_URL``    -- override for ``economic_events.parquet``
- ``R2_ACCESS_KEY_ID`` / ``R2_SECRET_ACCESS_KEY`` / ``R2_ENDPOINT_URL`` -- R2 credentials
- ``R2_REGION``          -- optional, defaults to ``auto``

Functions return ``str`` (never ``Path``) so that ``s3://`` URLs are not mangled
(``Path("s3://x/y")`` collapses the ``//`` to ``s3:/x/y``).
"""

from __future__ import annotations

import os
from pathlib import Path

LOCAL_TICK_PATH = "input-data/merged_nq_ticks.parquet"
LOCAL_ECON_EVENTS_PATH = "input-data/economic_events.parquet"

_REMOTE_SCHEMES = ("s3://", "r2://", "gs://", "az://", "http://", "https://")


def tick_data_url() -> str:
    """Location of the tick parquet: ``$TICK_DATA_URL`` or the local default."""
    return os.environ.get("TICK_DATA_URL", LOCAL_TICK_PATH)


def econ_events_url() -> str:
    """Location of the economic-events parquet: ``$ECON_EVENTS_URL`` or local default."""
    return os.environ.get("ECON_EVENTS_URL", LOCAL_ECON_EVENTS_PATH)


def minute_nq_url(local_default: str) -> str:
    """Location of the canonical NQ minute parquet.

    Returns ``$MINUTE_NQ_URL`` when set (e.g. the data-lake ohlcv-1m object on R2),
    else the caller's existing local default (callers differ: ``nq_minute_base.parquet``
    vs ``nq_1m.parquet``), so local behavior is preserved per-script.
    """
    return os.environ.get("MINUTE_NQ_URL", local_default)


def storage_options() -> dict[str, str] | None:
    """polars ``storage_options`` dict for R2, or ``None`` when running locally.

    Returns ``None`` unless all three R2 credential variables are present, so a
    local run (or a run pointed at a local file) passes ``storage_options=None``
    to polars, which is identical to not passing it at all.
    """
    key = os.environ.get("R2_ACCESS_KEY_ID")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("R2_ENDPOINT_URL")
    if not (key and secret and endpoint):
        return None
    return {
        "aws_access_key_id": key,
        "aws_secret_access_key": secret,
        "aws_region": os.environ.get("R2_REGION", "auto"),
        "endpoint_url": endpoint,
    }


def is_remote(path: object) -> bool:
    """True when ``path`` is an object-storage / HTTP URL rather than a local path."""
    return str(path).startswith(_REMOTE_SCHEMES)


def source_exists(path: object) -> bool:
    """Whether an input is present.

    Local paths are stat-ed; remote URLs are assumed present (statting R2 per run
    is not worth a round-trip -- a genuinely missing object surfaces as a clear
    error when polars tries to read it).
    """
    if is_remote(path):
        return True
    return Path(path).exists()
