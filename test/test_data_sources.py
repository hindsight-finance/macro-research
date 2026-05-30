"""Tests for env-driven data-source resolution and R2 storage-option threading."""

from pathlib import Path

import polars as pl
import pytest

from utils import data_sources, tick_data

_R2_ENV = {
    "TICK_DATA_URL": "s3://bucket/source/merged_nq_ticks.parquet",
    "ECON_EVENTS_URL": "s3://bucket/source/economic_events.parquet",
    "R2_ACCESS_KEY_ID": "key",
    "R2_SECRET_ACCESS_KEY": "secret",
    "R2_ENDPOINT_URL": "https://acct.r2.cloudflarestorage.com",
}


@pytest.fixture
def _clear_env(monkeypatch):
    for name in (*_R2_ENV, "R2_REGION"):
        monkeypatch.delenv(name, raising=False)


def test_urls_default_to_local_when_env_unset(_clear_env):
    assert data_sources.tick_data_url() == data_sources.LOCAL_TICK_PATH
    assert data_sources.econ_events_url() == data_sources.LOCAL_ECON_EVENTS_PATH
    assert data_sources.storage_options() is None


def test_urls_and_storage_options_follow_env(monkeypatch):
    for name, value in _R2_ENV.items():
        monkeypatch.setenv(name, value)

    assert data_sources.tick_data_url() == _R2_ENV["TICK_DATA_URL"]
    assert data_sources.econ_events_url() == _R2_ENV["ECON_EVENTS_URL"]

    opts = data_sources.storage_options()
    assert opts == {
        "aws_access_key_id": "key",
        "aws_secret_access_key": "secret",
        "aws_region": "auto",
        "endpoint_url": "https://acct.r2.cloudflarestorage.com",
    }


def test_storage_options_none_when_credentials_incomplete(_clear_env, monkeypatch):
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")  # missing secret + endpoint
    assert data_sources.storage_options() is None


def test_is_remote_and_source_exists(tmp_path, _clear_env):
    assert data_sources.is_remote("s3://bucket/key.parquet")
    assert not data_sources.is_remote("input-data/x.parquet")
    # remote inputs are assumed present without a round-trip
    assert data_sources.source_exists("s3://bucket/key.parquet")

    local = tmp_path / "x.parquet"
    assert not data_sources.source_exists(local)
    local.write_text("")
    assert data_sources.source_exists(local)


def test_scan_source_forwards_storage_options_only_for_remote(monkeypatch):
    captured = {}

    def fake_scan_parquet(path, storage_options=None):
        captured["path"] = path
        captured["storage_options"] = storage_options
        return pl.LazyFrame()  # real LazyFrame so the price_ticks schema-peek works

    monkeypatch.setattr(tick_data.pl, "scan_parquet", fake_scan_parquet)
    for name, value in _R2_ENV.items():
        monkeypatch.setenv(name, value)

    # remote default path -> R2 options attached
    assert isinstance(tick_data.scan_source(), pl.LazyFrame)
    assert captured["path"] == _R2_ENV["TICK_DATA_URL"]
    assert captured["storage_options"]["endpoint_url"].endswith("r2.cloudflarestorage.com")

    # explicit local path -> options suppressed even though env creds are set
    tick_data.scan_source("input-data/merged_nq_ticks.parquet")
    assert captured["path"] == "input-data/merged_nq_ticks.parquet"
    assert captured["storage_options"] is None


def _write_price_float(path, ts="2025-01-02T20:50:00Z", price=18312.75):
    pl.DataFrame(
        {"ts_event": [ts], "intra_ts_rank": [0], "side": [2], "price": [price], "size": [5]}
    ).with_columns(pl.col("ts_event").str.to_datetime(time_zone="UTC")).write_parquet(path)


def test_scan_source_synthesizes_price_ticks_from_lake_price(tmp_path, _clear_env):
    path = tmp_path / "lake.parquet"
    _write_price_float(path)  # 18312.75 * 4 == 73251
    df = tick_data.scan_source(path).collect()
    assert "price_ticks" in df.columns
    assert df["price_ticks"].to_list() == [73251]


def test_get_tick_schema_accepts_price_only_lake_file(tmp_path, _clear_env):
    path = tmp_path / "lake.parquet"
    _write_price_float(path)
    names = set(tick_data.get_tick_schema(path).names)
    assert set(tick_data.TICK_COLUMNS) <= names  # price_ticks present despite source having only `price`


def test_iter_tick_batches_normalizes_and_spans_glob(tmp_path, _clear_env):
    _write_price_float(tmp_path / "2024_merged_nq.parquet", ts="2024-06-01T20:50:00Z", price=18000.0)
    _write_price_float(tmp_path / "2025_merged_nq.parquet", ts="2025-06-01T20:50:00Z", price=18100.25)
    glob = str(tmp_path / "*_merged_nq.parquet")
    batches = list(tick_data.iter_tick_batches(glob))
    for b in batches:
        assert "price_ticks" in b.columns and "price" not in b.columns
    assert sum(b.height for b in batches) == 2
    pts = sorted(v for b in batches for v in b["price_ticks"].to_list())
    assert pts == [72000, 72401]  # 18000*4, round(18100.25*4)


def test_minute_nq_url_env_override(monkeypatch):
    monkeypatch.delenv("MINUTE_NQ_URL", raising=False)
    assert data_sources.minute_nq_url("outputs/nq_minute_base.parquet") == "outputs/nq_minute_base.parquet"
    monkeypatch.setenv("MINUTE_NQ_URL", "s3://b/NQ/NQ.ohlcv-1m.parquet")
    assert data_sources.minute_nq_url("outputs/x.parquet") == "s3://b/NQ/NQ.ohlcv-1m.parquet"


def test_get_tick_schema_reads_local_fixture(tmp_path, _clear_env):
    path = tmp_path / "ticks.parquet"
    pl.DataFrame(
        {
            "ts_event": ["2025-01-02T20:50:00Z"],
            "intra_ts_rank": [0],
            "side": [2],
            "price_ticks": [84000],
            "size": [5],
        }
    ).with_columns(pl.col("ts_event").str.to_datetime(time_zone="UTC")).write_parquet(path)

    schema = tick_data.get_tick_schema(path)
    assert set(schema.names) == set(tick_data.TICK_COLUMNS)


def test_open_parquet_file_local_roundtrip(tmp_path, _clear_env):
    path = tmp_path / "ticks.parquet"
    pl.DataFrame({"a": [1, 2, 3]}).write_parquet(path)

    pf = tick_data.open_parquet_file(path)
    assert pf.metadata.num_rows == 3
