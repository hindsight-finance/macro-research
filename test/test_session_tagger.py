from pathlib import Path

import pandas as pd

from session_tagger import process_file


def test_process_file_writes_minute_base_with_datetime_utc(tmp_path: Path):
    input_path = tmp_path / "nq_1m.csv"
    output_dir = tmp_path / "outputs"
    pd.DataFrame(
        {
            "DateTime_ET": ["2020-09-01 15:50:00", "2020-09-01 15:51:00"],
            "Open": [1.0, 2.0],
            "High": [2.0, 3.0],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [10, 11],
        }
    ).to_csv(input_path, index=False)

    out_path = process_file(str(input_path), str(output_dir))
    out = pd.read_parquet(out_path)

    assert out_path.name == "nq_minute_base.parquet"
    assert list(out.columns) == ["datetime_utc", "Open", "High", "Low", "Close", "Volume"]
    assert str(out["datetime_utc"].dtype) == "datetime64[ns, UTC]"
