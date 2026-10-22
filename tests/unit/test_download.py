import hashlib

import pytest

from sleep_staging.config import DataConfig
from sleep_staging.data.download import download_file, selected_files


def test_catalog_pairs_by_recording_prefix_and_selects_exact_records():
    entries = {
        "sleep-cassette/SC4001E0-PSG.edf": "a" * 64,
        "sleep-cassette/SC4001EC-Hypnogram.edf": "b" * 64,
        "sleep-cassette/SC4011E0-PSG.edf": "c" * 64,
    }
    pairs = selected_files(
        entries, DataConfig(name="pilot", source="sleep-cassette", records=["SC4001"])
    )
    assert len(pairs) == 1
    assert pairs[0]["recording_id"] == "SC4001"
    with pytest.raises(ValueError, match="hipnograma"):
        selected_files(
            entries, DataConfig(name="bad", source="sleep-cassette", records=["SC4011"])
        )


def test_cached_file_is_verified_and_corruption_is_not_accepted(tmp_path):
    path = tmp_path / "data.edf"
    path.write_bytes(b"verified")
    digest = hashlib.sha256(b"verified").hexdigest()
    download_file("https://invalid.example", path, digest)
    with pytest.raises(ValueError, match="checksum"):
        download_file("https://invalid.example", path, "a" * 64)
