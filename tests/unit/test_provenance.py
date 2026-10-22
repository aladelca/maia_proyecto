from sleep_staging.provenance import fingerprint, hash_file


def test_fingerprint_changes_when_inputs_change_but_not_key_order(tmp_path):
    file = tmp_path / "signal"
    file.write_bytes(b"first")
    a = fingerprint({"file": hash_file(file), "config": {"a": 1, "b": 2}})
    assert a == fingerprint({"config": {"b": 2, "a": 1}, "file": hash_file(file)})
    file.write_bytes(b"second")
    assert a != fingerprint({"file": hash_file(file), "config": {"a": 1, "b": 2}})
