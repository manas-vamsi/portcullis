"""API keys for the server: random, shown once, stored only as hashes."""

import json

import pytest
from pydantic import ValidationError

from portcullis.keys import KeyEntry, authenticate, hash_key, load_keys, new_key


def entry(key: str, tenant="acme", role="agent", name="runtime-1") -> KeyEntry:
    return KeyEntry(sha256=hash_key(key), tenant=tenant, role=role, name=name)


def test_new_keys_are_long_random_and_prefixed():
    keys = {new_key() for _ in range(50)}
    assert len(keys) == 50
    assert all(k.startswith("pc_") and len(k) >= 40 for k in keys)


def test_a_valid_key_authenticates_as_its_tenant_role_and_name():
    key = new_key()
    principal = authenticate([entry(key, "acme", "approver", "alice")], key)
    assert (principal.tenant, principal.role, principal.name) == ("acme", "approver", "alice")


@pytest.mark.parametrize("presented", [None, "", "pc_wrong", "x" * 10_000])
def test_a_missing_or_wrong_key_authenticates_as_nobody(presented):
    assert authenticate([entry(new_key())], presented) is None


def test_the_right_key_is_found_among_several():
    first, second = new_key(), new_key()
    entries = [entry(first, "acme"), entry(second, "globex")]
    assert authenticate(entries, second).tenant == "globex"


def test_a_key_that_is_a_hash_of_a_stored_hash_is_not_accepted():
    key = new_key()
    assert authenticate([entry(key)], hash_key(key)) is None  # the stored value cannot be replayed as a key


@pytest.mark.parametrize(
    "fields",
    [
        {"sha256": "short"},
        {"sha256": "G" * 64},
        {"tenant": ""},
        {"role": "admin"},
        {"name": ""},
        {"name": "bad name\n"},
        {"name": "x" * 65},
        {"extra": 1},
    ],
)
def test_a_malformed_entry_is_rejected(fields):
    good = {"sha256": "a" * 64, "tenant": "acme", "role": "agent", "name": "ok-1"}
    with pytest.raises(ValidationError):
        KeyEntry(**{**good, **fields})


def test_keys_load_from_a_json_file(tmp_path):
    key = new_key()
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"keys": [entry(key).model_dump()]}), encoding="utf-8")
    assert authenticate(load_keys(path), key).tenant == "acme"


@pytest.mark.parametrize(
    "content",
    ["not json", '{"keys": 5}', '{"keys": []}', "[]", '{"keys": [{"sha256": "x"}]}'],
    ids=["not_json", "not_a_list", "empty", "not_an_object", "bad_entry"],
)
def test_an_unusable_key_file_is_an_error_not_an_open_server(tmp_path, content):
    path = tmp_path / "keys.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_keys(path)


def test_a_missing_key_file_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        load_keys(tmp_path / "nope.json")


def test_two_entries_for_the_same_key_are_an_error(tmp_path):
    key = new_key()
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"keys": [entry(key).model_dump(), entry(key, "globex").model_dump()]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_keys(path)
