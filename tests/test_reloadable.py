"""A file whose parsed contents are reloaded when it changes, and refused (fail closed) when the new version is bad."""

import os

import pytest

from portcullis.reloadable import ConfigUnavailable, Reloadable


def load(path):
    text = path.read_text(encoding="utf-8")
    if text.startswith("bad"):
        raise ValueError("this version is not valid")
    return text.strip()


def bump(path, text):
    """Write a new version and make sure the modification time visibly changes."""
    path.write_text(text, encoding="utf-8")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))


def test_the_first_version_is_read_at_startup_and_a_bad_one_stops_startup(tmp_path):
    good = tmp_path / "a.txt"
    good.write_text("one", encoding="utf-8")
    assert Reloadable(good, load).get() == "one"
    bad = tmp_path / "b.txt"
    bad.write_text("bad", encoding="utf-8")
    with pytest.raises(ValueError):
        Reloadable(bad, load)
    with pytest.raises(OSError):
        Reloadable(tmp_path / "missing.txt", load)


def test_an_unchanged_file_is_not_read_again(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one", encoding="utf-8")
    reads = []
    source = Reloadable(path, lambda p: reads.append(1) or "one")
    for _ in range(5):
        source.get()
    assert len(reads) == 1


def test_a_changed_file_is_picked_up_on_the_next_request(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one", encoding="utf-8")
    source = Reloadable(path, load)
    bump(path, "two")
    assert source.get() == "two"


def test_a_bad_new_version_refuses_everything_until_a_good_one_arrives(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one", encoding="utf-8")
    source = Reloadable(path, load)

    bump(path, "bad edit")
    with pytest.raises(ConfigUnavailable):  # the old version is NOT kept: it might be the looser one
        source.get()
    with pytest.raises(ConfigUnavailable):
        source.get()

    bump(path, "three")
    assert source.get() == "three"


def test_a_file_that_disappears_refuses_everything(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one", encoding="utf-8")
    source = Reloadable(path, load)
    path.unlink()
    with pytest.raises(ConfigUnavailable):
        source.get()
    path.write_text("back", encoding="utf-8")
    assert source.get() == "back"


def test_the_error_never_quotes_the_files_contents(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one", encoding="utf-8")
    source = Reloadable(path, lambda p: (_ for _ in ()).throw(ValueError("secret-in-message")) if "x" in p.read_text() else "ok")
    bump(path, "x")
    with pytest.raises(ConfigUnavailable) as raised:
        source.get()
    assert "secret-in-message" not in str(raised.value) and "ValueError" in str(raised.value)


def test_a_transient_failure_is_retried_on_the_next_request_without_the_file_changing(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one", encoding="utf-8")
    attempts = []

    def flaky(p):
        attempts.append(1)
        if len(attempts) == 2:
            raise OSError("too many open files")
        return "ok"

    source = Reloadable(path, flaky)
    bump(path, "two")
    with pytest.raises(ConfigUnavailable):
        source.get()
    assert source.get() == "ok"  # the same file, read again, and now it works


def test_a_file_replaced_by_one_of_the_same_size_and_time_is_still_noticed(tmp_path):
    path, other = tmp_path / "a.txt", tmp_path / "b.txt"
    path.write_text("aaaa", encoding="utf-8")
    source = Reloadable(path, load)
    before = path.stat()
    other.write_text("bbbb", encoding="utf-8")
    os.utime(other, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(other, path)  # same size, same modification time, different file
    assert source.get() == "bbbb"
