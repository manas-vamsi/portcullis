"""The decision records stay in shape and keep pointing at things that exist."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
ADR_DIR = ROOT / "docs" / "adr"
RECORDS = sorted(p for p in ADR_DIR.glob("[0-9][0-9][0-9][0-9]-*.md"))


def test_the_records_are_numbered_without_gaps_from_one():
    assert [int(p.name[:4]) for p in RECORDS] == list(range(1, len(RECORDS) + 1)) and len(RECORDS) >= 10


@pytest.mark.parametrize("record", RECORDS, ids=lambda p: p.name)
def test_each_record_has_the_same_sections_and_a_matching_number_in_its_title(record):
    text = record.read_text(encoding="utf-8")
    assert text.startswith(f"# ADR-{record.name[:4]}: ")
    assert "Status: accepted." in text
    sections = re.findall(r"^## (.+)$", text, re.M)
    assert sections == ["Context", "Decision", "Consequences", "Where"]
    assert all(len(part.strip()) > 30 for part in re.split(r"^## .+$", text, flags=re.M)[1:])  # none is a stub


@pytest.mark.parametrize("record", RECORDS, ids=lambda p: p.name)
def test_every_path_a_record_names_under_where_exists(record):
    where = record.read_text(encoding="utf-8").split("## Where", 1)[1]
    paths = [t for t in re.findall(r"`([^`]+)`", where) if "/" in t]
    assert paths, "a record must say where its decision is built"
    for path in paths:
        assert (ROOT / path).exists(), f"{record.name} names {path}, which does not exist"


def test_the_index_lists_every_record_with_a_working_link():
    index = (ADR_DIR / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)]+\.md)\)", index)
    assert links == [p.name for p in RECORDS]
    assert all((ADR_DIR / link).exists() for link in links)


def test_no_record_claims_exactly_once_or_tamper_proof():
    banned = re.compile(r"exactly[- ]once|tamper[- ]proof", re.I)
    for record in RECORDS:
        for line in record.read_text(encoding="utf-8").splitlines():
            if banned.search(line):  # allowed only when it is being denied
                assert re.search(r"\bnot\b|\bis not\b|never|cannot|not achievable|worse", line, re.I), (record.name, line)
