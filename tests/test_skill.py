"""The portable skill file: valid front matter, and every rule it names is a rule the code really produces."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SKILL = ROOT / "skill" / "portcullis" / "SKILL.md"
SOURCES = "".join(p.read_text(encoding="utf-8") for p in (ROOT / "src" / "portcullis").glob("*.py"))


def parts():
    text = SKILL.read_text(encoding="utf-8")
    match = re.match(r"---\n(.*?)\n---\n(.*)", text, re.S)
    assert match, "the file must start with front matter between --- lines"
    fields = dict(line.split(": ", 1) for line in match.group(1).splitlines())
    return fields, match.group(2)


def test_the_front_matter_has_a_valid_name_and_a_description_of_sensible_length():
    fields, _ = parts()
    assert set(fields) == {"name", "description"}
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", fields["name"]) and len(fields["name"]) <= 64
    assert fields["name"] == SKILL.parent.name  # the folder is named for the skill
    assert 50 <= len(fields["description"]) <= 1024
    assert "Use when" in fields["description"]  # says when to load it


def test_every_rule_code_in_the_table_is_one_the_code_can_really_produce():
    _, body = parts()
    rows = re.findall(r"^\| ([^|]+) \|", body, re.M)
    codes = {code for row in rows for code in re.findall(r"`([a-z_]+)`", row)}
    assert len(codes) >= 12, codes
    for code in codes:
        assert f'"{code}"' in SOURCES, f"{code} is named in the skill but no code produces it"


def test_the_command_it_teaches_matches_the_real_cli():
    _, body = parts()
    assert "agentctl check" in body and "--tenant" in body and "--tool" in body and "--args" in body
    from portcullis.cli import _parser

    options = {a for action in _parser()._subparsers._group_actions[0].choices["check"]._actions for a in action.option_strings}
    assert {"--tenant", "--tool", "--args"} <= options


@pytest.mark.parametrize("promise", ["Exit code 0 means allow, 1 means deny, 3 means a human must approve"])
def test_the_exit_codes_it_states_are_the_ones_the_cli_uses(promise):
    from portcullis.cli import EXIT_ASK, EXIT_FAIL, EXIT_OK

    assert promise in parts()[1] and (EXIT_OK, EXIT_FAIL, EXIT_ASK) == (0, 1, 3)
