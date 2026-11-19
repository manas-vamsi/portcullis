"""The dashboard the server serves: strict headers, a fixed list of files, and code that can only show text."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_server import KEYS, make_policy

from portcullis.server import create_app

ASSETS = Path(__file__).parent.parent / "src" / "portcullis" / "web"


@pytest.fixture(scope="module")
def client():
    # No `with`: the lifespan (and so the database) is not started. These routes serve static files only.
    return TestClient(create_app(make_policy(), KEYS, "postgresql://u:p@127.0.0.1:1/portcullis"))


def test_the_bare_path_redirects_to_the_page(client):
    response = client.get("/dashboard", follow_redirects=False)
    assert (response.status_code, response.headers["location"]) in ((307, "/dashboard/"), (308, "/dashboard/"))


@pytest.mark.parametrize(
    "path, content_type",
    [("/dashboard/", "text/html"), ("/dashboard/app.js", "text/javascript"), ("/dashboard/lib.js", "text/javascript"),
     ("/dashboard/style.css", "text/css")],
)
def test_each_file_is_served_with_its_type_and_the_security_headers(client, path, content_type):
    response = client.get(path)
    assert response.status_code == 200 and response.headers["content-type"].startswith(content_type)
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store"
    assert "camera=()" in response.headers["permissions-policy"]


def test_the_content_security_policy_allows_only_this_server_and_nothing_inline(client):
    policy = client.get("/dashboard/").headers["content-security-policy"]
    directives = dict(part.strip().split(" ", 1) for part in policy.split(";") if part.strip())
    assert directives["default-src"] == "'none'"
    assert directives["script-src"] == directives["style-src"] == directives["connect-src"] == "'self'"
    assert directives["frame-ancestors"] == "'none'" and directives["base-uri"] == "'none'"
    assert "unsafe" not in policy and "*" not in policy and "http" not in policy


@pytest.mark.parametrize(
    "path",
    ["/dashboard/nope", "/dashboard/__init__.py", "/dashboard/..%2fserver.py", "/dashboard/%2e%2e/server.py",
     "/dashboard/APP.JS", "/dashboard/app.js%00", "/dashboard/a/b"],
)
def test_nothing_but_the_four_files_is_ever_served(client, path):
    assert client.get(path).status_code == 404


def test_the_page_itself_needs_no_key_but_the_api_behind_it_still_does(client):
    assert client.get("/dashboard/").status_code == 200
    assert client.get("/v1/approvals").status_code == 401


def test_the_page_has_no_inline_code_no_external_references_and_no_inline_style():
    html = (ASSETS / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"https?://|//[a-z]", html)
    assert not re.search(r"\son[a-z]+\s*=", html)  # no onclick= and the like
    assert not re.search(r"\sstyle\s*=", html)
    scripts = re.findall(r"<script\b[^>]*>", html)
    assert scripts and all('src="' in tag for tag in scripts)
    assert not re.search(r"<script\b[^>]*>\s*[^<\s]", html)  # no script text between the tags


FORBIDDEN_IN_JS = {
    "innerHTML": r"innerHTML|outerHTML|insertAdjacentHTML|createContextualFragment",
    "document.write": r"document\.write",
    "dynamic code": r"\beval\s*\(|new\s+Function|setTimeout\s*\(\s*[\"'`]|setInterval\s*\(\s*[\"'`]",
    "event attributes": r"setAttribute\s*\(\s*[\"']on",
    "navigation by text": r"\.href\s*=|location\s*=|location\.(href|assign|replace)|javascript:",
    "persisting the key": r"localStorage|sessionStorage|document\.cookie|indexedDB",
    "other origins": r"https?://",
}


@pytest.mark.parametrize("name", ["app.js", "lib.js"])
@pytest.mark.parametrize("label, pattern", FORBIDDEN_IN_JS.items(), ids=list(FORBIDDEN_IN_JS))
def test_the_scripts_cannot_turn_log_text_into_markup_or_code_or_keep_the_key(name, label, pattern):
    source = (ASSETS / name).read_text(encoding="utf-8")
    assert not re.search(pattern, source), f"{name} uses {label}"


def test_every_request_the_page_makes_goes_to_this_servers_api_with_the_key_in_a_header():
    source = (ASSETS / "app.js").read_text(encoding="utf-8")
    assert len(re.findall(r"\bfetch\(", source)) == 1  # one place that talks to the network
    paths = re.findall(r'api\(\s*"(?:GET|POST)"\s*,\s*[`"](/[^`"]*)', source)
    assert paths and all(p.startswith("/v1/") for p in paths)
    assert "Authorization: `Bearer ${apiKey}`" in source and "?key=" not in source


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
def test_the_logic_the_page_uses_passes_its_node_tests():
    done = subprocess.run(
        ["node", "--test", "tests/js/lib.test.mjs"], cwd=ASSETS.parents[2], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stdout[-2000:]
