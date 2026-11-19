"""Serves the dashboard: one HTML page and three small static files, from the server itself.

There is no build step and no third-party code. What keeps it safe:
- Only four named files are ever served, read into memory once at startup. No request ever touches the
  file system, so there is no path to traverse.
- A strict Content-Security-Policy: only this server may be talked to, no inline script or style, nothing
  framed, no base tag, no form posts. A page that cannot run inline code cannot be made to by log text.
- The script only ever puts text on the page with textContent. Everything it shows from the audit log is
  chosen by a model, so it is treated as untrusted text; tests ban the DOM calls that would make it markup.
- The page holds the API key in memory only. The files themselves need no key (they contain no data); the
  API behind them still does.
"""

from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import RedirectResponse

ASSET_DIR = Path(__file__).parent / "web"
TYPES = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "lib.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
}
HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def mount(app: FastAPI) -> None:
    assets = {name: (ASSET_DIR / name).read_bytes() for name in TYPES}

    def serve(name: str) -> Response:
        return Response(assets[name], media_type=TYPES[name], headers=HEADERS)

    @app.get("/dashboard", include_in_schema=False)
    def bare() -> RedirectResponse:
        return RedirectResponse("/dashboard/")

    @app.get("/dashboard/", include_in_schema=False)
    def index() -> Response:
        return serve("index.html")

    @app.get("/dashboard/{name}", include_in_schema=False)
    def asset(name: str) -> Response:
        if name not in assets:
            return Response(status_code=404)
        return serve(name)
