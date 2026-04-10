"""
Label Evolution Visualizer — stdlib-only HTTP server.

Usage:
    python visualizer.py [--port 7860] [--run-id <id>]

Opens http://localhost:7860 in the default browser.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

RUNS_DIR = Path(__file__).parent / "runs"
STATIC_DIR = Path(__file__).parent / "static"


def _load_run(run_id: str) -> dict | None:
    path = RUNS_DIR / run_id / "state.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _list_runs() -> list[dict]:
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        state_file = run_dir / "state.json"
        if not state_file.exists():
            continue
        try:
            d = json.loads(state_file.read_text(encoding="utf-8"))
            if d["stats"].get("steps_completed", 0) == 0:
                continue
            runs.append({
                "run_id": d["run_id"],
                "created_at": d["created_at"],
                "steps": d["stats"]["steps_completed"],
                "models": d["config"]["models"],
                "labeler": d["config"]["labeler"],
                "max_labels": d["config"]["max_labels"],
                "seed_file": d["config"].get("seed_file"),
                "n_labels": len(d["taxonomy"]["active"]),
                "n_graveyard": len(d["taxonomy"].get("graveyard", {})),
                "stats": d["stats"],
            })
        except Exception as e:
            runs.append({"run_id": run_dir.name, "error": str(e)})
    return runs


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress access logs
        pass

    def _send_json(self, data: dict | list, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        if not path.exists():
            self.send_response(404)
            self.end_headers()
            return
        ext = path.suffix.lower()
        mime = {
            ".html": "text/html",
            ".js": "application/javascript",
            ".css": "text/css",
            ".json": "application/json",
        }.get(ext, "application/octet-stream")
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)

        # API routes
        if path == "/api/runs":
            self._send_json(_list_runs())

        elif path.startswith("/api/runs/"):
            run_id = path[len("/api/runs/"):]
            data = _load_run(run_id)
            if data is None:
                self._send_json({"error": "not found"}, 404)
            else:
                self._send_json(data)

        # Static files
        elif path == "/" or path == "/index.html":
            self._send_file(STATIC_DIR / "index.html")

        else:
            # Try static dir
            rel = path.lstrip("/")
            f = STATIC_DIR / rel
            if f.exists() and f.is_file():
                self._send_file(f)
            else:
                self.send_response(404)
                self.end_headers()


def open_browser(port: int, run_id: str | None, delay: float = 0.8):
    time.sleep(delay)
    url = f"http://localhost:{port}"
    if run_id:
        url += f"?run={run_id}"
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="Label Evolution Visualizer")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--run-id", default=None, help="Open directly to a specific run")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    server = HTTPServer(("localhost", args.port), Handler)
    print(f"Visualizer running at http://localhost:{args.port}")
    if args.run_id:
        print(f"Opening run: {args.run_id}")

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port, args.run_id), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
