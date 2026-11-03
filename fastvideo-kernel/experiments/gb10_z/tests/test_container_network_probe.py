"""CPU checks for the container network probe, against a local HTTP server."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading

import pytest

CODE = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE))
from container_network_probe import check, probe


class _Handler(BaseHTTPRequestHandler):

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path == '/ok':
            self.send_response(206)
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'ok')
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{httpd.server_address[1]}'
    httpd.shutdown()


def closed_port_url():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return f'http://127.0.0.1:{s.getsockname()[1]}/'


def test_any_http_response_counts_as_reachable(server):
    assert probe(f'{server}/ok', 5)['status'] == 206
    missing = probe(f'{server}/missing', 5)
    assert (missing['ok'], missing['status']) == (True, 404)


def test_connection_failure_is_retried_and_reported():
    tries = check(closed_port_url(), 5, attempts=3)
    assert len(tries) == 3 and not any(t['ok'] for t in tries) and 'URLError' in tries[0]['error']


def test_success_stops_retrying(server):
    assert len(check(f'{server}/ok', 5, attempts=3)) == 1


def test_cli_prints_json_and_exit_code(server):
    ok = subprocess.run([sys.executable, str(CODE / 'container_network_probe.py'), '--url', f'{server}/ok'],
                        capture_output=True, text=True)
    assert ok.returncode == 0 and json.loads(ok.stdout)['all_ok']
    bad = subprocess.run([sys.executable, str(CODE / 'container_network_probe.py'), '--url', f'{server}/ok', '--url',
                          closed_port_url(), '--attempts', '1'], capture_output=True, text=True)
    assert bad.returncode == 1 and not json.loads(bad.stdout)['all_ok']
