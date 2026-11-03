"""CPU checks for the container network probe, against a local HTTP server."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from container_network_probe import probe


class _Handler(BaseHTTPRequestHandler):

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path == '/wheel':
            body = b'PK' * 512
            self.send_response(206)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{httpd.server_address[1]}'
    httpd.shutdown()


def test_wheel_must_return_bytes(server):
    assert probe(f'{server}/wheel', 5, need_bytes=True) | {'seconds': 0} == {
        'url': f'{server}/wheel', 'ok': True, 'status': 206, 'bytes': 1024, 'error': None, 'seconds': 0}
    missing = probe(f'{server}/missing', 5, need_bytes=True)
    assert (missing['ok'], missing['status']) == (False, 404)


def test_any_http_response_counts_as_reachable(server):
    assert probe(f'{server}/missing', 5, need_bytes=False)['ok']


def test_connection_failure_is_reported():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    result = probe(f'http://127.0.0.1:{port}/', 5, need_bytes=False)
    assert not result['ok'] and result['status'] is None and 'URLError' in result['error']
