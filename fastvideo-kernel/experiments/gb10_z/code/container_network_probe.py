"""Check that HTTPS downloads needed by the CI image build and the lane setup complete, and print the result as JSON.

Run inside a container (or on the host) with the standard library only. A URL counts as reachable when an HTTP
response arrives, whatever its status; the FlashAttention wheel must also return its first bytes (status 200 or 206).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

PYPI_URLS = ('https://pypi.org/simple/setuptools/', 'https://files.pythonhosted.org/')


def probe(url: str, timeout: float, need_bytes: bool) -> dict:
    request = urllib.request.Request(url, headers={'Range': 'bytes=0-1023', 'User-Agent': 'fastvideo-ci-repro-probe'})
    start = time.monotonic()
    result: dict = {'url': url, 'ok': False, 'status': None, 'bytes': 0, 'error': None}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result['status'] = response.status
            result['bytes'] = len(response.read(1024))
    except urllib.error.HTTPError as exc:
        result['status'] = exc.code
    except Exception as exc:  # noqa: BLE001 - every failure is reported, none is fatal here
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['seconds'] = round(time.monotonic() - start, 2)
    if result['status'] is not None:
        result['ok'] = result['status'] in (200, 206) and result['bytes'] > 0 if need_bytes else True
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wheel-url', required=True)
    p.add_argument('--timeout', type=float, default=60.0)
    a = p.parse_args()
    results = [probe(a.wheel_url, a.timeout, need_bytes=True)]
    results += [probe(url, a.timeout, need_bytes=False) for url in PYPI_URLS]
    print(json.dumps({'all_ok': all(r['ok'] for r in results), 'results': results}, indent=2))
    sys.exit(0 if all(r['ok'] for r in results) else 1)


if __name__ == '__main__':
    main()
