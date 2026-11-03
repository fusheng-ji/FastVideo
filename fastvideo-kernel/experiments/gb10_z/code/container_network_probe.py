"""Check that HTTPS endpoints answer from inside a container, retrying each a few times, and print the result as JSON.

Standard library only. A URL counts as reachable when any HTTP response arrives, whatever its status. Each URL is tried up
to --attempts times and passes if one attempt does; every attempt is recorded.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def probe(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={'Range': 'bytes=0-1023', 'User-Agent': 'fastvideo-ci-repro-probe'})
    start = time.monotonic()
    result: dict = {'ok': False, 'status': None, 'error': None}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result['status'] = response.status
    except urllib.error.HTTPError as exc:
        result['status'] = exc.code
    except Exception as exc:  # noqa: BLE001 - every failure is reported, none is fatal here
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['ok'] = result['status'] is not None
    result['seconds'] = round(time.monotonic() - start, 2)
    return result


def check(url: str, timeout: float, attempts: int) -> list[dict]:
    tries = []
    for _ in range(attempts):
        tries.append(probe(url, timeout))
        if tries[-1]['ok']:
            break
    return tries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', action='append', required=True)
    p.add_argument('--timeout', type=float, default=60.0)
    p.add_argument('--attempts', type=int, default=3)
    a = p.parse_args()
    results = {url: check(url, a.timeout, a.attempts) for url in a.url}
    all_ok = all(tries[-1]['ok'] for tries in results.values())
    print(json.dumps({'all_ok': all_ok, 'results': results}, indent=2))
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
