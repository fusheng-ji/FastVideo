"""Record a lane container's environment and check that what the tests will import comes from the checkout under test.

FastVideo must be the editable install of the checkout. fastvideo-kernel must have been installed from the wheel that
ci_lane_container.sh just built from the checkout (PEP 610 direct_url.json), and every Python file of its two packages
must be byte-identical to the checkout. Prints JSON; exits 1 and lists the problems on stderr if a check fails.
"""
from __future__ import annotations

import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

KERNEL_PACKAGES = {
    'fastvideo_kernel': 'fastvideo-kernel/python/fastvideo_kernel',
    'attn_qat_infer': 'fastvideo-kernel/attn_qat_infer',
}
VERSIONS = ('torch', 'triton', 'flash-attn', 'flashinfer-python', 'fastvideo-kernel', 'nvidia-cusparselt-cu13')


def compare_tree(source: Path, installed: Path) -> dict:
    """Compare every .py file under `source` with the file at the same relative path under `installed`."""
    files = sorted(p.relative_to(source) for p in source.rglob('*.py'))
    missing = [str(f) for f in files if not (installed / f).is_file()]
    differ = [
        str(f) for f in files
        if (installed / f).is_file() and (installed / f).read_bytes() != (source / f).read_bytes()
    ]
    return {'installed': str(installed), 'files': len(files), 'missing': missing, 'differ': differ}


def package_dir(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def check(src: Path, kernel_wheel_dir: str) -> tuple[dict, list[str]]:
    problems = []
    fastvideo = importlib.util.find_spec('fastvideo')
    fastvideo_origin = fastvideo.origin if fastvideo else None
    if not (fastvideo_origin or '').startswith(f'{src}/'):
        problems.append(f'fastvideo resolves to {fastvideo_origin}, not the checkout {src}')
    try:
        direct_url = json.loads(md.distribution('fastvideo-kernel').read_text('direct_url.json') or '{}')
    except md.PackageNotFoundError:
        direct_url = {}
    kernel_url = direct_url.get('url', '')
    if not kernel_url.startswith(f'file://{kernel_wheel_dir}/'):
        problems.append(f'fastvideo-kernel was installed from {kernel_url or "an unknown source"}, '
                        f'not from a wheel built in {kernel_wheel_dir}')
    trees = {}
    for name, relative in KERNEL_PACKAGES.items():
        installed = package_dir(name)
        if installed is None:
            problems.append(f'{name} is not installed as a package')
            continue
        trees[name] = compare_tree(src / relative, installed)
        if not trees[name]['files'] or trees[name]['missing'] or trees[name]['differ']:
            problems.append(f'{name} at {installed} does not match {src / relative}: {trees[name]}')
    return {'fastvideo_origin': fastvideo_origin, 'kernel_install_url': kernel_url, 'kernel_trees': trees}, problems


def main() -> int:
    src = Path(os.environ.get('REPRO_SRC', '/src'))
    kernel_wheel_dir = os.environ.get('REPRO_KERNEL_WHEEL_DIR', '/tmp/fastvideo-kernel-wheel')
    record, problems = check(src, kernel_wheel_dir)
    record['versions'] = {name: version(name) for name in VERSIONS}
    record['uv'] = subprocess.run(['uv', '--version'], capture_output=True, text=True).stdout.strip()
    try:
        import torch
        record.update({'capability': list(torch.cuda.get_device_capability()), 'device': torch.cuda.get_device_name(),
                       'cuda': torch.version.cuda})
        # The package import runs fastvideo_kernel/__init__.py; attn_qat_train is the module the GB10 PRs route through.
        from fastvideo_kernel.triton_kernels import attn_qat_train  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - recorded as a failed check, with the JSON still written
        problems.append(f'import failed: {type(exc).__name__}: {exc}')
    record['checks_passed'] = not problems
    record['problems'] = problems
    print(json.dumps(record, indent=2))
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
