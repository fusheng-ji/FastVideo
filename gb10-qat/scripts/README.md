# Reproduce the A/B figures

Run on a GB10 (SM121) GPU. `PYTHONPATH` selects the kernel source being measured:

```bash
PYTHONPATH=<main>/fastvideo-kernel/python   python measure_b.py b_before.json
PYTHONPATH=<branch>/fastvideo-kernel/python python measure_b.py b_after.json
PYTHONPATH=<main>/fastvideo-kernel/python   python measure_a.py a_before.json
PYTHONPATH=<branch>/fastvideo-kernel/python python measure_a.py a_after.json
python plot.py   # needs matplotlib; writes fig-a-split-pv.png and fig-b-nvfp4-mask.png
```
