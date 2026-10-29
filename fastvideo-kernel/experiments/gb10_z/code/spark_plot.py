"""Plot the published scalar summaries without loading tensors or initializing CUDA."""
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.results / 'summary.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, kind in zip(axes, ['kernel', 'model']):
        for j, policy in enumerate(['dense', 'recompute', 'Z']):
            rows = [r for r in summary[kind] if r['policy'] == policy]
            values = [r['saved_statistics_bytes'] / 2**30 if kind == 'kernel'
                      else r['peak_allocated_GiB'] for r in rows]
            axis.bar([i + (j - 1) * .24 for i in range(3)], values, .24, label=policy)
        axis.set_xticks(range(3), ['8192', '16384', '31200'])
        axis.set_xlabel('Tokens')
        axis.set_ylabel('GiB')
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    axes[0].set_title('Saved statistics per attention call (H=12)')
    axes[1].set_title('Complete Wan step: peak allocated')
    axes[1].set_ylim(0, 34)
    axes[0].legend(frameon=False)
    fig.suptitle('Smaller statistics do not always lower the model peak', fontsize=12)
    fig.tight_layout()
    fig.savefig(args.results / 'memory-summary.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
