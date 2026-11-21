#!/usr/bin/env python3
"""Rebuild all 17 report entries from saved JSON, even when weights are missing."""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'run/vbdf'))
from report_results import FLEET, summarize, write_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=FLEET, help='directory containing per-seed report JSON')
    args = parser.parse_args()
    summary = summarize(args.out)
    write_summary(args.out, summary)
    print(f"Summarized {len(summary['cells'])} report entries from saved results.")
    missing = [name for name, c in summary['cells'].items() if not c['bundle']['complete']]
    print('Missing original bundles:', ', '.join(missing) or 'none')


if __name__ == '__main__':
    main()
