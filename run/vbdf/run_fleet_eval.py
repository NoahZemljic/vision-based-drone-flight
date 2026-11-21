#!/usr/bin/env python3
"""Evaluate report policies with their saved overrides; write new results under outputs/."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from report_results import CELLS, FLEET, MANIFEST, MODELS, REPO, bundle_status, load_rows, summarize, write_summary


def isaac_python(required=True):
    path = os.environ.get('ISAAC_PYTHON', '/workspace/isaaclab/_isaac_sim/python.sh')
    if required and not Path(path).is_file():
        raise ValueError('Isaac runtime unavailable. Set ISAAC_PYTHON or use the matching course container.')
    return path


def parse_command(path):
    tokens = shlex.split(path.read_text())
    if '--task' not in tokens:
        raise ValueError(f'No --task in {path}')
    task = tokens[tokens.index('--task') + 1]
    return task, [t for t in tokens if t.startswith('env.')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runnable_cells(selected):
    ready, missing = [], []
    for spec in selected:
        status = bundle_status(spec)
        if not status['complete']:
            missing.append(spec['id'])
            continue
        folder = MODELS / spec['id']
        task, overrides = parse_command(folder / 'command.txt')
        if task != spec['task']:
            raise ValueError(f"{spec['id']}: saved task differs from report manifest")
        ready.append({**spec, 'overrides': overrides, 'checkpoint': folder / 'ckpt/model.zip',
                      'normalization': folder / 'ckpt/model_vecnormalize.pkl'})
    return ready, missing


def seed_done(out, cell, seed, steps):
    name = cell['id'].split('/')[1]
    path = out / cell['id'] / f'{name}_{seed}.json'
    if not path.is_file():
        return False
    try:
        row = load_rows(cell, out, [seed], strict=False)[str(seed)]
        return (row['steps'] == steps and row.get('checkpoint_sha256') == sha(cell['checkpoint'])
                and row.get('normalization_sha256') == sha(cell['normalization'])
                and sorted(row.get('environment_overrides', [])) == sorted(cell['overrides']))
    except (ValueError, KeyError, TypeError):
        return False


def run_cell(cell, out, seeds, write_csv, steps_override, dry_run):
    steps = steps_override or cell['steps']
    name = cell['id'].split('/')[1]
    pending = [s for s in seeds if not seed_done(out, cell, s, steps)
               or (write_csv and not (out / cell['id'] / f'{name}_{s}.csv').is_file())]
    if not pending:
        print(f"OK {cell['id']}: matching results already present")
        return 0
    dest = out / cell['id']
    cmd = [isaac_python(required=not dry_run), str(REPO / 'run/vbdf/eval_seeded.py'),
           '--task', cell['task'], '--checkpoint', str(cell['checkpoint']),
           '--steps', str(steps), '--num_envs', str(MANIFEST['num_envs']),
           '--seeds', ','.join(map(str, pending)), '--delay_seed', str(MANIFEST['delay_seed']),
           '--device', 'cuda:0', '--viz', 'none', '--out_dir', str(dest), '--out_name', name]
    if write_csv:
        cmd.append('--write_csv')
    cmd += cell['overrides']
    print(shlex.join(cmd), flush=True)
    if dry_run:
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / f'{name}.log').open('a') as log:
        log.write(f'\n{datetime.now(timezone.utc).isoformat()}\n{shlex.join(cmd)}\n')
        log.flush()
        proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        print(f"FAIL {cell['id']}: see {dest / (name + '.log')}")
        return proc.returncode
    return 0 if all(seed_done(out, cell, s, steps) for s in pending) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=REPO / 'outputs/eval_fleet')
    parser.add_argument('--only', nargs='+', help='Report cell names, e.g. H0-base S5-RS V3-final')
    parser.add_argument('--seeds', default='123,456,789')
    parser.add_argument('--write-csv', action='store_true', help='Also save environment-0 trajectories')
    parser.add_argument('--steps', type=int, help='Override horizon for a smoke check; not a report evaluation')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--available-only', action='store_true', help='Explicitly allow skipping missing model bundles')
    args = parser.parse_args()
    out = args.out.resolve()
    if out == FLEET.resolve() or FLEET.resolve() in out.parents:
        parser.error('Use a separate --out directory; deliverable/eval_fleet contains the published evidence.')
    seeds = [int(s) for s in args.seeds.split(',')]
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error('Provide distinct evaluation seeds.')
    if args.steps is not None and args.steps <= 0:
        parser.error('--steps must be positive')
    selected = CELLS
    if args.only:
        wanted = set(args.only)
        selected = [c for c in CELLS if c['id'] in wanted or c['id'].split('/')[1] in wanted]
        unknown = wanted - {s for c in selected for s in (c['id'], c['id'].split('/')[1])}
        if unknown:
            parser.error(f'Unknown report cells: {sorted(unknown)}')
    ready, missing = runnable_cells(selected)
    if missing:
        print('MISSING original model bundles: ' + ', '.join(missing))
        if not args.available_only:
            parser.error('Full report rerun is incomplete. Restore the bundles or explicitly use --available-only.')
    if not ready:
        parser.error('No complete model bundles selected')
    failed = [c['id'] for c in ready if run_cell(c, out, seeds, args.write_csv, args.steps, args.dry_run)]
    if not args.dry_run:
        summary = summarize(out, ready, seeds, strict=False)
        summary['missing_model_bundles'] = missing
        summary['failed_cells'] = failed
        write_summary(out, summary)
    if failed:
        sys.exit('Failed cells: ' + ', '.join(failed))
    print(f"{'Planned' if args.dry_run else 'Completed'} {len(ready)} cells; missing bundles: {len(missing)}")


if __name__ == '__main__':
    try:
        main()
    except ValueError as exc:
        sys.exit(str(exc))
