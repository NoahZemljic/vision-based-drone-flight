"""Exercise one published checkpoint from each task family; outputs stay temporary."""
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path


def main():
    repo = Path('/worktree')
    with tempfile.TemporaryDirectory(prefix='vbdf-report-smoke-') as tmp:
        cmd = [os.environ.get('ISAAC_PYTHON', '/workspace/isaaclab/_isaac_sim/python.sh'),
               str(repo / 'run/vbdf/run_fleet_eval.py'), '--only', 'H0-base', 'S5-RS', 'V3-final',
               '--steps', '50', '--seeds', '123', '--out', tmp]
        subprocess.run(cmd, cwd=repo, check=True)
        result = json.loads((Path(tmp) / 'summary.json').read_text())
        if len(result['cells']) != 3:
            raise RuntimeError('Expected all three task families')
        for cell in result['cells'].values():
            if not all(math.isfinite(v) for v in cell['mean'].values()):
                raise RuntimeError('Non-finite rollout metrics')
        print('PASS: hover, state and bounding-box policies completed 50 steps each.')
        print('This is a runtime smoke check, not a replication of the published evaluation.')


if __name__ == '__main__':
    main()
