"""Regression checks for report scope, protocol validation and evaluator definitions."""
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'run/vbdf'))
import report_results as results
import run_fleet_eval as runner
spec = importlib.util.spec_from_file_location('seeded', REPO / 'run/vbdf/eval_seeded.py')
seeded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seeded)


class ReportWorkflowTests(unittest.TestCase):
    def test_summary_includes_100m_without_any_models(self):
        with patch.object(results, 'MODELS', Path('/nonexistent-report-bundles')):
            summary = results.summarize()
        self.assertEqual(len(summary['cells']), 17)
        self.assertEqual(summary['cells']['state/S5-100m']['n_seeds'], 3)
        self.assertEqual(summary['cells']['vision/V3-100m']['n_seeds'], 3)
        self.assertTrue(all(not c['bundle']['complete'] for c in summary['cells'].values()))
        self.assertAlmostEqual(summary['cells']['state/S5-RS']['mean']['all_envs_dist_err'], 0.108381, places=6)

    def test_invalid_or_incomplete_evidence_is_rejected(self):
        cell = results.CELLS[0]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'Missing report evidence'):
                results.summarize(out, [cell])
            shutil.copytree(results.FLEET / cell['id'], out / cell['id'])
            path = out / cell['id'] / 'H0-base_123.json'
            original = json.loads(path.read_text())
            for key, value in [('steps', 50), ('seed', 999), ('num_envs', 1), ('delay_seed', 0),
                               ('all_envs_omex_std2', float('nan')), ('all_envs_crash_episodes', 100)]:
                with self.subTest(key=key):
                    path.write_text(json.dumps({**original, key: value}))
                    with self.assertRaises(ValueError):
                        results.summarize(out, [cell])
            path.write_text(json.dumps(original))
            self.assertEqual(len(results.summarize(out, [cell])['cells']), 1)

    def test_evaluator_metric_scope(self):
        distance = np.array([[1.5, 2.5], [1.5, 0.5], [3.5, 1.5], [1.5, 1.5]])
        rates = np.zeros((4, 2, 3))
        rates[:, :, 0] = [[0, 100], [2, 100], [0, 10], [2, 10]]
        view = np.array([[0, 0], [np.pi/2, np.pi/2], [0, 0], [0, 0]])
        box = np.array([[1, 1], [1, 1], [0, 1], [1, 1]])
        score = seeded.score_fleet(distance, rates, view, box)
        self.assertEqual(score['all_envs_dist_err'], 0.5)
        self.assertEqual(score['all_envs_omex_std2'], 0.5)
        self.assertEqual(score['all_envs_view_angle_deg'], 22.5)
        self.assertEqual(score['all_envs_in_view_45'], 0.75)
        self.assertEqual(score['all_envs_bbox_lost_frac'], 0.25)

    def test_eval_dry_run_uses_saved_settings(self):
        proc = subprocess.run([sys.executable, str(REPO / 'run/vbdf/run_fleet_eval.py'),
                               '--only', 'S5-RS', 'V3-final', '--dry-run'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('env.act_lpf_alpha=0.3', proc.stdout)
        self.assertIn('env.act_rate_coeff=0.006', proc.stdout)
        self.assertIn('--delay_seed 42', proc.stdout)
        self.assertIn('--num_envs 32', proc.stdout)

    def test_runner_exposes_missing_bundle_and_preserves_published_results(self):
        for args, message in [(['--out', str(results.FLEET)], 'published evidence'),
                              (['--only', 'no-such-model'], 'Unknown report cells')]:
            proc = subprocess.run([sys.executable, str(REPO / 'run/vbdf/run_fleet_eval.py'),
                                   '--dry-run', *args], capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn(message, proc.stderr)
        with patch.object(runner, 'bundle_status', return_value={'complete': False}):
            ready, missing = runner.runnable_cells(results.CELLS)
        self.assertEqual(ready, [])
        self.assertEqual(len(missing), 17)


if __name__ == '__main__':
    unittest.main()
