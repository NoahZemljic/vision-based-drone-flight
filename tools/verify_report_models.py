from pathlib import Path
import csv
import hashlib
import json
import re
import shlex
import statistics
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'run/vbdf'))
from report_results import CELLS, FLEET, MANIFEST, MODELS, SEEDS, load_rows

OUT = ROOT / 'deliverable'
PDF = ROOT / MANIFEST['report']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def top_scalars(path):
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r'^([A-Za-z_]\w*):\s+([^#!].*)$', line)
        if m:
            v = m[2].strip()
            if v == 'true': value = True
            elif v == 'false': value = False
            else:
                try: value = json.loads(v)
                except (ValueError, TypeError): value = v
            out[m[1]] = value
    return out

models = []
raw_checks = []
protocol_checks = []
for cell in CELLS:
    rel = cell['id']
    stage, name = rel.split('/')
    paths = [FLEET / rel / f'{name}_{seed}.json' for seed in SEEDS]
    seed_rows = load_rows(cell)
    rows = [seed_rows[str(seed)] for seed in SEEDS]
    metric_keys = sorted(set.intersection(*(set(r) for r in rows)))
    means = {k: statistics.mean(r[k] for r in rows) for k in metric_keys
             if k.startswith('all_envs_') and all(isinstance(r[k], (float, int)) for r in rows)}
    row = {'report_label': cell['label'], 'relative_model_folder': rel, 'report_reference': cell['report_reference'],
           'evaluation_folder': str((FLEET / rel).relative_to(ROOT)),
           'means_across_three_seeds': means,
           'evaluation_json': [{'path':str(p.relative_to(ROOT)), 'sha256':sha(p)} for p in paths]}
    ckpt = MODELS / rel / 'ckpt/model.zip'
    row['checkpoint_present'] = ckpt.is_file()
    if ckpt.is_file():
        norm = ckpt.with_name('model_vecnormalize.pkl')
        cmd = MODELS / rel / 'command.txt'
        params = MODELS / rel / 'params/env.yaml'
        tokens = shlex.split(cmd.read_text())
        overrides = {t.split('=',1)[0]:t.split('=',1)[1] for t in tokens if t.startswith(('env.', 'agent.'))}
        env = top_scalars(params)
        with zipfile.ZipFile(ckpt) as z:
            metadata = json.loads(z.read('data'))
            bad_member = z.testzip()
        obs = metadata['observation_space']['_shape'][0]
        k = env['cmd_hist_len']
        expected_obs = 15 + 4*k if stage == 'hover' else 25 + 4*k + 3*env['actor_vel_hist'] if stage == 'state' else 19 + 4*env['bbox_hist_len'] + 4*env.get('bbox_vel_hist',0) + 2 + 4*k
        row.update({'checkpoint_sha256':sha(ckpt), 'normalization_present':norm.is_file(),
                    'normalization_sha256':sha(norm) if norm.is_file() else None,
                    'observation_dimension':obs, 'expected_dimension_from_saved_cfg':expected_obs,
                    'dimensions_match':obs==expected_obs, 'checkpoint_crc_pass':bad_member is None,
                    'actual_training_transitions':metadata['num_timesteps'],
                    'training_envs':metadata['n_envs'], 'training_seed':metadata['seed'],
                    'training_command':cmd.read_text(), 'overrides':overrides,
                    'evaluation_checkpoint_paths_match':all(r['checkpoint'].endswith(f'deliverable/models/{rel}/ckpt/model.zip') for r in rows),
                    'saved_env_settings':{key:env.get(key) for key in ['cmd_hist_len','actor_vel_hist','bbox_hist_len','use_body_frame','use_applied_cmd_obs','act_lpf_alpha','act_rate_coeff','delay','ctrl_dt'] if key in env}})
    else:
        row['expected_checkpoint_path'] = str(ckpt.relative_to(ROOT))
        row['recorded_checkpoint_paths'] = sorted({r['checkpoint'] for r in rows})
    models.append(row)
    for seed, jp, d in zip(SEEDS, paths, rows):
        per_env = d['all_envs_crash_per_env']
        protocol_checks.append({'cell':rel,'seed':seed,
            'pass':d['crash_episodes']==per_env[0]})
        cp = jp.with_suffix('.csv')
        if not cp.is_file():
            continue
        with cp.open() as f:
            rr = list(csv.DictReader(f))
        cols = {k:np.array([float(r[k]) for r in rr]) for k in rr[0]}
        n=len(rr); h=n//2
        checks={}
        for axis in 'xyz':
            key=f'ome{axis}_std2'
            checks[key]=bool(np.isclose(cols[f'ome{axis}'][h:].std(), d[key],rtol=1e-9,atol=1e-10))
        pos=np.stack([cols[k] for k in ('px','py','pz')],axis=1)
        if stage=='hover':
            dist=np.linalg.norm(pos-np.array([0,0,1]),axis=1)
            checks['env0_dist']=bool(np.isclose(dist[h:].mean(),d['dist_mean_2nd_half'],rtol=1e-9,atol=1e-10))
        else:
            actor=np.stack([cols[k] for k in ('other_pos_x','other_pos_y','other_pos_z')],axis=1)
            dist=np.linalg.norm(pos-actor,axis=1)
            checks['env0_dist']=bool(np.isclose(np.abs(dist[h:]-1.5).mean(),d['dist_err'],rtol=1e-9,atol=1e-10))
        checks['crash_sum']=sum(per_env)==d['all_envs_crash_episodes']
        checks['crashed_environments']=sum(v>0 for v in per_env)==d['all_envs_crash_n_envs']
        checks['event_count']=len(d['crash_events'])==d['all_envs_crash_episodes']
        checks['env0_crash_count']=int(np.count_nonzero(cols['done']))==d['crash_episodes']==per_env[0]
        checks['length']=n==d['steps']
        checks['fleet_size']=len(per_env)==32
        raw_checks.append({'cell':rel,'seed':seed,'csv':str(cp.relative_to(ROOT)),
                           'csv_sha256':sha(cp),'checks':checks,'pass':all(checks.values())})

by_cell={m['relative_model_folder']:m for m in models}
published_checks=[]
def check(ref, cell, key, printed):
    value=by_cell[cell]['means_across_three_seeds'][key]
    digits=len(printed.split('.')[1]) if '.' in printed else 0
    actual=f'{value:.{digits}f}'
    published_checks.append({'reference':ref,'cell':cell,'metric':key,'reported':printed,
                             'recomputed':value,'rounded':actual,'pass':actual==printed})
table2_keys=['all_envs_dist_err','all_envs_view_angle_deg','all_envs_omex_std2',
             'all_envs_omey_std2','all_envs_omez_std2','all_envs_crash_episodes','all_envs_crash_n_envs']
for cell,values in [
    ('state/S5-RS',['0.108','14.94','0.477','0.414','0.329','6.0','6.0']),
    ('state/S5-100m',['0.080','14.56','0.479','0.438','0.378','7.7','6.7']),
    ('vision/V3-final',['0.205','15.25','0.536','0.517','0.396','8.0','6.7']),
    ('vision/V3-100m',['0.212','14.73','0.662','0.670','0.466','14.3','10.7'])]:
    for key,value in zip(table2_keys,values): check('Table II (and Figure 1)',cell,key,value)
for cell,values in [('hover/H0-base',['0.052','6.403','0.0']),
                    ('hover/H1-delay',['0.188','0.010','0.0']),
                    ('hover/H2-K',['0.956','1.704','36.3'])]:
    for key,value in zip(['all_envs_dist_mean_2nd_half','all_envs_omey_std2','all_envs_crash_episodes'],values):
        check('Table III',cell,key,value)
for cell,dist,crash in [('state/S1-body','0.090','11.3'),('state/S-K-abl','0.103','27.0'),
                        ('vision/V0-base','0.199','17.7'),('vision/V-K-ablation','0.243','27.0')]:
    check('Section III-C command-history ablation',cell,'all_envs_dist_err',dist)
    check('Section III-C command-history ablation',cell,'all_envs_crash_episodes',crash)
for cell,roll in [('state/S3-PV','3.098'),('state/S4-LPF','1.327'),('state/S5-RS','0.477'),
                   ('vision/V2-LPF','1.333'),('vision/V3-final','0.536')]:
    check('Section III-C filtering/rate penalty',cell,'all_envs_omex_std2',roll)
for cell,dist in [('state/S4-LPF','0.089'),('state/S5-RS','0.108'),('vision/V2-LPF','0.210'),('vision/V3-final','0.205')]:
    check('Section III-C filtering/rate penalty',cell,'all_envs_dist_err',dist)

# Remaining numerical claims in Section III-B: seed ranges, percentage changes,
# and missing synthetic-box fractions.
def check_derived(ref, cell, metric, value, printed):
    digits = len(printed.split('.')[1]) if '.' in printed else 0
    rounded = f'{value:.{digits}f}'
    published_checks.append({'reference': ref, 'cell': cell, 'metric': metric,
                             'reported': printed, 'recomputed': value,
                             'rounded': rounded, 'pass': rounded == printed})
for cell, low, high in [('state/S5-RS', '0.104', '0.112'),
                        ('state/S5-100m', '0.077', '0.082')]:
    name = cell.split('/')[1]
    vals = [json.loads((FLEET/cell/f'{name}_{seed}.json').read_text())['all_envs_dist_err']
            for seed in SEEDS]
    check_derived('Section III-B distance seed range', cell, 'minimum', min(vals), low)
    check_derived('Section III-B distance seed range', cell, 'maximum', max(vals), high)
def metric(cell, key):
    return by_cell[cell]['means_across_three_seeds'][key]
check_derived('Section III-B distance reduction', 'state/S5-100m vs S5-RS', 'percent',
              100*(1-metric('state/S5-100m','all_envs_dist_err')/metric('state/S5-RS','all_envs_dist_err')), '26')
for key, printed in zip(['all_envs_omex_std2','all_envs_omey_std2','all_envs_omez_std2'], ['24','30','18']):
    check_derived('Section III-B body-rate increase', 'vision/V3-100m vs V3-final', key,
                  100*(metric('vision/V3-100m',key)/metric('vision/V3-final',key)-1), printed)
for cell, printed in [('vision/V3-final','0.039'),('vision/V3-100m','0.158')]:
    check_derived('Section III-B missing synthetic boxes', cell, 'percent',
                  100*metric(cell,'all_envs_bbox_lost_frac'), printed)

output={'report':str(PDF.relative_to(ROOT)),'report_sha256':sha(PDF),
        'scope':'Read-only artifact/code consistency audit. No simulation or policy inference executed.',
        'models':models,'published_number_checks':published_checks,
        'protocol_checks':protocol_checks,'raw_csv_and_crash_checks':raw_checks,
        'limitations':[
            'Evaluation JSON records checkpoint paths but no checkpoint content hash; path/config/metadata agreement is not cryptographic proof of historical weights.',
            'All-32-environment distance and rate aggregates come from the recorded per-seed JSON. Optional CSV checks cover environment 0 only.',
            '100M checkpoint weights, normalization and saved training config bundles are absent; only evaluation artifacts and worker recipe are present.',
            'No Isaac runtime is available for a fresh rollout.'
        ]}
(OUT/'verification.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
print('Models referenced:',len(models),'; checkpoints present:',sum(m['checkpoint_present'] for m in models))
print('Published rounded number matches:',sum(c['pass'] for c in published_checks),'/',len(published_checks))
print('JSON protocol and crash consistency:',sum(c['pass'] for c in protocol_checks),'/',len(protocol_checks))
if raw_checks:
    print('Raw CSV and crash consistency:',sum(c['pass'] for c in raw_checks),'/',len(raw_checks))
print('Checkpoint dimensions match:',sum(m.get('dimensions_match',False) for m in models),'/',sum(m['checkpoint_present'] for m in models))
for c in published_checks+protocol_checks+raw_checks:
    if not c['pass']:print('FAIL',c)

# Integrity is checked against the frozen cleanup manifest, not filenames alone.
for cell, model in zip(MANIFEST['cells'], models):
    if model['checkpoint_present']:
        for key, actual in [('checkpoint_sha256_at_cleanup', 'checkpoint_sha256'),
                            ('normalization_sha256_at_cleanup', 'normalization_sha256')]:
            expected = cell.get(key)
            if expected and model[actual] != expected:
                raise SystemExit(f"Hash mismatch: {cell['id']} {actual}")
        if not all(model.get(key) for key in ['normalization_present', 'dimensions_match',
                                             'checkpoint_crc_pass', 'evaluation_checkpoint_paths_match']):
            raise SystemExit(f"Model verification failed: {cell['id']}")
if sha(PDF) != MANIFEST['report_sha256']:
    raise SystemExit('The report PDF changed relative to the manifest.')
if not all(c['pass'] for c in published_checks + protocol_checks + raw_checks):
    raise SystemExit('Report verification failed.')
if '--require-complete' in sys.argv and not all(m['checkpoint_present'] for m in models):
    raise SystemExit('Incomplete reproduction: recover the S5-100m and V3-100m model bundles.')
