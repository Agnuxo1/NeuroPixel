"""Full-cohort independent recount and figures; refuses partial cohorts, no NN."""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import platform
import statistics
from scipy.stats import t

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLAN_SHA = 'a4093bb8d559404b9f68587b753f8cea180a1e75d33363a04c1f2c7ebcb22dc0'
SEEDS = (200, 201, 202)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError('Three finite paired initialization values required')
    mean = statistics.mean(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / 2)
    half = float(t.ppf(.975, 2)) * sd / math.sqrt(3)
    return dict(values=list(values), mean=mean, sd=sd,
                t95=[mean - half, mean + half], training_initializations=3)


def direct_contrasts(cells):
    """Nested differences independent of auditor's orthogonal-weight algorithm."""
    if len(cells) != 8:
        raise ValueError('All eight factorial cells required')
    effects = {k: [] for k in ('budget_primary', 'query', 'school', 'query_school',
                               'budget_query', 'budget_school', 'triple_interaction')}
    for i in range(3):
        v = lambda b, p, s: cells[(b, p, s)][i]
        q = lambda b, s: v(b, 'paired', s) - v(b, 'repeated', s)
        school = lambda b, p: v(b, p, .3) - v(b, p, 0.)
        interaction = lambda b: q(b, .3) - q(b, 0.)
        effects['budget_primary'].append(sum(v(16384, p, s) - v(8192, p, s)
                                             for p in ('paired', 'repeated') for s in (0., .3)) / 4)
        effects['query'].append(sum(q(b, s) for b in (8192, 16384) for s in (0., .3)) / 4)
        effects['school'].append(sum(school(b, p) for b in (8192, 16384)
                                     for p in ('paired', 'repeated')) / 4)
        effects['query_school'].append((interaction(8192) + interaction(16384)) / 2)
        effects['budget_query'].append(sum(q(16384, s) - q(8192, s) for s in (0., .3)) / 2)
        effects['budget_school'].append(sum(school(16384, p) - school(8192, p)
                                            for p in ('paired', 'repeated')) / 2)
        effects['triple_interaction'].append(interaction(16384) - interaction(8192))
    return {k: stats(values) for k, values in effects.items()}


def require_full_audit(audit):
    if (audit['issues'] or audit['missing'] or audit['cases_recounted'] != 12
            or audit['status'] != 'verified_complete_continuation_arrays_checkpoint_replay_pending'):
        raise ValueError('Complete twelve-case array audit required; no partial-cohort statistics')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path,
                        default=ROOT / 'results/research/OPT03_QTRAIN_U16_recovery/37771588722/cohort')
    parser.add_argument('--review', type=pathlib.Path,
                        default=ROOT / 'results/research/OPT03_QTRAIN_U16_review/37771588722')
    parser.add_argument('--audit', type=pathlib.Path)
    args = parser.parse_args()
    plan_path = ROOT / 'docs/research/OPT03_QTRAIN_U16_plan.json'
    if sha(plan_path) != PLAN_SHA:
        raise ValueError('Frozen U16 plan differs')
    plan = read(plan_path)
    audit_path = args.audit or args.review / 'audit_full.json'
    audit = read(audit_path)
    require_full_audit(audit)  # Reject before reading partial outcome values or creating output.
    final_path = args.review / 'final_receipts_verification.json'
    final = read(final_path)
    if (final['status'] != 'verified_final_cohort_receipts_and_workflow_success'
            or final['issues'] or final['cases'] != 12
            or {p['label'] for p in final['proofs']} != {'science', 'verification'}):
        raise ValueError('Both terminal workflow successes and final cohort receipts required')
    inputs = [plan_path, audit_path, final_path]
    for proof in final['proofs']:
        path = ROOT / proof['path']
        if sha(path) != proof['sha256']:
            raise ValueError('Final anchored receipt changed')
        inputs.append(path)
    critical = float(t.ppf(.975, 2))
    residual = abs(.5 + critical / (2 * math.sqrt(critical ** 2 + 2)) - .975)
    if residual > 1e-12:
        raise ValueError('Frozen library quantile fails analytic CDF check')
    parent_root = ROOT / 'results/research/OPT03_QTRAIN_recovery/37728384380/cohort'
    records, cells, rows, replay_checks, replay_decisions = {}, {}, [], 0, 0
    for conf in plan['runs']:
        name = conf['run_id']
        current_path = args.root / name / 'result.json'
        parent_path = parent_root / conf['parent_case'] / 'result.json'
        current, parent = read(current_path), read(parent_path)
        replay_path = args.review / 'pinned_replay_receipts' / name / 'replay.json'
        anchor_path = replay_path.with_name('git_anchor.json')
        replay, anchor = read(replay_path), read(anchor_path)
        if (sha(replay_path) != anchor['sha256'] or replay['issues']
                or replay['status'] != 'verified_available_endpoint_replay'
                or replay['verification_returncode'] != 0 or replay['cases'] != [name]
                or replay['scientific_run_id'] != 37771588722
                or replay['scientific_source_commit'] != 'adab1f3dc0ded9da3c42a3e7e5adcb96cdaba67e'
                or replay['verification_source_commit'] != 'aa50cb08766fa0661719672b4dc364b31147da6f'
                or str(replay['verification_run_id']) != '37774614332'
                or replay['independent_external_replication'] is not False):
            raise ValueError('Exact, correctly scoped checkpoint replay required: ' + name)
        inputs.extend([current_path, parent_path, replay_path, anchor_path])
        replay_checks += replay['checks']
        replay_decisions += replay['individual_decisions_replayed']
        records[(conf['policy'], conf['school_weight'], conf['init_seed'])] = (parent, current)
        for budget, result in ((8192, parent), (16384, current)):
            panels = result['evaluations'][-1]['panels']
            probe = panels['probe']['metrics']['matched']['joint_query_binding']
            val = panels['validation']['metrics']['matched']['joint_query_binding']
            gate = probe >= .95 and val >= .90
            if gate != result['matched_joint_competence_gate_passed']:
                raise ValueError('Prespecified case gate differs')
            rows.append(dict(case=name, seed=conf['init_seed'], policy=conf['policy'],
                             school=conf['school_weight'], budget=budget,
                             probe_joint_pct=100 * probe, validation_joint_pct=100 * val,
                             competence_gate_passed=gate))
    if replay_decisions != audit['saved_decisions_recounted']:
        raise ValueError('Decision inventories differ')
    groups = {}
    for b in (8192, 16384):
        for p in ('paired', 'repeated'):
            for s in (0., .3):
                cells[(b, p, s)] = [records[(p, s, seed)][int(b == 16384)]['evaluations'][-1]
                                    ['panels']['validation']['metrics']['matched']['joint_query_binding']
                                    for seed in SEEDS]
                groups[f'u{b}_{p}_school{s}'] = stats(cells[(b, p, s)])
    effects = direct_contrasts(cells)
    checks = 0
    for computed, stored in ([(groups[k], audit['groups'][k]) for k in groups]
                             + [(effects[k], audit['effects'][k]) for k in effects]):
        for field in ('values', 'mean', 'sd', 't95'):
            x, y = computed[field], stored[field]
            if not isinstance(x, list):
                x, y = [x], [y]
            if len(x) != len(y) or max(abs(a - b) for a, b in zip(x, y)) > 1e-12:
                raise ValueError('Independent statistical recount differs: ' + field)
            checks += 1
    eligible = [f'{p}_school{s}' for p in ('paired', 'repeated') for s in (0., .3)
                if all(r['competence_gate_passed'] for r in rows
                       if r['budget'] == 16384 and r['policy'] == p and r['school'] == s)]
    if eligible != audit['eligible_conditions']:
        raise ValueError('Eligibility differs')
    summary = dict(status='verified_complete_exploratory_budget_continuation', groups=groups,
                   effects=effects, independent_statistics_checks=checks, t95_df=2,
                   quantile_cdf_residual=residual, recount_tolerance=1e-12,
                   array_checks=audit['checks'], replay_checks=replay_checks,
                   replay_decisions=replay_decisions, eligible_conditions=eligible,
                   selected_condition=None, test_opened=False, complete_scientific_task3=False,
                   independent_external_replication=False, new_initializations=0)
    output = args.review / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8', newline='\n')
    write_csv(output / 'cases.csv', rows)
    for name, panel in (('groups', groups), ('effects', effects)):
        write_csv(output / (name + '.csv'), [dict(contrast=k, seed200_pct_or_pp=100*v['values'][0],
                   seed201_pct_or_pp=100*v['values'][1], seed202_pct_or_pp=100*v['values'][2],
                   mean_pct_or_pp=100*v['mean'], t95_low_pct_or_pp=100*v['t95'][0],
                   t95_high_pct_or_pp=100*v['t95'][1], training_initializations=3)
                   for k, v in panel.items()])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt'] = 'NP-OPT03-QTRAIN-U16-20261008'
    colors = ('#0072B2', '#D55E00', '#009E73')
    policy_labels = {'paired': 'Cuatro consultas', 'repeated': 'Una consulta repetida'}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True, layout='constrained')
    for ax, (p, s) in zip(axes.flat, ((p, s) for p in ('paired', 'repeated') for s in (0., .3))):
        for seed, color in zip(SEEDS, colors):
            parent, current = records[(p, s, seed)]
            points = [(8192, parent['evaluations'][-1]['panels']['validation']['metrics']['matched']['joint_query_binding'])]
            points += [(v['update'], v['panels']['validation']['metrics']['matched']['joint_query_binding'])
                       for v in current['evaluations']]
            ax.plot([u for u, _ in points], [100*v for _, v in points], marker='o', color=color, label=str(seed))
        ax.axhline(90, color='#AA0000', linestyle='--', linewidth=.9)
        ax.set(title=f'{policy_labels[p]} / escuela {s:g}', xlabel='Actualizaciones',
               ylabel='Exactitud conjunta en validación (%)', ylim=(0, 101))
        ax.set_xticks([8192, 12288, 16384])
        ax.grid(alpha=.2)
    axes[0, 0].legend(title='Inicialización')
    fig.suptitle('Cada inicialización; umbral de validación: 90%; también se exige probe ≥95%')
    for ext in ('png', 'svg'):
        fig.savefig(output / ('joint_competence_continuation.' + ext), dpi=160,
                    metadata={'Date': None} if ext == 'svg' else {})
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for ax, (p, s) in zip(axes.flat, ((p, s) for p in ('paired', 'repeated') for s in (0., .3))):
        for seed, color in zip(SEEDS, colors):
            parent, current = records[(p, s, seed)]
            curve = parent['curve'] + current['curve']
            ax.plot([w['update'] for w in curve], [w['answer_loss_mean'] for w in curve],
                    color=color, label=str(seed))
        ax.axvline(8192, color='#777777', linestyle='--', linewidth=.8)
        ax.set(title=f'{policy_labels[p]} / escuela {s:g}', xlabel='Actualizaciones', ylabel='CE de respuesta')
        ax.grid(alpha=.2)
    axes[0, 0].legend(title='Inicialización')
    fig.suptitle('Estados continuados; el descenso de pérdida no prueba convergencia')
    for ext in ('png', 'svg'):
        fig.savefig(output / ('answer_loss_continuation.' + ext), dpi=160,
                    metadata={'Date': None} if ext == 'svg' else {})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    for i, (name, value) in enumerate(effects.items()):
        mean, low, high = (100*value['mean'], 100*value['t95'][0], 100*value['t95'][1])
        ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt='o', color='#444444', capsize=4)
        for j, (v, color) in enumerate(zip(value['values'], colors)):
            ax.scatter(100*v, i+(j-1)*.1, color=color, s=25)
    effect_labels = {'budget_primary': 'Presupuesto (primario)', 'query': 'Consultas', 'school': 'Escuela',
                     'query_school': 'Consultas × escuela', 'budget_query': 'Presupuesto × consultas',
                     'budget_school': 'Presupuesto × escuela', 'triple_interaction': 'Interacción triple'}
    ax.set_yticks(range(7), [effect_labels[key] for key in effects])
    ax.invert_yaxis()
    ax.axvline(0, color='#777777', linewidth=.8)
    ax.set(xlabel='Efecto en puntos porcentuales', title='Siete contrastes prospectivos; IC95 t, n=3')
    ax.grid(axis='x', alpha=.2)
    for ext in ('png', 'svg'):
        fig.savefig(output / ('prespecified_effects.' + ext), dpi=160,
                    metadata={'Date': None} if ext == 'svg' else {})
    plt.close(fig)
    # Matplotlib's SVG path serializer leaves spaces at line ends. Normalize
    # only generated figure text before hashing; experimental evidence is untouched.
    for path in output.glob('*.svg'):
        path.write_bytes(b'\n'.join(line.rstrip(b' \t\r') for line in path.read_bytes().splitlines()) + b'\n')
    manifest = dict(generator_sha256=sha(pathlib.Path(__file__)), python=platform.python_version(),
                    matplotlib=matplotlib.__version__, frozen_plan_sha256=PLAN_SHA,
                    inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
                    outputs={p.name: sha(p) for p in sorted(output.iterdir())
                             if p.is_file() and p.name != 'analysis_manifest.json'},
                    new_trainings=0, new_model_evaluations=0, complete_scientific_task3=False)
    (output / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: summary[k] for k in ('status', 'independent_statistics_checks',
                      'replay_decisions', 'eligible_conditions', 'complete_scientific_task3')}))


if __name__ == '__main__':
    main()
