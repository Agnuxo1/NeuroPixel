"""Recount frozen QTRAIN contrasts independently and export tables/figures; no fitting."""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import platform
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEEDS = (200, 201, 202)
PLAN_SHA = '20610e354288b48eeb7316367f61da5173fb2a124a01effa93e7d40e61bf3fb5'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    if len(values) != 3:
        raise ValueError('Exactly three training initializations required')
    mean = statistics.mean(values)
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / 2)
    # Independent analytic df2 quantile: F(t)=1/2+t/(2*sqrt(t*t+2)).
    critical = math.sqrt(2 * .95 ** 2 / (1 - .95 ** 2))
    half = critical * sd / math.sqrt(3)
    return {'values': list(values), 'mean': mean, 'sd': sd,
            't95': [mean - half, mean + half], 'training_initializations': 3}


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path,
                        default=ROOT / 'results/research/OPT03_QTRAIN_recovery/37728384380/cohort')
    parser.add_argument('--review', type=pathlib.Path,
                        default=ROOT / 'results/research/OPT03_QTRAIN_review/37728384380')
    args = parser.parse_args()
    plan_path = ROOT / 'docs/research/OPT03_QTRAIN_plan.json'
    if sha(plan_path) != PLAN_SHA:
        raise ValueError('Frozen plan differs')
    plan = read(plan_path)
    audit = read(args.review / 'audit_full.json')
    if audit['issues'] or audit['missing'] or audit['cases_recounted'] != 12:
        raise ValueError('Full array/control audit required')
    final_receipts = read(args.review / 'final_receipts_verification.json')
    if final_receipts['status'] != 'verified_final_cohort_receipts_and_workflow_success':
        raise ValueError('Final workflow/cohort receipt verification required')
    for item in final_receipts['proofs']:
        path = ROOT / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('Final receipt changed')
    replay_root = args.review / 'pinned_replay_receipts'
    inputs = [plan_path, args.review / 'audit_full.json',
              args.review / 'final_receipts_verification.json',
              args.review / 'final_receipt_anchors.json']
    rows, records, replay_checks, replay_decisions = [], {}, 0, 0
    for conf in plan['runs']:
        name = conf['run_id']
        result_path = args.root / name / 'result.json'
        result = read(result_path)
        replay_path = replay_root / name / 'replay.json'
        replay = read(replay_path)
        anchor = read(replay_root / name / 'git_anchor.json')
        if sha(replay_path) != anchor['sha256']:
            raise ValueError('Pinned replay bytes differ')
        if (replay['status'] != 'verified_available_endpoint_replay'
                or replay['issues'] or replay['verification_returncode'] != 0
                or replay['cases'] != [name]):
            raise ValueError('Exact endpoint replay required for each case')
        inputs.extend([result_path, replay_path, replay_root / name / 'git_anchor.json'])
        replay_checks += replay['checks']
        replay_decisions += replay['individual_decisions_replayed']
        final = result['evaluations'][-1]['panels']
        probe = final['probe']['metrics']['matched']['joint_query_binding']
        validation = final['validation']['metrics']['matched']['joint_query_binding']
        gate = probe >= .95 and validation >= .90
        if gate != result['matched_joint_competence_gate_passed']:
            raise ValueError('Frozen competence gate differs')
        row = {'case': name, 'seed': conf['init_seed'], 'policy': conf['policy'],
               'school_weight': conf['school_weight'], 'probe_joint_pct': 100 * probe,
               'validation_joint_pct': 100 * validation,
               'validation_nominal_binding_pct': 100 * final['validation']['metrics']['matched']['binding'],
               'competence_gate_passed': gate,
               'training_seconds_only': result['curve'][-1]['elapsed_training_seconds'],
               'minimum_recorded_available_ram_gib': min(x['available_ram_gib'] for x in result['resources'])}
        rows.append(row)
        records[(conf['policy'], conf['school_weight'], conf['init_seed'])] = result
    if replay_decisions != audit['saved_decisions_recounted']:
        raise ValueError('Replay/array inventories differ')
    cells, groups = {}, {}
    for policy in ('paired', 'repeated'):
        for school in (0., .3):
            values = [records[(policy, school, seed)]['evaluations'][-1]['panels']['validation']['metrics']['matched']['joint_query_binding'] for seed in SEEDS]
            cells[(policy, school)] = values
            groups[f'{policy}_school{school}'] = stats(values)
    effects = {
        'paired_minus_repeated': stats([sum(cells[('paired', w)][i] - cells[('repeated', w)][i] for w in (0., .3)) / 2 for i in range(3)]),
        'school': stats([sum(cells[(p, .3)][i] - cells[(p, 0.)][i] for p in ('paired', 'repeated')) / 2 for i in range(3)]),
        'interaction': stats([(cells[('paired', .3)][i] - cells[('repeated', .3)][i]) - (cells[('paired', 0.)][i] - cells[('repeated', 0.)][i]) for i in range(3)])}
    recount_checks = 0
    for calculated, stored in [(groups[k], audit['groups'][k]['validation_joint']) for k in groups] + [(effects[k], audit['effects'][k]) for k in effects]:
        for field in ('values', 'mean', 'sd', 't95'):
            x = calculated[field] if isinstance(calculated[field], list) else [calculated[field]]
            y = stored[field] if isinstance(stored[field], list) else [stored[field]]
            if len(x) != len(y) or max(abs(a - b) for a, b in zip(x, y)) > 1e-12:
                raise ValueError('Independent statistical recount differs: ' + field)
            recount_checks += 1
    eligible = [key for key in groups if all(row['competence_gate_passed'] for row in rows if f"{row['policy']}_school{row['school_weight']}" == key)]
    if eligible != audit['eligible_development_conditions']:
        raise ValueError('Eligibility differs')
    # Current completed phase has no eligible condition; do not select a failed arm.
    if eligible:
        raise ValueError('This negative-phase report must not substitute for prospective selection')
    summary = {'status': 'verified_complete_development_phase_no_eligible_condition',
               'groups': groups, 'effects': effects, 'independent_statistics_checks': recount_checks,
               'statistical_recount_tolerance': 1e-12, 'critical_df2': math.sqrt(2 * .95 ** 2 / (1 - .95 ** 2)),
               'array_checks': audit['checks'], 'replay_checks': replay_checks,
               'individual_decisions_replayed_exactly': replay_decisions,
               'eligible_development_conditions': eligible, 'selected_condition': None,
               'test_opened': False, 'scientific_task3_complete': False,
               'scope': 'Exploratory development, n3 training initializations, no external replication'}
    output = args.review / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8', newline='\n')
    write_csv(output / 'cases.csv', rows)
    for name, panel in [('groups', groups), ('effects', effects)]:
        write_csv(output / (name + '.csv'), [
            {'contrast': key, 'seed200_pct_or_pp': 100 * value['values'][0],
             'seed201_pct_or_pp': 100 * value['values'][1], 'seed202_pct_or_pp': 100 * value['values'][2],
             'mean_pct_or_pp': 100 * value['mean'], 'sd_pct_or_pp': 100 * value['sd'],
             't95_low_pct_or_pp': 100 * value['t95'][0], 't95_high_pct_or_pp': 100 * value['t95'][1],
             'training_initializations': 3} for key, value in panel.items()])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt'] = 'NP-OPT03-QTRAIN-20261008'
    colors = ('#0072B2', '#D55E00', '#009E73')
    names = ['Paired / escuela 0', 'Paired / escuela 0,3', 'Repeated / escuela 0', 'Repeated / escuela 0,3']
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), layout='constrained')
    for axis, panel, labels, xlabel in [(axes[0], groups, names, 'Exactitud conjunta en validación (%)'),
                                       (axes[1], effects, ['Paired − repeated', 'Escuela 0,3 − 0', 'Interacción'], 'Efecto en puntos porcentuales')]:
        for index, value in enumerate(panel.values()):
            mean, low, high = 100 * value['mean'], 100 * value['t95'][0], 100 * value['t95'][1]
            axis.errorbar(mean, index, xerr=[[mean - low], [high - mean]], fmt='o', color='#444444', capsize=5)
            for j, x in enumerate(value['values']):
                axis.scatter(100 * x, index + (j - 1) * .11, color=colors[j], s=25, zorder=3)
        axis.set_yticks(range(len(labels)), labels)
        axis.invert_yaxis()
        axis.axvline(0, color='#777777', linewidth=.9)
        axis.set_xlabel(xlabel)
        axis.grid(axis='x', alpha=.2)
    axes[0].axvline(90, color='#CC0000', linestyle='--', linewidth=1, label='Umbral validación: 90%')
    axes[0].legend(loc='lower right', fontsize=8)
    axes[0].set_title('Cuatro condiciones; tres semillas por condición')
    axes[1].set_title('Contrastes congelados; IC95 t, df=2, exploratorios')
    fig.suptitle('OPT03-QTRAIN: ninguna condición alcanza el gate de competencia')
    for extension in ('png', 'svg'):
        fig.savefig(output / ('joint_and_effects.' + extension), dpi=160, metadata={'Date': None} if extension == 'svg' else {})
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, layout='constrained')
    for axis, (key, _), title in zip(axes.flat, groups.items(), names):
        policy, school_text = key.split('_school')
        for seed, color in zip(SEEDS, colors):
            curve = records[(policy, float(school_text), seed)]['curve']
            axis.plot([x['update'] for x in curve], [x['answer_loss_mean'] for x in curve], color=color, label=str(seed))
        axis.set_title(title)
        axis.grid(alpha=.2)
        axis.set_xlabel('Actualizaciones')
        axis.set_ylabel('CE de respuesta; media por ventana de 128')
    axes[0, 0].legend(title='Semilla', fontsize=8)
    fig.suptitle('Curvas archivadas: descenso de pérdida no equivale a convergencia')
    for extension in ('png', 'svg'):
        fig.savefig(output / ('answer_loss_curves.' + extension), dpi=160, metadata={'Date': None} if extension == 'svg' else {})
    plt.close(fig)
    manifest = {'generator_sha256': sha(pathlib.Path(__file__)), 'python': platform.python_version(),
                'matplotlib': matplotlib.__version__, 'inputs': {str(p.relative_to(ROOT)): sha(p) for p in inputs},
                'outputs': {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'analysis_manifest.json'},
                'frozen_plan_sha256': PLAN_SHA, 'new_trainings': 0, 'new_model_evaluations': 0,
                'complete_scientific_task3': False}
    (output / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: summary[k] for k in ('status', 'independent_statistics_checks', 'array_checks', 'replay_checks', 'individual_decisions_replayed_exactly', 'selected_condition', 'scientific_task3_complete')}))


if __name__ == '__main__':
    main()
