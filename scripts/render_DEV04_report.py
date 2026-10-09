"""Render only a fully verified registered DEV04 summary; no new evaluations."""
import hashlib
import json
import math
import pathlib
import sys
from fractions import Fraction
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from summarize_DEV04 import PLAN,RUN
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    folder=ROOT/f'results/research/DEV04_review/{RUN}/analysis';summary_path=folder/'summary.json';manifest_path=folder/'analysis_manifest.json'
    if not summary_path.exists() or not manifest_path.exists():raise ValueError('Complete audited cohort analysis required before report')
    summary=json.loads(summary_path.read_bytes());manifest=json.loads(manifest_path.read_bytes())
    if summary['plan_sha256']!=PLAN or summary['body_fits']!=12 or summary['head_fits']!=24:raise ValueError('Wrong/incomplete registered summary')
    for name,digest in manifest['inputs_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Scientific analysis input changed')
    for name,digest in manifest['outputs_sha256'].items():
        if sha(folder/name)!=digest:raise ValueError('Analysis output changed before report')
    # Second estimator assembled from integral per-mask decision denominators.
    pairs=summary['pairs'];values=[];checks=0
    for policy in (101,102,103):
        rows=[x for x in pairs if x['partition_seed']==policy]
        if len(rows)!=4:raise ValueError('Four initializations per policy required')
        differences=[]
        for row in rows:
            for name in ('attention_joint_pp','local_joint_pp'):
                count=row[name]*16384/100
                if abs(count-round(count))>1e-8:raise ValueError('Validation accuracy not native decision fraction')
                checks+=1
            differences.append(Fraction(round(row['attention_joint_pp']*16384/100)-round(row['local_joint_pp']*16384/100),16384)*100)
            if abs(float(differences[-1])-row['difference_pp'])>1e-10:raise ValueError('Independent pair recount differs')
            checks+=1
        values.append(float(sum(differences,Fraction())/4))
        recorded=next(x['mean_difference_pp'] for x in summary['policies'] if x['partition_seed']==policy)
        if abs(values[-1]-recorded)>1e-10:raise ValueError('Independent policy recount differs')
        checks+=1
    mean=sum(values)/3;variance=sum((v-mean)**2 for v in values)/2;half=summary['primary']['t95_critical']*math.sqrt(variance/3)
    if max(abs(mean-summary['primary']['mean_pp']),abs(half-summary['primary']['half_width_pp']))>1e-10:raise ValueError('Independent interval recount differs')
    checks+=2
    primary=summary['primary'];low,high=primary['t95_pp'];competence=summary['competence_gate_all_twelve'];precision=primary['precision_target_passed']
    lines=['# DEV04: reproducción interna del candidato y precisión registrada','',
        f'Cohorte completa: 12 cuerpos nuevos y 24 lecturas. Atención menos control local: **{mean:+.3f} pp**, ICt95 exploratorio **[{low:.3f}; {high:.3f}] pp**, semianchura **{half:.3f} pp**.','',
        f'Gate de competencia: **{"superado" if competence else "no superado"}**, {summary["attention_gate_passed_cases"]}/12 candidatos cumplen probe ≥95% y validación ≥90%. Objetivo de precisión (semianchura ≤5 pp y límite inferior positivo): **{"superado" if precision else "no superado"}**.','',
        '## Los doce pares registrados','',
        '| Política | Inicialización | Atención validación % | Local validación % | Diferencia pp | Atención probe % | Gate |',
        '|---|---:|---:|---:|---:|---:|---|']
    for row in pairs:
        lines.append(f'| {row["partition_seed"]} | {row["init_seed"]} | {row["attention_joint_pp"]:.3f} | {row["local_joint_pp"]:.3f} | {row["difference_pp"]:+.3f} | {row["attention_probe_pp"]:.3f} | {"Pasa" if row["attention_gate"] else "No pasa"} |')
    lines.extend(['','## Promedios por política','', '| Política | Media pareada pp | Inicializaciones |','|---|---:|---:|'])
    for row in summary['policies']:lines.append(f'| {row["partition_seed"]} | {row["mean_difference_pp"]:+.3f} | 4 |')
    lines.extend(['','El estimador promedia cuatro diferencias pareadas dentro de cada política y luego las tres medias con igual peso. El intervalo df2 no se trunca, no se ajusta por multiplicidad y es exploratorio. Las políticas reutilizan un universo semántico de desarrollo; sus composiciones se solapan. Ni 24 cabezas ni máscaras, ejemplos, endpoints o reintentos se consideran réplicas independientes.','',
        '## Procedencia y auditoría','',
        f'Plan SHA-256: `{PLAN}`. Se preservan los 43 archivos gobernantes. Ejecución original `{RUN}`: nueve éxitos y tres fallos de transporte antes del entrenamiento. Recuperación operativa `37871777801`: exclusivamente los tres fallos bajo una nueva ejecución; cero casos exitosos reentrenados. Verificación interna de endpoints `{summary["verification_run_id"]}`, fuente `{summary["verification_source_commit"]}`.','',
        'Se recuperaron ZIP originales, manifiestos, checkpoints de todos los endpoints, optimizadores y tres streams RNG. Replay independiente de decisiones y recuentos, datasets regenerados, controles ACTO/LUGAR inalterados, mismos witnesses de batches/máscaras para cada pareja, finitud y ventanas de RAM ≥8 GiB. La tolerancia conserva decisiones exactas y NLL media ≤1e-4. La prueba completa contiene 9.732.096 decisiones individuales; no equivale a otros tantos sujetos experimentales.','',
        f'Recuento estadístico ortogonal con denominadores enteros: {checks} comprobaciones sin discrepancias. Los archivos derivados están vinculados por hashes al original, al replay y al análisis.','',
        '## Alcance y conclusiones','',
        'La evidencia se refiere únicamente a cuatro consultas, escuela 0,3 y nuevas inicializaciones de cuerpos bajo la receta seleccionada después de READ03. Es desarrollo interno y no una comparación nueva justa contra Transformer. Las 264 composiciones del test original se conservaron excluidas y no se puntuaron. H1 original permanece no soportada bajo su receta.','',
        'Las lecturas comparten cuerpo, decoder, datos, labels, batches, máscaras y presupuesto de updates. Ambas tienen 3.168 parámetros nominales activos; sus espacios funcionales y FLOPs difieren. El resultado no atribuye toda la capacidad a dinámica celular ni excluye atajos. Atención sobre estado visible tiene antecedentes; una ganancia no demuestra originalidad excepcional.','',
        'El replay y la reproducción de cuerpos pertenecen al mismo proyecto. No constituyen replicación externa, generalización visual, memoria continua, estabilidad prolongada, reparación, medición física de energía ni utilidad independiente. El coste completo y la energía siguen pendientes; el tiempo de updates excluye extracción de caches, evaluación, transporte y otras operaciones.','',
        'DEV04 puede cerrarse como receta completa auditada incluso si sus criterios no se superan. El cierre de la tarea 3 amplia exige una decisión separada sobre controles, límites de optimización y alcance promovible; este informe no la cierra automáticamente ni adelanta tareas 4–8. No se añaden semillas ni presupuesto para estrechar el intervalo retrospectivamente.','',
        '## Reproducir el análisis','',
        '`collect_DEV04_replay_receipts.py` requiere snapshots Git y de ejecuciones completas; `summarize_DEV04.py` verifica los doce ZIP y proofs antes de cualquier efecto; `render_DEV04_report.py` recontabiliza el estimador con denominadores enteros; `close_DEV04.py` verifica hashes, el informe y el alcance. Cada comando conserva los negativos y rechaza entradas incompletas.','',
        'Contexto primario y límites de comparación: [DEV04_context_literature_20261009.md](DEV04_context_literature_20261009.md).'])
    report=ROOT/'docs/research/DEV04_results.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    recount=dict(status='verified_independent_integer_denominator_recount',checks=checks,issues=[],summary_sha256=sha(summary_path),primary_mean_pp=mean,half_width_pp=half,scientific_task3_complete=False)
    (folder/'report_recount.json').write_text(json.dumps(recount,indent=2)+'\n',encoding='utf-8',newline='\n')
    outputs={str(report.relative_to(ROOT)):sha(report),'report_recount.json':sha(folder/'report_recount.json')}
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        colors={101:'#2878b5',102:'#e69f00',103:'#009e73'}
        fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        ax=axes[0]
        for i,row in enumerate(pairs):
            ax.plot([0,1],[row['local_joint_pp'],row['attention_joint_pp']],color=colors[row['partition_seed']],alpha=.6)
            ax.scatter([0,1],[row['local_joint_pp'],row['attention_joint_pp']],color=colors[row['partition_seed']],s=16)
        ax.set_xticks([0,1],['Control local','Atención']);ax.set_ylabel('Exactitud conjunta de validación (%)');ax.set_ylim(0,102);ax.set_title('Los doce pares de cuerpos')
        ax=axes[1]
        for i,row in enumerate(summary['policies']):ax.scatter(row['mean_difference_pp'],i,c=colors[row['partition_seed']],s=60)
        ax.errorbar(mean,3,xerr=half,fmt='ko',capsize=5);ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(4),['Política 101','Política 102','Política 103','Media / ICt95 df2']);ax.set_xlabel('Atención − control local (pp)');ax.set_title('Tres medias y su intervalo exploratorio')
        for suffix in ('png','svg'):
            path=folder/('registered_contrast.'+suffix);fig.savefig(path,dpi=160);outputs[path.name]=sha(path)
        plt.close(fig)
    except ModuleNotFoundError:
        # A missing plot library cannot manufacture results or obstruct the tabulated evidence.
        outputs['plot_status']='matplotlib_unavailable_tables_complete'
    (folder/'report_manifest.json').write_text(json.dumps(dict(source_sha256=sha(pathlib.Path(__file__)),summary_sha256=sha(summary_path),outputs=outputs),indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='verified_DEV04_report_written',report=str(report),recount_checks=checks,scientific_task3_complete=False)))

if __name__=='__main__':main()
