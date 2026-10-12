"""Preserve earlier documents and reconcile only admitted saved engineering evidence."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from completion_kit.common import read_json, verify_files, write_json

OUT=ROOT/'results/research/COMPLETION20261012'
energy=read_json(OUT/'energy_complete_summary.json')
admission=read_json(OUT/'energy_recovery_admission.json')
assert energy['unique_rows']==105 and energy['original_rows']==103
assert admission['old_row_canonical_bytes_verified'] and admission['new_measurements_this_task']==0
plan=read_json(ROOT/'docs/research/RENDER11_energy_plan.json')
documents=['README.md','docs/research/RENDER11_results.md','docs/research/RENDER10_results.md','docs/research/RESULTS_INDEX_20261009.md','docs/research/REPRODUCE_20261009.md','docs/research/programme_20261008.json','docs/papers/NeuroPixel_manuscript_20261009.md']
assert not (set(documents)&set(plan['source_sha256'])), 'Do not mutate frozen source governors'
archive=ROOT/'docs/archive/completion_20261012/G1_before'
assert not archive.exists(),'Preserve a fresh archive; never overwrite old versions'
archive.mkdir(parents=True)
(archive/'.gitattributes').write_text('* -text\n',encoding='utf-8',newline='\n')
records=[]
for rel in documents:
    raw=(ROOT/rel).read_bytes()
    target=archive/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(raw)
    records.append({'path':rel,'archived_path':target.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
write_json(archive/'manifest.json',{'scope':'Exact earlier document bytes before local reconciliation; historical failures remain available.','files':records},exclusive=True)

receipt_link='[recovered receipt](../../results/research/RENDER11_energy/recovered_20261009B/receipt.json)'
summary_link='[admitted table](../../results/research/COMPLETION20261012/energy_complete_summary.json)'
lines=['# Sustained device energy: complete registered cohort','',
       'The original RTX 3090 execution preserved 103 of 105 registered rows and all 77 output gates before the unchanged 8 GiB RAM floor stopped it. Its raw receipt remains unchanged. The previously completed recovery from Git archive `87c60227f59f2763e474b12e08f22d2c4104aafb` is now admitted locally: 105 unique keys, the 103 original rows identical, and zero repeated closed measurements. No physical measurement occurred during this admission.','',
       'Only the RGB8 retina, technical block 2, methods `render` and `vector_render` were absent from the original receipt. Their prior recovery uses the same frozen plan, source and registered inputs. Three technical blocks across interrupted sessions are not independent scientific replications.','',
       f'Original receipt SHA-256: `{admission["original_receipt_sha256"]}`. Recovered receipt SHA-256: `{admission["recovered_receipt_sha256"]}`. See the {receipt_link}, {summary_link} and [admission proof](../../results/research/COMPLETION20261012/energy_recovery_admission.json).','',
       '| Workload | Method | Median gross J/call | Technical minimum | Technical maximum |','|---|---|---:|---:|---:|']
for row in energy['table']:
    lines.append(f'| {row["case"]} | {row["method"]} | {row["median_j"]:.9f} | {row["minimum_j"]:.9f} | {row["maximum_j"]:.9f} |')
lines+=['','Whole-board NVML cumulative counters include background consumption. Active blocks last at least three seconds, with one-second idle observations before and after. Idle-adjusted estimates, including negative values, remain in the raw receipt and table without clipping. Baseline ranges are descriptive estimates, not confidence intervals or unique process attribution. CPU, RAM, PSU, training, installation and whole-system wall energy remain unmeasured. This complete device cohort does not establish universal efficiency or energy savings.']
(ROOT/'docs/research/RENDER11_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')

for rel in ['docs/research/RENDER10_results.md','docs/research/RESULTS_INDEX_20261009.md','docs/papers/NeuroPixel_manuscript_20261009.md']:
    path=ROOT/rel
    text=path.read_text(encoding='utf-8')
    if rel.endswith('RENDER10_results.md'):
        old='The separate sustained RENDER11 energy endpoint preserves 100/105 original blocks after a RAM-floor interruption; do not infer its complete comparative effect without all registered rows. Original failures and queue timeouts remain archived.'
        new='The separate sustained RENDER11 energy endpoint preserved 103/105 original blocks. Its previously completed recovery is now admitted with 105 unique keys, 103 unchanged original rows and zero repeated closed measurements. It measures GPU-board energy across interrupted sessions; whole-system wall energy and universal efficiency remain unestablished. Original failures and queue timeouts remain archived.'
        assert old in text
        text=text.replace(old,new)
    else:
        text=text.replace('RENDER11 retained100/105 original energy blocks after RAM8 stopped it; no complete comparative energy or wall-energy claim follows.','RENDER11 originally retained 103/105 energy blocks after the RAM floor stopped it. Its saved recovery is admitted with 105 unique keys, 103 unchanged original rows and zero repeated closed measurements; GPU-board energy is descriptive and whole-system wall energy remains unmeasured.')
    if rel.endswith('RESULTS_INDEX_20261009.md'):
        text=text.replace('143/245 timing rows retained after RAM interruption. Complete speed comparison pending.','245 registered timing rows complete; 143 original rows preserved without repeat. Interrupted sessions qualify interpretation.')
        table_end=text.index('\n\nAll raw archives')
        text=text[:table_end]+'\n| RENDER11 sustained board energy | [Report](RENDER11_results.md) | 105 admitted keys; 103 original rows unchanged, zero repeats; saved recovery 87c60227. Board energy only; wall energy unmeasured. |'+text[table_end:]
    if rel.endswith('NeuroPixel_manuscript_20261009.md'):
        text=text.replace('An additional OpenGL 4.6 compute backend passes all seven execution checks, but its timing cohort is incomplete.','The OpenGL 4.6 compute backend has a complete 245-row timing cohort and 77 output gates, retaining 143 original rows without repeat; workload-dependent outcomes and interrupted sessions limit interpretation. Sustained GPU-board energy has 105 admitted keys with 103 unchanged original rows, without establishing whole-system energy savings.')
        old='RENDER10 passes all 77 parity checks before timing, then stops at the unchanged 8 GiB RAM floor with 143 of 245 rows saved. A recovery process times out in the shared GPU queue before execution. No complete RENDER10 speed or energy effect is inferred. Recovery may only measure missing registered keys while preserving inputs, source and completed rows.'
        new='RENDER10 contains all 245 registered timing rows and 77 output gates. The original 143 rows remain unchanged and no completed timing key was repeated. Compute measures 2.084× relative to the best CUDA comparator for C16/grid128 and loses for C48/grid128. Historically, the original execution stopped at the unchanged 8 GiB RAM floor and a first recovery timed out in the GPU queue; those failed attempts remain archived. Recovery sessions are operational continuity, not additional scientific realizations.'
        assert old in text
        text=text.replace(old,new)
    path.write_text(text,encoding='utf-8',newline='\n')

readme=ROOT/'README.md'
text=readme.read_text(encoding='utf-8')
text=text.replace('with the per-case table in the recovered receipt folder. Full wall energy is unmeasured.','with the [admitted per-case table](results/research/COMPLETION20261012/energy_complete_summary.json), [recovered receipt](results/research/RENDER11_energy/recovered_20261009B/receipt.json) and [SHA-256 admission proof](results/research/COMPLETION20261012/energy_recovery_admission.json). This saved internal recovery adds no independent replicate. Full wall energy is unmeasured.')
readme.write_text(text,encoding='utf-8',newline='\n')
reproduce=ROOT/'docs/research/REPRODUCE_20261009.md'
text=reproduce.read_text(encoding='utf-8')
text=text.replace('OpenGL 4.6 compute execution passed 21 Torch-free checks across all seven archived workloads. Its five-method timing study passed all 77 output checks but stopped at the RAM floor, leaving an incomplete cohort. Read [RENDER10](RENDER10_results.md) and its original receipts before using the compute backend. No RENDER10 acceleration claim is inferred from partial measurements.','OpenGL 4.6 compute execution passed 21 Torch-free checks across all seven archived workloads. Its five-method timing study is complete: 245 registered rows, 77 output gates, 143 original rows unchanged and zero repeated timing keys. The original RAM-floor interruption and queue timeout remain archived history. Read [RENDER10](RENDER10_results.md) and its admitted recovery before using the compute backend. Workload-dependent point comparisons across interrupted sessions do not establish universal acceleration or independent replication.')
text+='\n## Admitted sustained-energy recovery — 12 October 2026\n\nDo not rerun the completed RENDER11 measurements. The original contains 103 rows; the admitted saved recovery contains 105 unique registered keys and preserves all original rows. Use the kit environment in E: to verify or summarize saved receipts:\n\n```powershell\n& E:/Proyecto-concursos/NeuroPixel-env-completion/Scripts/python.exe -m completion_kit.energy summarize --root . --original results/research/RENDER11_energy/original_20261009A/receipt.json --recovery results/research/RENDER11_energy/recovered_20261009B/receipt.json\n```\n\nSee `results/research/COMPLETION20261012/energy_recovery_admission.json` for original and recovered SHA-256 identities, archived commit 87c60227, matching sources/inputs and exact original-row preservation. This is saved board-energy evidence, not a new physical run, independent replication or wall-energy measurement.\n'
reproduce.write_text(text,encoding='utf-8',newline='\n')
programme=read_json(ROOT/'docs/research/programme_20261008.json')
programme['completion_energy_reconciliation_20261012']={'status':'admitted_saved_complete_board_energy','original_rows':103,'complete_unique_rows':105,'repeated_closed_keys':0,'new_measurements_this_reconciliation':0,'source_recovery_commit':admission['source_commit'],'admission_receipt':'results/research/COMPLETION20261012/energy_recovery_admission.json','summary':'results/research/COMPLETION20261012/energy_complete_summary.json','whole_system_wall_energy_measured':False,'programme_complete_inferred':False}
write_json(ROOT/'docs/research/programme_20261008.json',programme)
verify_files(ROOT,plan['source_sha256'])
verify_files(ROOT,plan['input_sha256'])
print(json.dumps({'updated_documents':documents,'archive':str(archive),'scope':'Local prepared documentation; no publication or public push.'}))
