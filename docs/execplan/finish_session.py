"""Write a factual session checkpoint and a concise shared-board handoff."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess

ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'docs/execplan'
utc=datetime.now(timezone.utc).isoformat(timespec='seconds')
status=json.loads(subprocess.check_output(['python',str(PLAN/'plan_status.py'),'status'],cwd=ROOT,text=True,encoding='utf-8'))
assert not status['problems'] and not status['in_progress']
assert status['next']=='A18.2.H' and not status['plan_complete']
resources=json.loads((PLAN/'evidence/session1_integrity_resources.json').read_text(encoding='utf-8'))
text=(PLAN/'EXECPLAN.md').read_text(encoding='utf-8')
text=text.replace('Rama local research/completion-20261012; no push ni publicacion ejecutados.','Version publicable anclada al commit ba9f3f84 y verificada en G1_owner_publication_locator.json; la rama actual contiene el parche T0 separado. No push ni publicacion ejecutados.')
text=text.replace('## Registro de decisiones\n','## Registro de decisiones\n\n- G2: JEV connected/provenance=jev, preservar frozen y parche acotado (1,0), bloquear carga bajo8GiB y continuar metadatos (0,98); G2_jev.txt.\n- A18: JEV connected/provenance=jev, congelar fuentes completas y clasificar por artefactos reales (1,0 en ambas); A18_jev.txt.\n',1)
text=text.replace('## Resultados y retrospectiva\n','## Resultados y retrospectiva\n\n- Sesion1: P0, G0 y G1 local cerradas; subpasos G2.1 y A18.1 cerrados.10 tareas cerradas de48 (37 originales preservadas,11 divisiones/dependencias agregadas).\n- Verificacion:63 pruebas;1797 blobs y44 arboles de origen; recovery105/103 sin nuevas mediciones; auditoria editorial posterior sin incidencias.\n- G2 no cerrado: solo parche AST, Torch y backend pendientes. Guard ejecutado rechaza RAM; ultima medida '+f'{resources["ram_available_gib"]:.2f}'+'GiB.\n- A18 no cerrado:25 fuentes y795 unidades conservadas; niveles aun no asignados. Siguiente grupoH305 unidades; despuesD234/G74/P182.\n- Sin cargas propias activas, sin procesos ajenos detenidos, sin acciones externas ni push. plan_complete:false; §19 final pendiente.\n',1)
text=text.replace('## Cambios a este plan\n','## Cambios a este plan\n\n- '+utc+' · 37 tareas originales conservadas y criterios intactos; G2 separado por CPU/Torch/GPU y A18 por inventario, semantica y auditoria; publicacionG1 separada externamente.48 tareas totales; evidencia de integridad en session1_integrity_resources.json.\n',1)
(PLAN/'EXECPLAN.md').write_text(text,encoding='utf-8',newline='\n')
entry=f'''\n## {utc} — Sesion 1
- Cerradas: P0.1–P0.4, G0, G1.1–G1.3, G2.1 y A18.1; 10/48, criterios originales intactos.
- Evidencia: 26 fuentes del kit, 38+25 pruebas OK, 1797 blobs y44 origenes; recovery105 filas/103 originales sin mediciones nuevas.
- Documentos reconciliados y versiones anteriores preservadas; auditoria posterior sin incidencias; publicacion no ejecutada, paquete78archivos anclado enba9f3f84.
- Bloqueos: G2.2 RAM<8GiB (ultima medida {resources['ram_available_gib']:.2f}GiB); G1.3.ext publicacion del propietario; ningun trabajo propio activo.
- Siguiente: python docs/execplan/plan_status.py start A18.2.H; revisar305 unidades historicas del lote, con claims_ledger completo aun pendiente. plan_complete:false; entrega§19 no escrita como final.
'''
with (PLAN/'SESSION_LOG.md').open('a',encoding='utf-8',newline='\n') as stream:
    stream.write(entry)
board=f'\n{utc} | CODEX NP-COMPLETION-20261012 | Sesion1:10/48 cerradas,63 pruebas,1797 blobs,RENDER11 recovery105/103 admitida sin medir. G2.2 bloqueadaRAM<8GiB; G1.3.ext paquete78archivos enba9f3f84 pendiente propietario. SiguienteA18.2.H (305 unidades), plan_complete:false. Plan:D:/PROJECTS/NeuroPixel-Completion-20261012/docs/execplan/EXECPLAN.md.\n'
for base in [ROOT,Path('D:/PROJECTS/196_NeuroPixel')]:
    with (base/'coordinacion/TABLON.md').open('a',encoding='utf-8',newline='\n') as stream:
        stream.write(board)
print(json.dumps({'session_checkpoint':utc,'status':status,'resources':resources['ram_available_gib']}))
