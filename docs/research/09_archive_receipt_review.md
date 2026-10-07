# Item 9: independent review of archived audit receipts

Status: verified for the documentary scope below; no blocking discrepancy found.

This review concerns the completed offline archive O=`2b8203f15ec5f6fe190876c80bf29232034603ce`, under `results/research/09_offline_runs/37603398040-1-offline`. It reviews the two saved audit receipts, the offline manifest, status, environment and recovery receipts. It does not rerun either auditor, execute models or tests, deserialize checkpoints, or inspect scientific outcome tables. Receipt bytes and SHA-256 values were independently checked through the GitHub read interface against the offline manifest and the status references.

## Evidence identity

| Receipt | UTF-8 bytes | SHA-256 |
|---|---:|---|
| `full_archive_audit_01.json` | 169659 | `6cefabacb7d5538e68886924968e6502617b8e62c5f09d848bc0a35863f3b4e1` |
| `gate_archive_audit_01.json` | 113988 | `461d8cab2143934ca5f8b417ae7c1640c8b15b7bc0751c2204c07489f2f9fcbc` |
| `offline_status.json` | 7091 | `d2f8c071dea72640bac1be8e47ffa555d05986d76dd877131fea23bf51c346bd` |
| `offline_environment.json` | 617 | `bf7dbc0da59395d3f470ca7dfb36b5acc0afa01a821d4288fb73bddb7753d84a` |
| `recovery_receipt.json` | 748 | `8d15c4dc6f8d78400bcac726bde93f75a07bb07bdd7b939fc4b33076dfcfba28` |

All five identities match. The offline manifest is explicitly final and completed. Its operational source is `87e1f1c0b388190474e8190952c148c0501c8abf`; its plan SHA-256 is `7cc8b995ebc388089d1668820a60174fa228de4e5bba1285b14e3fb35d4b6eb3`. The post-execution source receipt retains the same clean source and plan.

## Full archive audit

The saved full audit reports `verified`, with 1,100 true checks and no issues. Its inventory covers 134 manifested raw files, totaling 56,105,745 bytes excluding the manifest; the manifest itself is additional. It checks 348 regular source files in the source ZIP against the frozen source and verifies all 83 scientific source bindings. These are file and assertion counts, not independent experiments.

The scientific source is S=`83fe135e301f76bc0c74e30c66bb18e067ca5959`; the study plan SHA-256 is `7e9089ae013abbc4fa042e30606d5d1329dc8b7e89281fcab080addc832c7792`. The raw manifest SHA-256 is `49cda016746d6f020e22da2e10e09f8706b1f17f15491baf438d6952484f78e6`. The audit script identity is `2e83c184304ebaf15e9a7f13c1a081af9ca9551864fabca4dbd515af22ba5352`.

The receipt accepts the declared contract-test, train and final stages, their ordering, source guards and environment boundaries. There are 34 collected parent test cases, passing without skips. The JUnit suite aggregate of 137 is retained as a different reporting unit; it must not be presented as 137 independent test cases. The established interpretation is 34 parent cases plus 103 successful nested unittest subtest reports. This documentary review does not reconstruct an unsaved per-subtest event stream.

The recorded worker duration before its final archive is 3,061.636433142 seconds. RAM monitoring retained 4 contract-test samples, 2,947 train samples and 93 final samples, or 3,044 samples in total. The minimum sampled available RAM is 14.214973449707031 GiB, above the 8 GiB admission/reserve floor. These samples do not establish peak process RSS, continuous resource usage or energy consumption.

## Intermediate archive gate and chronology

The saved gate audit reports `verified`: all 690 checks pass, with zero issues and no dependency limits. Its script SHA-256 is `f4212055fd181b2f34d447d5d5a2e96ca76ce81a3b323f6583ba5343de82774a`.

The receipt authenticates the supplied identities:

- Intermediate archive G: `2e58f5deb47f5676e8663d666d9db7ea98e3e080`.
- Final scientific archive F: `b64d0e2ba222df76caf72bc9870c7602873e7236`.
- Independently supplied remote observation H: the same commit as F.

G is distinct from and an ancestor of F; F is an ancestor of H, with equality allowed for the latter check. The recorded descendant path is G -> `0a127eda8dac6a5556efc5f6c53cb4abeb2ee990` -> F. The runner's observed remote head agrees with H.

G contains 99 manifested files totaling 30,542,121 bytes, plus its manifest, for 100 regular Git files. The gate audit checks the final inventory, all ten checkpoint and summary pairs, their index, and the summaries' referenced training logs, probe and validation artifacts. The ten entries cover seeds 40 through 44 for both families, with the selected learning rates and 4,096 updates. These referenced bytes agree between G and F. No prohibited final artifact or gate receipt is present in G; `final_inventory.json` is the explicit permitted pre-final inventory, not final evaluation data.

The chronology below uses the recorded timestamps on 2026-10-07, in UTC:

| Recorded event | Time |
|---|---|
| Scientific source freeze | 08:22:31.852462 |
| Final inventory created | 09:16:11.700848 |
| Scientific training completed | 09:16:11.705613 |
| Training worker stage completed | 09:16:12.371434 |
| Intermediate copied-file manifest created | 09:16:12.470954 |
| Actions log `evidence_archived` event for G | 09:16:15.491317 |
| Local gate archive receipt written | 09:16:15.491798 |
| Final worker stage started | 09:16:15.580302 |
| First final heartbeat | 09:16:15.581847 |
| Final scientific process started | 09:16:16.957535 |
| Final access consumed | 09:16:17.038259 |
| Final scientific process completed | 09:17:47.378233 |
| Final worker stage completed | 09:17:48.107702 |
| Source-after check | 09:17:53.559503 |
| Worker completed | 09:17:53.632580 |
| Final copied-file manifest created | 09:17:53.787168 |

The complete decoded Actions job log is 44,137 bytes, SHA-256 `340b74f9322b20c682d8791ce2fa0e92f26badff92dcc21aa7cd7e3757257749`. The matching archive event occurs at line 408, before final heartbeats at lines 409 and 410. The worker does not emit a separate final-start stdout event: the first heartbeat corroborates the ordering but is not described as the exact start instant. The stage receipt supplies that separate start timestamp.

## Offline execution and scope limits

The offline run is 37603398040, job 112732965039. All four stages completed with return code zero and no supervision failure: full archive audit, gate audit, scientific saved-array analysis and figures. Their recorded durations are respectively 0.604246313, 1.203703496, 11.452072082 and 1.803723441 seconds. Total offline worker duration before its final archive is 27.802332284 seconds. Completion of the latter two stages is checked here as provenance only; their scientific outcomes and rendered figures are outside this review.

The reviewed offline environment records Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8 and psutil 7.2.2, with the six declared numerical thread environment variables set to one. This is the analysis runtime. It is distinct from the scientific runner's pinned Python 3.12.8 / Torch 2.6.0+cpu environment and its two intra-operation/one inter-operation thread policy, whose runtime checks are accepted in the full audit. The offline recovery receipt retains S, F, H, all 83 source bindings and the same raw-manifest and Actions-log identities.

The Git object identities and ancestry establish content and structural ordering. The frozen worker's push-and-confirm procedure, retained receipt and Actions log provide consistent operational evidence that the ten checkpoints were archived before final access. Git author/committer dates alone do not prove the time of a remote push; the retained timestamps are not independent wall-clock attestations. This is a review of internal evidence under the same project custody. It does not establish external replication, independent blinding or external custody, nor prove that no unrecorded access occurred elsewhere. No scientific outcome claim is added by this review.
