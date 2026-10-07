# Item 6: complete presentation tables

Post-freeze editorial rendering of two completed independent reports. This appendix performs no model evaluation, example recount, selection, or new scientific analysis. Status checks and inventory checks are presentation safeguards; they do not repeat either artifact audit.

All accuracy/binding values are fractions, CE is mean negative log likelihood in nats, and durations are seconds. Paired columns always show seed20 / seed21. The inferential unit is the initialization: n=2 on split0, df=1. The t95 formula is mean +/- cot(pi*0.025)*sample_SD/sqrt(2), without clipping. Display uses up to 12 significant digits; exact stored precision remains in the linked JSON. No rounded display value drives a decision.

| Evidence | SHA-256 |
| --- | --- |
| [Core verified JSON](../../results/research/06_root_review/37566497890-1/core/core_analysis.json) | 2c2ce1e5a584bd635bb6656c6f43c649d9b78ecb8836b4e0b1f0995cfe7c8c6f |
| [Growth verified JSON](../../results/research/06_root_review/37566497890-1/growth/06_growth_analysis.json) | 6f73d8df5b2ea98333f2c0bc5c5815104a1b3daf691faef7afc0cc1de548b948 |
| Editorial renderer SHA-256 | ec8105f133c01a372e7ab9bb2bc41add8e4fc9b4d4540662406ec56ed540192e |

Complete machine-readable score tables: [core_scores.csv](../../results/research/06_root_review/37566497890-1/core/core_scores.csv), [core_contrasts.csv](../../results/research/06_root_review/37566497890-1/core/core_contrasts.csv), [06_growth_runs.csv](../../results/research/06_root_review/37566497890-1/growth/06_growth_runs.csv), [06_growth_contrasts.csv](../../results/research/06_root_review/37566497890-1/growth/06_growth_contrasts.csv). These CSV links identify the analyzers' companion outputs; the renderer reads only the two JSON files. Per-role scores, other metrics, diagnostic arrays and provenance references are retained there, not replaced by these binding-focused tables.

## Core: all 17 paired evaluation conditions

A=tying, B=school (B0=0, B1=0.3), C=reinjection. Binding summaries below are copied from the verified analyzer.

| Condition | Global 20 / 21 | Binding 20 / 21 | Binding mean | SD | Range | t95, df1 | CE 20 / 21 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factorial: A0 B0 C0; train T16; 1024 updates; clean | 0.21630859375 / 0.30908203125 | 0.12646484375 / 0.0810546875 | 0.103759765625 | 0.0321098294191 | [0.0810546875, 0.12646484375] | [-0.184735605582, 0.392255136832] | 2.78874127903 / 2.04419084414 |
| factorial: A0 B0 C1; train T16; 1024 updates; clean | 0.195068359375 / 0.2587890625 | 0.1181640625 / 0.07568359375 | 0.096923828125 | 0.0300382275211 | [0.07568359375, 0.1181640625] | [-0.172958938488, 0.366806594738] | 3.19335461845 / 2.76295473565 |
| factorial: A0 B1 C0; train T16; 1024 updates; clean | 0.221435546875 / 0.267333984375 | 0.12451171875 / 0.10693359375 | 0.11572265625 | 0.012429611388 | [0.10693359375, 0.12451171875] | [0.00404702868596, 0.227398283814] | 2.33747197372 / 2.51182261911 |
| factorial: A0 B1 C1; train T16; 1024 updates; clean | 0.199951171875 / 0.294677734375 | 0.115234375 / 0.1015625 | 0.1083984375 | 0.00966747552403 | [0.1015625, 0.115234375] | [0.0215396160613, 0.195257258939] | 2.79627886599 / 2.90753778369 |
| factorial: A1 B0 C0; train T16; 1024 updates; clean | 0.1552734375 / 0.2705078125 | 0.1181640625 / 0.07763671875 | 0.097900390625 | 0.0286571595891 | [0.07763671875, 0.1181640625] | [-0.159573972925, 0.355374754175] | 2.78034167517 / 2.14381526728 |
| factorial: A1 B0 C1; train T16; 1024 updates; clean | 0.163818359375 / 0.23681640625 | 0.1318359375 / 0.07373046875 | 0.102783203125 | 0.0410867709771 | [0.07373046875, 0.1318359375] | [-0.266366787989, 0.471933194239] | 2.59873120264 / 2.31109029574 |
| factorial: A1 B1 C0; train T16; 1024 updates; clean | 0.152587890625 / 0.258544921875 | 0.11669921875 / 0.09326171875 | 0.10498046875 | 0.0165728151841 | [0.09326171875, 0.11669921875] | [-0.043920368002, 0.253881305502] | 2.44754147966 / 2.29719579218 |
| factorial: A1 B1 C1; train T16; 1024 updates; clean | 0.195068359375 / 0.28515625 | 0.140625 / 0.078125 | 0.109375 | 0.0441941738242 | [0.078125, 0.140625] | [-0.287693898005, 0.506443898005] | 4.94032685317 / 2.98135712776 |
| recurrence: A1 B0 C1; train T1; 1024 updates; clean | 0.11669921875 / 0.10791015625 | 0.10498046875 / 0.07958984375 | 0.09228515625 | 0.0179538831161 | [0.07958984375, 0.10498046875] | [-0.0690240835647, 0.253594396065] | 2.34741688603 / 2.33905435074 |
| recurrence: A1 B0 C1; train T4; 1024 updates; clean | 0.142822265625 / 0.13818359375 | 0.1171875 / 0.0849609375 | 0.10107421875 | 0.0227876208781 | [0.0849609375, 0.1171875] | [-0.103664431784, 0.305812869284] | 2.37197635154 / 2.3304423324 |
| damage: A1 B0 C1; train T16; 1024 updates; clean | 0.152587890625 / 0.23046875 | 0.12841796875 / 0.0810546875 | 0.104736328125 | 0.0334908973511 | [0.0810546875, 0.12841796875] | [-0.196167446145, 0.405640102395] | 2.4468816247 / 2.08215747386 |
| optimization: A1 B0 C1; train T16; 8192 updates; clean | 0.5322265625 / 0.507080078125 | 0.28955078125 / 0.1396484375 | 0.214599609375 | 0.105996963781 | [0.1396484375, 0.28955078125] | [-0.737745325685, 1.16694454443] | 2.6879608439 / 1.80619141543 |
| optimization: A1 B1 C1; train T16; 8192 updates; clean | 0.57373046875 / 0.541748046875 | 0.203125 / 0.2265625 | 0.21484375 | 0.0165728151841 | [0.203125, 0.2265625] | [0.065942913248, 0.363744586752] | 1.75101340963 / 2.39859649681 |
| factorial: A1 B0 C1; train T16; 1024 updates; deployment_truncation T1 | 0.1123046875 / 0.114990234375 | 0.099609375 / 0.0947265625 | 0.09716796875 | 0.00345266983001 | [0.0947265625, 0.099609375] | [0.0661469610933, 0.128188976407] | 2.41306769718 / 2.45445660962 |
| factorial: A1 B0 C1; train T16; 1024 updates; deployment_truncation T4 | 0.13916015625 / 0.160400390625 | 0.12255859375 / 0.072265625 | 0.097412109375 | 0.0355624992491 | [0.072265625, 0.12255859375] | [-0.222104269489, 0.416928488239] | 2.34173818344 / 2.35036914488 |
| factorial: A1 B0 C1; train T16; 1024 updates; fixed_lesion | 0.153564453125 / 0.2138671875 | 0.12451171875 / 0.0771484375 | 0.100830078125 | 0.0334908973511 | [0.0771484375, 0.12451171875] | [-0.200073696145, 0.401733852395] | 2.65489842712 / 2.37768934374 |
| damage: A1 B0 C1; train T16; 1024 updates; fixed_lesion | 0.154052734375 / 0.2353515625 | 0.1337890625 / 0.0888671875 | 0.111328125 | 0.0317645624361 | [0.0888671875, 0.1337890625] | [-0.174065145441, 0.396721395441] | 2.38798048758 / 2.089605239 |

## Core: all 38 binding contrasts

Signs, contrast scale, SD, range and intervals are those of the verified analyzer.

| Panel | Contrast | Delta 20 / 21 | Mean | SD | Range | t95, df1 |
| --- | --- | --- | --- | --- | --- | --- |
| factorial | tying | 0.0057373046875 / -0.0106201171875 | -0.00244140625 | 0.0115664439305 | [-0.0106201171875, 0.0057373046875] | [-0.1063617819, 0.1014789694] |
| factorial_conditional | tying__school0_reinjection0 | -0.00830078125 / -0.00341796875 | -0.005859375 | 0.00345266983001 | [-0.00830078125, -0.00341796875] | [-0.0368803826567, 0.0251616326567] |
| factorial_conditional | tying__school0_reinjection1 | 0.013671875 / -0.001953125 | 0.005859375 | 0.011048543456 | [-0.001953125, 0.013671875] | [-0.0934078495014, 0.105126599501] |
| factorial_conditional | tying__school1_reinjection0 | -0.0078125 / -0.013671875 | -0.0107421875 | 0.00414320379601 | [-0.013671875, -0.0078125] | [-0.047967396688, 0.026483021688] |
| factorial_conditional | tying__school1_reinjection1 | 0.025390625 / -0.0234375 | 0.0009765625 | 0.0345266983001 | [-0.0234375, 0.025390625] | [-0.309233514067, 0.311186639067] |
| factorial | school | 0.0006103515625 / 0.0179443359375 | 0.00927734375 | 0.0122569778965 | [0.0006103515625, 0.0179443359375] | [-0.100847233431, 0.119401920931] |
| factorial_conditional | school__tying0_reinjection0 | -0.001953125 / 0.02587890625 | 0.011962890625 | 0.0196802180311 | [-0.001953125, 0.02587890625] | [-0.164856853018, 0.188782634268] |
| factorial_conditional | school__tying0_reinjection1 | -0.0029296875 / 0.02587890625 | 0.011474609375 | 0.0203707519971 | [-0.0029296875, 0.02587890625] | [-0.171549335799, 0.194498554549] |
| factorial_conditional | school__tying1_reinjection0 | -0.00146484375 / 0.015625 | 0.007080078125 | 0.012084344405 | [-0.00146484375, 0.015625] | [-0.101493448673, 0.115653604923] |
| factorial_conditional | school__tying1_reinjection1 | 0.0087890625 / 0.00439453125 | 0.006591796875 | 0.00310740284701 | [0.00439453125, 0.0087890625] | [-0.021327110016, 0.034510703766] |
| factorial | reinjection | 0.0050048828125 / -0.0074462890625 | -0.001220703125 | 0.00880430806653 | [-0.0074462890625, 0.0050048828125] | [-0.0803242726495, 0.0778828663995] |
| factorial_conditional | reinjection__tying0_school0 | -0.00830078125 / -0.00537109375 | -0.0068359375 | 0.00207160189801 | [-0.00830078125, -0.00537109375] | [-0.025448542094, 0.011776667094] |
| factorial_conditional | reinjection__tying0_school1 | -0.00927734375 / -0.00537109375 | -0.00732421875 | 0.00276213586401 | [-0.00927734375, -0.00537109375] | [-0.0321410248753, 0.0174925873753] |
| factorial_conditional | reinjection__tying1_school0 | 0.013671875 / -0.00390625 | 0.0048828125 | 0.012429611388 | [-0.00390625, 0.013671875] | [-0.106792815064, 0.116558440064] |
| factorial_conditional | reinjection__tying1_school1 | 0.02392578125 / -0.01513671875 | 0.00439453125 | 0.0276213586401 | [-0.01513671875, 0.02392578125] | [-0.243773530003, 0.252562592503] |
| factorial | tying_by_school | 0.006103515625 / -0.015869140625 | -0.0048828125 | 0.0155370142351 | [-0.015869140625, 0.006103515625] | [-0.144477346955, 0.134711721955] |
| factorial_conditional | tying_by_school__reinjection0 | 0.00048828125 / -0.01025390625 | -0.0048828125 | 0.00759587362603 | [-0.01025390625, 0.00048828125] | [-0.0731290293447, 0.0633634043447] |
| factorial_conditional | tying_by_school__reinjection1 | 0.01171875 / -0.021484375 | -0.0048828125 | 0.0234781548441 | [-0.021484375, 0.01171875] | [-0.215825664565, 0.206060039565] |
| factorial | tying_by_reinjection | 0.027587890625 / -0.004150390625 | 0.01171875 | 0.0224423538951 | [-0.004150390625, 0.027587890625] | [-0.189917799768, 0.213355299768] |
| factorial_conditional | tying_by_reinjection__school0 | 0.02197265625 / 0.00146484375 | 0.01171875 | 0.0145012132861 | [0.00146484375, 0.02197265625] | [-0.118569482158, 0.142006982158] |
| factorial_conditional | tying_by_reinjection__school1 | 0.033203125 / -0.009765625 | 0.01171875 | 0.0303834945041 | [-0.009765625, 0.033203125] | [-0.261266117379, 0.284703617379] |
| factorial | school_by_reinjection | 0.004638671875 / -0.005615234375 | -0.00048828125 | 0.00725060664303 | [-0.005615234375, 0.004638671875] | [-0.065632397329, 0.064655834829] |
| factorial_conditional | school_by_reinjection__tying0 | -0.0009765625 / 0 | -0.00048828125 | 0.000690533966002 | [-0.0009765625, 0] | [-0.00669248278134, 0.00571592028134] |
| factorial_conditional | school_by_reinjection__tying1 | 0.01025390625 / -0.01123046875 | -0.00048828125 | 0.0151917472521 | [-0.01123046875, 0.01025390625] | [-0.136980714939, 0.136004152439] |
| factorial | tying_by_school_by_reinjection | 0.01123046875 / -0.01123046875 | 0 | 0.0158822812181 | [-0.01123046875, 0.01123046875] | [-0.142696635221, 0.142696635221] |
| recurrence_trained | trained_T1_minus_trained_T16 | -0.02685546875 / 0.005859375 | -0.010498046875 | 0.0231328878611 | [-0.02685546875, 0.005859375] | [-0.218338798175, 0.197342704425] |
| recurrence_deployment | same_T16_weights_eval_T1_minus_eval_T16 | -0.0322265625 / 0.02099609375 | -0.005615234375 | 0.0376341011471 | [-0.0322265625, 0.02099609375] | [-0.343744217833, 0.332513749083] |
| recurrence_trained | trained_T4_minus_trained_T16 | -0.0146484375 / 0.01123046875 | -0.001708984375 | 0.0182991500991 | [-0.0146484375, 0.01123046875] | [-0.166120324955, 0.162702356205] |
| recurrence_deployment | same_T16_weights_eval_T4_minus_eval_T16 | -0.00927734375 / -0.00146484375 | -0.00537109375 | 0.00552427172802 | [-0.00927734375, -0.00146484375] | [-0.0550047060007, 0.0442625185007] |
| damage | damage_training_minus_clean_training__eval_lesion0 | -0.00341796875 / 0.00732421875 | 0.001953125 | 0.00759587362603 | [-0.00341796875, 0.00732421875] | [-0.0662930918447, 0.0701993418447] |
| damage | damage_training_minus_clean_training__eval_lesion1 | 0.00927734375 / 0.01171875 | 0.010498046875 | 0.00172633491501 | [0.00927734375, 0.01171875] | [-0.00501245695334, 0.0260085507033] |
| damage | eval_lesion_minus_clean__training_damage0 | -0.00732421875 / 0.00341796875 | -0.001953125 | 0.00759587362603 | [-0.00732421875, 0.00341796875] | [-0.0701993418447, 0.0662930918447] |
| damage | eval_lesion_minus_clean__training_damage1 | 0.00537109375 / 0.0078125 | 0.006591796875 | 0.00172633491501 | [0.00537109375, 0.0078125] | [-0.00891870695334, 0.0221023007033] |
| damage | training_damage_by_eval_lesion | 0.0126953125 / 0.00439453125 | 0.008544921875 | 0.00586953871102 | [0.00439453125, 0.0126953125] | [-0.0441907911414, 0.0612806348914] |
| budget | updates8192_minus1024__school0 | 0.15771484375 / 0.06591796875 | 0.11181640625 | 0.0649101928042 | [0.06591796875, 0.15771484375] | [-0.471378537696, 0.695011350196] |
| budget | updates8192_minus1024__school1 | 0.0625 / 0.1484375 | 0.10546875 | 0.0607669890082 | [0.0625, 0.1484375] | [-0.440500984758, 0.651438484758] |
| budget | budget_by_school | -0.09521484375 / 0.08251953125 | -0.00634765625 | 0.125677181812 | [-0.09521484375, 0.08251953125] | [-1.13551233495, 1.12281702245] |
| budget_conditional | school03_minus0__updates8192 | -0.08642578125 / 0.0869140625 | 0.000244140625 | 0.122569778965 | [-0.08642578125, 0.0869140625] | [-1.10100163119, 1.10148991244] |

## Core: training diagnostics for all 13 paired conditions

Probe counts are reported by the trainer and checked for consistency; probe prediction arrays were not saved. The budget-limited flag denotes probe binding below 0.95. The last objective is a 128-update-window mean; answer loss is the last update only. Their difference is not an estimate of school loss. Times cover recorded expert training, not isolated gate fitting.

| Condition | Parameters 20 / 21 | Probe binding 20 / 21 | Validation binding 20 / 21 | Budget-limited 20 / 21 | Last window objective 20 / 21 | Last answer CE 20 / 21 | Training seconds 20 / 21 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factorial: A0 B0 C0; train T16; 1024 updates | 30384 / 30384 | 0.1171875 / 0.109375 | 0.115234375 / 0.0927734375 | true / true | 2.08633762505 / 1.90641395189 | 1.83141887188 / 1.62475168705 | 110.73913367 / 107.627662429 |
| factorial: A0 B0 C1; train T16; 1024 updates | 30384 / 30384 | 0.15234375 / 0.10546875 | 0.1259765625 / 0.0751953125 | true / true | 2.17374528479 / 1.92895164713 | 1.8492115736 / 1.63044631481 | 107.956180343 / 107.970054657 |
| factorial: A0 B1 C0; train T16; 1024 updates | 30384 / 30384 | 0.1171875 / 0.14453125 | 0.11328125 / 0.0986328125 | true / true | 2.23386381101 / 2.05427083373 | 1.99171459675 / 1.65952420235 | 111.361514792 / 111.527622556 |
| factorial: A0 B1 C1; train T16; 1024 updates | 30384 / 30384 | 0.1171875 / 0.08203125 | 0.126953125 / 0.0771484375 | true / true | 2.25722750835 / 1.91998419072 | 1.847219944 / 1.56901168823 | 111.390351977 / 111.474375547 |
| factorial: A1 B0 C0; train T16; 1024 updates | 29824 / 29824 | 0.09765625 / 0.09765625 | 0.1064453125 / 0.0849609375 | true / true | 2.25605027657 / 1.91422835737 | 1.99058234692 / 1.55668306351 | 107.21348905 / 107.314334686 |
| factorial: A1 B0 C1; train T16; 1024 updates | 29824 / 29824 | 0.15625 / 0.09765625 | 0.1513671875 / 0.09765625 | true / true | 2.29282572865 / 2.12010075524 | 2.07971262932 / 1.84096670151 | 107.296330746 / 107.454624159 |
| factorial: A1 B1 C0; train T16; 1024 updates | 29824 / 29824 | 0.12890625 / 0.0859375 | 0.1044921875 / 0.0927734375 | true / true | 2.33017616719 / 2.13457312621 | 2.11461949348 / 1.79418742657 | 112.155548453 / 111.794824986 |
| factorial: A1 B1 C1; train T16; 1024 updates | 29824 / 29824 | 0.140625 / 0.10546875 | 0.134765625 / 0.0732421875 | true / true | 2.19229365792 / 1.9868402062 | 1.7785320282 / 1.58927941322 | 112.027245088 / 111.896648656 |
| recurrence: A1 B0 C1; train T1; 1024 updates | 29824 / 29824 | 0.0859375 / 0.10546875 | 0.08984375 / 0.09375 | true / true | 2.87368415669 / 2.87340907566 | 2.90009188652 / 2.9380736351 | 8.081386339 / 8.265050979 |
| recurrence: A1 B0 C1; train T4; 1024 updates | 29824 / 29824 | 0.109375 / 0.12109375 | 0.0947265625 / 0.111328125 | true / true | 2.41146109439 / 2.4119785931 | 2.2063908577 / 2.22830748558 | 28.319070905 / 28.314228112 |
| damage: A1 B0 C1; train T16; 1024 updates | 29824 / 29824 | 0.12109375 / 0.109375 | 0.1064453125 / 0.0986328125 | true / true | 2.29293524846 / 2.09469064511 | 2.06134581566 / 1.84169256687 | 108.039180218 / 107.6531463 |
| optimization: A1 B0 C1; train T16; 8192 updates | 29824 / 29824 | 0.30078125 / 0.15234375 | 0.2822265625 / 0.1552734375 | true / true | 1.0057461774 / 1.31093079317 | 0.887842953205 / 1.20117807388 | 861.568289198 / 855.633534496 |
| optimization: A1 B1 C1; train T16; 8192 updates | 29824 / 29824 | 0.171875 / 0.2578125 | 0.1875 / 0.2265625 | true / true | 1.00794614432 / 1.00899664592 | 0.788447737694 / 0.757559299469 | 896.218071943 / 892.767981278 |

## Growth: all 17 bank/router pairs across four topic groups

There are 68 paired rows. Pooled means the fixed concatenation of A/B/C, 1,024 examples each. Binding summaries here are derived editorially from the two saved binding scores, with n=2, sample SD, range and the same unclipped t95 formula. Slots and topics do not enlarge the replication count.

| Bank | Router | Topic | Slots 20 / 21 | Binding 20 / 21 | Mean | SD | Range | t95, df1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_final | uniform | pooled | 1 / 1 | 0.0748697916667 / 0.0657552083333 | 0.0703125 | 0.00644498368269 | [0.0657552083333, 0.0748697916667] | [0.0124066190409, 0.128218380959] |
| single_final | uniform | A | 1 / 1 | 0 / 0 | 0 | 0 | [0, 0] | [0, 0] |
| single_final | uniform | B | 1 / 1 | 0 / 0 | 0 | 0 | [0, 0] | [0, 0] |
| single_final | uniform | C | 1 / 1 | 0.224609375 / 0.197265625 | 0.2109375 | 0.0193349510481 | [0.197265625, 0.224609375] | [0.0372198571226, 0.384655142877] |
| snapshots | uniform | pooled | 3 / 3 | 0.0690104166667 / 0.0794270833333 | 0.07421875 | 0.00736569563736 | [0.0690104166667, 0.0794270833333] | [0.00804060033242, 0.140396899668] |
| snapshots | uniform | A | 3 / 3 | 0 / 0.0625 | 0.03125 | 0.0441941738242 | [0, 0.0625] | [-0.365818898005, 0.428318898005] |
| snapshots | uniform | B | 3 / 3 | 0.056640625 / 0.04296875 | 0.0498046875 | 0.00966747552403 | [0.04296875, 0.056640625] | [-0.0370541339387, 0.136663508939] |
| snapshots | uniform | C | 3 / 3 | 0.150390625 / 0.1328125 | 0.1416015625 | 0.012429611388 | [0.1328125, 0.150390625] | [0.029925934936, 0.253277190064] |
| snapshots | random | pooled | 3 / 3 | 0.0859375 / 0.0794270833333 | 0.0826822916667 | 0.00460355977335 | [0.0794270833333, 0.0859375] | [0.0413209481244, 0.124043635209] |
| snapshots | random | A | 3 / 3 | 0.111328125 / 0.10546875 | 0.1083984375 | 0.00414320379601 | [0.10546875, 0.111328125] | [0.071173228312, 0.145623646688] |
| snapshots | random | B | 3 / 3 | 0.072265625 / 0.0703125 | 0.0712890625 | 0.001381067932 | [0.0703125, 0.072265625] | [0.0588806594373, 0.0836974655627] |
| snapshots | random | C | 3 / 3 | 0.07421875 / 0.0625 | 0.068359375 | 0.00828640759203 | [0.0625, 0.07421875] | [-0.00609104337602, 0.142809793376] |
| snapshots | scanner | pooled | 3 / 3 | 0.233072916667 / 0.224609375 | 0.228841145833 | 0.00598462770535 | [0.224609375, 0.233072916667] | [0.175071399228, 0.282610892438] |
| snapshots | scanner | A | 3 / 3 | 0.291015625 / 0.2578125 | 0.2744140625 | 0.0234781548441 | [0.2578125, 0.291015625] | [0.0634712104346, 0.485356914565] |
| snapshots | scanner | B | 3 / 3 | 0.20703125 / 0.244140625 | 0.2255859375 | 0.0262402907081 | [0.20703125, 0.244140625] | [-0.0101737206907, 0.461345595691] |
| snapshots | scanner | C | 3 / 3 | 0.201171875 / 0.171875 | 0.1865234375 | 0.0207160189801 | [0.171875, 0.201171875] | [0.000397391559941, 0.37264948344] |
| snapshots | learned | pooled | 3 / 3 | 0.229166666667 / 0.23828125 | 0.233723958333 | 0.00644498368269 | [0.229166666667, 0.23828125] | [0.175818077374, 0.291629839292] |
| snapshots | learned | A | 3 / 3 | 0.259765625 / 0.275390625 | 0.267578125 | 0.011048543456 | [0.259765625, 0.275390625] | [0.168310900499, 0.366845349501] |
| snapshots | learned | B | 3 / 3 | 0.203125 / 0.2421875 | 0.22265625 | 0.0276213586401 | [0.203125, 0.2421875] | [-0.0255118112534, 0.470824311253] |
| snapshots | learned | C | 3 / 3 | 0.224609375 / 0.197265625 | 0.2109375 | 0.0193349510481 | [0.197265625, 0.224609375] | [0.0372198571226, 0.384655142877] |
| duplicate_slots | uniform | pooled | 3 / 3 | 0.0690104166667 / 0.0807291666667 | 0.0748697916667 | 0.00828640759203 | [0.0690104166667, 0.0807291666667] | [0.000419373290643, 0.149320210043] |
| duplicate_slots | uniform | A | 3 / 3 | 0 / 0 | 0 | 0 | [0, 0] | [0, 0] |
| duplicate_slots | uniform | B | 3 / 3 | 0.20703125 / 0.2421875 | 0.224609375 | 0.0248592227761 | [0.20703125, 0.2421875] | [0.00125811987193, 0.447960630128] |
| duplicate_slots | uniform | C | 3 / 3 | 0 / 0 | 0 | 0 | [0, 0] | [0, 0] |
| duplicate_slots | random | pooled | 3 / 3 | 0.0846354166667 / 0.0904947916667 | 0.0875651041667 | 0.00414320379601 | [0.0846354166667, 0.0904947916667] | [0.0503398949787, 0.124790313355] |
| duplicate_slots | random | A | 3 / 3 | 0.111328125 / 0.10546875 | 0.1083984375 | 0.00414320379601 | [0.10546875, 0.111328125] | [0.071173228312, 0.145623646688] |
| duplicate_slots | random | B | 3 / 3 | 0.142578125 / 0.166015625 | 0.154296875 | 0.0165728151841 | [0.142578125, 0.166015625] | [0.00539603824795, 0.303197711752] |
| duplicate_slots | random | C | 3 / 3 | 0 / 0 | 0 | 0 | [0, 0] | [0, 0] |
| duplicate_slots | scanner | pooled | 3 / 3 | 0.166015625 / 0.16796875 | 0.1669921875 | 0.001381067932 | [0.166015625, 0.16796875] | [0.154583784437, 0.179400590563] |
| duplicate_slots | scanner | A | 3 / 3 | 0.291015625 / 0.2578125 | 0.2744140625 | 0.0234781548441 | [0.2578125, 0.291015625] | [0.0634712104346, 0.485356914565] |
| duplicate_slots | scanner | B | 3 / 3 | 0.20703125 / 0.24609375 | 0.2265625 | 0.0276213586401 | [0.20703125, 0.24609375] | [-0.0216055612534, 0.474730561253] |
| duplicate_slots | scanner | C | 3 / 3 | 0 / 0 | 0 | 0 | [0, 0] | [0, 0] |
| duplicate_slots | learned | pooled | 3 / 3 | 0.1640625 / 0.173828125 | 0.1689453125 | 0.00690533966002 | [0.1640625, 0.173828125] | [0.106903297187, 0.230987327813] |
| duplicate_slots | learned | A | 3 / 3 | 0.28515625 / 0.27734375 | 0.28125 | 0.00552427172802 | [0.27734375, 0.28515625] | [0.231616387749, 0.330883612251] |
| duplicate_slots | learned | B | 3 / 3 | 0.20703125 / 0.244140625 | 0.2255859375 | 0.0262402907081 | [0.20703125, 0.244140625] | [-0.0101737206907, 0.461345595691] |
| duplicate_slots | learned | C | 3 / 3 | 0 / 0 | 0 | 0 | [0, 0] | [0, 0] |
| adaptive_novelty | uniform | pooled | 3 / 3 | 0.0690104166667 / 0.0794270833333 | 0.07421875 | 0.00736569563736 | [0.0690104166667, 0.0794270833333] | [0.00804060033242, 0.140396899668] |
| adaptive_novelty | uniform | A | 3 / 3 | 0 / 0.0625 | 0.03125 | 0.0441941738242 | [0, 0.0625] | [-0.365818898005, 0.428318898005] |
| adaptive_novelty | uniform | B | 3 / 3 | 0.056640625 / 0.04296875 | 0.0498046875 | 0.00966747552403 | [0.04296875, 0.056640625] | [-0.0370541339387, 0.136663508939] |
| adaptive_novelty | uniform | C | 3 / 3 | 0.150390625 / 0.1328125 | 0.1416015625 | 0.012429611388 | [0.1328125, 0.150390625] | [0.029925934936, 0.253277190064] |
| adaptive_novelty | random | pooled | 3 / 3 | 0.0859375 / 0.0794270833333 | 0.0826822916667 | 0.00460355977335 | [0.0794270833333, 0.0859375] | [0.0413209481244, 0.124043635209] |
| adaptive_novelty | random | A | 3 / 3 | 0.111328125 / 0.10546875 | 0.1083984375 | 0.00414320379601 | [0.10546875, 0.111328125] | [0.071173228312, 0.145623646688] |
| adaptive_novelty | random | B | 3 / 3 | 0.072265625 / 0.0703125 | 0.0712890625 | 0.001381067932 | [0.0703125, 0.072265625] | [0.0588806594373, 0.0836974655627] |
| adaptive_novelty | random | C | 3 / 3 | 0.07421875 / 0.0625 | 0.068359375 | 0.00828640759203 | [0.0625, 0.07421875] | [-0.00609104337602, 0.142809793376] |
| adaptive_novelty | scanner | pooled | 3 / 3 | 0.233072916667 / 0.224609375 | 0.228841145833 | 0.00598462770535 | [0.224609375, 0.233072916667] | [0.175071399228, 0.282610892438] |
| adaptive_novelty | scanner | A | 3 / 3 | 0.291015625 / 0.2578125 | 0.2744140625 | 0.0234781548441 | [0.2578125, 0.291015625] | [0.0634712104346, 0.485356914565] |
| adaptive_novelty | scanner | B | 3 / 3 | 0.20703125 / 0.244140625 | 0.2255859375 | 0.0262402907081 | [0.20703125, 0.244140625] | [-0.0101737206907, 0.461345595691] |
| adaptive_novelty | scanner | C | 3 / 3 | 0.201171875 / 0.171875 | 0.1865234375 | 0.0207160189801 | [0.171875, 0.201171875] | [0.000397391559941, 0.37264948344] |
| adaptive_novelty | learned | pooled | 3 / 3 | 0.229166666667 / 0.23828125 | 0.233723958333 | 0.00644498368269 | [0.229166666667, 0.23828125] | [0.175818077374, 0.291629839292] |
| adaptive_novelty | learned | A | 3 / 3 | 0.259765625 / 0.275390625 | 0.267578125 | 0.011048543456 | [0.259765625, 0.275390625] | [0.168310900499, 0.366845349501] |
| adaptive_novelty | learned | B | 3 / 3 | 0.203125 / 0.2421875 | 0.22265625 | 0.0276213586401 | [0.203125, 0.2421875] | [-0.0255118112534, 0.470824311253] |
| adaptive_novelty | learned | C | 3 / 3 | 0.224609375 / 0.197265625 | 0.2109375 | 0.0193349510481 | [0.197265625, 0.224609375] | [0.0372198571226, 0.384655142877] |
| adaptive_resonance | uniform | pooled | 1 / 2 | 0.0748697916667 / 0.0716145833333 | 0.0732421875 | 0.00230177988667 | [0.0716145833333, 0.0748697916667] | [0.0525615157289, 0.0939228592711] |
| adaptive_resonance | uniform | A | 1 / 2 | 0 / 0.06640625 | 0.033203125 | 0.0469563096882 | [0, 0.06640625] | [-0.388682579131, 0.455088829131] |
| adaptive_resonance | uniform | B | 1 / 2 | 0 / 0 | 0 | 0 | [0, 0] | [0, 0] |
| adaptive_resonance | uniform | C | 1 / 2 | 0.224609375 / 0.1484375 | 0.1865234375 | 0.0538616493482 | [0.1484375, 0.224609375] | [-0.297404281944, 0.670451156944] |
| adaptive_resonance | random | pooled | 1 / 2 | 0.0748697916667 / 0.0826822916667 | 0.0787760416667 | 0.00552427172802 | [0.0748697916667, 0.0826822916667] | [0.029142429416, 0.128409653917] |
| adaptive_resonance | random | A | 1 / 2 | 0 / 0.154296875 | 0.0771484375 | 0.109104366628 | [0, 0.154296875] | [-0.903115404451, 1.05741227945] |
| adaptive_resonance | random | B | 1 / 2 | 0 / 0 | 0 | 0 | [0, 0] | [0, 0] |
| adaptive_resonance | random | C | 1 / 2 | 0.224609375 / 0.09375 | 0.1591796875 | 0.0925315514443 | [0.09375, 0.224609375] | [-0.672183317699, 0.990542692699] |
| adaptive_resonance | scanner | pooled | 1 / 2 | 0.0748697916667 / 0.147135416667 | 0.111002604167 | 0.0510995134842 | [0.0748697916667, 0.147135416667] | [-0.348108309152, 0.570113517485] |
| adaptive_resonance | scanner | A | 1 / 2 | 0 / 0.263671875 | 0.1318359375 | 0.186444170821 | [0, 0.263671875] | [-1.54329847596, 1.80697035096] |
| adaptive_resonance | scanner | B | 1 / 2 | 0 / 0 | 0 | 0 | [0, 0] | [0, 0] |
| adaptive_resonance | scanner | C | 1 / 2 | 0.224609375 / 0.177734375 | 0.201171875 | 0.0331456303681 | [0.177734375, 0.224609375] | [-0.0966297985041, 0.498973548504] |
| adaptive_resonance | learned | pooled | 1 / 2 | 0.0748697916667 / 0.158203125 | 0.116536458333 | 0.0589255650989 | [0.0748697916667, 0.158203125] | [-0.412888739007, 0.645961655674] |
| adaptive_resonance | learned | A | 1 / 2 | 0 / 0.27734375 | 0.138671875 | 0.196111646345 | [0, 0.27734375] | [-1.6233213599, 1.9006651099] |
| adaptive_resonance | learned | B | 1 / 2 | 0 / 0 | 0 | 0 | [0, 0] | [0, 0] |
| adaptive_resonance | learned | C | 1 / 2 | 0.224609375 / 0.197265625 | 0.2109375 | 0.0193349510481 | [0.197265625, 0.224609375] | [0.0372198571226, 0.384655142877] |

## Growth: all 18 stage decisions and retained expert counts

Indices are zero-based. K is the retained bank size after that stage; the fixed trajectory retains snapshots. These stage observations are not independent training replications. The two preallocation checks are construction identities, not extra seeds.

| Policy | Seed | Stage | Action | Parent | Target slot | K after stage | Resonance tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_novelty | 20 | A | create | none | 0 | 1 | none |
| adaptive_novelty | 20 | B | create | 0 | 1 | 2 | none |
| adaptive_novelty | 20 | C | create | 1 | 2 | 3 | none |
| adaptive_novelty | 21 | A | create | none | 0 | 1 | none |
| adaptive_novelty | 21 | B | create | 0 | 1 | 2 | none |
| adaptive_novelty | 21 | C | create | 1 | 2 | 3 | none |
| adaptive_resonance | 20 | A | create | none | 0 | 1 | 0.743478029966 |
| adaptive_resonance | 20 | B | update | 0 | 0 | 1 | 0.743478029966 |
| adaptive_resonance | 20 | C | update | 0 | 0 | 1 | 0.743478029966 |
| adaptive_resonance | 21 | A | create | none | 0 | 1 | 0.693457424641 |
| adaptive_resonance | 21 | B | create | 0 | 1 | 2 | 0.693457424641 |
| adaptive_resonance | 21 | C | update | 1 | 1 | 2 | 0.693457424641 |
| fixed_sequential | 20 | A | create | none | 0 | 1 | none |
| fixed_sequential | 20 | B | create | 0 | 1 | 2 | none |
| fixed_sequential | 20 | C | create | 1 | 2 | 3 | none |
| fixed_sequential | 21 | A | create | none | 0 | 1 | none |
| fixed_sequential | 21 | B | create | 0 | 1 | 2 | none |
| fixed_sequential | 21 | C | create | 1 | 2 | 3 | none |

### Adaptive creation margins

Margins are copied from the analyzer, not recomputed. A strictly positive rounded margin means creation; the sign convention already reverses for resonance. Numerical display rounding never drives decisions. Full score vectors and tie/lineage information remain in the source JSON and its artifact links.

| Policy | Seed | Stage | Selected raw | Selected round4 | Threshold | Raw margin | Rounded margin | Rounding changes comparison |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_novelty | 20 | B | 0.222439229488 | 0.2224 | 0.1 | 0.122439229488 | 0.1224 | false |
| adaptive_novelty | 20 | C | 0.185329854488 | 0.1853 | 0.1 | 0.0853298544884 | 0.0853 | false |
| adaptive_novelty | 21 | B | 0.297743052244 | 0.2977 | 0.1 | 0.197743052244 | 0.1977 | false |
| adaptive_novelty | 21 | C | 0.244574651122 | 0.2446 | 0.1 | 0.144574651122 | 0.1446 | false |
| adaptive_resonance | 20 | B | 0.774003982544 | 0.774 | 0.743478029966 | -0.0305259525776 | -0.0305219700336 | false |
| adaptive_resonance | 20 | C | 0.797236084938 | 0.7972 | 0.743478029966 | -0.0537580549717 | -0.0537219700336 | false |
| adaptive_resonance | 21 | B | 0.690889894962 | 0.6909 | 0.693457424641 | 0.00256752967834 | 0.00255742464066 | false |
| adaptive_resonance | 21 | C | 0.752250432968 | 0.7523 | 0.693457424641 | -0.0587930083275 | -0.0588425753593 | false |

## Growth: all 28 pooled binding contrasts

Differences use the analyzer's left-minus-right convention. Saved means, sample SDs and t95 intervals are retained; ranges are derived only as min/max of its two saved seed differences. The complete growth contrast CSV contains 448 rows: 28 comparisons x 4 topic groups x 4 metrics. Negative cross-entropy differences favor the left arm, unlike accuracy differences.

| Contrast | Delta 20 / 21 | Mean | SD | Range | t95, df1 |
| --- | --- | --- | --- | --- | --- |
| snapshots:scanner_minus_uniform | 0.1640625 / 0.145182291667 | 0.154622395833 | 0.0133503233427 | [0.145182291667, 0.1640625] | [0.0346744995584, 0.274570292108] |
| snapshots:scanner_minus_random | 0.147135416667 / 0.145182291667 | 0.146158854167 | 0.001381067932 | [0.145182291667, 0.147135416667] | [0.133750451104, 0.15856725723] |
| snapshots:scanner_minus_learned | 0.00390625 / -0.013671875 | -0.0048828125 | 0.012429611388 | [-0.013671875, 0.00390625] | [-0.116558440066, 0.106792815066] |
| duplicate_slots:scanner_minus_uniform | 0.0970052083333 / 0.0872395833333 | 0.0921223958333 | 0.00690533966002 | [0.0872395833333, 0.0970052083333] | [0.0300803805187, 0.154164411148] |
| duplicate_slots:scanner_minus_random | 0.0813802083333 / 0.0774739583333 | 0.0794270833333 | 0.00276213586401 | [0.0774739583333, 0.0813802083333] | [0.0546102772075, 0.104243889459] |
| duplicate_slots:scanner_minus_learned | 0.001953125 / -0.005859375 | -0.001953125 | 0.00552427172802 | [-0.005859375, 0.001953125] | [-0.0515867372517, 0.0476804872517] |
| adaptive_novelty:scanner_minus_uniform | 0.1640625 / 0.145182291667 | 0.154622395833 | 0.0133503233427 | [0.145182291667, 0.1640625] | [0.0346744995584, 0.274570292108] |
| adaptive_novelty:scanner_minus_random | 0.147135416667 / 0.145182291667 | 0.146158854167 | 0.001381067932 | [0.145182291667, 0.147135416667] | [0.133750451104, 0.15856725723] |
| adaptive_novelty:scanner_minus_learned | 0.00390625 / -0.013671875 | -0.0048828125 | 0.012429611388 | [-0.013671875, 0.00390625] | [-0.116558440066, 0.106792815066] |
| adaptive_resonance:scanner_minus_uniform | 0 / 0.0755208333333 | 0.0377604166667 | 0.0534012933709 | [0, 0.0755208333333] | [-0.442031168433, 0.517552001766] |
| adaptive_resonance:scanner_minus_random | 0 / 0.064453125 | 0.0322265625 | 0.0455752417562 | [0, 0.064453125] | [-0.377250738576, 0.441703863576] |
| adaptive_resonance:scanner_minus_learned | 0 / -0.0110677083333 | -0.00553385416667 | 0.00782605161469 | [-0.0110677083333, 0] | [-0.0758481381899, 0.0647804298566] |
| snapshots_uniform_minus_single_final_uniform | -0.005859375 / 0.013671875 | 0.00390625 | 0.01381067932 | [-0.005859375, 0.013671875] | [-0.120177780629, 0.127990280629] |
| snapshots_minus_duplicate_slots:uniform | 1.38777878078e-17 / -0.00130208333333 | -0.000651041666667 | 0.00092071195467 | [-0.00130208333333, 1.38777878078e-17] | [-0.00892331037528, 0.00762122704195] |
| adaptive_novelty_minus_fixed_snapshots:uniform | 0 / 0 | 0 | 0 | [0, 0] | [0, 0] |
| adaptive_resonance_minus_fixed_snapshots:uniform | 0.005859375 / -0.0078125 | -0.0009765625 | 0.00966747552403 | [-0.0078125, 0.005859375] | [-0.0878353839405, 0.0858822589405] |
| snapshots_random_minus_single_final_uniform | 0.0110677083333 / 0.013671875 | 0.0123697916667 | 0.00184142390934 | [0.0110677083333, 0.013671875] | [-0.00417474575056, 0.0289143290839] |
| snapshots_minus_duplicate_slots:random | 0.00130208333333 / -0.0110677083333 | -0.0048828125 | 0.00874676356936 | [-0.0110677083333, 0.00130208333333] | [-0.0834693652318, 0.0737037402318] |
| adaptive_novelty_minus_fixed_snapshots:random | 0 / 0 | 0 | 0 | [0, 0] | [0, 0] |
| adaptive_resonance_minus_fixed_snapshots:random | -0.0110677083333 / 0.00325520833333 | -0.00390625 | 0.0101278315014 | [-0.0110677083333, 0.00325520833333] | [-0.0949012057948, 0.0870887057948] |
| snapshots_scanner_minus_single_final_uniform | 0.158203125 / 0.158854166667 | 0.158528645833 | 0.000460355977335 | [0.158203125, 0.158854166667] | [0.154392511479, 0.162664780188] |
| snapshots_minus_duplicate_slots:scanner | 0.0670572916667 / 0.056640625 | 0.0618489583333 | 0.00736569563736 | [0.056640625, 0.0670572916667] | [-0.00432919133558, 0.128027108002] |
| adaptive_novelty_minus_fixed_snapshots:scanner | 0 / 0 | 0 | 0 | [0, 0] | [0, 0] |
| adaptive_resonance_minus_fixed_snapshots:scanner | -0.158203125 / -0.0774739583333 | -0.117838541667 | 0.0570841411895 | [-0.158203125, -0.0774739583333] | [-0.630719201601, 0.395042118267] |
| snapshots_learned_minus_single_final_uniform | 0.154296875 / 0.172526041667 | 0.163411458333 | 0.0128899673654 | [0.154296875, 0.172526041667] | [0.0475996964127, 0.279223220254] |
| snapshots_minus_duplicate_slots:learned | 0.0651041666667 / 0.064453125 | 0.0647786458333 | 0.000460355977335 | [0.064453125, 0.0651041666667] | [0.060642511479, 0.0689147801876] |
| adaptive_novelty_minus_fixed_snapshots:learned | 0 / 0 | 0 | 0 | [0, 0] | [0, 0] |
| adaptive_resonance_minus_fixed_snapshots:learned | -0.154296875 / -0.080078125 | -0.1171875 | 0.0524805814162 | [-0.154296875, -0.080078125] | [-0.588706816391, 0.354331816391] |

## Scope and unavailable measurements

All comparisons remain exploratory; H1 is not reconsidered. Two seeds do not establish generalization across training randomness or datasets. Saved example-bootstrap intervals concern their individual checkpoints and are not substituted for seed-level intervals. Adaptive contrasts condition on realized K and lineage; ABC versus ABB matches nominal slots, not unique functional capacity. Router fitting adds supervised optimization. Neither exclusive gate-fit elapsed time nor peak process RSS was measured. Available-RAM minima are minima of recorded samples, not continuous minima. No FLOPs, energy or physical inference-latency claim is introduced.
