# Reviewed contraction plan evidence

These files are the plans used for the recorded Week 9 results and figures. They contain index scopes, contraction steps, retained indices, work, tensor sizes, method and search settings. They do not contain downloaded UAI factor tables, raw BP traces or complete run records.

| Directory | Evidence |
| --- | --- |
| `asia-ad-random/` | AD evidence scalar plan: random order, seed 42, plus per-step scalar complexity. |
| `asia-ad-greedy/` | AD evidence scalar plan: omeco greedy order, plus per-step scalar complexity. |
| `promedus-greedy/` | Greedy scalar plan and scalar complexity; query plans for variable IDs 59, 102 and 139. |
| `promedus-treesa/` | Time-optimized TreeSA scalar plan and scalar complexity; query plans for IDs 59, 102 and 139, with seed and search settings in the plans. |

Each `order.json` is a scalar evidence plan. Each `complexity.json` gives its independently recountable step costs. For query plans the work and tensor sizes are recorded directly in `query-<id>-order.json`. `manifest.json` lists the 14 files, their original local paths and SHA-256 checksums.

For these recorded scalar plans:

| Plan | Work | Largest input or intermediate | Largest generated intermediate |
| --- | ---: | ---: | ---: |
| ASIA AD random | 200 | 32 | 32 |
| ASIA AD greedy | 38 | 8 | 4 |
| Promedus greedy | 5,492,329,020 | 16,777,216 | 16,777,216 |
| Promedus TreeSA | 10,538,082 | 131,072 | 131,072 |

Run the following from the week9 directory. The first command independently recomputes all plan costs and checksums using only the saved evidence and Python's standard library. The second also checks that every saved file is byte-identical to its current ignored run source.

```bash
python3 scripts/export_plan_evidence.py --verify
python3 scripts/export_plan_evidence.py
```

To replay or analyze a saved scalar plan after installing `infer` and downloading the data as described in the main README:

```bash
infer --model asia.uai --evidence AD --analyze evidence/plans/asia-ad-random/order.json --out artifacts/plan-analysis
infer --model asia.uai --evidence AD --plan evidence/plans/asia-ad-greedy/order.json --out artifacts/plan-replay
infer --model promedus/Promedus_14.uai --evidence promedus/Promedus_14.uai.evid --analyze evidence/plans/promedus-treesa/order.json --out artifacts/promedus-plan-analysis
```

The current CLI accepts `--plan` and `--analyze` for **scalar** plans. It does not accept a query plan through those options. The query plans are still independently checked for costs by `--verify`; `scripts/audit_exact_paths.py` also replays the TreeSA query plans from the local run directory with NumPy.

These files are snapshots. Rerunning the inference may choose different valid greedy trees because of tie-breaking. After intentionally rerunning the ASIA and Promedus experiments, run the checks and use `python3 scripts/export_plan_evidence.py --refresh` to update these snapshots, then regenerate the plots and review their reported costs together. The update command never commits or pushes.
