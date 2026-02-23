# PUMS Enrichment Pipeline

Key Phase 2 controls for large runs:

- --phase2-sub-batch N: Bound per-shard sub-batch size (defaults ~500–2000).
- --phase2-max-candidates N: Cap candidate matches per PUMS record (overrides config).

Examples:

- Run Phase 2 streaming with tighter sub-batches:
	pwsh.exe -File main.py --phase 2 --streaming --phase2-sub-batch 800 --sample-size 100000

- Cap candidate list size to 150 per record:
	pwsh.exe -File main.py --phase 2 --streaming --phase2-max-candidates 150 --sample-size 100000

Notes:

- Phase 2 strictly consumes Phase 1 output/shards.
- Comparison outputs are numeric-only by default to avoid object-dtype blowups.
