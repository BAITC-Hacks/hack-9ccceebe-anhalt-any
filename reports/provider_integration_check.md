# Provider integration check — 2026-09-23

No retraining and no edits to `models/power_model.joblib`, its metadata, or holdout.
Existing branch started clean at `5f7b569`; no uncommitted work needed stashing.
Remote fetched before implementation. No AGENTS.md found in workspace ancestors or either branch.

## Results

- Standalone ML suite: **61 passed, 1 skipped**, DeprecationWarning treated as error.
  The skip is the optional agent-integration module: standalone ML does not own/install agent code.
- Agent integration overlay: **2 passed** against unmodified MVP source
  `bc67f05864a2a5e7a6ef12c060872c9ccfc427c7`.
- `python -m scripts.verify_ml_provider --report reports/provider_verification.json`:
  frozen hashes verified; 47006 holdout rows reproduce original MODE A raw and clipped metrics.
- Provider raw kW MAE: 82.9046933529376; RMSE: 145.1903279207735; R²: 0.9530159122008534.
- Real provider holdout outputs include 210 negative and 1 above-rated prediction; retained unchanged.

## What the integration tests prove

`tests/test_provider_agent_contract.py` runs unchanged in a checkout containing both agent and ML.
For this check, a git archive of the MVP ref was unpacked to a separate work directory and the ML
files overlaid. Neither branch's agent/orchestrator/rules implementation was modified.

Test fixture forces raw normalized values [-0.1, 1.2, NaN] through the real saved model wrapper,
the provider, deterministic rules and unmodified orchestrator. Rules observe [-205, 2460, NaN] kW
and emit negative_power, above_rated_power and non_finite_power; JSON represents NaN as null.
No clipping or silent energy total is introduced. The old DataFrame API remains clipped by default.

The fixture explicitly supplies Kelmarsh context in feature attrs. A second call removes turbine_id
and confirms that the current MVP fails closed: actual context handoff still belongs to C.
Weather loading is mocked in this plumbing test; no weather API, LLM call, forecast metric or
Goldwind transfer claim is made. Hourly test fixtures test the interface only, not model accuracy.

Standalone verification environment: published Python 3.13.13 training environment.
Overlay environment: Python 3.12, NumPy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1, joblib 1.6.0,
pydantic 2.13.5, python-dotenv 1.2.3, OpenAI SDK 2.54.0 (import only; no API calls).

## C: integration handoff and merge ownership

Target is the current open MVP branch `codex/energy-agent-mvp`; main still contains only the initial README.
Contract proposal was posted on PR #1; confirmation of attrs and additional direction field is pending.
Draft adapter PR is for review, not permission to merge or deploy to Goldwind.

A read-only `git merge-tree` check identified pre-existing branch-root conflicts:

| File | Suggested resolution by C |
|---|---|
| README.md | Keep MVP overview, link README_ML.md and docs/ML_INFERENCE_CONTRACT.md; update stale "no ML data" text. |
| .gitignore | Union both lists; retain .env protection and raw ZIP exclusion. Keep the intentionally tracked model/reports. |
| requirements.txt | Union compatible agent/ML dependencies, including matplotlib and tzdata; preserve a training-compatible sklearn environment. |

These common files were not rewritten by B to disguise conflicts; no merge or force-push was performed.
Source ML branch contains no edits to src/agent, src/api, src/app or Person A weather implementation.
Required C changes: authoritative turbine context, preservation of wind height, a Kelmarsh registry,
and an honest observed-SCADA scenario/cadence path. A must supply native direction semantics or explicitly
resolve the provider convention; do not silently substitute a yaw angle or fabricated forecast.
