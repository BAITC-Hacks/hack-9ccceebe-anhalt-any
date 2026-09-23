# Team integration checklist

Initial repository audit: `main` contained only README, commit `794eed5`.
No existing datasets, dependencies, model, .gitignore or teammate branches were present.
Work is isolated in `codex/energy-agent-mvp`.

1. Person A/B publish their work to team branches/PRs; provide branch names.
2. Review their diff and artifacts for credentials, units, schema, model training cutoff,
   dependency changes and weather provenance. Do not copy source files by hand.
3. Use the team's chosen merge or cherry-pick workflow; do not merge unrelated branches automatically.
   Example after reviewing a branch: `git merge --no-ff origin/<reviewed-branch>`.
4. Set DATA_MODULE, ML_MODULE, MODEL_PATH and agreed feature schema. Update adapters if names differ.
5. Install merged dependencies, check imports, then run:

```bash
python -m pytest -q
python scripts/run_forecast.py --date 2026-02-10 --turbine T1 --horizon 24 --no-agent
python scripts/run_backtest.py --turbine T1 --actuals data/actuals.csv
```

6. Run one `--with-agent` forecast only with a configured key/model; verify structured output and fallback.
7. Smoke-test Streamlit. Review .gitignore and staged changes. Keep .env, credentials and private data out of git.
8. Merge the reviewed integration through the team's workflow.

Outstanding external dependencies: Person A weather API adapter, Person B real trained model,
actual generation, training/validation metadata, verified turbine cut-in/cut-out thresholds,
OpenAI credentials and a model available to the team's account.
