# Evaluation: MODE A

Observed weather -> power; does not measure weather forecast skill.

Target: normalized_power; units: Normalized power.
Test: 2017-11-07T13:10:00+00:00 through 2017-12-31T23:50:00+00:00.
Selected candidate: hist_baseline; turbine strategy: global.
Model fitted on training only; validation used for selection/early stopping; test held out.

Served metrics: {"MAE": 0.04039838023174107, "RMSE": 0.07081149638978826, "R2": 0.9530332300684281, "normalized_MAE": 0.04039838023174107}
Raw metrics before clipping: {"MAE": 0.040441313830701264, "RMSE": 0.07082455020525537, "R2": 0.9530159122008534, "normalized_MAE": 0.040441313830701264}
Targets below zero: 28014; above nominal: 3400 (full dataset).
Largest observed wind-bin MAE: 12-20 m/s, MAE=0.0931966, n=1948.
Importance: see feature_importance.csv; top entries: wind_speed, temperature, wind_dir_cos, turbine_0, day_of_year_sin.

Wind bins are diagnostic groups, not inferred cut-in/rated/cut-out thresholds.
Use empirical_power_curve.csv and turbine specifications/operating status to assess these regions.
Missing regimes/seasons cannot be evaluated. Irregular sampling and temporal correlation limit metric interpretation.
No uncertainty intervals or forecast-skill claims are produced from mock data.
