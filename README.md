# hack-9ccceebe-anhalt-any
Hackathon team repository for Anhalt ANY


## ML MODE A

Trained Kelmarsh weather-to-power model, data and evaluation artifacts.

- [ML setup and usage](README_ML.md)
- [Work report (Russian)](reports/WORK_REPORT_RU.md)
- [Dataset provenance and license](data/README.md)
- [Evaluation](reports/evaluation.md)

Run `python -m scripts.evaluate_model data/holdout.csv --mode A` from this branch root.
Agent/API integration and transfer to other turbine types require a separate adapter and validation.
