# Реальное обучение Kelmarsh 2017 — 2026-09-23

Источник: https://doi.org/10.5281/zenodo.16807551, Charlie Plumley и Roberta Takeuchi,
Cubico Sustainable Investments Ltd, CC BY 4.0. Подготовка и исключения — data/preparation_report.json.
Исходный архив и паспортные данные сохранены, контрольные суммы совпали с Zenodo.

311564 пригодных observations, шесть турбин, timestamp UTC, номинал 2050 kW/турбину.
Цель — Power (kW) / 2050. Выборка делится по времени 70/15/15 без перемешивания.
Общая HistGradientBoosting выбрана по validation RMSE:

| Кандидат | Validation normalized RMSE |
|---|---:|
| Mean | 0.301220 |
| HistGradientBoosting, общая модель | 0.038160 |
| LightGBM, общая модель | 0.045196 |
| Отдельные модели по турбинам | 0.041216 |

Test — 47006 observations с 2017-11-07 13:10 до 2017-12-31 23:50 UTC.
На test модель не подбиралась. Загрузка сохранённого artifact отдельной командой evaluation
в пользовательском Python 3.13 environment воспроизвела метрики.

| Метрика | Raw | После clipping [0,1] |
|---|---:|---:|
| normalized MAE | 0.040441 | 0.040398 |
| normalized RMSE | 0.070825 | 0.070811 |
| R² | 0.953016 | 0.953033 |

После clipping MAE ≈82.82 kW и RMSE ≈145.16 kW на турбину. Это доли номинальной
мощности, не MAPE. Для test используются фактические SCADA weather observations: **MODE A**.
MODE B не оценивался — archived forecasts отсутствуют.

## Физическая проверка и ограничения

На power_curve.png видна область низкой мощности примерно до 3 m/s, нелинейный рост
примерно 3–11/12 m/s и скопление работающих турбин около номинала при более сильном ветре.
Это наблюдения по scatter/эмпирическим bins, а не подтверждённые паспортные cut-in/cut-out thresholds.
Нулевые мощности встречаются и при сильном ветре; чтобы различить простои, curtailment и неисправности,
нужны status/events из исходного архива. Их причины модель только по weather не определяет.
Данных test при 20+ m/s нет: cut-out и экстремальные режимы на test не оценены.

Наибольшая MAE — 12–20 m/s: 0.09320 номинала (1948 observations).
В 8–12 m/s MAE=0.05949 (15931 observations); в 3–8 m/s MAE=0.02768 (27483).
Кривая predictions при сильном ветре занижена и немонотонна: это недостаток текущей модели,
а не доказанный физический cut-out. Настройка по этому test потребовала бы нового независимого holdout.
Начальная версия пригодна для интеграции/backtesting, но не подтверждает перенос на другие ветропарки.

В полном dataset сохранены 28014 отрицательных и 3400 выше-номинальных наблюдений.
Raw prediction и метрики отдельно от clipping позволяют увидеть его эффект.
Наиболее важные признаки по permutation importance: wind_speed, temperature, wind_dir_cos,
turbine_0, day_of_year_sin. Значимость рассчитана диагностически на ограниченной test-подвыборке;
она не доказывает причинность.

Artifacts: models/power_model.joblib, models/power_model.metadata.json.
Четыре PNG, metrics.json, EDA, ошибки по ветру/турбине/сезону и feature importance — в reports/.
Версии обучения: requirements-trained.txt и metadata модели; тестовая среда описана отдельно.
