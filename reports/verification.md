# Проверка реализации — 2026-09-23

Windows, Python 3.12; pandas 2.3.3; NumPy 2.5.3; scikit-learn 1.9.1;
LightGBM 4.7.0; joblib 1.6.0; matplotlib 3.11.2; pytest 9.1.1.
Точные проверенные зависимости: requirements-tested.txt (Windows / Python 3.12).

**30 tests passed**, включая запуск с `-W error::DeprecationWarning`. Проверены:

- Групповой временной split без пересечения timestamps.
- Исключение target/непроверенных lag-колонок из признаков, train-only vocabulary.
- Неизменность predictions и выбора модели при изменении только test-target.
- LightGBM callbacks early stopping и fallback при отсутствии LightGBM.
- Global/per-turbine comparison, unknown turbine fallback.
- Absolute/normalized target, raw predictions до clipping, joblib round-trip.
- Пропуски, отрицательный wind speed, infinity, timezone/units/mapping, дубликаты.
- Forecast vintage, lead time, cutoff model selection, отсутствие boundary gap.
- Circular mean, неопределённое противоположное направление, час/день/месяц.
- Генерация четырёх PNG, EDA/metrics/importance reports.
- Понятные CLI-ошибки для отсутствующих config/dataset и незаполненного target_source;
  чтение Windows JSON с UTF-8 BOM.
- Evaluation CLI проверяет наличие trained artifact и holdout до загрузки модели.

Тестовая power curve визуально проверена. Все искусственные модели, графики и метрики
этого прогона находятся только в рабочей test-директории, не входят в deliverable/production reports.
Предупреждения Timedelta устранены явным `np.timedelta64(..., "h")` в test fixtures;
в последнем прогоне предупреждений нет. Предупреждение устаревающего LightGBM eval_set устранено:
используется eval_X/eval_y при наличии в сигнатуре и eval_set для прежних версий.

Из-за особенностей sandbox ACL стандартная установка через venv/ensurepip не завершилась.
Для проверки зависимости установлены отдельно в work/python_packages, без изменения системного Python;
временный bootstrap обходит Windows ACL временных директорий. Этот workaround не является
зависимостью ML-проекта и не входит в архив. Обычная установка описана в README.

Это техническая проверка реализации, не validation на реальном энергетическом dataset.
Не проверены: реальная точность, NASA network integration, качество transfer на новые турбины,
архивные operational forecasts, физические cut-in/rated/cut-out области.
