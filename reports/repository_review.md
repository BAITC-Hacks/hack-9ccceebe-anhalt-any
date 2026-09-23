# Первичное исследование — 2026-09-23

Исследована предоставленная рабочая директория. До реализации присутствовали только пустые outputs/ и work/.

| Объект | Результат |
|---|---|
| Git / исходники / requirements / pyproject | Не найдены |
| Data/weather/NASA pipeline | Не найден |
| CSV / Parquet / SCADA / historical target | Не найдены |
| Existing ML / tests | Не найдены |
| Timestamp / timezone / date range / frequency | Не определяются без dataset |
| Turbine count / rated power / units | Не подтверждены |
| Missing values / power curve regimes | Не определяются без dataset |

Решение: минимальный изолированный ML-проект в outputs/energy_ml; адаптерный contract для Person A,
стабильный API для Person B. Реальные модели и метрики не генерировались.

Для следующего шага: реальный dataset с power или подтверждённым normalized_power,
единицы, timezone и provenance. NASA погода не заменяет фактическую мощность турбины.
