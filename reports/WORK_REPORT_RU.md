# Отчёт о выполненной работе — HackAlem AI Energy Agent

Дата: 23 сентября 2026 года.
Репозиторий: https://github.com/BAITC-Hacks/hack-9ccceebe-anhalt-any
Ветка публикации: `ml-model-mode-a`. Ветка `main` и agent/API-ветка не изменяются.

## Результат

Создан и проверен локальный ML-слой для зависимости наблюдаемая погода → активная мощность.
Скачан реальный SCADA dataset, выполнены подготовка, временная оценка и обучение.
Сохранена воспроизводимая модель с metadata, подготовлены API, CLI, тесты и отчёты.
Результаты относятся к **MODE A**, а не к прогнозированию с архивной forecast weather.

## Данные

- Kelmarsh SCADA 2017, официальная публикация https://doi.org/10.5281/zenodo.16807551.
- Авторы Charlie Plumley и Roberta Takeuchi; Cubico Sustainable Investments Ltd; CC BY 4.0.
- Шесть Senvion MM92 по 2050 kW; номинал подтверждён паспортным CSV.
- UTC, шаг 10 минут, период 2017-01-01–2017-12-31.
- 315360 исходных строк; 311564 оставлены; 3796 без ветра или мощности исключены с построчным отчётом.
- 28014 отрицательных и 3400 выше-номинальных значений мощности сохранены.
- Цель — фактическая Power (kW), нормированная на 2050 kW. NASA не использовался как power target.
- Подготовленные CSV включены в Git. Исходный ZIP 174.6 MB остаётся у официального издателя
  и локально: в обычный Git не включён из-за размера. `python -m scripts.download_kelmarsh`
  восстанавливает его и проверяет MD5; `python -m scripts.prepare_kelmarsh` воспроизводит преобразование.

## ML-реализация

- API: train_model, predict, save_model, load_model; preprocessing сохраняется вместе с estimator.
- Features: wind_speed, temperature, sin/cos wind direction и времени, train-only turbine encoding.
- Split 70/15/15 по уникальным timestamp без shuffle; test не используется для выбора.
- Сравнение mean baseline, HistGradientBoosting, LightGBM с early stopping и per-turbine моделей.
- Автоматический sklearn fallback при недоступном LightGBM.
- Raw prediction сохраняется до normalized clipping [0,1].
- Hourly/daily/monthly aggregation с circular mean направления.
- MODE B contract проверяет issued_at, lead time и временные границы; фактических MODE B metrics нет.

## Полученные результаты

Выбрана **общая HistGradientBoostingRegressor**. Validation normalized RMSE:

| Вариант | RMSE |
|---|---:|
| Mean baseline | 0.301220 |
| Общая HistGradientBoosting | 0.038160 |
| Общая LightGBM | 0.045196 |
| Отдельные модели турбин | 0.041216 |

Test: 47006 строк, 2017-11-07 13:10 — 2017-12-31 23:50 UTC.

| Метрика | Raw | После clipping |
|---|---:|---:|
| Normalized MAE | 0.040441 | 0.040398 |
| Normalized RMSE | 0.070825 | 0.070811 |
| R² | 0.953016 | 0.953033 |

После clipping MAE ≈82.82 kW, RMSE ≈145.16 kW на турбину.
Это не MAPE: проценты обозначают долю номинальной мощности.
Наибольшая ошибка — 12–20 m/s; при сильном ветре наблюдается занижение prediction.
Test не покрывает ветер 20+ m/s и весь год по сезонам. Без дополнительного holdout
не следует заявлять качество на экстремальных режимах или донастраивать модель по текущему test.

## Проверки и исправления

- 30 автоматических тестов: leakage, временной split, fallback, сериализация, MODE B origins,
  units/timezone, пропуски, дубликаты, нормализация, circular mean и генерация отчётов.
- Отдельный evaluation CLI загрузил сохранённую модель и воспроизвёл test metrics.
- Исправлены DeprecationWarning timedelta явными единицами; тесты проверялись с warnings-as-errors.
- CLI выдаёт понятные сообщения при отсутствующих config, dataset, model и holdout.
- Убрана вводившая в заблуждение команда с YOUR_FORECAST_HOLDOUT.csv.
- Проверены реальные графики power curve; сохранены EDA, feature importance, ошибки по ветру,
  турбинам и сезонам. Зависимости обучения записаны в requirements-trained.txt.

## Состав ветки

Код src/ml, CLI scripts, tests, config, requirements, примеры входа, обученный joblib,
metadata, подготовленные CSV, source attribution и отчёты/PNG. Окружения .venv,
кэши, секреты и большой raw ZIP не включаются. Корневой README команды сохраняется;
ML-инструкция размещается в README_ML.md.

## Интеграция с командой: что ещё требуется

Проверена отдельная ветка `codex/energy-agent-mvp`; её файлы не изменены и не объединены.
Контракты сейчас различаются: agent ожидает ndarray мощности в kW и weather names
wind_speed_ms/temperature_c с DatetimeIndex, тогда как ML API возвращает DataFrame
и использует timestamp/wind_speed/temperature/wind_direction/turbine_id.
Agent также ожидает отсутствие clipping, а текущий публичный ML output по умолчанию clipped;
raw_prediction доступен отдельно. Перед подключением нужен согласованный adapter.

В agent registry указаны Goldwind GW109/2500 (T1/T2), а модель обучена на Kelmarsh Senvion MM92.
Нельзя просто переименовать турбины или масштабировать номинал и считать перенос проверенным.
Нужна валидация/переобучение на целевом парке; при отсутствии данных допускается только
явно помеченная демонстрация на Kelmarsh. Источник будущей/архивной forecast weather
и корректная оценка MODE B остаются отдельной задачей Person A и команды.

## Запуск из корня ML-ветки

```powershell
python -m pip install -r requirements-trained.txt
python -m pytest -q
python -m scripts.evaluate_model data/holdout.csv --mode A
```

Для повторного обучения:

```powershell
python -m scripts.train_model data/scada.csv --config config.kelmarsh.json
```

Модель уже обучена; повторное обучение для проверки API не требуется.
