# HackAlem AI Energy Agent — ML layer

## Goldwind T1/T2: обучение подготовлено, финальной модели пока нет

`src.ml.goldwind_provider` — отдельный провайдер raw кВт для T1/T2. Полученные почасовые
CSV требуют подтверждения времени, единиц, нормализации, покрытия и идентичности турбин.
Команды подготовки/обучения/оценки, настройки и блокировки описаны в
[GOLDWIND_MODEL_CONTRACT](docs/GOLDWIND_MODEL_CONTRACT.md).
Kelmarsh ниже остаётся отдельным проверенным примером, не заменой целевой модели.

## Agent provider (Person B)

`ML_MODULE=src.ml.provider`, `MODEL_PATH=models/power_model.joblib`.
Точный DATA-контракт, согласование A/C и ограничения: [ML_INFERENCE_CONTRACT](docs/ML_INFERENCE_CONTRACT.md).
Адаптер принимает DatetimeIndex, три исходных погодных столбца и явный turbine context в attrs;
возвращает **raw кВт без clipping**. Существующий DataFrame API не изменён.
Текущий orchestrator ещё должен передавать turbine_id: одного изменения ML_MODULE недостаточно.
T1/T2 Goldwind отклоняются; перенос модели Kelmarsh на них не подтверждён.

Проверка опубликованного artifact без переобучения:

```powershell
python -m scripts.verify_ml_provider --report reports/provider_verification.json
```

## Статус

Рабочая папка при исследовании содержала только пустые `outputs/` и `work/`.
Git repository, data/weather/NASA modules, requirements, datasets и tests отсутствовали.
Поэтому проект создан здесь с минимальной структурой. Чужие модули не изменялись.

Теперь скачан реальный **Kelmarsh SCADA 2017**: шесть турбин, шаг 10 минут, UTC,
311564 пригодных строки, наблюдаемая активная мощность в kW. Исходный архив сохранён,
контрольные суммы проверены. Происхождение, CC BY 4.0 и все исключения описаны в `data/README.md`.
Файлы `data/scada.csv`, `data/holdout.csv` и заполненный `config.kelmarsh.json` готовы к запуску.
`config.json` также заполнен, если ранее он был пустым шаблоном.
Обучение выполнено; `models/power_model.joblib` и отчёты созданы. Выбрана общая
HistGradientBoosting по validation RMSE. Test MODE A: MAE=0.04040, RMSE=0.07081,
R²=0.95303 в normalized target. Это оценка по наблюдаемой погоде, не operational forecast.
Проверена загрузка artifact командой evaluate_model; environment обучения — `requirements-trained.txt`.

Примеры JSON по-прежнему вымышленные weather inputs без target; кривая внутри tests служит
только проверке кода. Реальное обучение использует исключительно target из скачанного SCADA.

## Windows: установка и запуск

В PowerShell, из директории этого проекта:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Необязательная основная модель; без неё автоматически используется sklearn:
.\.venv\Scripts\python.exe -m pip install -r requirements-lightgbm.txt
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
```

Для скачанного Kelmarsh используйте заполненный `config.kelmarsh.json` или `config.json`.
Для другого dataset создайте отдельный config из `examples/config.json`, указав подтверждённые
target_source, power_unit, timezone и mapping. Null в шаблоне намеренно блокирует обучение.

```powershell
.\.venv\Scripts\python.exe -m scripts.train_model data/scada.csv --config config.json
.\.venv\Scripts\python.exe -m scripts.evaluate_model data/holdout.csv --mode A
```

MODE B пока недоступен на скачанном dataset. Он требует отдельного реального архива
прогнозов с issued_at, наблюдаемой мощностью для valid timestamp и указанным источником прогноза.
SCADA holdout содержит фактическую погоду и подходит только для MODE A.

Для Parquet отдельно установите `pyarrow`; для CSV он не нужен. CLI запускаются через `-m`
из корня проекта. После установки пакета через `pip install -e .` API доступен извне.
Training создаёт `models/power_model.joblib` и `models/power_model.metadata.json`.
**Оценку запускайте только после успешного обучения.** Запуск pytest не создаёт production artifact:
тестовые модели существуют только во временных директориях. Если обучение завершилось ошибкой,
`models/power_model.joblib` не появится. Для оценки нужны и сохранённая модель, и реальный holdout.
Если модель уже есть в другом
месте, укажите её явно через `--model "полный/путь/power_model.joblib"`.
После скачивания Kelmarsh файлы `data/scada.csv` и `data/holdout.csv` существуют;
`data/forecast_holdout.csv` по-прежнему отсутствует: архивных прогнозов нет.
Все turbine-модели и глобальный fallback сохраняются в одном переносимом bundle.
Joblib загружайте только из доверенного источника. Для воспроизводимости сохраняйте environment
(`pip freeze`), конфигурацию и неизменяемую версию исходного dataset; версии библиотек также в metadata.

## API для Person B

```python
import pandas as pd
from src.ml import MLConfig, train_model, save_model, load_model, predict, aggregate_predictions

config = MLConfig(
    columns={"timestamp": "Date/Time", "wind_speed": "Wind Speed (m/s)",
             "temperature": "Ambient Temperature", "power": "Active Power (kW)"},
    timezone="UTC",  # заменить на документированную timezone источника
    power_unit="kW",
    target_source="SCADA export / actual dataset identifier",
    weather_source="measured SCADA weather",
)
model = train_model(pd.read_csv("data/scada.csv"), config)
save_model(model, "models/power_model.joblib")

model = load_model("models/power_model.joblib")
predictions = predict(model, features)  # DataFrame в schema источника из config
daily = aggregate_predictions(predictions, "daily", timezone="UTC")
```

В API нет зависимости от выбранного backend. `train_model` возвращает PowerModel;
`predict` — DataFrame с исходным порядком/index, UTC timestamp, turbine_id,
weather-полями, raw_prediction и predicted_power и/или predicted_normalized_power.
raw_prediction имеет единицы target (при normalized target это доля, не kW).
Time features извлекаются в config.timezone; naive timestamps локализуются в неё,
aware timestamps переводятся в UTC. Неоднозначные DST timestamps требуют исправления upstream.

## Data contract для Person A

Mapping имеет направление **logical_name → source_column**. Явно указывайте единицы:
wind speed `m/s` или `km/h`, temperature `C` или `K`, power/rated power в одинаковых `W/kW/MW`.
Timestamp — момент валидности измерения/прогноза, не момент загрузки файла.
Интерфейс `WeatherProvider` и выбор доступного forecast vintage находятся в `src/ml/adapters.py`.
Можно напрямую передавать результат существующего loader как DataFrame.

NASA/weather не является target. Второй downloader не создан: автономные проверки ML
не требуют API и географические координаты не предоставлены. NASA POWER возвращает
историческую метеорологию; её нельзя обозначать archived operational forecast.
Перед подключением Person A должен подтвердить доступность параметров для выбранной temporal API
и высоту, близкую к hub height. В официальном каталоге подтверждены WS50M, WD50M и T2M;
100 m параметр в исследованных документах не подтверждён, имя WS100M не предполагается.
Если используется 50 m, это метеопризнак на 50 m, без неявной экстраполяции на hub height.
Сохраните фактически запрошенные имена в `nasa_parameters`, источник в `weather_source`;
явно задайте UTC (у hourly API есть также Local Solar Time). Fill values API обрабатываются upstream,
не превращаются в реальные отрицательные температуру/ветер. NASA API не требует ключа для обычных запросов.

Официальные источники:
- https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
- https://gis.earthdata.nasa.gov/portal/home/item.html?id=8f19b884dbfd4441beeb2ae9e26abae5
- https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html

## Target и качество данных

- power + подтверждённый rated_power → normalized_power. Для подтверждения задайте
  `rated_power_verified=true`, `rated_power_source` и `rated_power` (число или dict turbine_id → capacity).
  Если capacity берётся из колонки, она нужна и при inference. При меняющейся capacity передавайте колонку.
- power без подтверждения capacity → absolute_power; номинал из максимума power не оценивается.
- только normalized_power → требуется `normalized_target_confirmed=true` и реальный target_source.
- Отсутствующий/нечисловой target, отрицательный/отсутствующий wind speed, бесконечности и конфликтующие
  дубликаты вызывают явную ошибку. Точные дубликаты удаляются с логированием.
- NaN temperature/direction поддерживаются tree-моделями. Wind direction приводится к [0,360).
  Подозрительные температуры/ветер и отрицательная мощность отражаются в quality/report, не удаляются.
- Выбросы мощности видны в quantiles и эмпирической кривой; без паспорта физический максимум не придумывается.
  Значения normalized target >1 сохраняются и подсчитываются. Raw prediction сохраняется всегда;
  normalized output по умолчанию ограничен [0,1], `clip_normalized=false` отключает ограничение.
  Absolute output не ограничивается неизвестным номиналом.

## Leakage и оценка

Данные сортируются по timestamp. Split 70/15/15 разделяет уникальные временные отметки,
включая все турбины одной отметки в один блок. Случайного split нет. Декабрь–январь
не выбран автоматически без реального диапазона дат. Нужны минимум 30 train и 5 val/test rows;
это технический минимум, не гарантия статистической репрезентативности.

Pipeline сохраняет feature transformer вместе с regressor. Признаки: wind speed, temperature,
sin/cos направления, часа, месяца и дня года, train-only one-hot turbine ID.
Сырые timestamps, power, normalized_power и любые сторонние lag/rolling-колонки исключены allowlist.
Лаги не включены без подтверждённого availability contract. Unknown turbine использует глобальную модель
с предупреждением: её качество переноса следует измерить отдельно.

Mean и HistGradientBoosting baselines сравниваются с LightGBM на validation RMSE served output;
лучший кандидат выбирается до test. LightGBM использует callbacks early stopping на validation.
HistGradientBoosting использует фиксированные 250 итераций и отключённую случайную internal validation.
При недоступности/native failure LightGBM fallback выполняется автоматически.
Если несколько турбин, сравниваются global и per-turbine модели на той же validation;
переключение на per-turbine требует минимум 1% выигрыша RMSE. Малые группы остаются на global.
Обучение после выбора не повторяется на test или validation. Нет обещания, что сложная модель победит baseline.

MODE A: фактически наблюдаемая погода → мощность, без оценки качества weather forecast.
MODE B: архивный прогноз → мощность, обязательны issued_at и forecast_provenance.
По умолчанию lead >=1 h. В test/evaluation origins должны быть позже validation cutoff,
а в training MODE B validation origins — позже training cutoff. Добавляйте gap между блоками upstream;
ошибка при пересечении не маскируется удалением строк. Для operational availability учитывайте задержку
публикации в issued_at. При evaluation MODE B модель может быть обучена в MODE A: это измеряет
полную систему, но распределение forecast weather может отличаться от observed weather.
`select_forecast_vintage` выбирает последний реально доступный прогноз при явном origin.
Несколько vintages одного valid time требуют выбора upstream, иначе training/aggregation откажут.

`evaluate_model` принимает только данные после validation; MAE, RMSE, R² для raw/served
возвращаются раздельно. Для normalized target возвращается normalized MAE; R²=null для постоянного target.
Test уже использован для отчёта: последующий tuning по нему потребует нового holdout.

## Отчёты после обучения на реальных данных

- `reports/eda.json`: объём, диапазон дат, частоты по турбинам, NaN, quantiles и quality counters.
- `reports/figures/power_curve.png`: actual scatter и усреднённые по wind bins predictions, по турбинам.
- `actual_vs_predicted.png`, `error_vs_wind_speed.png`, `time_series_prediction.png` в той же папке.
- `reports/feature_importance.csv`: LightGBM split importance или test permutation importance
  (не используется при подборе модели; максимум 500 observations, 3 повтора).
- `reports/errors_by_wind_bin.csv`, `errors_by_turbine_id.csv`, `errors_by_season.csv`:
  counts, MAE/RMSE; сезонные названия DJF/MAM/JJA/SON — календарные группы, без предположения о полушарии.
- `reports/empirical_power_curve.csv`, `evaluation.md`, `metrics.json`, `test_predictions.csv`.

Диапазоны ветра 0–3/3–8/8–12/12–20/20+ используются только для error analysis.
Cut-in/growth/rated/cut-out нельзя надёжно объявлять по фиксированным bins без SCADA и паспорта.
Окончательный физический анализ этих режимов остаётся до получения реального dataset.

## Агрегации для API

`aggregate_predictions(predictions, frequency="hourly|daily|monthly", timezone="UTC")`
возвращает statistical_time (начало bucket), turbine_id, mean_wind_speed_ms,
mean_wind_direction_deg, predicted_active_power, normalized_active_power,
mean_ambient_temperature_c, observation_count. Это средняя мощность по samples каждой турбины,
не энергия и не сумма электростанции. Для irregular sampling нужен регулярный grid upstream;
counts показывают покрытие, но не заменяют проверку частоты. Направление — circular mean,
359° + 1° → 0°; для противоположных направлений среднее неопределённо (NaN).
Без capacity отсутствующая normalized/absolute величина остаётся NaN, не выдумывается.
