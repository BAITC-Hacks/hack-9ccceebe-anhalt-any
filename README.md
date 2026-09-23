# AI Energy Agent

HackAlem AI 2026 · Anhalt ANY

Агент помогает анализировать прогноз генерации двух ветрогенераторов Goldwind GW109/2500.
Мощность рассчитывает локальная ML-модель. LLM объясняет результат и флаги правил,
но не рассчитывает и не изменяет основной прогноз.

**Текущий статус:** интегрированы ML-ветка B и реальный сценарий **Kelmarsh MODE A**:
наблюдаемая SCADA-погода → сохранённая модель → сравнение с фактической мощностью → анализ.
Синтетический offline demo сохранён отдельно. Погодный API Person A и валидированный перенос
на Goldwind T1/T2 ещё не подключены. Результаты Kelmarsh нельзя выдавать за качество Goldwind.

## Запуск на реальных данных (без внешнего API)

После установки зависимостей:

```bash
python scripts/run_observed.py --date 2017-11-08 --turbine "Kelmarsh 1" --output results/observed.json
python scripts/run_observed_backtest.py --turbine "Kelmarsh 1"
streamlit run src/app/streamlit_app.py
```

UI по умолчанию открывает «Реальные данные Kelmarsh». Выберите турбину и дату,
нажмите «Рассчитать и сравнить». Показываются raw-прогноз ML и факт, MAE/RMSE,
покрытие, флаги и анализ. Данные сохранены в репозитории; ключи и переобучение не нужны.

Реестр: `config/kelmarsh_turbines.json`; шесть Senvion MM92 по 2050 кВт,
высота 78,5 м (1/2/4/5) или 68,5 м (3/6). Это отдельная станция, не T1/T2.
Период полных суток holdout: **2017-11-08…2017-12-31**, горизонт 1–72 ч должен помещаться в него.
Сохраняется исходный шаг **10 минут**. Энергия — оценка суммы кВт × 10/60 ч,
не показание счётчика; start/end convention timestamp источника не подтверждён.
Пропуски и невалидная мощность делают полный итог энергии null, значения не обрезаются.
Метрики используют все конечные пары raw/actual, включая отрицательные и выше номинала,
в соответствии с протоколом B. Это MODE A по наблюдаемой погоде, не прогноз будущего.

Проверка неизменности artifact/holdout и воспроизведения метрик B:

```bash
python -m scripts.verify_ml_provider --report results/provider_verification.json
```

На всех 47006 holdout-строках: raw MAE **82,9047 кВт**, RMSE **145,1903 кВт**,
R² **0,953016**. Метрики выбранной турбины/дня отличаются от общей оценки.
Дневной backtest сохраняет results.csv, summary.json и daily.json в results/observed-backtest.
При пропусках покрытия CLI возвращает код 2, сохраняя результаты и описание пропусков.
LLM опционален через `--with-agent` (до одного запроса на день).

Источник: Charlie Plumley, Roberta Takeuchi / Cubico Sustainable Investments Ltd,
[Kelmarsh SCADA, CC BY 4.0](https://doi.org/10.5281/zenodo.16807551).
Подготовка и исключения описаны в [data/README.md](data/README.md).
Подробности ML: [README_ML.md](README_ML.md),
[контракт адаптера](docs/ML_INFERENCE_CONTRACT.md).

## Станция

| Турбина | Широта | Долгота | Мощность | Высота ступицы |
| --- | --- | --- | --- | --- |
| T1 | 43.645150 | 78.535604 | 2,5 МВт | 80 м |
| T2 | 43.643198 | 78.538828 | 2,5 МВт | 80 м |

Общая номинальная мощность — **5 МВт**. Параметры предоставлены командой.
Внутренние единицы: UTC, м/с, °C, кВт, кВт·ч. Прогноз запускается отдельно для каждой турбины.

## Что реализовано

- Orchestrator: валидация → погода/кэш → признаки → локальная модель → статистика → правила → анализ.
- Контракты Person A (DATA) и Person B (ML), импорт через настройки без зависимости агента от модели.
- Structured Output через официальный OpenAI Python SDK, `responses.parse` и Pydantic.
- Один короткий запрос анализа на запуск; в backtest — на день только при `--with-agent`.
- Локальный fallback без ключа, при отказе API, таймауте или некорректном ответе; автоматические retries отключены.
- Правила: пропуски/NaN, невозможная погода, отрицательная/выше номинала мощность,
  скачки более 50% номинала, сильный ветер ниже cut-out при почти нулевой генерации.
- CLI, backtest с CSV/JSON, Streamlit UI, FastAPI с `/forecast` и `/health`.
- Сохранённые синтетические данные и локальная demo ML-модель; сеть для demo не нужна.

## Архитектура

```text
config/turbines.json          Реестр T1/T2
src/data/contracts.py        Контракт Person A
src/data/demo.py             Чтение сохранённой синтетической погоды
src/ml/contracts.py          Контракт Person B
src/ml/demo.py               Загрузка локальной demo-модели
src/agent/tools.py           Загрузка провайдеров, валидация погоды, кэш
src/agent/orchestrator.py    End-to-end pipeline
src/agent/rules.py           Числа и детерминированные флаги
src/agent/analyzer.py        OpenAI / локальный анализ
src/agent/schemas.py         Pydantic request/result/analysis
src/agent/backtest.py        Прогон, actual join, метрики
src/api/main.py              FastAPI
src/app/streamlit_app.py     UI
scripts/                    Forecast, backtest, генерация demo
models/demo/                Модель и metadata
data/demo/weather.csv       Синтетический почасовой кэш
tests/                     Проверки интеграции и отказов
```

Технологии: Python 3.12, pandas, NumPy, scikit-learn/joblib, Pydantic, OpenAI SDK,
Streamlit, FastAPI, pytest. Demo — небольшой RandomForestRegressor, обученный на 3000
синтетических примеров идеализированной кривой мощности. Это проверка pipeline,
а не валидированная модель Goldwind.

## Установка

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
cp .env.example .env
```

`requirements.lock.txt` фиксирует проверенное окружение Python 3.12/macOS arm64;
на другой платформе могут потребоваться совместимые версии из `requirements.txt`.
При смене версии scikit-learn перегенерируйте только demo-артефакты:

```bash
python scripts/prepare_demo.py
```

Ключи не нужны для автономного демо. `.env` не попадает в git.

## Демо за 30 секунд после установки

```bash
python scripts/run_forecast.py --demo --date 2026-02-10 --turbine T1 --horizon 24 --output results/demo.json
```

CLI печатает структурированный JSON: 24 точки мощности и ветра, энергию, флаги и анализ.
Информационные логи идут в stderr. Exit code 0 — полный валидный прогноз, 2 — ошибка/неполное покрытие.
Демо поддерживает даты **2026-01-31…2026-02-28**, горизонт 1–72 ч, T1/T2.
Обе турбины используют один синтетический ряд — пространственные различия не моделируются.

```bash
streamlit run src/app/streamlit_app.py
```

В UI выберите турбину и дату, нажмите «Рассчитать прогноз». Появятся мощность,
ветер, энергия, анализ, флаги и скачивание JSON. Для синтетического режима выберите «Синтетическое демо» в боковом меню;
по умолчанию UI показывает реальные наблюдения Kelmarsh. В CLI/API production выбран по умолчанию, demo включается явно.

## OpenAI и настройки

| Переменная | Назначение |
| --- | --- |
| `OPENAI_API_KEY` | Ключ команды, только через окружение/.env |
| `OPENAI_MODEL` | Доступная команде модель с Responses Structured Outputs; дефолта нет |
| `DEMO_MODE` | `true` для offline demo; по умолчанию `false` |
| `DATA_MODULE` | Python-модуль Person A с get_archival_weather/build_features |
| `ML_MODULE` | Python-модуль Person B с load_model/predict |
| `MODEL_PATH` | Локальный путь модели; по умолчанию models/production |
| `TURBINES_FILE` | По умолчанию config/turbines.json |
| `WEATHER_CACHE_DIR` | По умолчанию data/cache |

```bash
python scripts/run_forecast.py --demo --date 2026-02-10 --with-agent
```

`--with-agent` разрешает один API-запрос при наличии ключа и модели. Для demo агент
по умолчанию локальный; для обычного CLI-прогноза OpenAI включён при конфигурации.
`--no-agent` отключает API. SDK проверен: openai 2.54.0, callable responses.parse.
Реальный OpenAI-вызов не проверен без ключа/имени модели; SDK-вызов и сценарии отказов
проверены подставными ответами. Использован официальный
[Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

## Backtest

По умолчанию: **31 января — 28 февраля 2026 включительно**, 29 ежедневных запусков,
каждый на 24 часа от 00:00 UTC. LLM по умолчанию выключен.

```bash
python scripts/run_backtest.py --demo
python scripts/run_backtest.py --demo --turbine T2 --output-dir results/backtest-t2
# После подключения реальных провайдеров и фактов:
python scripts/run_backtest.py --actuals data/actuals.csv
# Опционально: до одного запроса LLM на день
python scripts/run_backtest.py --demo --with-agent
```

Выход: `results/backtest/results.csv`, `summary.json`, а с агентом — `analyses.json`.
Actuals CSV: `timestamp,turbine_id,actual_power_kw`; timestamp с UTC offset,
мощность — средняя за часовую ячейку. Сопоставление строго по турбине и времени.
Дубликаты и некорректные actuals отклоняются; пропуски не заполняются нулями.
Метрики MAE/RMSE/bias выводятся только при наличии совпавших валидных actuals, вместе с покрытием.
Без фактов метрики `null`. Ошибки отдельных дней записываются, прогон продолжается,
а exit code 2 не даёт спутать неполный backtest с успешным.

## Data/API и интеграция

Подробные [контракты](docs/CONTRACTS.md) и [порядок интеграции веток](docs/INTEGRATION.md).
Провайдер Person A должен отдавать ветер на 80 м и происхождение данных.
Кэш различает координаты, модуль, дату выпуска и горизонт. Сохранённый запрос работает
при недоступности API; при отсутствии кэша production возвращает понятную ошибку.
Демо не обращается к погодному API и не подставляется молча в production.

```bash
DEMO_MODE=true uvicorn src.api.main:app --host 127.0.0.1 --port 8000
curl -X POST http://127.0.0.1:8000/forecast \
  -H 'Content-Type: application/json' \
  -d '{"turbine_id":"T1","forecast_date":"2026-02-10","horizon_h":24}'
```

Дополнительно `POST /observed` принимает `{"turbine_id":"Kelmarsh 1","observation_date":"2017-11-08","horizon_h":24}`
и возвращает native samples, actuals, raw metrics и анализ.

API `?with_agent=true` включает анализ; по умолчанию внешних LLM-вызовов нет.
API/UI рассчитаны на локальную демонстрацию; аутентификация для публичного сервиса не реализована.
Deployment URL отсутствует.

## Проверка

```bash
python -m pytest -q
python scripts/run_forecast.py --demo --date 2026-02-10 --output results/demo.json
python scripts/run_backtest.py --demo
```

Тесты проверяют интеграцию MODE A и demo, единицы/границы мощности, выравнивание времени, пропуски,
контракты признаков, высоту ветра, cache hit при отказе API, отсутствие погодной утечки
из будущего, structured analysis/fallback, actual join/метрики, частичные прогоны и API.
UI дополнительно проверяется Streamlit AppTest.

## Ограничения и оставшаяся работа

- Подключить погодный адаптер A и реальный архив прогнозов. ML-ветка B интегрирована;
  модель обучена на Kelmarsh, а не на Goldwind. Для Goldwind нужны целевые данные/валидация.
- Получить actual generation и подтвердить training cutoff модели. Реанализ даёт hindcast,
  а не честную проверку прогноза, доступного на дату выпуска.
- Пороговые скорости 3/25 м/с — предварительные настройки, не подтверждённые характеристики GW109.
- Нет вероятностных интервалов, калибровки неопределённости, SCADA/curtailment/maintenance контекста.
- Нет суммарного прогноза станции, расписания обновлений или production deployment.
- Прогноз — почасовая средняя мощность, поэтому сумма × 1 ч даёт кВт·ч.
  При некорректных часах полная энергия `null`, а не искусственно заниженная сумма.
- Детерминированные флаги — основания для проверки, не диагноз поломки.
- OpenAI улучшает объяснение; численные результаты и локальный fallback работают без него.
