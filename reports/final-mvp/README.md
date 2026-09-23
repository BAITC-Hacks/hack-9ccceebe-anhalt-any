# Финальная интеграция MVP — 23.09.2026

Объединены main с интерфейсом WindAgent, ветка A `feature/data-weather` (`13e3edb`)
и ветка B `feature/goldwind-model` (`202b6ba`) через Git merge.
Работает демонстрационный полный цикл, включая локальную ML-модель и анализ.
**Реальный исторический прогноз Goldwind пока не выполнен.**

## Проверено

- 293 Python-теста прошли; 1 пропущен (необязательный LightGBM).
  Остаётся предупреждение зависимости Starlette/httpx.
- Обе JavaScript-проверки данных и состояния интерфейса прошли.
- A → B → C: настоящие адаптеры признаков/модели и orchestrator проверены вместе
  для 24/48 часов на искусственных тестовых данных. Это проверка совместимости,
  а не измерение качества Goldwind.
- CLI demo T1: 24/24 часа, T2: 48/48 часов, статус `ok`.
- Демонстрационный backtest 31 января–28 февраля: 29/29 дней и 696/696 часов
  **на каждую турбину**, без ошибок. MAE/RMSE пустые: фактов для этого демо нет.
- Kelmarsh MODE A 08.11.2017: 144/144 наблюдения, MAE 51,3933 кВт,
  RMSE 71,4567 кВт. Это наблюдаемая погода другой станции, не прогноз Goldwind.
- Основной forecast и replay возвращают ожидаемый `blocked`/код 2;
  ни синтетика, ни Kelmarsh не подставляются вместо целевых данных.
- В браузере: время +05:00 из общей конфигурации, заблокированный основной режим,
  успешный демо-расчёт 71,4 МВт·ч за 24 часа для T1+T2.

Машинный отчёт с командами и кодами завершения: [verification.json](verification.json).
Сохранены [T1 demo](demo-T1.json), [T2 demo](demo-T2.json),
[T1 backtest](backtest-T1/results.csv), [T2 backtest](backtest-T2/results.csv),
[готовность](target-readiness.json) и [попытка целевого прогноза](target-48h.json).
696 часов legacy-demo включают 31 января; строгий февральский replay отдельно
оценивает 672 часа февраля и сохраняет перекрывающиеся горизонты 24/48 часов.

## Запуск

Из корня репозитория с установленными зависимостями:

```bash
source .venv/bin/activate
python scripts/run_forecast.py --demo --date 2026-02-10 --turbine T1 --horizon 24 --with-agent --output results/demo.json
python scripts/run_backtest.py --demo --turbine T1 --output-dir results/backtest-T1
python scripts/run_backtest.py --demo --turbine T2 --output-dir results/backtest-T2
python scripts/run_ui.py
```

Откройте http://127.0.0.1:8000/ui/ и выберите «Синтетическое демо».
Для анализа реальных наблюдений: `python scripts/run_observed.py --date 2017-11-08`.
Streamlit с переключателем OpenAI: `streamlit run src/app/streamlit_app.py`.

Демо не требует ключей или внешней сети. Для необязательного LLM нужны
`OPENAI_API_KEY`, `OPENAI_MODEL` и флаг `--with-agent` (либо переключатель Streamlit).
Используется официальный SDK openai 2.54.0 и `responses.parse` с Pydantic-схемой,
`store=False`, ограниченным ответом и без автоматических повторов SDK.
Интерфейс проверен по [официальной документации Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
Ключ и имя модели в проверенном окружении отсутствуют: живой API не проверен,
подтверждён fallback `openai_not_configured` с сохранением численного прогноза.
`.env` не включается в Git; образец — `.env.example`.

Целевые провайдеры A/B подключены по умолчанию: `FORECAST_DATA_MODULE=src.data.weather`,
`FORECAST_ML_MODULE=src.ml.goldwind_provider`. Пути модели/manifest/protocol задаются
через `FORECAST_MODEL_PATH`, `FORECAST_MANIFEST_PATH`, `FORECAST_PROTOCOL_PATH`.
Шаблон доступности погоды: `WEATHER_AVAILABILITY_PATH`.

## Процесс

Входы → выпуск погоды/кэш → признаки A → локальная модель B → статистика и правила →
компактный структурированный анализ → почасовые результаты, флаги и рекомендации.
LLM не рассчитывает мощность. В целевом пути один анализ относится ко всей ВЭС,
а `--refresh` пересчитывает изменившиеся допустимые входы с сохранением прежней версии.

## Что добавлено и изменено

- Интегрированы `src/data/`, `scripts/prepare_data.py`, `scripts/cache_weather.py`
  и отчёт проверки архива A; `src/ml/goldwind_model.py`, `goldwind_provider.py`,
  `scripts/train_goldwind.py`, `evaluate_goldwind.py` и почасовой контракт B.
- Исправлены настройки `src/forecast/`, `.env.example`, календарь в
  `src/config.py`, API, Streamlit и browser UI. Календарь источника модели отделён
  от календаря выполнения; UTC-сравнения границ обучения сохранены.
- Синхронизированы оба шаблона обучения с первым origin команды; добавлена
  сквозная проверка модулей A/B/C, обновлены UI/schedule тесты.
- Обновлены README, контракты/передача команды, добавлены результаты в этой папке.

## До полного соответствия ТЗ

Нужны подтверждённые семантика CSV и доступность исторических погодных версий,
обучение реальной модели T1/T2, настоящий февральский replay и (если доступны)
факты февраля для оценки. Скачанные 58 ECMWF выпусков имеют проверенный локальный
кэш, но сама загрузка не доказывает доступность этих версий в январе–феврале.
Нет публичного deployment и автономного фонового опроса; обновление запускается явно.
Этот отчёт подтверждает работоспособность MVP, а не 100% выполнения целевого кейса.
