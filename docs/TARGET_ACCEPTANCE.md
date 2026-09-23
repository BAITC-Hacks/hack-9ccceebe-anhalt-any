# Приёмка прогнозного сценария T1/T2

Состояние на 23 сентября 2026 года: строгий интеграционный сценарий реализован, но **реальный прогноз целевой ВЭС ещё не принят**. Получены агрегированные CSV с метками T1/T2; проверены 50 784 строки и SHA256. Подтверждённые временные определения/нормализация/идентичность, архивный погодный провайдер и обученная целевая модель пока отсутствуют. Подробности в [аудите поступивших файлов](GOLDWIND_DELIVERY_REVIEW.md). Проверки на фикстурах доказывают поведение кода, а не доступность исторических прогнозов или качество Goldwind.

Итоговый полный прогон после интеграции PR #3 и исправления перестановки строк Goldwind: **212 passed, 1 skipped**. Пропущен необязательный LightGBM-тест; одно предупреждение относится к Starlette TestClient. `pip check` и `git diff --check` пройдены. Публикация кода в `main` не означает приёмку реального прогнозного сценария.

## Требования и доказательства

В колонке «результат» различаются автономная контрактная проверка и запуск на реальных целевых данных.

| Требование ТЗ | Реализация | Команда / проверка | Подтверждённый результат или блокер |
| --- | --- | --- | --- |
| История организаторов: март 2023 — 31 января 2026, T1/T2 | Подготовка данных в `src/ml/goldwind_data.py`, `scripts/prepare_goldwind.py`; явная карта файлов и метаданных | `python -m pytest tests/test_goldwind_data.py -q`; [контракт dataset](GOLDWIND_MODEL_CONTRACT.md) | Получены почасовые CSV за 11.03.2023–31.01.2026; первичный аудит выполнен. Исходных 10-минутных файлов и метаданных нет; обучение не подтверждено. |
| Не угадывать timezone, смысл timestamp и normalized power | `ForecastProtocol`, `config/forecast_protocol.json`; конфигурация остаётся неподтверждённой | `test_missing_target_config_never_uses_demo`; `test_target_default_blocks_without_model_and_keeps_other_modes` | Фикстурные проверки пройдены. Реальный UI показывает блокеры и не включает основной расчёт. Координаты/номинал не подменяют словарь данных. |
| Разделить origin, issue, availability, valid_time и lead | `src/forecast/contracts.py`, `src/forecast/weather.py` | `test_invalid_origin_and_horizon`; `test_future_issue_or_payload_revision_rejected`; `test_naive_provenance_time_rejected` | Фикстуры подтверждают aware UTC, почасовые границы и запрет более позднего выпуска/ревизии. Реальное расписание запуска пока не согласовано. |
| Получать архивный прогноз, доступный в прошлом | Новый интерфейс `get_forecast_weather`, выбор одного допустимого выпуска и ревизии | `test_latest_issue_and_revision_selected_as_a_single_vintage`; `test_non_archived_weather_rejected` | Контрактная логика пройдена. Ни один реальный источник пока не принят. Реанализ, наблюдения и synthetic отвергаются в основном сценарии. |
| Почасовой горизонт 24 и 48 часов | `expected_times`, `run_target_forecast`; интервалы `[valid_time, valid_time + 1h)` | `test_24_48_and_farm_alignment`; `test_48_hour_horizon_reindexes_and_preserves_missing` | На фикстурах ровно 24/48 часов для каждой турбины, пропуски остаются видимыми. Реальный целевой origin не выполнен. |
| Целевая модель, единицы кВт, T1/T2 | `ModelManifest`, проверка artifact SHA256, ёмкости, высот и feature schema | `test_hash_mismatch_blocks_unverified_model`; `test_source_target_cannot_be_in_feature_manifest` | На фикстурах модель и метаданные проверяются. Реальный Goldwind artifact/provider отсутствует; модель Kelmarsh не является заменой. |
| Исключить утечку обучения и выбора модели | Проверяются availability cutoff и конец целевых интервалов для fit и selection | `test_future_fit_or_selection_data_blocked`; тесты availability в `tests/test_goldwind_data.py` | Контрактные проверки пройдены. B ещё должен доказать временной split и происхождение реальных fit/validation данных. Данные февраля не могут участвовать в выборе модели. |
| Ежедневные исторические запуски с 31 января | `src/forecast/replay.py`: локальные календарные даты, проверка DST и почасовых границ | `test_full_replay_preserves_origins_overlap_and_local_february`; `test_origins_follow_local_calendar_across_dst` | Полный цикл проверен на фикстурах. Реальный replay заблокирован до формирования расписания: **0 запущенных origins**. |
| Оценивать только часы 1–28 февраля | Окно по `valid_time` в подтверждённой timezone, отдельно от даты запуска | `tests/test_forecast_replay.py`; `configuration.evaluation_window` в успешном manifest | Проверено на фикстурах. Реальное окно нельзя объявить подтверждённым до получения timezone. CSV сохраняет также прогнозные часы вне окна, но они не входят в февральскую оценку. |
| Сохранять перекрывающиеся прогнозы | Ключи включают origin и valid_time; actuals присоединяются many-to-one | `test_actuals_match_many_vintages_without_selecting_best`; `test_duplicate_predictions_are_contract_failure_and_not_scored` | Фикстуры пройдены. Прогнозы разных origins не усредняются и не выбираются по лучшей ошибке постфактум. |
| Прогноз всей ВЭС и корректная энергия | `farm_rows`: T1+T2 для одного origin/часа; кВт и кВт·ч раздельно | `test_one_turbine_failure_farm_null`; `test_raw_negative_not_clipped_but_farm_flagged`; partial UI test | Фикстуры пройдены. Отсутствующая/невалидная турбина не становится нулём; полная энергия недоступна при неполном покрытии. |
| Анализ результата и agentic pipeline | Погода → признаки → local ML → правила → один анализ станции → сохранение | `tests/test_target_forecast.py`; `tests/test_pipeline.py` | Полный путь проверен с заглушками. Детерминированный анализ работает; живой вызов OpenAI для целевого сценария не подтверждён. LLM не изменяет численные прогнозы. |
| Пересчёт при обновлении входов | `--refresh`, immutable weather/result snapshots, версии модели/погоды/кода | `test_updated_inputs_create_new_immutable_version`; `test_unchanged_refresh_no_new_inference_or_llm`; `test_inference_failure_can_retry_same_weather` | Фикстуры пройдены: новые допустимые входы дают новую версию; неизменные не вызывают повторный ML/LLM; временный отказ inference допускает повтор. |
| Обновление не создаёт утечку будущего | Проверка доступности фактической ревизии и изменений известного vintage | `test_refresh_cannot_backdate_changes_to_known_payload_revision`; `test_future_revision_is_blocked_not_silently_cached` | Проверено на фикстурах. Первое обращение к реальному архиву всё равно требует внешнего доказательства политики публикации и исправлений. |
| Метрики с фактами и покрытием | Raw MAE/RMSE/bias/R² по турбинам, станции и горизонтам; флаги не скрывают конечные ошибки | `test_negative_raw_predictions_and_actuals_are_scored_including_farm`; `test_missing_actual_turbine_never_becomes_zero_farm_actual` | Расчёт метрик проверен на фикстурах. Факты февраля не получены, **реальные метрики T1/T2 отсутствуют**. Отсутствие фактов блокирует оценку, но не должно блокировать прогноз при готовых A/B. |
| CLI/API/UI и воспроизводимая поставка | `run_target_forecast.py`, `replay_february.py`, `/readiness`, `/target-forecast`, отдельный UI-режим | Команды ниже; `tests/test_ui.py`; [сохранённая попытка приёмки](../reports/target-acceptance/README.md) | Реальные команды завершились exit 2/blocked без подмены данных. JSON/manifest описывают блокеры; CSV содержит только заголовки. Это отчёт об отказе, а не прогнозная поставка. |

## Фактически выполненные команды

Команды запускались с текущей конфигурацией репозитория; сохранены [readiness](../reports/target-acceptance/target-readiness.json), [один origin](../reports/target-acceptance/target-origin.json), [replay manifest](../reports/target-acceptance/manifest.json) и CSV в `reports/target-acceptance/`.

```bash
python scripts/run_target_forecast.py --check
python scripts/run_target_forecast.py --origin 2026-01-31T23:00:00Z --horizon 48
python scripts/replay_february.py --horizon 48 --output-dir results/target-february
```

`2026-01-31T23:00:00Z` — диагностический пример, **не подтверждённое время запуска организаторов**. Readiness вернул `ready=false`; один origin — `status=blocked`, 0 прогнозных строк. Replay — `status=blocked`, пустые `runs`/`run_ids`, 0 прогнозов, `metrics=null`. Без timezone/расписания ожидаемое число строк также не вычисляется. Заголовки CSV не означают успешное покрытие февраля.

Отдельно проверен работающий Kelmarsh MODE A за 2017-11-08: 144/144 десятиминутных интервала, MAE 51,39 кВт, RMSE 71,46 кВт для показанной турбины. Это регрессионная проверка существующего сценария по наблюдаемой погоде; она не доказывает качество Goldwind или архивного прогноза. Синтетическое демо остаётся отдельным режимом.

## Что нужно получить для реальной приёмки

**От организаторов/команды:** точные файлы T1/T2 с историей, словарь колонок, timezone, начало/конец статистического интервала, длительность, единицы, определение нормализации, высота ветра, задержка публикации SCADA и согласованное время ежедневного origin. `confirmed` нельзя менять только ради прохождения readiness.

**От A:** импортируемый `FORECAST_DATA_MODULE` с `get_forecast_weather`/`build_features`; архивный источник с названием продукта/модели, URL и доказательством доступности каждого возвращаемого выпуска/ревизии. Обязательны aware `weather_issued_at`, `weather_available_at`, `availability_basis`, `availability_evidence`, ветер на согласованной высоте и реальные часы февраля. Пересчёт высоты должен быть согласован с B. Точное API и provenance — в [TARGET_FORECAST_CONTRACT.md](TARGET_FORECAST_CONTRACT.md).

**От B:** воспроизводимое обучение именно на organizer SCADA, adapter с raw 1D кВт и отдельный artifact Goldwind. Manifest должен содержать SHA256 артефакта, T1/T2, по 2500 кВт, подтверждённые высоты/timezone, ordered features, шаг 60 минут, нормализацию, fit/selection availability cutoffs и окончания соответствующих target-интервалов. Все использованные для fit/selection данные должны быть доступны к первому origin. Подготовленный загрузчик из PR #3 этого результата ещё не заменяет.

**От C после получения A/B:** проверить подтверждения, запустить реальный origin на 24/48h, полный февральский replay, проверить покрытие/версии, а при наличии февральских фактов — метрики. Затем повторить итоговый suite, UI-проверку и обновить проверенную сборку в `main`. До этого формулировка «100% ТЗ выполнено» не подтверждена.
