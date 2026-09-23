import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt
import pandas as pd
import streamlit as st

from src.agent.observed import kelmarsh_turbines, run_observed_analysis
from src.agent.orchestrator import run_forecast
from src.agent.schemas import ForecastError
from src.config import Settings
from src.forecast.orchestrator import readiness, run_target_forecast


def render_target_mode():
    st.caption("Goldwind GW109/2500 · T1/T2 · 2 × 2500 кВт · архивные выпуски погоды")
    st.info("Основной сценарий ТЗ: прогноз обеих турбин и всей ВЭС на 24–48 часов. "
            "Время прогноза отделено от времени выпуска и доступности погодных данных.")
    state = readiness()
    if not state["ready"]:
        st.error("Основной прогноз пока недоступен: требуется подтверждённая конфигурация и целевая модель.")
        for blocker in state["blockers"]:
            st.error(str(blocker))
    else:
        st.caption("Конфигурация найдена. Доступность архивного выпуска и граница обучающих данных "
                   "будут проверены для выбранного момента запуска.")
    st.caption("Время ниже — редактируемый пример. Расписание, timezone и смысл часового интервала "
               "должны соответствовать подтверждённому протоколу организаторов. "
               "Графики и выгрузка показывают время в UTC.")
    with st.form("target_forecast"):
        origin = st.text_input("Момент запуска (ISO 8601 с часовым поясом)",
                               "2026-01-31T23:00:00+00:00", key="target_origin")
        horizon = st.selectbox("Горизонт, часов", [24, 48], index=1, key="target_horizon")
        refresh = st.checkbox("Проверить обновление погодных данных", value=False, key="target_refresh")
        with_agent = st.checkbox("Анализ через OpenAI", value=False, key="target_agent")
        submitted = st.form_submit_button("Рассчитать прогноз ВЭС", type="primary", disabled=not state["ready"])
    if submitted:
        st.session_state.pop("target_result", None)
        try:
            with st.spinner("Архивный выпуск → признаки → T1/T2 → сумма ВЭС → анализ…"):
                st.session_state.target_result = run_target_forecast(
                    origin, horizon, refresh=refresh, with_agent=with_agent)
        except (ForecastError, ValueError) as exc:
            st.error(str(exc))
    result = st.session_state.get("target_result")
    if result is None:
        return
    st.subheader(f"ВЭС T1 + T2 · {result['forecast_origin']} · {result['horizon_h']} ч")
    st.caption(f"Запуск: {result['run_id']} · статус: {result['status']} · "
               f"{'Сохранённый результат' if result.get('cache_hit') else 'Новый расчёт'}")
    if result["status"] != "ok":
        st.warning("Полного прогноза ВЭС нет. Пропуски и ошибки сохранены; отсутствующая турбина не считается нулём.")
    for turbine_id, error in result.get("errors", {}).items():
        st.error(f"{turbine_id}: {error}")
    farm = pd.DataFrame(result.get("farm_rows", []))
    rows = pd.DataFrame(result.get("rows", []))
    complete_hours = int(farm.complete.sum()) if not farm.empty else 0
    complete_energy = (farm.energy_kwh.sum() if complete_hours == result["horizon_h"]
                       and not farm.energy_kwh.isna().any() else None)
    cols = st.columns(3)
    cols[0].metric("Энергия ВЭС, кВт·ч", f"{complete_energy:,.1f}" if complete_energy is not None else "Нет полного расчёта")
    cols[1].metric("Полных часов ВЭС", f"{complete_hours}/{result['horizon_h']}")
    cols[2].metric("Номинальная мощность ВЭС, кВт", "5000")
    series = []
    if not rows.empty:
        series.append(rows[["valid_time", "turbine_id", "power_kw"]].rename(columns={"turbine_id": "series"}))
    if not farm.empty:
        series.append(farm[["valid_time", "power_kw"]].assign(series="ВЭС T1 + T2"))
    if series:
        for item in series:
            item["power_kw"] = pd.to_numeric(item.power_kw, errors="coerce").astype(float)
        chart_frame = pd.concat(series, ignore_index=True)
        chart_frame["valid_time"] = pd.to_datetime(chart_frame.valid_time, utc=True)
        chart_frame["timestamp_utc"] = chart_frame.valid_time.dt.strftime("%Y-%m-%d %H:%M UTC")
        chart = alt.Chart(chart_frame).mark_line().encode(
            x=alt.X("valid_time:T", title="Начало часового интервала, UTC", scale=alt.Scale(type="utc"),
                    axis=alt.Axis(format="%d.%m %H:%M")),
            y=alt.Y("power_kw:Q", title="Мощность, кВт"), color=alt.Color("series:N", title=None),
            tooltip=[alt.Tooltip("timestamp_utc:N", title="UTC"), alt.Tooltip("series:N", title="Ряд"),
                     alt.Tooltip("power_kw:Q", title="кВт")])
        st.subheader("Мощность T1, T2 и станции")
        st.altair_chart(chart, width="stretch")
        if not farm.empty:
            st.subheader("Почасовой прогноз ВЭС")
            st.dataframe(farm, hide_index=True)
        if not rows.empty:
            with st.expander("Прогнозы турбин и происхождение данных"):
                st.dataframe(rows, hide_index=True)
    analysis = result.get("analysis") or {}
    if analysis:
        st.subheader("Анализ")
        st.caption(f"Источник: {result.get('analysis_source', '—')} · Риск: {analysis.get('risk_level', '—')}")
        st.write(analysis.get("summary", ""))
        st.write(analysis.get("recommendation", ""))
        if analysis.get("confidence_note"):
            st.info(analysis["confidence_note"])
    st.download_button("Скачать JSON прогноза ВЭС", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                       "target_forecast.json", "application/json")

st.set_page_config(page_title="AI Energy Agent", page_icon="🌬️", layout="wide")
st.title("AI Energy Agent")
mode = st.sidebar.radio("Сценарий", ["Прогноз ВЭС T1/T2", "Реальные данные Kelmarsh", "Синтетическое демо"])
if mode == "Прогноз ВЭС T1/T2":
    render_target_mode()
    st.stop()
observed = mode == "Реальные данные Kelmarsh"
demo = mode == "Синтетическое демо"
settings = Settings.from_env(demo)
try:
    turbines = kelmarsh_turbines() if observed else settings.turbines()
except (OSError, ValueError):
    st.error("Не удалось прочитать конфигурацию турбин")
    st.stop()
if observed:
    st.caption("HackAlem AI 2026 · Kelmarsh · Senvion MM92 · 2050 кВт на турбину · UTC")
    st.info("Реальные SCADA-данные и обученная ML-модель. MODE A: расчёт по наблюдаемой погоде, а не прогноз будущего. Шаг 10 минут.")
elif demo:
    st.caption("Goldwind GW109/2500 · 2 × 2,5 МВт · UTC")
    st.warning("ДЕМО: синтетическая погода и модель, обученная на синтетике. Это не реальный прогноз станции.")
with st.form(f"run_{mode}"):
    c1, c2, c3 = st.columns(3)
    turbine = c1.selectbox("Турбина", list(turbines), key=f"turbine_{mode}")
    day = c2.date_input("Дата наблюдений, UTC" if observed else "Дата, UTC",
        value=date(2017, 11, 8) if observed else date(2026, 2, 10), key=f"date_{mode}")
    horizon = c3.slider("Горизонт, часов", 1, 72, 24, key=f"hours_{mode}")
    with_agent = st.checkbox("Анализ через OpenAI", value=False, key=f"agent_{mode}")
    submitted = st.form_submit_button("Рассчитать и сравнить" if observed else "Рассчитать прогноз", type="primary")
if submitted:
    st.session_state.pop("forecast_result", None)
    try:
        with st.spinner("Данные → признаки → ML → проверка → анализ…"):
            result = (run_observed_analysis(turbine, day, horizon, with_agent=with_agent) if observed
                      else run_forecast(turbine, day, horizon, settings=settings, with_agent=with_agent))
            st.session_state.forecast_result = (mode, result)
    except ForecastError as exc:
        st.error(str(exc))
stored = st.session_state.get("forecast_result")
if isinstance(stored, tuple) and stored[0] == mode:
    result = stored[1]
    run_date = result.request.observation_date if observed else result.request.forecast_date
    st.subheader(f"{result.request.turbine_id} · {run_date} · {result.request.horizon_h} ч")
    stats = result.statistics
    cols = st.columns(3)
    cols[0].metric("Оценка энергии, кВт·ч" if observed else "Энергия, кВт·ч",
        f"{stats.predicted_energy_kwh:,.1f}" if stats.predicted_energy_kwh is not None else "Нет полного расчёта")
    cols[1].metric("Средний ветер, м/с", f"{stats.avg_wind_ms:.1f}" if stats.avg_wind_ms is not None else "Нет данных")
    cols[2].metric("Валидных часов", f"{stats.valid_hours:g}/{stats.requested_hours}")
    if result.status != "ok":
        st.warning("Есть пропуски или физически подозрительные результаты. Полная энергия не рассчитана; raw значения сохранены.")
    if observed:
        met = result.metrics
        mc = st.columns(3)
        mc[0].metric("MAE, кВт · выбранный период", f"{met.mae_kw:.2f}" if met.mae_kw is not None else "—")
        mc[1].metric("RMSE, кВт · выбранный период", f"{met.rmse_kw:.2f}" if met.rmse_kw is not None else "—")
        mc[2].metric("Пар модель / факт", f"{met.matched_samples}/{met.expected_samples}")
        st.caption("Метрики по raw-мощности: отрицательные и выше-номинальные значения не скрыты clipping.")
    points = result.samples if observed else result.hourly
    chart_frame = pd.DataFrame([p.model_dump() for p in points])
    chart_frame["timestamp_utc"] = pd.to_datetime(chart_frame.timestamp, utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
    def chart(columns, title):
        long = chart_frame.melt(id_vars=["timestamp", "timestamp_utc"], value_vars=columns,
                                var_name="series", value_name="value")
        long["series"] = long.series.replace({"power_kw": "ML, raw", "actual_power_kw": "Факт", "wind_speed_ms": "Ветер"})
        return alt.Chart(long).mark_line().encode(
            x=alt.X("timestamp:T", title="Время, UTC", scale=alt.Scale(type="utc"), axis=alt.Axis(format="%d.%m %H:%M")),
            y=alt.Y("value:Q", title=title), color=alt.Color("series:N", title=None),
            tooltip=[alt.Tooltip("timestamp_utc:N", title="UTC"), alt.Tooltip("series:N", title="Ряд"), alt.Tooltip("value:Q", title=title)])
    st.subheader("Модель и фактическая мощность, кВт" if observed else "Мощность, кВт")
    st.altair_chart(chart(["power_kw", "actual_power_kw"] if observed else ["power_kw"], "кВт"), width="stretch")
    height = result.hub_height_m if observed else turbines[result.request.turbine_id].hub_height_m
    st.subheader(f"Ветер на высоте {height:g} м, м/с")
    st.altair_chart(chart(["wind_speed_ms"], "м/с"), width="stretch")
    st.subheader("Анализ")
    st.caption(f"Источник: {result.analysis_source} · {result.analysis_reason} · Риск: {result.analysis.risk_level}")
    st.write(result.analysis.summary)
    st.write(result.analysis.recommendation)
    st.info(result.analysis.confidence_note)
    if result.anomalies:
        st.subheader("Флаги правил")
        st.dataframe(pd.DataFrame([a.model_dump() for a in result.anomalies]), hide_index=True)
    else:
        st.write("Детерминированные правила не обнаружили аномалий.")
    st.caption(f"Данные: {result.source if observed else result.weather_source} · Модель: {result.model_id}")
    if observed:
        st.caption(result.limitation)
        st.markdown("Данные: Charlie Plumley, Roberta Takeuchi / Cubico Sustainable Investments Ltd · [Kelmarsh, CC BY 4.0](https://doi.org/10.5281/zenodo.16807551)")
    st.download_button("Скачать JSON", result.model_dump_json(indent=2), "analysis.json", "application/json")
