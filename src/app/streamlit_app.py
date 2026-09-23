import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt
import pandas as pd
import streamlit as st

from src.agent.orchestrator import run_forecast
from src.agent.schemas import ForecastError
from src.config import Settings

st.set_page_config(page_title="AI Energy Agent", page_icon="🌬️", layout="wide")
st.title("AI Energy Agent")
st.caption("HackAlem AI 2026 · Goldwind GW109/2500 · 2 × 2,5 МВт · UTC")
demo = st.sidebar.toggle("Демо на синтетических данных", value=True)
settings = Settings.from_env(demo)
try:
    turbines = settings.turbines()
except (OSError, ValueError):
    st.error("Не удалось прочитать конфигурацию турбин")
    st.stop()
if demo:
    st.warning("ДЕМО: синтетическая погода и модель, обученная на синтетике. Это не реальный прогноз станции.")
with st.form("forecast"):
    c1, c2, c3 = st.columns(3)
    turbine = c1.selectbox("Турбина", list(turbines))
    day = c2.date_input("Дата, UTC", value=date(2026, 2, 10))
    horizon = c3.slider("Горизонт, часов", 1, 72, 24)
    with_agent = st.checkbox("Анализ через OpenAI (нужны ключ и OPENAI_MODEL)", value=False)
    submitted = st.form_submit_button("Рассчитать прогноз", type="primary")
if submitted:
    st.session_state.pop("forecast_result", None)
    try:
        with st.spinner("Погода → признаки → ML → проверка → анализ…"):
            st.session_state.forecast_result = run_forecast(turbine, day, horizon,
                settings=settings, with_agent=with_agent)
    except ForecastError as exc:
        st.error(str(exc))
result = st.session_state.get("forecast_result")
if result is not None and result.demo == demo:
    st.subheader(f"{result.request.turbine_id} · {result.request.forecast_date} · {result.request.horizon_h} ч")
    stats = result.statistics
    cols = st.columns(3)
    cols[0].metric("Энергия, кВт·ч", f"{stats.predicted_energy_kwh:,.1f}" if stats.predicted_energy_kwh is not None else "Нет полного прогноза")
    cols[1].metric("Средний ветер, м/с", f"{stats.avg_wind_ms:.1f}" if stats.avg_wind_ms is not None else "Нет данных")
    cols[2].metric("Валидных часов", f"{stats.valid_hours}/{stats.requested_hours}")
    frame = pd.DataFrame([p.model_dump() for p in result.hourly]).set_index("timestamp")
    chart_frame = frame.reset_index()
    chart_frame["timestamp_utc"] = pd.to_datetime(chart_frame.timestamp, utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
    def chart(column, title):
        return alt.Chart(chart_frame).mark_line().encode(
            x=alt.X("timestamp:T", title="Время, UTC", scale=alt.Scale(type="utc"),
                    axis=alt.Axis(format="%d.%m %H:%M")),
            y=alt.Y(f"{column}:Q", title=title),
            tooltip=[alt.Tooltip("timestamp_utc:N", title="UTC"), alt.Tooltip(f"{column}:Q", title=title)],
        )
    st.subheader("Мощность, кВт")
    st.altair_chart(chart("power_kw", "кВт"), width="stretch")
    st.subheader("Ветер на высоте 80 м, м/с")
    st.altair_chart(chart("wind_speed_ms", "м/с"), width="stretch")
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
    st.caption(f"Погода: {result.weather_source} · Модель: {result.model_id}")
    st.download_button("Скачать JSON", result.model_dump_json(indent=2), "forecast.json", "application/json")
