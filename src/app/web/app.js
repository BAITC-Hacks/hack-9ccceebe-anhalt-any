(() => {
  "use strict";

  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const mean = (values) => {
    const usable = values.filter(finite);
    return usable.length ? usable.reduce((sum, value) => sum + value, 0) / usable.length : null;
  };
  const sumComplete = (values) => values.length && values.every(finite)
    ? values.reduce((sum, value) => sum + value, 0) : null;
  const safeList = (value) => Array.isArray(value) ? value : [];
  const numberOrNull = (value) => finite(value) ? value : null;
  const toMW = (value) => finite(value) ? value / 1000 : null;
  const unique = (values) => [...new Set(values.filter(Boolean))];

  // Preserve raw model values and nulls. Validity determines integration, never chart clipping.
  function normalize(payload, mode, requestedHours) {
    if (mode === "observed") {
      return {
        mode, payload, horizon: requestedHours, intervalHours: (payload.sampling_interval_minutes || 10) / 60,
        series: [{ key: "model", label: "Модель", color: "#39c07f" }, { key: "actual", label: "Факт SCADA", color: "#6a91ff" }],
        rows: safeList(payload.samples).map((point) => ({
          time: point.timestamp, model: numberOrNull(point.power_kw), actual: numberOrNull(point.actual_power_kw),
          valid: { model: point.valid === true }, wind: numberOrNull(point.wind_speed_ms),
          energy: { model: point.valid === true && finite(point.power_kw) ? point.power_kw * (payload.sampling_interval_minutes || 10) / 60 : null },
        })),
      };
    }
    const byTime = new Map();
    const add = (time, turbine, point) => {
      if (!time || !["T1", "T2"].includes(turbine)) return;
      if (!byTime.has(time)) byTime.set(time, { time, T1: null, T2: null, total: null, valid: {}, energy: {}, winds: {} });
      const row = byTime.get(time);
      row[turbine] = numberOrNull(point.power_kw);
      row.valid[turbine] = point.valid === true;
      row.energy[turbine] = point.valid === true
        ? numberOrNull(mode === "target" ? point.energy_kwh : point.power_kw) : null;
      row.winds[turbine] = numberOrNull(point.wind_speed_ms);
    };
    if (mode === "demo") {
      ["T1", "T2"].forEach((key) => safeList(payload.results?.[key]?.hourly)
        .forEach((point) => add(point.timestamp, key, point)));
    } else {
      safeList(payload.rows).forEach((point) => add(point.valid_time, point.turbine_id, point));
    }
    const farm = new Map(safeList(payload.farm_rows).map((row) => [row.valid_time, row]));
    const rows = [...byTime.values()].sort((a, b) => Date.parse(a.time) - Date.parse(b.time));
    rows.forEach((row) => {
      const reported = farm.get(row.time);
      row.total = reported ? numberOrNull(reported.raw_power_kw) : sumComplete([row.T1, row.T2]);
      row.valid.total = reported ? reported.complete === true : row.valid.T1 === true && row.valid.T2 === true;
      row.energy.total = row.valid.total
        ? (reported ? numberOrNull(reported.energy_kwh) : sumComplete([row.energy.T1, row.energy.T2])) : null;
      row.wind = mean(Object.values(row.winds));
    });
    return {
      mode, payload, horizon: requestedHours, intervalHours: 1, rows,
      series: [{ key: "T1", label: "T1", color: "#39c07f" }, { key: "T2", label: "T2", color: "#6a91ff" },
        { key: "total", label: "Всего", color: "#dfe6f2", dashed: true }],
    };
  }

  function summarize(data, selected = "all", registry = {}) {
    const key = data.mode === "observed" ? "model" : selected === "all" ? "total" : selected;
    const rows = data.rows;
    const expectedSamples = Math.round(data.horizon / data.intervalHours);
    const energies = rows.map((row) => row.energy[key]);
    const integrated = rows.length === expectedSamples ? sumComplete(energies) : null;
    const validHours = rows.filter((row) => row.valid[key] === true).length * data.intervalHours;
    const selectedRegistry = selected === "all" ? [registry.T1, registry.T2] : [registry[selected]];
    const rating = data.mode === "observed" ? null : sumComplete(selectedRegistry.map((entry) => entry?.rated_power_kw));
    const winds = rows.flatMap((row) => data.mode === "observed" ? [row.wind]
      : selected === "all" ? [row.winds.T1, row.winds.T2] : [row.winds[selected]]).filter(finite);
    return {
      energy: integrated,
      average: mean(rows.map((row) => row[key])),
      capacity: finite(integrated) && finite(rating) && rating > 0 ? integrated / data.horizon / rating * 100 : null,
      validHours, expectedSamples, wind: mean(winds), maxWind: winds.length ? Math.max(...winds) : null,
    };
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { normalize, summarize, finite, sumComplete };
  if (typeof document === "undefined") return;

  const $ = (id) => document.getElementById(id);
  const text = (id, value) => { if ($(id)) $(id).textContent = value ?? "—"; };
  const hidden = (id, value) => { if ($(id)) $(id).hidden = value; };
  const format = (value, digits = 1) => finite(value)
    ? new Intl.NumberFormat("ru-RU", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value) : "—";
  const metric = (key, value) => document.querySelectorAll(`[data-metric="${key}"]`)
    .forEach((node) => { node.textContent = value; });
  const chart = $("forecast-chart");
  const modeControl = $("data-mode");
  const dateControl = $("forecast-date");
  const originControl = $("origin");
  const horizonControl = $("horizon");
  const turbineControl = $("turbine-filter");
  const observedControl = $("observed-turbine");
  const state = {
    mode: "target", busy: false, loadingConfig: true, readiness: null, config: null, data: null,
    requestId: 0, controller: null, selectedPoint: 0, chart: null, readinessId: 0,
    dates: { demo: "2026-02-10", observed: "2017-11-08" },
  };
  if (modeControl) modeControl.value = "target";
  const horizon = () => horizonControl?.value === "48" ? 48 : 24;
  const selected = () => state.mode === "observed" ? "all"
    : ["T1", "T2"].includes(turbineControl?.value) ? turbineControl.value : "all";
  const scope = () => JSON.stringify({ mode: state.mode, horizon: horizon(),
    date: state.mode === "target" ? originControl?.value : dateControl?.value,
    turbine: state.mode === "observed" ? observedControl?.value : "all" });
  const datetime = (value) => {
    const stamp = new Date(value);
    if (!Number.isFinite(stamp.getTime())) return String(value || "—");
    return new Intl.DateTimeFormat("ru-RU", { timeZone: "UTC", day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" }).format(stamp);
  };
  const announce = (message) => text("run-status", message);
  const error = (message) => { text("request-error", message || ""); hidden("request-error", !message); };

  async function request(path, { body, signal } = {}) {
    const controller = new AbortController();
    let timedOut = false;
    const abort = () => controller.abort(signal?.reason);
    if (signal?.aborted) abort();
    else signal?.addEventListener("abort", abort, { once: true });
    const timeout = window.setTimeout(() => { timedOut = true; controller.abort(); }, 60000);
    try {
      const response = await fetch(path, {
        method: body ? "POST" : "GET", signal: controller.signal,
        headers: body ? { "Content-Type": "application/json" } : {},
        ...(body ? { body: JSON.stringify(body) } : {}),
      });
      let result;
      try { result = await response.json(); } catch (cause) {
        if (cause.name === "AbortError") throw cause;
        throw new Error(`Сервер вернул некорректный ответ (${response.status}).`);
      }
      if (!response.ok) {
        const detail = typeof result.detail === "string" ? result.detail
          : Array.isArray(result.detail) ? result.detail.map((item) => item.msg || "Ошибка параметров").join("; ")
            : `Ошибка запроса (${response.status})`;
        throw new Error(detail);
      }
      return result;
    } catch (cause) {
      if (timedOut) throw new Error("Сервер не ответил за 60 секунд. Повторите расчёт или проверьте доступность источников.");
      throw cause;
    } finally {
      window.clearTimeout(timeout);
      signal?.removeEventListener("abort", abort);
    }
  }

  function controls() {
    ["data-mode", "horizon", "origin", "forecast-date", "observed-turbine", "turbine-filter"]
      .forEach((id) => { if ($(id)) $(id).disabled = state.busy; });
    if (turbineControl) turbineControl.disabled = state.busy || state.mode === "observed";
    if (observedControl) observedControl.disabled = state.busy || state.loadingConfig;
    if ($("run-forecast")) {
      $("run-forecast").disabled = state.busy || state.loadingConfig
        || (state.mode === "target" && state.readiness?.ready !== true)
        || (state.mode === "observed" && !observedControl?.value);
      $("run-forecast").setAttribute("aria-busy", String(state.busy));
    }
    ["export-forecast", "export-json"].forEach((id) => {
      if ($(id)) $(id).disabled = state.busy || !state.data;
    });
    if ($("export-forecast")) $("export-forecast").disabled = state.busy || !state.data?.rows.length;
    if ($("refresh-readiness")) $("refresh-readiness").disabled = state.busy;
  }

  function renderReadiness() {
    hidden("readiness-panel", state.mode !== "target" || state.readiness?.ready === true);
    const blockers = $("blockers");
    if (blockers) {
      blockers.replaceChildren();
      const messages = state.readiness ? safeList(state.readiness.blockers) : ["Проверяем доступность протокола, погоды и модели…"];
      messages.forEach((message) => { const item = document.createElement("li"); item.textContent = message; blockers.append(item); });
    }
    if (!state.data && !state.busy) {
      text("data-status", state.mode === "target"
        ? state.readiness?.ready === true ? "Готов к запуску" : "Расчёт недоступен"
        : "Ожидает запуска");
    }
    controls();
  }

  const modeInfo = {
    target: ["Целевой прогноз", "T1 и T2 · архивный прогноз погоды и модель Goldwind. Расчёт доступен после проверки протокола и источников."],
    demo: ["Демонстрация", "Синтетическая погода и демонстрационная модель. Результаты показывают работу системы и не являются прогнозом для ВЭС."],
    observed: ["Историческая проверка", "Kelmarsh · модель и фактическая SCADA на 10-минутных наблюдениях. Это оценка на исторических данных, а не прогноз будущей погоды."],
  };

  function renderMode() {
    text("run-forecast", state.mode === "observed" ? "Рассчитать и сравнить" : "Рассчитать прогноз");
    text("mode-label", modeInfo[state.mode][0]);
    text("mode-description", modeInfo[state.mode][1]);
    text("forecast-heading", state.mode === "observed" ? "Модель и факт" : "Прогноз выработки");
    text("workspace-context", state.mode === "observed" ? "Kelmarsh / SCADA · шаг 10 минут" : "Goldwind GW109/2500 / 2 × 2,5 МВт");
    document.querySelectorAll("[data-mode-field]").forEach((node) => {
      const field = node.dataset.modeField;
      node.hidden = field === "date" ? state.mode === "target"
        : field === "forecast" ? state.mode === "observed" : field !== state.mode;
    });
    if (dateControl) {
      dateControl.min = state.mode === "demo" ? "2026-01-31" : "";
      dateControl.max = state.mode === "demo" ? (horizon() === 48 ? "2026-02-27" : "2026-02-28") : "";
    }
    hidden("turbines-panel", state.mode === "observed");
    hidden("observed-metrics", state.mode !== "observed");
    renderReadiness();
    renderLegend();
  }

  function clearResults() {
    state.data = null;
    state.chart = null;
    state.selectedPoint = 0;
    document.querySelectorAll("[data-metric]").forEach((node) => { node.textContent = "—"; });
    ["date-range", "weather-source", "model-source", "analysis-source"].forEach((id) => text(id, "—"));
    text("analysis-summary", "Запустите расчёт, чтобы получить проверяемые результаты.");
    text("analysis-recommendation", "Рекомендации появятся после анализа данных.");
    text("confidence-note", "Числовая оценка уверенности не предоставлена.");
    text("agent-status", "Ожидает запуска");
    ["step-weather", "step-data", "step-model", "step-check"].forEach((id) => text(id, "Не запущен"));
    text("run-results", "");
    text("observed-metrics", "MAE, RMSE и покрытие появятся после исторической проверки.");
    text("chart-detail", "Нет результатов для выбранных параметров.");
    text("chart-note", "Пропуски не заменяются нулями. Мощность — МВт, энергия — МВт·ч.");
    if (chart) {
      const note = document.createElement("div"); note.className = "empty-chart";
      const title = document.createElement("strong"); title.textContent = "От данных — к прогнозу";
      const instruction = document.createElement("span"); instruction.textContent = "Выберите источник и запустите расчёт.";
      note.append(title, instruction); chart.replaceChildren(note);
    }
    renderReadiness();
  }

  function invalidate(message = "Параметры изменены. Запустите расчёт заново.") {
    state.requestId += 1;
    state.controller?.abort();
    state.controller = null;
    state.busy = false;
    clearResults();
    error("");
    announce(message);
    renderMode();
  }

  function visibleSeries() {
    const series = state.data?.series || (state.mode === "observed"
      ? [{ key: "model", label: "Модель", color: "#39c07f" }, { key: "actual", label: "Факт SCADA", color: "#6a91ff" }]
      : [{ key: "T1", label: "T1", color: "#39c07f" }, { key: "T2", label: "T2", color: "#6a91ff" }, { key: "total", label: "Всего", color: "#dfe6f2", dashed: true }]);
    return state.mode === "observed" || selected() === "all" ? series : series.filter((item) => item.key === selected());
  }

  function renderLegend() {
    const legend = $("series-legend");
    if (!legend) return;
    legend.replaceChildren();
    visibleSeries().forEach((item) => {
      const label = document.createElement("span");
      const swatch = document.createElement("i");
      swatch.style.backgroundColor = item.color;
      label.append(swatch, document.createTextNode(item.label));
      legend.append(label);
    });
  }

  function renderAnalysis() {
    const data = state.data;
    if (!data) return;
    const reports = data.mode === "demo"
      ? (selected() === "all" ? ["T1", "T2"] : [selected()]).map((key) => ({ key, result: data.payload.results?.[key] })).filter((item) => item.result)
      : [{ key: "", result: data.payload }];
    const analysisText = (key, fallback) => reports.filter((item) => item.result.analysis?.[key])
      .map((item) => `${item.key ? item.key + ": " : ""}${item.result.analysis[key]}`).join("\n\n") || fallback;
    text("analysis-summary", analysisText("summary", "Анализ недоступен: прогноз не сформирован."));
    text("analysis-recommendation", analysisText("recommendation", "Проверьте причины недоступности расчёта."));
    text("confidence-note", analysisText("confidence_note", "Числовая оценка уверенности не предоставлена."));
    const sources = unique(reports.map((item) => item.result.analysis_source));
    text("analysis-source", sources.map((source) => ({ deterministic: "Проверки по правилам", openai: "OpenAI", unavailable: "Недоступен" })[source] || source).join(" · ") || "—");
    text("agent-status", sources.includes("openai") ? "Анализ OpenAI завершён"
      : sources.includes("deterministic") ? "Проверки по правилам завершены · LLM не вызывался" : "Не выполнен");
    const weather = data.mode === "observed" ? [data.payload.source]
      : data.mode === "demo" ? reports.map((item) => item.result.weather_source)
        : Object.values(data.payload.weather_snapshots || {}).map((item) => item.source || item.weather_version);
    text("weather-source", unique(weather).join(" · ") || "Источник не получен");
    const models = data.mode === "demo" ? reports.map((item) => item.result.model_id)
      : [data.payload.model_id || data.payload.provenance?.model_manifest?.model_version];
    text("model-source", unique(models).join(" · ") || "Модель не получена");
    const hasWeather = data.rows.some((row) => finite(row.wind));
    const hasModel = data.rows.some((row) => data.series.some((item) => item.key !== "actual" && finite(row[item.key])));
    text("step-weather", hasWeather ? data.mode === "demo" ? "Синтетическая" : data.mode === "observed" ? "SCADA" : "Архив получен" : "Нет данных");
    text("step-data", data.rows.length ? "Обработаны" : "Нет данных");
    text("step-model", hasModel ? "Выполнен" : "Не выполнен");
    text("step-check", reports.some((item) => item.result.analysis) ? "Выполнены" : "Не выполнены");
    const issues = data.mode === "target" ? Object.values(data.payload.errors || {}) : [];
    text("run-results", [data.payload.limitation, ...issues,
      data.payload.run_id ? `ID расчёта: ${data.payload.run_id}${data.payload.cache_hit ? " · сохранённый результат" : ""}` : ""].filter(Boolean).join("\n"));
  }

  function renderResults() {
    const data = state.data;
    if (!data) return;
    const summary = summarize(data, selected(), state.config?.turbines || {});
    metric("energy", format(toMW(summary.energy)));
    metric("average", format(toMW(summary.average), 2));
    metric("capacity", finite(summary.capacity) ? `${format(summary.capacity)}%` : "—");
    metric("wind", format(summary.wind));
    metric("max-wind", format(summary.maxWind));
    metric("valid-hours", `${format(summary.validHours, summary.validHours % 1 ? 1 : 0)} / ${data.horizon}`);
    ["T1", "T2"].forEach((key) => {
      const item = data.mode === "observed" ? null : summarize(data, key, state.config?.turbines || {});
      metric(key.toLowerCase(), item ? format(toMW(item.average), 2) : "—");
      metric(`${key.toLowerCase()}-capacity`, finite(item?.capacity) ? `${format(item.capacity)}%` : "—");
    });
    if (data.rows.length) text("date-range", `${datetime(data.rows[0].time)} — ${datetime(data.rows.at(-1).time)} · UTC`);
    const complete = summary.validHours >= data.horizon && data.rows.length === summary.expectedSamples;
    text("data-status", !data.rows.length ? "Расчёт заблокирован" : complete ? "Расчёт завершён" : "Неполные / некорректные данные");
    if (data.mode === "observed") {
      const metrics = data.payload.metrics || {};
      text("observed-metrics", `MAE ${format(metrics.mae_kw)} кВт · RMSE ${format(metrics.rmse_kw)} кВт · покрытие ${finite(metrics.coverage) ? format(metrics.coverage * 100) + "%" : "—"} (${metrics.matched_samples ?? "—"} / ${metrics.expected_samples ?? "—"} точек). Шаг: ${data.payload.sampling_interval_minutes || 10} мин.`);
    }
    text("chart-note", `${data.mode === "demo" ? "Демонстрационные данные. " : ""}Показаны исходные значения модели; некорректные значения не обрезаются. Пропуски — разрывы линий. ${finite(summary.energy) ? "Энергия рассчитана по всем интервалам." : "Энергия за полный период недоступна: есть пропуски или некорректные интервалы."}`);
    renderLegend();
    renderAnalysis();
    renderChart();
    controls();
  }

  function chartSize() {
    return { width: Math.max(280, Math.round(chart?.clientWidth || 620)), height: Math.max(220, Math.round(chart?.clientHeight || 270)) };
  }

  function showPoint(index) {
    if (!state.chart || !state.data?.rows.length) return;
    const { svg, cursor, markers, x, y } = state.chart;
    const rows = state.data.rows;
    state.selectedPoint = Math.max(0, Math.min(rows.length - 1, index));
    const row = rows[state.selectedPoint];
    const parts = visibleSeries().map((item) => `${item.label}: ${format(toMW(row[item.key]), 3)} МВт${row.valid[item.key] === false ? " (некорректный интервал)" : ""}`);
    const label = `${datetime(row.time)} UTC · ${parts.join(" · ")}`;
    text("chart-detail", label);
    svg.setAttribute("aria-valuenow", String(state.selectedPoint));
    svg.setAttribute("aria-valuetext", label);
    cursor.setAttribute("x1", x(state.selectedPoint));
    cursor.setAttribute("x2", x(state.selectedPoint));
    markers.forEach(({ element, key }) => {
      element.setAttribute("visibility", finite(row[key]) ? "visible" : "hidden");
      if (finite(row[key])) { element.setAttribute("cx", x(state.selectedPoint)); element.setAttribute("cy", y(toMW(row[key]))); }
    });
  }

  function renderChart() {
    if (!chart || !state.data) return;
    const rows = state.data.rows;
    if (!rows.length) {
      const note = document.createElement("p"); note.className = "empty-chart"; note.textContent = "Прогноз не сформирован. Проверьте причины блокировки.";
      chart.replaceChildren(note); state.chart = null; text("chart-detail", "Нет прогнозных значений."); return;
    }
    const series = visibleSeries();
    const values = rows.flatMap((row) => series.map((item) => toMW(row[item.key]))).filter(finite);
    const min = Math.min(0, ...values);
    const max = Math.max(0.1, ...values);
    const range = Math.max(0.1, max - min);
    const low = min < 0 ? min - range * 0.06 : 0;
    const high = max + range * 0.08;
    const { width, height } = chartSize();
    const margin = { left: 42, right: 15, top: 24, bottom: 36 };
    const bottom = height - margin.bottom;
    const x = (index) => margin.left + (rows.length > 1 ? index / (rows.length - 1) : 0.5) * (width - margin.left - margin.right);
    const y = (value) => bottom - (value - low) / (high - low) * (bottom - margin.top);
    const create = (tag, attributes = {}, content) => {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
      if (content !== undefined) node.textContent = content;
      return node;
    };
    const restoreFocus = document.activeElement === state.chart?.svg;
    const svg = create("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height: "100%", tabindex: 0, role: "slider",
      "aria-label": "Мощность по времени. Стрелки меняют точку; Home и End — начало и конец",
      "aria-valuemin": 0, "aria-valuemax": rows.length - 1, "aria-orientation": "horizontal", "aria-describedby": "chart-detail" });
    svg.style.display = "block"; svg.style.borderRadius = "10px";
    svg.append(create("title", {}, `${modeInfo[state.mode][0]} · ${horizon()} ч · мощность в МВт`));
    const graphics = create("g", { "aria-hidden": true }); svg.append(graphics);
    for (let index = 0; index <= 4; index += 1) {
      const value = low + (high - low) * index / 4;
      graphics.append(create("line", { x1: margin.left, y1: y(value), x2: width - margin.right, y2: y(value), stroke: "#2c3743" }));
      graphics.append(create("text", { x: margin.left - 8, y: y(value) + 4, fill: "#a5b0be", "font-size": 11, "text-anchor": "end" }, format(value, high > 15 ? 0 : 1)));
    }
    graphics.append(create("text", { x: margin.left, y: 13, fill: "#a5b0be", "font-size": 11 }, "МВт"));
    const tickCount = width < 600 ? 3 : 5;
    const tickIndexes = unique(Array.from({ length: tickCount }, (_, index) => String(Math.round(index * (rows.length - 1) / (tickCount - 1))))).map(Number);
    tickIndexes.forEach((index, position) => {
      const label = new Intl.DateTimeFormat("ru-RU", { timeZone: "UTC", day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }).format(new Date(rows[index].time));
      graphics.append(create("text", { x: x(index), y: height - 10, fill: "#a5b0be", "font-size": 10,
        "text-anchor": position === 0 ? "start" : position === tickIndexes.length - 1 ? "end" : "middle" }, label));
    });
    series.forEach((item) => {
      let penDown = false;
      const segments = rows.map((row, index) => {
        if (!finite(row[item.key])) { penDown = false; return ""; }
        const segment = `${penDown ? "L" : "M"}${x(index)},${y(toMW(row[item.key]))}`;
        penDown = true; return segment;
      }).join(" ");
      graphics.append(create("path", { d: segments, fill: "none", stroke: item.color, "stroke-width": item.dashed ? 1.8 : 2.6,
        "stroke-dasharray": item.dashed ? "5 5" : "none", "stroke-linecap": "round", "stroke-linejoin": "round" }));
    });
    const cursor = create("line", { y1: margin.top, y2: bottom, stroke: "#a5b0be", "stroke-dasharray": "3 5" }); graphics.append(cursor);
    const markers = series.map((item) => { const element = create("circle", { r: 4, fill: item.color, stroke: "#171d24", "stroke-width": 2 }); graphics.append(element); return { element, key: item.key }; });
    chart.replaceChildren(svg);
    state.chart = { svg, cursor, markers, x, y, width, height };
    const pointer = (event) => {
      const point = svg.createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
      const matrix = svg.getScreenCTM(); if (!matrix) return;
      const local = point.matrixTransform(matrix.inverse());
      showPoint(Math.round((local.x - margin.left) / (width - margin.left - margin.right) * (rows.length - 1)));
    };
    svg.addEventListener("pointermove", pointer);
    svg.addEventListener("click", (event) => { pointer(event); svg.focus({ preventScroll: true }); });
    svg.addEventListener("focus", () => showPoint(state.selectedPoint));
    svg.addEventListener("keydown", (event) => {
      const offsets = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1, PageUp: 6, PageDown: -6 };
      if (event.key in offsets) showPoint(state.selectedPoint + offsets[event.key]);
      else if (event.key === "Home") showPoint(0);
      else if (event.key === "End") showPoint(rows.length - 1);
      else return;
      event.preventDefault();
    });
    showPoint(state.selectedPoint);
    if (restoreFocus) svg.focus({ preventScroll: true });
  }

  async function run() {
    if (state.busy) return;
    if (state.mode === "target" && state.readiness?.ready !== true) { announce("Целевой расчёт недоступен. Проверьте причины блокировки."); return; }
    const mode = state.mode;
    let path, body;
    if (mode === "target") {
      const origin = originControl?.value.trim();
      if (!origin || !/(Z|[+-]\d{2}:\d{2})$/.test(origin) || !Number.isFinite(Date.parse(origin))) { error("Укажите дату и время ISO с часовым поясом, например 2026-01-31T23:00:00+05:00."); return; }
      path = "/target-forecast"; body = { forecast_origin: origin, horizon_h: horizon(), refresh: false, with_agent: false };
    } else {
      const date = dateControl?.value;
      if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) { error("Укажите дату расчёта."); return; }
      if (mode === "observed") { path = "/observed"; body = { turbine_id: observedControl?.value, observation_date: date, horizon_h: horizon() }; }
      else { path = "/demo-forecast"; body = { forecast_date: date, horizon_h: horizon() }; }
    }
    state.controller?.abort();
    const controller = new AbortController(); state.controller = controller;
    const id = ++state.requestId, requestScope = scope(), requestedHours = horizon();
    clearResults(); state.busy = true; error(""); controls();
    announce("Выполняется расчёт…"); text("data-status", "Выполняется расчёт");
    try {
      const payload = await request(path, { body, signal: controller.signal });
      if (id !== state.requestId || requestScope !== scope()) return;
      state.data = normalize(payload, mode, requestedHours);
      state.data.scope = requestScope;
      renderResults();
      announce(payload.status === "blocked" ? "Расчёт заблокирован. Прогноз не сформирован." : "Расчёт завершён. Результаты получены от сервера.");
    } catch (cause) {
      if (id !== state.requestId || requestScope !== scope() || cause.name === "AbortError") return;
      error(cause.message || "Не удалось выполнить расчёт.");
      announce("Расчёт не выполнен. Исправьте параметры или проверьте доступность данных.");
      text("data-status", "Ошибка расчёта");
    } finally {
      if (id === state.requestId) { state.busy = false; state.controller = null; controls(); }
    }
  }

  async function refreshReadiness() {
    const id = ++state.readinessId;
    if ($("refresh-readiness")) $("refresh-readiness").disabled = true;
    try {
      const result = await request("/readiness");
      if (id !== state.readinessId) return;
      state.readiness = result;
      renderReadiness();
      if (!state.data && !state.busy && state.mode === "target") announce(result.ready ? "Проверка готовности пройдена. Можно запустить расчёт." : "Целевой прогноз заблокирован: необходимые источники ещё не настроены.");
    } catch (cause) {
      if (id !== state.readinessId) return;
      state.readiness = { ready: false, blockers: [`Не удалось проверить готовность: ${cause.message}`] };
      renderReadiness();
    } finally { if (id === state.readinessId) controls(); }
  }

  function download(content, mime, extension) {
    const url = URL.createObjectURL(new Blob([content], { type: mime }));
    const link = document.createElement("a"); link.href = url;
    const selection = state.mode === "observed" ? (observedControl?.value || "Kelmarsh").replaceAll(" ", "-")
      : selected() === "all" ? "T1-T2" : selected();
    link.download = `windagent-${state.mode}-${selection}-${state.mode === "target" ? originControl.value.slice(0, 10) : dateControl.value}-${horizon()}h.${extension}`;
    document.body.append(link); link.click(); link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  $("export-json")?.addEventListener("click", () => {
    if (!state.data || state.busy || state.data.scope !== scope()) return;
    download(JSON.stringify({ ui_mode: state.mode, ...state.data.payload }, null, 2), "application/json;charset=utf-8", "json");
    announce("JSON с исходными результатами и происхождением данных подготовлен.");
  });
  $("export-forecast")?.addEventListener("click", () => {
    if (!state.data?.rows.length || state.busy || state.data.scope !== scope()) return;
    const series = visibleSeries();
    const headings = ["mode", "timestamp", ...series.flatMap((item) => [`${item.key}_power_kw`, `${item.key}_valid`]), "wind_speed_ms"];
    const cell = (value) => value === null || value === undefined ? "" : '"' + String(value).replaceAll('"', '""') + '"';
    const lines = [headings, ...state.data.rows.map((row) => [state.mode, row.time,
      ...series.flatMap((item) => [row[item.key], row.valid[item.key] ?? null]),
      state.mode !== "observed" && selected() !== "all" ? row.winds[selected()] : row.wind])];
    download("\uFEFF" + lines.map((row) => row.map(cell).join(",")).join("\r\n"), "text/csv;charset=utf-8", "csv");
    announce(`CSV подготовлен: ${state.data.rows.length} интервалов. Пропуски оставлены пустыми; единицы указаны в заголовках.`);
  });

  $("run-forecast")?.addEventListener("click", run);
  $("refresh-readiness")?.addEventListener("click", refreshReadiness);
  modeControl?.addEventListener("change", () => {
    if (state.mode !== "target" && dateControl?.value) state.dates[state.mode] = dateControl.value;
    state.mode = ["demo", "observed"].includes(modeControl.value) ? modeControl.value : "target";
    if (state.mode !== "target" && dateControl) dateControl.value = state.dates[state.mode];
    invalidate("Режим изменён. Запустите расчёт для выбранных данных.");
  });
  [horizonControl, originControl, dateControl, observedControl].filter(Boolean)
    .forEach((control) => control.addEventListener(control.tagName === "INPUT" ? "input" : "change", () => invalidate()));
  turbineControl?.addEventListener("change", () => { if (state.data) renderResults(); else renderLegend(); });

  let resizeTimer;
  const resize = () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => {
      if (!state.chart) return;
      const { width, height } = chartSize();
      if (Math.abs(width - state.chart.width) >= 2 || Math.abs(height - state.chart.height) >= 2) renderChart();
    }, 100);
  };
  if (chart && "ResizeObserver" in window) new window.ResizeObserver(resize).observe(chart);
  else window.addEventListener("resize", resize);

  clearResults(); renderMode();
  refreshReadiness();
  request("/dashboard/config").then((config) => {
    state.config = config;
    if (originControl && !originControl.value) originControl.value = config.first_forecast_origin || "";
    if (observedControl) {
      observedControl.replaceChildren();
      safeList(config.observed_turbines).forEach((name) => { const option = document.createElement("option"); option.value = name; option.textContent = name; observedControl.append(option); });
    }
    document.querySelectorAll("[data-turbine-detail]").forEach((node) => {
      const turbine = config.turbines?.[node.dataset.turbineDetail];
      if (turbine) node.textContent = `${turbine.model_name} · ${format(toMW(turbine.rated_power_kw))} МВт · ${format(turbine.hub_height_m, 0)} м · ${format(turbine.latitude, 5)}°, ${format(turbine.longitude, 5)}°`;
    });
  }).catch((cause) => { error(`Не удалось загрузить конфигурацию: ${cause.message}`); })
    .finally(() => { state.loadingConfig = false; controls(); });
})();
