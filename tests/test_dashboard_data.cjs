const assert = require('node:assert/strict');
const { normalize, summarize, finite } = require('../src/app/web/app.js');
const registry = { T1: { rated_power_kw: 2500 }, T2: { rated_power_kw: 2500 } };
const hourly = (value) => Array.from({ length: 24 }, (_, index) => ({
  timestamp: new Date(Date.UTC(2026, 1, 10, index)).toISOString(),
  power_kw: value, wind_speed_ms: 8, valid: true,
}));
const demo = { mode: 'demo', results: { T1: { hourly: hourly(1000) }, T2: { hourly: hourly(1500) } } };
assert.equal(finite(null), false);
let data = normalize(demo, 'demo', 24);
assert.deepEqual(summarize(data, 'all', registry), {
  energy: 60000, average: 2500, capacity: 50, validHours: 24, expectedSamples: 24, wind: 8, maxWind: 8,
});
assert.equal(summarize(data, 'T1', registry).energy, 24000);
assert.equal(summarize(data, 'T2', registry).capacity, 60);

const missing = structuredClone(demo);
missing.results.T2.hourly[8].power_kw = null;
missing.results.T2.hourly[8].valid = false;
data = normalize(missing, 'demo', 24);
assert.equal(data.rows[8].T2, null);
assert.equal(data.rows[8].total, null);
assert.equal(summarize(data, 'all', registry).energy, null);
assert.equal(summarize(data, 'all', registry).capacity, null);
assert.equal(summarize(data, 'all', registry).validHours, 23);
assert.equal(summarize(data, 'T1', registry).energy, 24000);

const invalid = structuredClone(demo);
invalid.results.T1.hourly[0].power_kw = -300;
invalid.results.T1.hourly[0].valid = false;
invalid.results.T1.hourly[1].power_kw = 2700;
invalid.results.T1.hourly[1].valid = false;
data = normalize(invalid, 'demo', 24);
assert.equal(data.rows[0].T1, -300);
assert.equal(data.rows[1].T1, 2700);
assert.equal(summarize(data, 'T1', registry).energy, null);
assert.equal(summarize(data, 'T1', registry).validHours, 22);

const absent = structuredClone(demo);
absent.results.T2.hourly.splice(10, 1);
data = normalize(absent, 'demo', 24);
assert.equal(data.rows.length, 24);
assert.equal(data.rows[10].T2, null);
assert.equal(summarize(data, 'all', registry).energy, null);

const observed = { sampling_interval_minutes: 10, samples: Array.from({ length: 144 }, (_, index) => ({
  timestamp: new Date(Date.UTC(2017, 10, 8, 0, index * 10)).toISOString(),
  power_kw: 600, actual_power_kw: index === 0 ? null : 590, wind_speed_ms: 7, valid: true,
})) };
data = normalize(observed, 'observed', 24);
assert.deepEqual(data.series.map((series) => series.key), ['model', 'actual']);
assert.equal(data.rows[0].actual, null);
assert.equal(summarize(data, 'all', registry).energy, 14400);
assert.equal(summarize(data, 'all', registry).validHours, 24);
assert.equal(summarize(data, 'all', registry).capacity, null);
assert.equal(summarize(data, 'all', registry).expectedSamples, 144);

const target = { rows: ['T1', 'T2'].flatMap((turbine_id) => hourly(1000).map((point) => ({
  ...point, valid_time: point.timestamp, turbine_id, energy_kwh: 1000,
}))), farm_rows: hourly(2000).map((point) => ({ valid_time: point.timestamp, raw_power_kw: 2000, energy_kwh: 2000, complete: true })) };
target.farm_rows[3].raw_power_kw = null;
target.farm_rows[3].energy_kwh = null;
target.farm_rows[3].complete = false;
data = normalize(target, 'target', 24);
assert.equal(data.rows[3].total, null);
assert.equal(summarize(data, 'all', registry).energy, null);
assert.equal(summarize(data, 'T1', registry).energy, 24000);
data = normalize({ status: 'blocked', rows: [], farm_rows: [] }, 'target', 24);
assert.equal(summarize(data, 'all', registry).energy, null);
assert.equal(summarize(data, 'all', registry).average, null);
console.log('PASS: complete and partial energy, null/gap preservation, raw invalid predictions, native 10-minute integration, per-turbine filtering, blocked target');
