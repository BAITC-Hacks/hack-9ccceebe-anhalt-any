const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase(); this.style = {}; this.dataset = {}; this.children = [];
    this.attributes = {}; this.events = {}; this.value = ''; this.textContent = ''; this.hidden = false;
    this.disabled = false; this.clientWidth = 300; this.clientHeight = 245;
  }
  append(...nodes) { this.children.push(...nodes); if (this.tagName === 'SELECT' && !this.value) this.value = nodes[0]?.value || ''; }
  replaceChildren(...nodes) { this.children = []; if (this.tagName === 'SELECT') this.value = ''; this.append(...nodes); }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  addEventListener(name, fn) { this.events[name] = fn; }
  remove() {}
  click() { return this.events.click?.({}); }
  focus() { document.activeElement = this; this.events.focus?.(); }
}
const ids = Object.fromEntries(['forecast-chart','chart-detail','data-mode','forecast-date','origin','horizon','turbine-filter','observed-turbine',
  'run-status','request-error','run-forecast','export-forecast','export-json','refresh-readiness','readiness-panel','blockers','data-status',
  'mode-label','mode-description','turbines-panel','observed-metrics','date-range','weather-source','model-source','analysis-source',
  'analysis-summary','analysis-recommendation','confidence-note','agent-status','step-weather','step-data','step-model','step-check',
  'run-results','chart-note','series-legend'].map(id => [id,new Element(['data-mode','horizon','turbine-filter','observed-turbine'].includes(id) ? 'select' : ['origin','forecast-date'].includes(id) ? 'input' : 'div')]));
ids.horizon.value = '24'; ids['turbine-filter'].value = 'all';
const metrics = Object.fromEntries(['energy','average','capacity','wind','max-wind','valid-hours','t1','t2','t1-capacity','t2-capacity'].map(key => [key,new Element()]));
const document = { activeElement: null, getElementById: id => ids[id], body: new Element('body'),
  createElement: tag => new Element(tag), createElementNS: (ns,tag) => new Element(tag),
  createTextNode: content => ({ textContent: content }),
  querySelectorAll(selector) { if (selector === '[data-metric]') return Object.values(metrics); const match = selector.match(/^\[data-metric="(.+)"\]$/); return match ? [metrics[match[1]]].filter(Boolean) : []; },
};
const registry = { T1: { rated_power_kw: 2500 }, T2: { rated_power_kw: 2500 } };
let pending, capturedBlob, respectAbort = false;
let timerId = 0;
const timers = new Map();
const requests = [];
const fetch = async (path, options) => {
  requests.push({ path, options });
  if (path === '/dashboard/config') return { ok: true, json: async () => ({ turbines: registry, observed_turbines: ['Kelmarsh 1','Kelmarsh 2'] }) };
  if (path === '/readiness') return { ok: true, json: async () => ({ ready: false, blockers: ['Model missing'] }) };
  return await new Promise((resolve, reject) => {
    pending = payload => resolve({ ok: true, json: async () => payload });
    if (respectAbort) options.signal.addEventListener('abort', () => reject(Object.assign(new Error('Aborted'), { name: 'AbortError' })), { once: true });
  });
};
const context = { document, fetch, Intl, Date, Math, String, Map, Set, Object, Array, JSON, Number, AbortController, Blob,
  URL: { createObjectURL: blob => { capturedBlob = blob; return 'blob:download'; }, revokeObjectURL() {} },
  window: { addEventListener() {}, clearTimeout(id) { timers.delete(id); }, setTimeout(callback, delay) { timers.set(++timerId, { callback, delay }); return timerId; } },
};
vm.runInNewContext(fs.readFileSync(require.resolve('../src/app/web/app.js'),'utf8'),context);
const flush = () => new Promise(resolve => setImmediate(resolve));
(async () => {
  await flush();
  assert.equal(ids['data-mode'].value, 'target');
  assert.equal(ids['run-forecast'].disabled, true);
  assert.equal(ids['export-forecast'].disabled, true);
  assert.equal(metrics.energy.textContent, '—');
  assert.equal(ids.blockers.children[0].textContent,'Model missing');
  ids['data-mode'].value = 'demo'; ids['data-mode'].events.change();
  assert.equal(ids['forecast-date'].value, '2026-02-10');
  assert.equal(ids['run-forecast'].disabled,false);
  const staleRun = ids['run-forecast'].click();
  assert.equal(ids.horizon.disabled,true);
  assert.equal(JSON.parse(requests.at(-1).options.body).forecast_date,'2026-02-10');
  ids.horizon.value = '48'; ids.horizon.events.change();
  pending({ mode: 'demo', results: {} }); await staleRun;
  assert.equal(metrics.energy.textContent,'—');
  assert.equal(ids['export-json'].disabled,true);

  ids['data-mode'].value = 'observed'; ids['data-mode'].events.change();
  assert.equal(ids['forecast-date'].value,'2017-11-08');
  assert.equal(ids['turbines-panel'].hidden,true);
  ids.horizon.value = '24'; ids.horizon.events.change();
  const observedRun = ids['run-forecast'].click();
  const samples = Array.from({ length: 144 }, (_, index) => ({ timestamp: new Date(Date.UTC(2017,10,8,0,index*10)).toISOString(),
    power_kw: index === 0 ? null : 500, actual_power_kw: 490, wind_speed_ms: 6, valid: index !== 0 }));
  pending({ mode:'observed_scada', samples, sampling_interval_minutes:10, status:'partial', source:'SCADA', model_id:'actual-model-id',
    metrics:{ mae_kw:10,rmse_kw:10,coverage:143/144,matched_samples:143,expected_samples:144 },
    analysis:{ summary:'<script>untrusted</script>',recommendation:'Review gaps',confidence_note:'No confidence percentage' }, analysis_source:'deterministic' });
  await observedRun;
  assert.equal(metrics.energy.textContent,'—');
  assert.equal(ids['model-source'].textContent,'actual-model-id');
  assert.equal(ids['analysis-summary'].textContent,'<script>untrusted</script>');
  assert.equal(ids['forecast-chart'].children[0].attributes.viewBox,'0 0 300 245');
  const svg = ids['forecast-chart'].children[0];
  svg.events.keydown({ key:'End', preventDefault() {} });
  assert.equal(svg.attributes['aria-valuenow'],'143');
  assert.equal(ids['export-forecast'].disabled,false);
  ids['export-forecast'].click();
  const csv = await capturedBlob.text();
  const lines = csv.trim().split('\r\n');
  assert.equal(lines.length,145);
  assert(lines[0].includes('model_power_kw'));
  assert(!lines[0].includes('T1'));
  assert(lines[1].includes(',,"false"'));
  ids['forecast-date'].value = '2017-11-09'; ids['forecast-date'].events.input();
  assert.equal(ids['export-forecast'].disabled,true);
  assert.equal(metrics.average.textContent,'—');
  ids['data-mode'].value = 'demo'; ids['data-mode'].events.change();
  const demoRun = ids['run-forecast'].click();
  const hourly = (wind, power) => Array.from({ length: 24 }, (_, index) => ({ timestamp: new Date(Date.UTC(2026,1,10,index)).toISOString(), power_kw:power, wind_speed_ms:wind, valid:true }));
  pending({ mode:'demo', results:{ T1:{ hourly:hourly(4,600) }, T2:{ hourly:hourly(8,900) } } });
  await demoRun;
  ids['turbine-filter'].value = 'T1'; ids['turbine-filter'].events.change();
  ids['export-forecast'].click();
  const selectedCSV = await capturedBlob.text();
  assert(selectedCSV.split('\r\n')[1].endsWith(',"4"'));
  assert(document.body.children.at(-1).download.includes('-T1-'));
  assert(!selectedCSV.split('\r\n')[0].includes('T2'));

  respectAbort = true;
  const timeoutRun = ids['run-forecast'].click();
  const timer = [...timers.values()].find(item => item.delay === 60000);
  assert(timer);
  timer.callback();
  await timeoutRun;
  assert(ids['request-error'].textContent.includes('60 секунд'));
  assert.equal(ids['run-forecast'].disabled,false);
  assert.equal(ids['export-forecast'].disabled,true);
  assert.equal([...timers.values()].some(item => item.delay === 60000),false);
  console.log('PASS: blocked default, controls, date defaults, stale-response discard, observed provenance, safe server text, adaptive chart, keyboard, CSV nulls/filter, result invalidation, timeout recovery');
})().catch(error => { console.error(error); process.exitCode = 1; });
