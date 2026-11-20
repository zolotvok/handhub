/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Schmiz Otto */
'use strict';
/* HandHub web UI. Vanilla JS, no dependencies; talks only to the hub it was served from. */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const E = new Proxy({}, { get: (c, id) => (c[id] ??= document.getElementById(id)) }); // E.someId -> element
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fixed = (v, d = 0) => (v == null || Number.isNaN(v) ? '--' : v.toFixed(d));
const setText = (el, t) => { if (el._t !== t) el.textContent = el._t = t; }; // skip no-op DOM writes
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable: fine */ } },
};
const refs = (root, key) => Object.fromEntries($$(`[data-${key}]`, root).map((n) => [n.dataset[key], n])); // data-<key>="name" -> {name: element}
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'text') el.textContent = v; else el.setAttribute(k === 'cls' ? 'class' : k, v);
  }
  el.append(...kids);
  return el;
}

// One entry per camera, in display order. `id` is the key in snapshot.cameras and in /stream/<id>.mjpg,
// `prefix` is the camera's config path. The wall tile comes from #tpl-tile and the settings section from #tpl-cam;
// elements marked data-only="<ids>" stay on those cameras. An `optional` camera (off by default) has its tile and
// status chip only while it is enabled; its settings section is always there, so it can be switched on.
const CAMS = [
  { id: 'webcam', title: 'Laptop webcam', short: 'Webcam', prefix: 'camera', noun: 'your webcam' },
  { id: 'phone', title: 'Phone (USB)', short: 'Phone', prefix: 'extra_cameras.phone', noun: 'the phone camera' },
  { id: 'pico', title: 'Pico camera', short: 'Pico', prefix: 'extra_cameras.pico', noun: 'the Pico camera' },
  { id: 'headset', title: 'Headset camera', short: 'Headset', prefix: 'extra_cameras.headset', noun: 'the headset camera', optional: true },
  { id: 'ir', title: 'Infrared camera', short: 'Infrared', prefix: 'extra_cameras.ir', noun: 'the infrared camera', optional: true },
];
const PHONE_CAMS = ['phone', 'headset']; // cameras that are an Android phone over adb
const camName = (id) => CAMS.find((c) => c.id === id)?.short ?? id;
const DEFAULT_AR = 4 / 3; // width / height of a preview until its camera reports a size
const ROTATIONS = [[0, '0° (as captured)'], [90, '90° (phone standing upright)'], [180, '180° (upside down)'], [270, '270° (turned the other way)']];
const LIGHT = [['off', 'Off'], ['gamma', 'Brighten'], ['local', 'Brighten + even out']]; // light_correction
const ZOOMS = [[0, 'default'], [1, '1x main'], [0.67, '0.67x ultrawide']]; // headset camera zoom (a number)
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

const SIDE = ['L', 'R'];
const HANDS = ['Left', 'Right'];
const BUTTONS = ['Trigger', 'Grip', 'A', 'B']; // button bits 1, 2, 4, 8
const EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
const COL = {}; // theme colours for the canvases, read from CSS variables
const readColors = () => {
  const cs = getComputedStyle(document.documentElement);
  for (const n of ['bg', 'panel', 'panel-2', 'grid', 'muted', 'text', 'left', 'right', 'accent', 'line', 'ok', 'warn', 'bad']) COL[n] = cs.getPropertyValue('--' + n).trim();
};

/* ===================================================================== config */

// [group, config path, label, min, max, step, unit, hint]
const SLIDERS = [
  ...CAMS.map((c) => [`${c.id}-main`, `${c.prefix}.hfov_deg`, 'Field of view', 40, 120, 1, '°', `How wide ${c.noun} sees. A wrong value makes hands drift.`]),
  ['pico-main', 'extra_cameras.pico.ae_target', 'Exposure target', 16, 110, 1, '', 'How bright the Pico aims to make the picture. Raise it if a lamp in view darkens the hands.'],
  ['pico-main', 'extra_cameras.pico.jpeg_scale', 'Picture compression (lower = sharper)', 2, 40, 1, '', 'Lower numbers give a sharper picture but larger frames, which can slow the camera down.'],
  ['webcam-pos', 'camera.pos.0', 'Left / right (x)', -1.5, 1.5, 0.01, ' m', 'Plus is to your right.'],
  ['webcam-pos', 'camera.pos.1', 'Up / down (y)', -1.5, 1.5, 0.01, ' m', 'Plus is up. The camera is usually below eye level.'],
  ['webcam-pos', 'camera.pos.2', 'Front / back (z)', -1.5, 1.5, 0.01, ' m', 'Minus is in front of you.'],
  ['webcam-angle', 'camera.pitch_deg', 'Pitch (tilt)', -60, 60, 1, '°', 'Fine-tune if the camera is tilted.'],
  ['webcam-angle', 'camera.yaw_deg', 'Yaw (turn)', -60, 60, 1, '°', 'Fine-tune if the camera is turned.'],
  ['webcam-hand', 'tracking.hand_scale', 'Hand size', 0.7, 1.3, 0.01, '×', '1.00 is an average hand. Smaller hands need a lower value.'],
  ['gest-trigger', 'gestures.trigger_click', 'Click at', 0, 1, 0.01, '', 'The trigger is pressed above this level.'],
  ['gest-trigger', 'gestures.trigger_release', 'Release at', 0, 1, 0.01, '', 'The trigger is let go below this level.'],
  ['gest-grip', 'gestures.grip_click', 'Click at', 0, 1, 0.01, '', 'The grip is pressed above this level.'],
  ['gest-grip', 'gestures.grip_release', 'Release at', 0, 1, 0.01, '', 'The grip is let go below this level.'],
  ['gest-pinch', 'gestures.pinch_near', 'Fully pinched at', 0, 0.6, 0.01, '', 'Thumb to index distance that counts as a full pinch.'],
  ['gest-pinch', 'gestures.pinch_far', 'Fully open at', 0.1, 1.5, 0.01, '', 'Thumb to index distance that counts as no pinch.'],
  ['gest-curl', 'gestures.curl_open', 'Straight finger at', 1.2, 2.5, 0.01, '', 'Reading of a fully straight finger.'],
  ['gest-curl', 'gestures.curl_closed', 'Bent finger at', 0.3, 1.5, 0.01, '', 'Reading of a fully bent finger.'],
  ['smooth', 'smoothing.min_cutoff', 'Smoothing strength (min cutoff)', 0.1, 5, 0.05, ' Hz', 'Lower is smoother but slower to follow.'],
  ['smooth', 'smoothing.beta', 'Fast-move boost (beta)', 0, 0.2, 0.001, '', 'Higher means less lag during quick moves.'],
  ['smooth', 'tracking.lost_hold_ms', 'Hold when tracking is lost', 0, 1000, 10, ' ms', 'How long to keep the hand in place after the camera loses it.'],
];
const PAIRS = [['gestures.trigger_release', 'gestures.trigger_click'], ['gestures.grip_release', 'gestures.grip_click'],
  ['gestures.pinch_near', 'gestures.pinch_far'], ['gestures.curl_closed', 'gestures.curl_open']]; // [must be lower, than this]

let cfg = null; // last known config (kept in sync with the server)
let devices = []; // webcams from /api/cameras
let phones = []; // adb phones from /api/cameras
const camViews = {}; // camera id -> {c, el: settings section, r: its elements by data-r, tile, t: tile elements by data-t, ar, live, retryAt}
const pending = new Map(); // dotted key -> value, waiting for the debounced PUT
let saveTimer = 0;
let toastTimer = 0;

const getPath = (o, p) => p.split('.').reduce((a, k) => (a == null ? a : a[k]), o);
function setPath(o, p, v) {
  const ks = p.split('.');
  const last = ks.pop();
  ks.reduce((a, k) => (a[k] ??= {}), o)[last] = v;
}

function toast(msg, bad = false) {
  E.toast.textContent = msg;
  E.toast.dataset.k = bad ? 'err' : 'ok';
  E.toast.dataset.show = '1';
  clearTimeout(toastTimer);
  if (!bad) toastTimer = setTimeout(() => { E.toast.dataset.show = '0'; }, 1500);
}

function buildSliders() {
  for (const [group, path, label, min, max, step, unit, hint] of SLIDERS) {
    const id = 'sl-' + path.replaceAll('.', '-');
    const out = h('output', { cls: 'val', for: id, text: '--' });
    const input = h('input', { type: 'range', id, min, max, step, 'data-path': path });
    input.dataset.unit = unit;
    input.dataset.digits = (String(step).split('.')[1] || '').length;
    input._out = out;
    input._warn = h('small', { cls: 'warn-text', hidden: '' });
    $(`[data-group="${group}"]`).append(h('div', { cls: 'field' },
      h('div', { cls: 'field-head' }, h('label', { for: id, text: label }), out),
      input, h('small', { cls: 'hint', text: hint }), input._warn));
  }
}

const showValue = (el) => { el._out.textContent = (+el.value).toFixed(+el.dataset.digits) + el.dataset.unit; };
const slider = (path) => $(`input[data-path="${path}"]`);

function checkPairs() {
  for (const [lo, hi] of PAIRS) {
    const a = slider(lo), b = slider(hi);
    a._warn.textContent = `Should be lower than "${$('label', b.closest('.field')).textContent}".`;
    a._warn.hidden = !(+a.value >= +b.value);
  }
}

// A control is "busy" while the user is dragging or has just touched it (or is typing in it): never overwrite it from the server.
const busy = (el) => el.dataset.drag === '1' || performance.now() - (el._touch ?? -1e9) < 800
  || ((el.type === 'text' || el.type === 'number') && el === document.activeElement);

// A number box can show the value in another unit (data-scale="100": metres in the config, centimetres on screen) and
// show one config value as an empty box (data-empty="0": 0 means unknown).
const shown = (el, v) => (el.dataset.empty !== undefined && v === +el.dataset.empty ? '' : el.dataset.scale ? Math.round(v * +el.dataset.scale * 10) / 10 : v);

function syncControls() {
  for (const el of $$('[data-path]')) {
    const v = getPath(cfg, el.dataset.path);
    el.disabled = v === undefined; // not in this hub's config (older hub): leave it off
    if (v === undefined || busy(el)) continue;
    if (el.type === 'checkbox') el.checked = !!v;
    else if (el.type === 'radio') el.checked = el.value === v;
    else el.value = shown(el, v);
    if (el.type === 'range') showValue(el);
  }
  checkPairs();
}

// Fill a <select> from [value, text] items. The configured value stays selectable even when it is not listed (unplugged device).
function fillSelect(sel, items, cur, missing) {
  const list = items.slice();
  if (cur != null && !list.some(([v]) => String(v) === String(cur))) list.push([cur, missing(cur)]);
  sel.replaceChildren(...list.map(([value, text]) => h('option', { value, text })));
  if (cur != null) sel.value = cur;
}

function renderPickers() {
  const ex = cfg?.extra_cameras;
  const cams = devices.map((d) => [d.device, `${d.name} (#${d.device})`]);
  fillSelect(camViews.webcam.r.device, cams.length ? cams : [['', 'No cameras found']], cfg?.camera.device, (v) => `Camera not found (#${v})`);
  for (const id of PHONE_CAMS) { // the phone and the headset's phone choose from the same adb list
    const r = camViews[id].r, c = ex?.[id];
    fillSelect(r.serial, [['', 'Choose a phone...'], ...phones.map((p) => [p.serial, `${p.model || 'Phone'} (${p.serial})`])], c?.serial, (v) => `Not connected (${v})`);
    fillSelect(r.camId, [['0', 'Back camera (0)'], ['1', 'Front camera (1)'], ['2', '2 (ultrawide)']], c?.camera_id, (v) => `Camera ${v}`);
  }
  fillSelect(camViews.headset.r.zoom, ZOOMS, ex?.headset?.zoom, (v) => `${v}x`);
  fillSelect(camViews.pico.r.speed, [[12, 'Normal, ~17 fps'], [8, '~25 fps'], [7, 'Fast, ~29 fps']], ex?.pico.xclk_div, (v) => `Custom (divider ${v})`);
  for (const c of CAMS) {
    fillSelect(camViews[c.id].r.rotate, ROTATIONS, getPath(cfg, c.prefix + '.rotate_deg'), (v) => `${v}°`);
    fillSelect(camViews[c.id].r.light, LIGHT, getPath(cfg, c.prefix + '.light_correction'), String);
  }
}

async function loadCameras() {
  const btns = $$('[data-act="refresh"]');
  btns.forEach((b) => { b.disabled = true; });
  try {
    const j = await (await fetch('/api/cameras')).json();
    devices = j.devices || []; phones = j.phones || [];
  } catch { devices = []; phones = []; }
  btns.forEach((b) => { b.disabled = false; });
  renderPickers();
}

let loading = false;
async function loadConfig() {
  if (loading) return;
  loading = true;
  try {
    const r = await fetch('/api/config');
    if (!r.ok) throw new Error(r.status);
    const c = await r.json();
    if (!pending.size) { cfg = c; syncControls(); renderPickers(); }
  } catch {
    toast('Could not load settings. Retrying...', true);
    setTimeout(loadConfig, 2000);
  }
  loading = false;
}

// Record one changed value; only that key goes into the next PUT.
function change(path, value) {
  setPath(cfg, path, value);
  const ks = path.split('.');
  if (/^\d+$/.test(ks.at(-1))) ks.pop(); // list item (camera.pos.1): the server wants the whole list
  const key = ks.join('.');
  pending.set(key, structuredClone(getPath(cfg, key)));
  clearTimeout(saveTimer);
  saveTimer = setTimeout(flush, 150);
}

async function flush() {
  const patch = {};
  for (const [k, v] of pending) setPath(patch, k, v);
  pending.clear();
  try {
    const r = await fetch('/api/config', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch) });
    if (!r.ok) throw new Error(r.status);
    const c = await r.json();
    if (!pending.size) { cfg = c; syncControls(); }
    toast('Saved');
  } catch {
    toast('Could not save. Is the hub still running?', true);
  }
}

function onControl(ev) {
  const el = ev.target;
  if (!cfg || !el.dataset || !el.dataset.path) return;
  const range = el.type === 'range';
  if (ev.type !== (range ? 'input' : 'change')) return; // sliders save while moving, the rest on change
  el._touch = performance.now();
  let v;
  if (el.type === 'checkbox') v = el.checked;
  else if (el.type === 'radio') { if (!el.checked) return; v = el.value; }
  else if (el.dataset.type === 'text') v = el.value.trim(); // serials, camera ids and ports are strings
  else if (el.value === '' && el.dataset.empty !== undefined) v = +el.dataset.empty; // an emptied box means "unknown"
  else if (el.value === '' || !Number.isFinite(+el.value)) return;
  else {
    v = +el.value;
    if (el.type === 'number' && el.min !== '' && el.max !== '' && !(el.dataset.empty !== undefined && v === +el.dataset.empty)) {
      v = clamp(v, +el.min, +el.max); // out of range: the nearest allowed value
      el.value = v;
    }
    if (el.dataset.scale) v = Math.round(v / +el.dataset.scale * 1e4) / 1e4; // on screen cm, in the config m
  }
  if (range) { showValue(el); checkPairs(); }
  change(el.dataset.path, v);
}

function bindControls() {
  document.addEventListener('input', onControl);
  document.addEventListener('change', onControl);
  document.addEventListener('pointerdown', (e) => { if (e.target.matches?.('input[type="range"]')) e.target.dataset.drag = '1'; });
  const release = () => $$('[data-drag]').forEach((el) => { delete el.dataset.drag; el._touch = performance.now(); });
  addEventListener('pointerup', release);
  addEventListener('pointercancel', release);
  document.addEventListener('click', (e) => { if (e.target.closest('[data-act="refresh"]')) loadCameras(); });
}

/* ===================================================================== tabs */

const TABS = ['cameras', 'view', 'gestures', 'smoothing', 'calibration'];
let activeTab = TABS[0];

// Only stream while it can be seen: Cameras tab open, camera enabled and online.
function syncPreview(v) {
  const cam = camOf(v.c.id);
  const want = activeTab === 'cameras' && !!cam?.online && camEnabled(v.c, cam);
  if (want && !v.live && performance.now() >= v.retryAt) {
    v.t.img.src = `/stream/${v.c.id}.mjpg?t=${Date.now()}`;
    v.live = true;
  } else if (!want && v.live) {
    v.t.img.removeAttribute('src');
    v.live = false;
  }
}
const syncPreviews = () => CAMS.forEach((c) => syncPreview(camViews[c.id]));

function setTab(id, focus = false) {
  activeTab = TABS.includes(id) ? id : TABS[0];
  document.body.dataset.tab = activeTab; // the Cameras page uses the full window width
  for (const t of TABS) {
    const on = t === activeTab;
    $(`#tab-${t}`).setAttribute('aria-selected', on);
    $(`#tab-${t}`).tabIndex = on ? 0 : -1;
    $(`#page-${t}`).hidden = !on;
  }
  store.set('handhub.tab', activeTab);
  syncPreviews();
  if (focus) $(`#tab-${activeTab}`).focus();
}

// The camera wall is sized to the window minus the header; publish the header's height to CSS (--top-h).
function trackTopbar() {
  const bar = $('.topbar');
  const publish = () => document.documentElement.style.setProperty('--top-h', bar.offsetHeight + 'px');
  new ResizeObserver(publish).observe(bar);
  publish();
}

function bindTabs() {
  E.tabs.addEventListener('click', (e) => { const b = e.target.closest('.tab'); if (b) setTab(b.id.slice(4)); });
  E.tabs.addEventListener('keydown', (e) => {
    const i = TABS.indexOf(activeTab);
    const to = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: TABS.length - 1 }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    setTab(TABS[(to + TABS.length) % TABS.length], true);
  });
}

/* ===================================================================== live data */

let snap = null; // latest WebSocket snapshot (null while disconnected)
let wsUp = false, wsFailed = false, lastMsg = 0, ws = null;
const WINDOW = 10000; // jitter graph length in ms
const jit = [0, 1].map(() => ({ buf: [], hist: [], last: 0, value: null }));
const clicks = [0, 1].map(() => ({ prev: 0, n: [0, 0, 0, 0] }));

function jitterMm(buf) { // 3D standard deviation of the last second of positions
  if (buf.length < 5) return null;
  let v = 0;
  for (const a of [1, 2, 3]) {
    const m = buf.reduce((s, p) => s + p[a], 0) / buf.length;
    v += buf.reduce((s, p) => s + (p[a] - m) ** 2, 0) / buf.length;
  }
  return Math.sqrt(v) * 1000;
}

function ingest(s) { // runs per message: cheap bookkeeping only, no drawing
  const now = performance.now();
  s.hands.forEach((hd, i) => {
    const j = jit[i], c = clicks[i];
    if (hd.tracked) {
      j.buf.push([now, ...hd.pos]);
      while (now - j.buf[0][0] > 1000) j.buf.shift();
    } else j.buf.length = 0;
    j.value = jitterMm(j.buf);
    if (now - j.last >= 100) {
      j.last = now;
      j.hist.push([now, j.value]);
      while (now - j.hist[0][0] > WINDOW + 500) j.hist.shift();
    }
    const b = hd.tracked ? hd.buttons : 0;
    BUTTONS.forEach((_, k) => { if ((b >> k) & 1 && !((c.prev >> k) & 1)) c.n[k]++; });
    c.prev = b;
  });
}

function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => {
    const reconnect = wsFailed;
    wsUp = true; lastMsg = performance.now();
    if (reconnect) { loadConfig(); loadCameras(); } // the hub may have restarted (the previews restart with the next snapshot)
  };
  ws.onmessage = (e) => {
    try { snap = JSON.parse(e.data); } catch { return; }
    lastMsg = performance.now();
    ingest(snap);
  };
  ws.onclose = () => { wsUp = false; wsFailed = true; snap = null; setTimeout(connect, 1000); };
  ws.onerror = () => ws.close();
}

/* ===================================================================== rendering */

function fit(canvas) { // size the backing store for devicePixelRatio; returns [ctx, cssWidth, cssHeight]
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, hh = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(hh * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(hh * dpr);
  }
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, hh);
  return [g, w, hh];
}

const line = (g, x1, y1, x2, y2) => { g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke(); };
function halo(g, text, x, y, color = COL.text) { // text with a dark halo so it stays readable over lines
  g.lineWidth = 3; g.lineJoin = 'round'; g.strokeStyle = COL['panel-2']; g.strokeText(text, x, y);
  g.fillStyle = color; g.fillText(text, x, y);
}

function paintChip(c, state, text, tip = '') {
  if (c.dataset.s !== state) c.dataset.s = state;
  setText($('.chip-state', c), text);
  if (c.title !== tip) c.title = tip;
}
const setChip = (id, ...args) => paintChip(E['chip-' + id], ...args);

// This camera's entry in the snapshot (older hubs only have `camera`, which is the webcam).
const camOf = (id) => snap?.cameras?.[id] ?? (id === 'webcam' ? snap?.camera : undefined);
const camEnabled = (c, cam) => !!(cam?.enabled ?? getPath(cfg, c.prefix + '.enabled'));

function camChip(c) { // -> [state, text, tooltip] for the status bar
  const cam = camOf(c.id);
  if (!cam) return ['off', 'Unknown', 'This hub does not report this camera.'];
  if (!camEnabled(c, cam)) return ['off', 'Off', 'The camera is switched off.'];
  if (cam.error || !cam.online) return ['bad', cam.error ? 'Error' : 'Offline', cam.error || 'No video frames are arriving.'];
  if (!(cam.hands ?? snap.hands.filter((x) => x.tracked).length)) return ['warn', 'No hand', 'The camera works but no hand is visible.'];
  return ['ok', 'Tracking', c.id !== 'webcam' && !cam.calibrated ? 'Not calibrated yet, so this camera is not used for the hand pose.' : ''];
}

const calibrated = (c, cam) => c.id === 'webcam' || cam?.calibrated === true; // the webcam is the origin of the room, so it always is

// How well a camera agrees with the others (px of reprojection error): a word and a chip shape go with every value, never colour alone.
function agreement(px) {
  if (!Number.isFinite(px)) return null;
  if (px < 10) return { state: 'ok', word: 'good', short: 'good' };
  return px <= 25 ? { state: 'warn', word: 'doubtful', short: 'doubtful' } : { state: 'bad', word: 'bad — recalibrate', short: 'bad' };
}

const fmtPx = (px) => fixed(px, px >= 100 ? 0 : 1) + ' px';

function renderStatus() {
  E.banner.hidden = wsUp || !wsFailed;
  const s = snap;
  for (const c of CAMS) { // an optional camera has a chip only while it is enabled
    const show = !c.optional || camEnabled(c, camOf(c.id));
    if (E['chip-' + c.id].hidden === show) E['chip-' + c.id].hidden = !show;
  }
  if (!s) {
    for (const id of ['runtime', 'hmd', ...CAMS.map((c) => c.id)]) setChip(id, 'off', 'Unknown');
    setText(E.readout, '-- fps · -- ms');
    return;
  }
  const d = s.driver, cam = camOf('webcam');
  if (d.rx_error) setChip('runtime', 'bad', 'Port error', `Cannot listen for the runtime: ${d.rx_error}`);
  else setChip('runtime', d.connected ? 'ok' : 'warn', d.connected ? 'Connected' : 'Waiting', d.connected ? '' : 'The VR runtime driver is not talking to the hub yet.');
  setChip('hmd', s.hmd.valid ? 'ok' : 'warn', s.hmd.valid ? 'Found' : 'Not found');
  for (const c of CAMS) setChip(c.id, ...camChip(c));
  setText(E.readout, cam?.online ? `${fixed(cam.fps)} fps · ${fixed(cam.latency_ms)} ms` : '-- fps · -- ms');
}

/* ---- cameras page: the wall (one tile per CAMS entry, from #tpl-tile) and one collapsible settings section each (from #tpl-cam) ---- */
function buildCameras() {
  for (const c of CAMS) {
    const tile = document.importNode(E['tpl-tile'].content.firstElementChild, true);
    const el = document.importNode(E['tpl-cam'].content.firstElementChild, true);
    for (const root of [tile, el]) $$('[data-only]', root).forEach((n) => { if (!n.dataset.only.split(' ').includes(c.id)) n.remove(); });
    $$('[id]', el).forEach((n) => { n.id += '-' + c.id; }); // ids are per camera
    $$('label[for]', el).forEach((n) => { n.htmlFor += '-' + c.id; });
    $$('[data-group]', el).forEach((n) => { n.dataset.group = `${c.id}-${n.dataset.group}`; });
    $$('[data-key]', el).forEach((n) => { n.dataset.path = `${c.prefix}.${n.dataset.key}`; });
    const r = refs(el, 'r'), t = refs(tile, 't');
    const v = { c, el, r, tile, t, ar: DEFAULT_AR, live: false, retryAt: 0, visible: !c.optional };
    tile.hidden = !v.visible; // an optional camera's tile appears when the camera is enabled (renderCam)
    r.title.textContent = c.title;
    t.name.textContent = c.short;
    t.img.alt = `Live preview of the ${c.title} with the detected hand skeleton`;
    t.img.addEventListener('error', () => { // stream dropped: try again shortly
      t.img.removeAttribute('src');
      v.live = false;
      v.retryAt = performance.now() + 2000;
    });
    const openKey = `handhub.cam.${c.id}.open`; // each section remembers whether it was open
    el.open = store.get(openKey) === '1';
    el.addEventListener('toggle', () => store.set(openKey, el.open ? '1' : '0'));
    camViews[c.id] = v;
    E.camWall.append(tile);
    E.camSections.append(el);
  }
}

// Width / height of a camera's picture: the size it reports, else the configured capture size (swapped when rotated a quarter turn), else 4:3.
function aspectOf(c, cam) {
  let [w, hh] = Array.isArray(cam?.size) ? cam.size : [getPath(cfg, c.prefix + '.width'), getPath(cfg, c.prefix + '.height')];
  if (!cam?.size && (getPath(cfg, c.prefix + '.rotate_deg') ?? 0) % 180 === 90) [w, hh] = [hh, w];
  return w > 0 && hh > 0 ? clamp(Math.round(w / hh * 1000) / 1000, 0.3, 3) : DEFAULT_AR;
}

const setData = (el, key, val) => { if (el.dataset[key] !== String(val)) el.dataset[key] = val; }; // skip no-op attribute writes

function paintBadge(el, name, on) { // filled + letter = seen, dashed outline + letter = not seen (never colour alone)
  if (el.dataset.on === String(+on)) return;
  el.dataset.on = +on;
  el.setAttribute('aria-label', `${name} hand: ${on ? 'seen' : 'not seen'}`);
}

function renderCam(v) {
  const { c, r, t } = v;
  const cam = camOf(c.id), hands = snap?.hands || [];
  const on = !!cam?.online, enabled = camEnabled(c, cam);
  v.visible = !c.optional || enabled;
  if (v.tile.hidden === v.visible) v.tile.hidden = !v.visible;
  syncPreview(v);
  const shown = v.live && t.img.naturalWidth > 0; // the first frame of the stream has arrived
  t.img.style.visibility = shown ? 'visible' : 'hidden';
  t.empty.hidden = shown;
  setData(t.empty, 'k', cam?.error && enabled ? 'err' : '');
  setText(t.empty, !snap ? 'No connection to the hub' : !cam ? 'Not reported by this hub' : !enabled ? 'Off'
    : cam.error ? cam.error : 'Waiting for video…');
  const ar = aspectOf(c, cam);
  if (v.ar !== ar) { v.ar = ar; v.tile.style.setProperty('--ar', ar); }
  const status = !cam ? 'off' : !enabled ? 'off' : on ? 'ok' : 'bad';
  const label = !cam ? 'Unknown' : !enabled ? 'Off' : on ? 'Online' : 'Offline';
  paintChip(t.online, status, label, cam?.error || '');
  setText(t.fps, on ? fixed(cam.fps) : '--');
  const dark = on && cam.brightness != null && cam.brightness < 50;
  const small = on && cam.hands !== 0 && hands.some((x) => x.tracked && x.height_px < 100) && c.id === 'webcam';
  const warn = dark ? 'Too dark' : small ? 'Hand is small in frame: move closer' : '';
  t.warn.hidden = !warn;
  if (warn) setText(t.warn, 'Warning: ' + warn);
  // Which hands this camera sees; an older hub has no `sees`, so only the count of hands is known.
  const sees = on && Array.isArray(cam.sees) ? cam.sees : null;
  const countOnly = on && !sees && cam.hands != null;
  SIDE.forEach((s, i) => { t['see' + s].hidden = countOnly; paintBadge(t['see' + s], HANDS[i], !!sees?.[i]); });
  t.seeN.hidden = !countOnly;
  if (countOnly) {
    setText(t.seeN, plural(cam.hands, 'hand'));
    setData(t.seeN, 'on', +(cam.hands > 0));
  }
  const ok = calibrated(c, cam); // `calibrated` is missing on an older hub: not calibrated
  const agree = ok && enabled ? agreement(cam?.agreement_px) : null; // compact badge; for the phone and Pico it stands in for the "Calibrated" chip (a camera with a value is calibrated)
  t.agree.hidden = !agree;
  if (agree) paintChip(t.agree, agree.state, `${fmtPx(cam.agreement_px)} ${agree.short}`, `Calibrated. Agreement with the other cameras: ${fmtPx(cam.agreement_px)}, ${agree.word}`);
  if (t.calib) {
    t.calib.hidden = !!agree;
    paintChip(t.calib, !cam ? 'off' : ok ? 'ok' : 'warn', !cam ? 'Unknown' : ok ? 'Calibrated' : 'Not calibrated');
    r.calibHint.hidden = !cam || ok;
  }
  r.err.hidden = !cam?.error;
  if (cam?.error) setText(r.err, cam.error);
  r.dark.hidden = !dark;
  if (r.small) r.small.hidden = !small;
  setText(r.fps, on ? fixed(cam.fps) : '--');
  setText(r.capFps, on ? fixed(cam.capture_fps) : '--');
  setText(r.proc, on ? fixed(cam.proc_ms, 1) + ' ms' : '--');
  setText(r.latency, on ? fixed(cam.latency_ms) + ' ms' : '--');
  setText(r.size, on && cam.size ? `${cam.size[0]}×${cam.size[1]}` : '--');
  const seenText = Array.isArray(cam?.sees) ? SIDE.filter((_, i) => cam.sees[i]).join(' + ') : ''; // which hands (missing on an older hub)
  setText(r.seen, on && cam.hands != null ? String(cam.hands) + (seenText && cam.hands ? ` (${seenText})` : '') : '--');
  setText(r.bright, on ? fixed(cam.brightness) + ' / 255' : '--');
  if (r.confL) {
    SIDE.forEach((s, i) => {
      const hd = hands[i], tr = !!hd?.tracked;
      setText(r['conf' + s], tr ? Math.round(hd.confidence * 100) + ' %' : 'not tracked');
      setText(r['height' + s], tr ? hd.height_px + ' px' : 'not tracked');
    });
  }
}

// The columns follow the pictures of the cameras that are shown: widths in the ratio of each camera's aspect ratio, so all tiles
// get the same height and no side bars. With more tiles the row gets narrower per tile, so the tiles shrink to fit the window.
let wallKey = '';
function renderWall() {
  const shown = CAMS.map((c) => camViews[c.id]).filter((v) => v.visible), ars = shown.map((v) => v.ar);
  const head = camViews.headset.visible, key = ars.join(' ') + (head ? ' head' : '');
  if (key === wallKey) return;
  wallKey = key;
  E.camTop.style.setProperty('--cols', ars.map((a) => `minmax(0, ${a}fr)`).join(' '));
  E.camTop.style.setProperty('--sum', ars.reduce((s, a) => s + a, 0));
  E.camTop.style.setProperty('--n', ars.length);
  E.camTop.dataset.n = ars.length; // CSS: how many tiles share the row
  E.camTop.dataset.head = +head; // CSS: the headset calibration strip is shown under the first one
}

// One line per hand (and the head): tracked or not, and which cameras made the pose.
function handLine(hd) {
  if (!hd?.tracked) return 'not tracked';
  const from = (hd.views || []).map(camName).join(' + ');
  if (!from) return 'tracked';
  // Modes: "triangulated", "single", "single+body" (distance taken from the skeleton), each
  // optionally with "+snapped" (the hand was pulled back onto the skeleton's wrist).
  const mode = String(hd.mode || '');
  const base = mode.startsWith('triangulated') ? `triangulated from ${from}`
    : mode.startsWith('single+body') ? `one camera (${from}), distance from the body`
    : mode.startsWith('single+kept') ? `one camera (${from}), distance kept from a moment ago`
    : mode.startsWith('single') ? `single view: ${from}` : `tracked (${from})`;
  return base + (mode.endsWith('+snapped') ? ', pulled back to the wrist' : '');
}
function headLine(hd) {
  if (hd.enabled === false) return 'tracking is off';
  if (!hd.tracked) return 'not seen';
  const from = (hd.views || []).map(camName).join(' + ');
  return from ? `seen by ${from}` + (hd.mode === 'triangulated' ? ' (triangulated)' : hd.mode === 'single' ? ' (single view)' : '') : 'seen';
}
function renderLines() {
  SIDE.forEach((s, i) => {
    const hd = snap?.hands[i];
    setData(E['line' + s], 'on', +!!hd?.tracked);
    setText(E['line' + s + 'Text'], snap ? handLine(hd) : '--');
  });
  const head = snap?.head; // missing on an older hub: no head line
  E.lineHead.hidden = !head;
  if (head) {
    setData(E.lineHead, 'on', +!!head.tracked);
    setText(E.lineHeadText, headLine(head));
  }
}

function renderCameras() {
  CAMS.forEach((c) => renderCam(camViews[c.id]));
  renderWall();
  renderLines();
  renderCollectors();
}

/* ---- 3D view: a perspective room on one canvas, projected by hand (no libraries) ---- */
// Room axes (metres, the webcam is the origin): x = the person's right, y = up, z = from the webcam towards the person.
// A virtual camera orbits a target point. Yaw 0 looks from the webcam's side towards +z, a true third-person view where the
// person's right hand is on the viewer's left; yaw +90 deg looks from the person's right. Pitch is the height above the horizon.
const FLOOR_DEFAULT = -1.2; // floor height until feet have been seen: 1.2 m below the webcam
const GRID = 0.5, GRID_HALF = 3; // floor cell (m) and how far the grid reaches around the view target
const FOV = 35 * Math.PI / 180; // vertical field of view
const NEAR = 0.15; // closer than this (m) counts as behind the virtual camera
const DIST = [0.8, 14], PITCH = [0.02, 1.55]; // limits: zoom (m from the target) and elevation (rad)
const PRESETS = { front: { yaw: 0, pitch: 0.14, dist: 3.4 }, side: { yaw: Math.PI / 2, pitch: 0.14, dist: 3.7 }, top: { yaw: 0, pitch: PITCH[1], dist: 5.5 } };
const VIEW_KEY = 'handhub.view3d';
const LIMB = 0.045, JOINT = 0.036, FOOT_JOINT = 0.024, HEAD_R = 0.1, HAND_BONE = 0.011, HAND_JOINT = 0.008; // thicknesses (m), drawn in perspective
// MediaPipe pose bones [from, to, side] (side 0 = person's left, 1 = right, 2 = middle) and joints [landmark, side, radius]
const BONES = [[11, 12, 2], [11, 13, 0], [13, 15, 0], [12, 14, 1], [14, 16, 1], [11, 23, 0], [12, 24, 1], [23, 24, 2], [23, 25, 0], [25, 27, 0],
  [24, 26, 1], [26, 28, 1], [27, 29, 0], [29, 31, 0], [27, 31, 0], [28, 30, 1], [30, 32, 1], [28, 32, 1]];
const JOINTS = [[11, 0, JOINT], [12, 1, JOINT], [13, 0, JOINT], [14, 1, JOINT], [15, 0, JOINT], [16, 1, JOINT], [23, 0, JOINT], [24, 1, JOINT],
  [25, 0, JOINT], [26, 1, JOINT], [27, 0, JOINT], [28, 1, JOINT], [29, 0, FOOT_JOINT], [30, 1, FOOT_JOINT], [31, 0, FOOT_JOINT], [32, 1, FOOT_JOINT]];
const FOOT_DROP = { 27: 0.09, 28: 0.09, 29: 0.02, 30: 0.02, 31: 0.02, 32: 0.02 }; // how far above the floor each foot landmark sits (m)
const PALM = [0, 1, 5, 9, 13, 17];

const V = { ...PRESETS.front, follow: true, goal: null, tgt: [0, FLOOR_DEFAULT + 0.9, 1.5], tgtLive: false, feet: [], seen: null, floorLow: null, floor: FLOOR_DEFAULT, t: 0, pinch: 0, saveTimer: 0 };
const pointers = new Map(); // pointers down on the canvas -> [x, y]
const viewRows = {}; // camera id -> {li, val, chip} of the agreement list
const sideColor = (s) => (s === 0 ? COL.left : s === 1 ? COL.right : COL.text);
const wrapPi = (a) => a - 2 * Math.PI * Math.round(a / (2 * Math.PI));
const okPt = (p) => Array.isArray(p) && p.length >= 3 && Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]);
const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const madd = (a, b, k) => [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k];
const unit = (v) => { const n = Math.hypot(v[0], v[1], v[2]); return n > 1e-9 ? [v[0] / n, v[1] / n, v[2] / n] : null; };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

function loadView() {
  try {
    const s = JSON.parse(store.get(VIEW_KEY));
    if ([s.yaw, s.pitch, s.dist].every(Number.isFinite)) { V.yaw = wrapPi(s.yaw); V.pitch = clamp(s.pitch, ...PITCH); V.dist = clamp(s.dist, ...DIST); }
    if (typeof s.follow === 'boolean') V.follow = s.follow;
  } catch { /* nothing (valid) stored: the front view */ }
  E.viewFollow.checked = V.follow;
}
function saveView(now = false) {
  clearTimeout(V.saveTimer);
  const put = () => { const g = V.goal ?? V; store.set(VIEW_KEY, JSON.stringify({ yaw: +wrapPi(g.yaw).toFixed(3), pitch: +g.pitch.toFixed(3), dist: +g.dist.toFixed(2), follow: V.follow })); };
  if (now) put(); else V.saveTimer = setTimeout(put, 300);
}
const orbit = (dyaw, dpitch, zoom = 1) => {
  V.goal = null;
  V.yaw = wrapPi(V.yaw + dyaw); V.pitch = clamp(V.pitch + dpitch, ...PITCH); V.dist = clamp(V.dist * zoom, ...DIST);
  saveView();
};

function bindView() {
  const cv = E.view3d;
  cv.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    try { cv.setPointerCapture(e.pointerId); } catch { /* pointer already gone */ }
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    V.pinch = 0;
    cv.dataset.drag = '1';
  });
  cv.addEventListener('pointermove', (e) => {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p[0], dy = e.clientY - p[1];
    p[0] = e.clientX; p[1] = e.clientY;
    if (pointers.size === 1) orbit(dx * 0.008, dy * 0.008); // drag right: the room turns right
    else if (pointers.size === 2) { // pinch
      const [a, b] = [...pointers.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (V.pinch > 0 && d > 0) orbit(0, 0, V.pinch / d);
      V.pinch = d;
    }
  });
  const up = (e) => {
    pointers.delete(e.pointerId);
    V.pinch = 0;
    if (!pointers.size) delete cv.dataset.drag;
  };
  cv.addEventListener('pointerup', up);
  cv.addEventListener('pointercancel', up);
  cv.addEventListener('wheel', (e) => {
    e.preventDefault();
    orbit(0, 0, Math.exp(e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 100 : 1) * 0.0015));
  }, { passive: false });
  cv.addEventListener('keydown', (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const step = e.shiftKey ? 0.03 : 0.1; // arrows move the virtual camera; shift for fine steps
    const act = { ArrowLeft: [step, 0, 1], ArrowRight: [-step, 0, 1], ArrowUp: [0, step, 1], ArrowDown: [0, -step, 1],
      '+': [0, 0, 0.88], '=': [0, 0, 0.88], '-': [0, 0, 1 / 0.88], _: [0, 0, 1 / 0.88] }[e.key];
    if (!act) return;
    e.preventDefault();
    orbit(...act);
  });
  E.viewBar.addEventListener('click', (e) => {
    const p = PRESETS[e.target.closest('[data-preset]')?.dataset.preset];
    if (!p) return;
    V.goal = { ...p };
    saveView(true);
  });
  E.viewFollow.addEventListener('change', () => { V.follow = E.viewFollow.checked; saveView(true); });
}

function buildView() {
  for (const c of CAMS) {
    const val = h('span', { cls: 'val mono', text: '--' });
    const chip = h('span', { cls: 'chip', 'data-s': 'off' }, h('span', { cls: 'dot', 'aria-hidden': 'true' }), h('span', { cls: 'chip-state', text: '--' }));
    const li = h('li', { cls: 'agree-row', hidden: '' }, h('b', { text: c.short }), val, chip);
    viewRows[c.id] = { li, val, chip };
    E.vsCams.append(li);
  }
  V.btns = $$('[data-preset]', E.viewBar);
  loadView();
  bindView();
}

// What to draw this frame, from the latest snapshot. Every field may be missing (older hub) or null (point not seen).
function viewModel() {
  const s = snap, b = s?.body, P = b?.tracked && Array.isArray(b.points) ? b.points : null;
  const bp = (i) => (P && okPt(P[i]) ? P[i] : null);
  const m = { bp, floorLow: Infinity, hands: [] };
  for (const i in FOOT_DROP) { const p = bp(+i); if (p) m.floorLow = Math.min(m.floorLow, p[1] - FOOT_DROP[i]); }
  m.sh = bp(11) && bp(12) ? mid(bp(11), bp(12)) : null;
  m.hip = bp(23) && bp(24) ? mid(bp(23), bp(24)) : null;
  const ears = bp(7) && bp(8) ? mid(bp(7), bp(8)) : null, eyes = s?.head?.tracked && okPt(s.head.pos_room) ? s.head.pos_room : null;
  m.head = ears || bp(0) || (eyes && [eyes[0], eyes[1] + 0.05, eyes[2]]) || null; // ears' midpoint, else the nose, else the eyes of the head tracker
  m.face = ears && bp(0) ? unit(sub(bp(0), ears)) : null; // where the face points
  SIDE.forEach((_, i) => {
    const hd = s?.hands?.[i], lm = hd?.tracked && Array.isArray(hd.lm_room) && hd.lm_room.length >= 21 && okPt(hd.lm_room[0]) ? hd.lm_room : null;
    m.hands.push({ hd, lm });
  });
  const wrists = m.hands.filter((x) => x.lm).map((x) => x.lm[0]);
  m.anchor = m.hip && m.sh ? mid(m.hip, m.sh) : m.hip || m.sh || m.head || (wrists.length ? wrists.reduce((a, p) => madd(a, p, 1 / wrists.length), [0, 0, 0]) : null);
  m.feet = m.floorLow < Infinity;
  m.any = !!(b?.tracked || s?.head?.tracked || s?.hands?.some((x) => x.tracked));
  return m;
}

// Per frame: the floor estimate, the preset animation and the follow target, all smoothed.
function stepView(m, dt) {
  const k = (tau) => 1 - Math.exp(-dt / tau);
  if (snap && snap !== V.seen && m.feet) { // floor: the low end (5th percentile) of the lowest foot point over the last minute, so one wild frame cannot move it
    V.feet.push(m.floorLow);
    if (V.feet.length > 1800) V.feet.shift();
    V.floorLow = [...V.feet].sort((a, b) => a - b)[Math.floor(V.feet.length * 0.05)];
  }
  V.seen = snap;
  V.floor += ((V.floorLow == null ? FLOOR_DEFAULT : V.floorLow - 0.02) - V.floor) * k(0.6); // a 2 cm margin, smoothed
  const g = V.goal;
  if (g) {
    const a = k(0.12);
    V.yaw = wrapPi(V.yaw + wrapPi(g.yaw - V.yaw) * a); V.pitch += (g.pitch - V.pitch) * a; V.dist += (g.dist - V.dist) * a;
    if (Math.abs(wrapPi(g.yaw - V.yaw)) + Math.abs(g.pitch - V.pitch) + Math.abs(g.dist - V.dist) < 0.002) { Object.assign(V, g); V.goal = null; }
  }
  const a = m.anchor;
  if (!V.follow || !a) return; // not following, or nobody to follow: the view stays where it is
  const goal = m.feet && m.head ? [a[0], (V.floor + m.head[1] + 0.12) / 2, a[2]] : a; // full figure: centre it between floor and head
  if (!V.tgtLive) { V.tgt = goal.slice(); V.tgtLive = true; } else for (let i = 0; i < 3; i++) V.tgt[i] += (goal[i] - V.tgt[i]) * k(0.5);
}

// The virtual camera: eye, forward / right / up unit vectors, focal length in px (fitted to the smaller side), image centre.
const C = { ex: 0, ey: 0, ez: 0, fx: 0, fy: 0, fz: 1, rx: 1, ry: 0, rz: 0, ux: 0, uy: 1, uz: 0, k: 1, cx: 0, cy: 0 };
function placeCamera(w, hh) {
  const cp = Math.cos(V.pitch), sp = Math.sin(V.pitch), sy = Math.sin(V.yaw), cy = Math.cos(V.yaw);
  Object.assign(C, { fx: -sy * cp, fy: -sp, fz: cy * cp, rx: -cy, ry: 0, rz: -sy, ux: -sy * sp, uy: cp, uz: cy * sp,
    k: Math.min(hh, w / 1.2) / 2 / Math.tan(FOV / 2), cx: w / 2, cy: hh / 2 });
  C.ex = V.tgt[0] - C.fx * V.dist; C.ey = V.tgt[1] - C.fy * V.dist; C.ez = V.tgt[2] - C.fz * V.dist;
}
const cp3 = [0, 0, 0], cq3 = [0, 0, 0], S = { x1: 0, y1: 0, x2: 0, y2: 0, d: 0 }; // scratch: camera-space points, projected segment
function toCam(p, o) {
  const dx = p[0] - C.ex, dy = p[1] - C.ey, dz = p[2] - C.ez;
  o[0] = dx * C.rx + dy * C.ry + dz * C.rz; o[1] = dx * C.ux + dy * C.uy + dz * C.uz; o[2] = dx * C.fx + dy * C.fy + dz * C.fz;
}
function clipNear(p, q) { // move p onto the near plane along p-q when it is behind it
  if (p[2] >= NEAR) return;
  const t = (NEAR - p[2]) / (q[2] - p[2]);
  p[0] += (q[0] - p[0]) * t; p[1] += (q[1] - p[1]) * t; p[2] = NEAR;
}
function projSeg(a, b) { // world segment -> S; false when it is entirely behind the virtual camera
  toCam(a, cp3); toCam(b, cq3);
  if (cp3[2] < NEAR && cq3[2] < NEAR) return false;
  clipNear(cp3, cq3); clipNear(cq3, cp3);
  S.x1 = C.cx + C.k * cp3[0] / cp3[2]; S.y1 = C.cy - C.k * cp3[1] / cp3[2];
  S.x2 = C.cx + C.k * cq3[0] / cq3[2]; S.y2 = C.cy - C.k * cq3[1] / cq3[2];
  S.d = (cp3[2] + cq3[2]) / 2;
  return true;
}

// Drawables are collected, sorted back to front and painted: t 0 = line, 1 = dot, 2 = polygon.
const draws = [], labels = [];
function addSeg(a, b, thick, minPx, col, alpha = 1, outline = 0) {
  if (projSeg(a, b)) draws.push({ t: 0, d: S.d, x1: S.x1, y1: S.y1, x2: S.x2, y2: S.y2, w: Math.max(minPx, thick * C.k / S.d), col, a: alpha, o: outline });
}
function addDot(p, rad, minPx, col, alpha = 1, outline = 0, sphere = false) {
  toCam(p, cp3);
  if (cp3[2] < NEAR) return;
  draws.push({ t: 1, d: cp3[2] - rad, x: C.cx + C.k * cp3[0] / cp3[2], y: C.cy - C.k * cp3[1] / cp3[2], r: Math.max(minPx, rad * C.k / cp3[2]), col, a: alpha, o: outline, s: sphere });
}
function addPoly(pts, col, alpha) {
  const xy = [];
  let d = 0;
  for (const p of pts) {
    toCam(p, cp3);
    if (cp3[2] < NEAR) return;
    xy.push(C.cx + C.k * cp3[0] / cp3[2], C.cy - C.k * cp3[1] / cp3[2]);
    d += cp3[2] / pts.length;
  }
  draws.push({ t: 2, d: d + 0.01, xy, col, a: alpha });
}
function addLabel(p, text, col, up, badge) { // text near a world point; `up` px above it
  toCam(p, cp3);
  if (cp3[2] >= NEAR) labels.push({ x: C.cx + C.k * cp3[0] / cp3[2], y: C.cy - C.k * cp3[1] / cp3[2] - up, text, col, badge });
}

let shadowImg = null;
function shadow(g, p, rad, alpha) { // soft blob on the floor under p, with a thin drop line up to p
  const f = [p[0], V.floor, p[2]];
  toCam(f, cp3);
  if (cp3[2] < NEAR) return;
  if (!shadowImg) {
    shadowImg = document.createElement('canvas');
    shadowImg.width = shadowImg.height = 64;
    const x = shadowImg.getContext('2d'), gr = x.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(0,0,0,.95)'); gr.addColorStop(.55, 'rgba(0,0,0,.45)'); gr.addColorStop(1, 'rgba(0,0,0,0)');
    x.fillStyle = gr; x.fillRect(0, 0, 64, 64);
  }
  const lift = Math.max(0, p[1] - V.floor), s = C.k / cp3[2], rx = rad * (1 + lift * 0.2) * s, ry = Math.max(rx * Math.sin(V.pitch), rx * 0.14);
  g.globalAlpha = alpha / (1 + lift * 0.9);
  g.drawImage(shadowImg, C.cx + s * cp3[0] - rx, C.cy - s * cp3[1] - ry, rx * 2, ry * 2);
  if (lift > 0.05 && projSeg(p, f)) {
    g.globalAlpha = 0.3; g.strokeStyle = COL.muted; g.lineWidth = 1; g.lineCap = 'butt';
    line(g, S.x1, S.y1, S.x2, S.y2);
  }
  g.globalAlpha = 1;
}

function drawGrid(g) { // world-anchored, so walking shows as moving across it; fades out with distance from the view target
  const n = Math.round(GRID_HALF / GRID), x0 = Math.round(V.tgt[0] / GRID) * GRID, z0 = Math.round(V.tgt[2] / GRID) * GRID;
  const levels = [[], [], [], []], axes = [], A = [0, V.floor, 0], B = [0, V.floor, 0];
  const add = (ax, az, bx, bz) => {
    A[0] = ax; A[2] = az; B[0] = bx; B[2] = bz;
    if (!projSeg(A, B)) return;
    const f = 1 - Math.hypot((ax + bx) / 2 - V.tgt[0], (az + bz) / 2 - V.tgt[2]) / (GRID_HALF + GRID);
    if (f > 0) (ax === 0 && bx === 0 || az === 0 && bz === 0 ? axes : levels[Math.min(3, Math.floor(f * 4))]).push(S.x1, S.y1, S.x2, S.y2);
  };
  for (let i = -n; i <= n; i++) {
    for (let j = -n; j < n; j++) {
      add(x0 + i * GRID, z0 + j * GRID, x0 + i * GRID, z0 + (j + 1) * GRID); // runs along z
      add(x0 + j * GRID, z0 + i * GRID, x0 + (j + 1) * GRID, z0 + i * GRID); // runs along x
    }
  }
  const stroke = (a) => { g.beginPath(); for (let i = 0; i < a.length; i += 4) { g.moveTo(a[i], a[i + 1]); g.lineTo(a[i + 2], a[i + 3]); } g.stroke(); };
  g.strokeStyle = COL.muted; g.lineWidth = 1; g.lineCap = 'butt';
  levels.forEach((a, i) => { g.globalAlpha = 0.08 + i * 0.09; stroke(a); });
  g.globalAlpha = 0.6; g.lineWidth = 2; stroke(axes); // the lines through the webcam: x = 0 and z = 0
  g.globalAlpha = 1;
}

function addCamera(c, cam) { // a small pyramid at the camera, its narrow end at the lens, pointing along its viewing direction
  if (c.optional && !camEnabled(c, cam)) return; // an optional camera that is switched off is not drawn
  const p = cam.pos_room, f = okPt(cam.dir_room) && unit(cam.dir_room);
  if (!okPt(p) || !f) return;
  const r = unit(cross(f, [0, 1, 0])) ?? [-1, 0, 0], u = cross(r, f);
  const ar = Array.isArray(cam.size) && cam.size[0] > 0 && cam.size[1] > 0 ? clamp(cam.size[0] / cam.size[1], 0.4, 2.5) : 4 / 3;
  const hw = ar >= 1 ? 0.085 : 0.085 * ar, hh = ar >= 1 ? 0.085 / ar : 0.085, base = madd(p, f, 0.2);
  const corners = [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([a, b]) => madd(madd(base, r, a * hw), u, b * hh));
  const on = !!cam.online && camEnabled(c, cam), a = on ? 0.7 : 0.35; // see-through, so a camera in front of the person does not hide them
  addPoly(corners, COL.text, 0.2 * a);
  corners.forEach((q, i) => { addSeg(p, q, 0.008, 1.5, COL.text, a, 1); addSeg(q, corners[(i + 1) % 4], 0.008, 1.5, COL.text, a, 1); });
  addDot(p, 0.03, 3, COL.text, a, 1.5);
  const ag = camEnabled(c, cam) && calibrated(c, cam) ? agreement(cam.agreement_px) : null;
  addLabel(p, camName(c.id) + (!camEnabled(c, cam) ? ' (off)' : !cam.online ? ' (offline)' : '') + (ag && ag.state !== 'ok' ? ` · ${ag.short}` : ''), COL.text, 16, false);
}

function addFigure(m) { // the stick figure, the head and the hands; returns whether anything was drawn
  const n0 = draws.length, { bp } = m;
  for (const [a, b, side] of BONES) { const p = bp(a), q = bp(b); if (p && q) addSeg(p, q, LIMB, 4, sideColor(side), 1, 2.5); }
  if (m.head && m.sh) { // neck: from the shoulders to the lower edge of the head
    const dn = unit(sub(m.sh, m.head));
    if (dn) addSeg(m.sh, madd(m.head, dn, HEAD_R * 0.9), 0.05, 5, COL.text, 1, 2.5);
  }
  for (const [i, side, rad] of JOINTS) { const p = bp(i); if (p) addDot(p, rad, 3, sideColor(side), 1, 2.5); }
  if (m.head) {
    addDot(m.head, HEAD_R, 8, COL.text, 1, 3, true);
    if (m.face) addSeg(madd(m.head, m.face, HEAD_R * 0.7), madd(m.head, m.face, HEAD_R + 0.07), 0.03, 5, COL.ok, 1, 2.5); // nose tick
  }
  m.hands.forEach(({ lm }, i) => {
    if (!lm) return;
    const col = i ? COL.right : COL.left, pts = PALM.map((k) => lm[k]);
    if (pts.every(okPt)) addPoly(pts, col, 0.4);
    for (const [a, b] of EDGES) if (okPt(lm[a]) && okPt(lm[b])) addSeg(lm[a], lm[b], HAND_BONE, 2.5, col, 1, 1.5);
    lm.forEach((p, k) => { if (okPt(p)) addDot(p, k % 4 === 0 && k ? HAND_JOINT * 1.3 : HAND_JOINT, 2.5, col, 1, 1.5); });
    addLabel(lm[0], SIDE[i], col, 0, true);
  });
  return draws.length > n0;
}

function paintDraws(g) {
  draws.sort((p, q) => q.d - p.d); // far first
  g.lineCap = g.lineJoin = 'round';
  for (const it of draws) {
    g.globalAlpha = it.a;
    if (it.t === 0) {
      g.beginPath(); g.moveTo(it.x1, it.y1); g.lineTo(it.x2, it.y2);
      if (it.o) { g.strokeStyle = COL.bg; g.lineWidth = it.w + it.o * 2; g.stroke(); }
      g.strokeStyle = it.col; g.lineWidth = it.w; g.stroke();
    } else if (it.t === 1) {
      g.beginPath(); g.arc(it.x, it.y, it.r, 0, 7);
      if (it.s) { // the head: a lit sphere
        const gr = g.createRadialGradient(it.x - it.r * 0.35, it.y - it.r * 0.35, it.r * 0.1, it.x, it.y, it.r);
        gr.addColorStop(0, '#fff'); gr.addColorStop(1, COL.muted);
        g.fillStyle = gr;
      } else g.fillStyle = it.col;
      g.fill();
      if (it.o) { g.strokeStyle = COL.bg; g.lineWidth = it.o; g.stroke(); }
    } else {
      g.beginPath(); g.moveTo(it.xy[0], it.xy[1]);
      for (let i = 2; i < it.xy.length; i += 2) g.lineTo(it.xy[i], it.xy[i + 1]);
      g.closePath(); g.fillStyle = it.col; g.fill();
    }
  }
  g.globalAlpha = 1;
}

function paintLabels(g, w, hh) {
  const fs = clamp(hh / 40, 13, 22);
  g.textAlign = 'center'; g.textBaseline = 'middle'; g.lineJoin = 'round';
  const placed = []; // camera names already drawn: a name that would sit on another one moves up
  for (const l of labels) {
    g.font = l.badge ? `800 ${fs * 1.15}px system-ui, sans-serif` : `700 ${fs}px system-ui, sans-serif`;
    const hw = l.badge ? fs : g.measureText(l.text).width / 2 + 6, x = clamp(l.x, hw, w - hw); // keep it inside the canvas
    let y = clamp(l.y, fs, hh - fs);
    if (!l.badge) {
      for (let n = 0; n < 4 && placed.some((q) => Math.abs(q.y - y) < fs * 1.2 && Math.abs(q.x - x) < q.hw + hw); n++) y -= fs * 1.25;
      placed.push({ x, y, hw });
    }
    if (l.badge) { // hand marker: a filled disc with the letter
      g.beginPath(); g.arc(x, y - fs * 1.4, fs * 0.95, 0, 7); g.fillStyle = l.col; g.fill();
      g.lineWidth = 2.5; g.strokeStyle = COL.bg; g.stroke();
      g.fillStyle = COL.bg; g.fillText(l.text, x, y - fs * 1.4 + 1);
    } else halo(g, l.text, x, y, l.col);
  }
  g.textBaseline = 'alphabetic'; g.textAlign = 'left';
}

function paintMessage(g, w, hh, text) { // centred note over the room
  const fs = clamp(hh / 26, 17, 30), maxW = Math.min(w - 48, 620), lines = [];
  g.font = `600 ${fs}px system-ui, sans-serif`;
  let cur = '';
  for (const word of text.split(' ')) {
    const t = cur ? cur + ' ' + word : word;
    if (cur && g.measureText(t).width > maxW) { lines.push(cur); cur = word; } else cur = t;
  }
  lines.push(cur);
  const bw = Math.max(...lines.map((l) => g.measureText(l).width)) + 44, bh = lines.length * fs * 1.35 + 30, x = (w - bw) / 2, y = (hh - bh) / 2;
  g.beginPath();
  if (g.roundRect) g.roundRect(x, y, bw, bh, 12); else g.rect(x, y, bw, bh);
  g.fillStyle = COL.panel; g.globalAlpha = 0.94; g.fill(); g.globalAlpha = 1;
  g.strokeStyle = COL.line; g.lineWidth = 1.5; g.stroke();
  g.fillStyle = COL.text; g.textAlign = 'center'; g.textBaseline = 'middle';
  lines.forEach((l, i) => g.fillText(l, w / 2, y + 15 + fs * 0.68 + i * fs * 1.35));
  g.textAlign = 'left'; g.textBaseline = 'alphabetic';
}

function bodyLine(b) {
  if (!b) return 'not reported by this hub';
  if (!b.tracked) return 'not seen';
  const from = (b.views || []).map(camName).join(' + ');
  if (b.source === 'triangulated') return from ? `3D from ${from}` : '3D from several cameras';
  if (b.source === 'single') return from ? `estimated from ${from} only` : 'estimated from one camera';
  return from ? `tracked (${from})` : 'tracked';
}

function paintLine(row, text, on, msg) {
  setData(row, 'on', +on);
  setText(text, msg);
}

const NO_AGREEMENT = { headset: 'moves with your head', ir: 'fixed to webcam' }; // cameras with no agreement value of their own
function renderViewStatus() {
  const s = snap;
  paintLine(E.vsBody, E.vsBodyText, !!s?.body?.tracked, s ? bodyLine(s.body) : '--');
  E.vsHead.hidden = !s?.head; // missing on an older hub
  if (s?.head) paintLine(E.vsHead, E.vsHeadText, !!s.head.tracked, headLine(s.head));
  SIDE.forEach((side, i) => paintLine(E['vs' + side], E['vs' + side + 'Text'], !!s?.hands?.[i]?.tracked, s ? handLine(s.hands[i]) : '--'));
  for (const c of CAMS) {
    const row = viewRows[c.id], cam = camOf(c.id);
    row.li.hidden = !cam || (c.optional && !camEnabled(c, cam));
    if (row.li.hidden) continue;
    const ag = calibrated(c, cam) ? agreement(cam.agreement_px) : null;
    const [state, word, val] = !camEnabled(c, cam) ? ['off', 'off', '--'] : !calibrated(c, cam) ? ['warn', 'not calibrated', '--']
      : ag ? [ag.state, ag.word, fmtPx(cam.agreement_px)] : ['off', NO_AGREEMENT[c.id] ?? 'no data yet', '--'];
    setText(row.val, val);
    paintChip(row.chip, state, word, ag ? `Agreement with the other cameras: ${fmtPx(cam.agreement_px)} (good is under 10, bad is over 25)` : '');
  }
  for (const b of V.btns) {
    const p = PRESETS[b.dataset.preset], g = V.goal ?? V, on = Math.abs(wrapPi(g.yaw - p.yaw)) < 0.03 && Math.abs(g.pitch - p.pitch) < 0.03;
    if (b.getAttribute('aria-pressed') !== String(on)) b.setAttribute('aria-pressed', on);
  }
}

function render3d() {
  const now = performance.now(), dt = clamp((now - V.t) / 1000, 0, 0.5), m = viewModel();
  V.t = now;
  stepView(m, dt);
  renderViewStatus();
  const [g, w, hh] = fit(E.view3d);
  if (w < 20 || hh < 20) return;
  placeCamera(w, hh);
  draws.length = labels.length = 0;
  drawGrid(g);
  if (m.anchor) shadow(g, m.anchor, 0.3, 0.85); // under the hips (else the shoulders, head or hands) and under each hand
  m.hands.forEach(({ lm }) => { if (lm) shadow(g, lm[0], 0.1, 0.75); });
  for (const c of CAMS) { const cam = camOf(c.id); if (cam) addCamera(c, cam); }
  m.hands.forEach(({ lm }, i) => { // thin lines from every camera that sees a hand to its wrist
    if (!lm) return;
    for (const c of CAMS) { const cam = camOf(c.id); if (cam?.online && cam.sees?.[i] && okPt(cam.pos_room)) addSeg(cam.pos_room, lm[0], 0, 1.6, i ? COL.right : COL.left, 0.6); }
  });
  const drawn = addFigure(m);
  paintDraws(g);
  paintLabels(g, w, hh);
  if (!snap) paintMessage(g, w, hh, 'No connection to the hub');
  else if (!m.any) paintMessage(g, w, hh, 'No person seen — stand where at least one camera can see you');
  else if (!drawn) paintMessage(g, w, hh, 'Tracked, but this hub sends no 3D positions to draw');
}

/* ---- gestures ---- */
const BAR_ROWS = ['Trigger', 'Grip', 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky'];
const MARK_KEYS = [['trigger_click', 'trigger_release'], ['grip_click', 'grip_release']];
const gv = []; // per hand: bar elements and test-area tiles

function buildGestures() {
  HANDS.forEach((name, i) => {
    const rows = BAR_ROWS.map((text, r) => {
      const fill = h('div', { cls: 'fill' }), val = h('span', { cls: 'bar-val', text: '0.00' });
      const marks = r < 2 ? [h('i', { cls: 'mk click', title: 'Click level' }), h('i', { cls: 'mk release', title: 'Release level' })] : [];
      return { row: h('div', { cls: 'bar-row' }, h('span', { cls: 'bar-name', text }), h('div', { cls: 'bar' }, fill, ...marks), val), fill, val, marks };
    });
    const state = h('span', { cls: 'pill' }), pinch = h('small', { cls: 'hint' });
    const root = h('div', { cls: `hand hand-${SIDE[i]}` }, h('h4', {}, h('b', { cls: 'key', text: SIDE[i] }), `${name} hand`, state), ...rows.map((r) => r.row), pinch);
    E.liveHands.append(root);
    const tiles = BUTTONS.map((text) => ({ el: h('div', { cls: 'tile' }, h('b', { text }), h('small', { text: 'up' }), h('span', { cls: 'n', text: '0' })) }));
    E.tiles.append(h('div', { cls: `test-hand hand-${SIDE[i]}` }, h('h4', {}, h('b', { cls: 'key', text: SIDE[i] }), `${name} hand`),
      h('div', { cls: 'tiles' }, ...tiles.map((t) => t.el))));
    gv.push({ root, rows, state, pinch, tiles });
  });
  E.resetClicks.addEventListener('click', () => clicks.forEach((c) => c.n.fill(0)));
}

function renderGestures() {
  const gc = cfg?.gestures;
  gv.forEach((v, i) => {
    const hd = snap?.hands[i], on = !!hd?.tracked;
    const vals = on ? [hd.trigger, hd.grip, ...hd.curl] : Array(7).fill(0);
    v.root.dataset.off = on ? '0' : '1';
    setText(v.state, on ? 'Tracked' : snap ? 'Not tracked' : 'No data');
    v.rows.forEach((r, k) => {
      r.fill.style.transform = `scaleX(${clamp(vals[k], 0, 1)})`;
      setText(r.val, vals[k].toFixed(2));
      if (gc && r.marks.length) r.marks.forEach((m, n) => { m.style.left = clamp(gc[MARK_KEYS[k][n]], 0, 1) * 100 + '%'; });
    });
    setText(v.pinch, on ? `Thumb to index distance: ${hd.pinch.toFixed(2)} (compare with the Pinch sliders)` : '');
    v.tiles.forEach((t, k) => {
      const down = on && ((hd.buttons >> k) & 1) === 1;
      const el = t.el;
      if (el.dataset.on !== String(+down)) el.dataset.on = +down;
      setText(el.children[1], down ? 'down' : 'up');
      setText(el.children[2], String(clicks[i].n[k]));
    });
  });
}

/* ---- smoothing ---- */
function renderSmoothing() {
  SIDE.forEach((s, i) => setText(E['jit' + s], !snap ? '--' : jit[i].value == null ? 'Not tracked' : jit[i].value.toFixed(2) + ' mm'));
  const [g, w, hh] = fit(E.jitterGraph);
  const L = 44, R = 10, T = 10, B = 22, pw = w - L - R, ph = hh - T - B, now = performance.now();
  const peak = Math.max(0, ...jit.flatMap((j) => j.hist.map((p) => p[1] ?? 0)));
  const ymax = [1, 2, 5, 10, 20, 50, 100].find((m) => m >= peak) ?? Math.ceil(peak);
  const X = (t) => L + pw * (1 - (now - t) / WINDOW), Y = (v) => T + ph * (1 - v / ymax);
  g.font = '11px system-ui, sans-serif'; g.lineWidth = 1; g.fillStyle = COL.muted;
  for (const f of [0, 0.5, 1]) {
    g.strokeStyle = COL.grid; line(g, L, Y(ymax * f), L + pw, Y(ymax * f));
    g.textAlign = 'right'; g.fillText(`${ymax * f} mm`, L - 5, Y(ymax * f) + 4);
  }
  g.textAlign = 'left'; g.fillText('-10 s', L, hh - 6);
  g.textAlign = 'right'; g.fillText('now', L + pw, hh - 6);
  jit.forEach((j, i) => {
    g.strokeStyle = i ? COL.right : COL.left; g.lineWidth = 2; g.setLineDash(i ? [6, 3] : []);
    g.beginPath();
    let pen = false, tail = null;
    for (const [t, v] of j.hist) {
      if (v == null) { pen = false; continue; }
      if (pen) g.lineTo(X(t), Y(v)); else g.moveTo(X(t), Y(v));
      pen = true; tail = [X(t), Y(v)];
    }
    g.stroke(); g.setLineDash([]);
    if (tail && now - j.hist.at(-1)[0] < 1000) { g.font = 'bold 13px system-ui, sans-serif'; g.textAlign = 'right'; halo(g, SIDE[i], Math.min(tail[0] - 4, L + pw - 4), tail[1] + (i ? 16 : -6), g.strokeStyle); }
  });
}

/* ---- calibration ---- */
async function post(kind) {
  try {
    const r = await fetch('/api/calibrate/' + kind, { method: 'POST' });
    const j = await r.json().catch(() => ({}));
    return r.ok ? j : { ok: false, error: typeof j.detail === 'string' && j.detail ? j.detail : `The hub answered with error ${r.status}.` };
  } catch { return { ok: false, error: 'Could not reach the hub.' }; }
}
function showResult(el, ok, msg) {
  el.hidden = false;
  el.dataset.k = ok ? 'ok' : 'err';
  el.textContent = (ok ? 'Done. ' : 'Failed. ') + msg;
}
const toDeg = (rad) => (rad * 180 / Math.PI).toFixed(1) + '°';

// What a finished countdown calibration says: `kind` -> text from the hub's answer.
const signed = (v, d = 1) => { const t = Math.abs(v).toFixed(d); return (v < 0 && +t !== 0 ? '−' : '') + t; }; // with a true minus sign, and no "−0.0"
const COUNT_DONE = {
  rest: (r) => `Camera position set to ${r.pos.map((v) => v.toFixed(2)).join(', ')} m.${r.hands_used === 1 ? ' Only one hand was tracked.' : ''}`,
  level: (r) => {
    const deg = (v) => (Number.isFinite(v) ? signed(v) + '°' : '--');
    const f = Number(r.scale_factor), height = getPath(cfg, 'tracking.user_height_m') ?? 0;
    const size = r.scale_factor != null && Number.isFinite(f) ? `size corrected by ${f < 1 ? '−' : '+'}${Math.abs(Math.round((f - 1) * 100))} %`
      : r.scale_note ? String(r.scale_note)
      : r.measured_nose_to_ankle_m != null ? (Math.abs((height - 0.2) / r.measured_nose_to_ankle_m - 1) <= 0.02 ? 'size already fits your height'
        : `size not corrected: nose to ankle measured ${r.measured_nose_to_ankle_m} m does not fit your height`)
      : !(height > 0.5) ? 'enter your height to correct the size as well' : '';
    return `Standing calibration: camera tilt ${deg(r.pitch_deg)}, roll ${deg(r.roll_deg)}${size ? '; ' + size : ''}.`;
  },
};
COUNT_DONE.head = COUNT_DONE.rest;

// Countdown calibrations: `kind` is the API name and the prefix of the element ids.
const countTimers = {};
function resetCountUi(kind) {
  clearInterval(countTimers[kind]); countTimers[kind] = 0;
  E[kind + 'Count'].textContent = ''; E[kind + 'Btn'].textContent = 'Start countdown';
}
function countdownCalibration(kind) {
  if (countTimers[kind]) { resetCountUi(kind); return; } // second press cancels
  const btn = E[kind + 'Btn'], count = E[kind + 'Count'], result = E[kind + 'Result'];
  let n = 3;
  result.hidden = true;
  btn.textContent = 'Cancel';
  count.textContent = n;
  countTimers[kind] = setInterval(async () => {
    if (--n > 0) { count.textContent = n; return; }
    resetCountUi(kind);
    btn.disabled = true;
    const r = await post(kind);
    btn.disabled = false;
    if (r.ok) { showResult(result, true, COUNT_DONE[kind](r)); loadConfig(); }
    else showResult(result, false, r.error || 'Unknown error.');
  }, 1000);
}

async function alignHeadset() {
  E.alignBtn.disabled = true;
  const r = await post('align');
  E.alignBtn.disabled = false;
  if (r.ok) showResult(E.alignResult, true, `Yaw offset is ${toDeg(r.yaw_offset)}.`);
  else showResult(E.alignResult, false, r.error || 'Unknown error.');
}

// Calibrations the hub collects over time: the camera-to-camera one (progress and result in snapshot.multicam) and the headset
// camera one (snapshot.headcam, missing on an older hub). Pressing Start on the camera one first counts down (time to get into
// position), then POSTs. The headset one is not a procedure: the hub fits it continuously, so it is a live status line plus a
// "Learn again" button that POSTs at once (`live: true`, drawn by renderLiveCollector).
// Each is mounted twice from #tpl-multi (strip under the wall, card on the Calibrate tab, see buildCollectors) and both mounts are
// drawn by renderCollector from the one state object per calibration below.
const COUNT_MS = 3000; // ms between pressing Start and the request
const failed = (res) => `Failed. ${res.error || 'Calibration did not succeed.'}`;
const COLLECTORS = {
  cameras: {
    api: 'cameras', total: 30, // `total`: seconds, from the POST answer
    state: () => snap?.multicam,
    waiting: 'No samples yet. Stay where two cameras can see you.',
    rows: (m) => Object.entries(m.pairs || {}).map(([k, n]) => [`${k.split('+').map(camName).join(' + ')}: ${plural(n, 'sample')}`]),
    done: (res) => (!res.ok ? [failed(res), []] : ['Done. Calibration finished.', Object.entries(res.cameras || {}).map(([id, cr]) => (cr.ok
      ? [`${camName(id)}: calibrated, ` + [Number.isFinite(cr.rms_px) && `error ${cr.rms_px.toFixed(1)} px`, Number.isFinite(cr.samples) && plural(cr.samples, 'sample')].filter(Boolean).join(', '), 'ok']
      : [`${camName(id)}: failed` + (cr.error ? ` - ${cr.error}` : ''), 'err']))]),
  },
  headset: {
    api: 'headset', live: true,
    state: () => snap?.headcam,
    waiting: 'Waiting for a hand that the headset camera and two other cameras both see. Wear the headset and look around a little.', // when the hub gives no note
    // why Learn again cannot be pressed: the camera is off, the hub is too old, or the hub said there is no VR app (cleared when there is one)
    off: (c) => (!camEnabled(CAMS.find((x) => x.id === 'headset'), camOf('headset')) ? 'Turn on the headset camera first (Headset camera settings below).'
      : !snap?.headcam ? 'This hub cannot calibrate the headset camera.' : c.blocked ? 'Start a VR app first.' : ''),
  },
};
for (const [id, c] of Object.entries(COLLECTORS)) {
  Object.assign(c, { id, goAt: 0, timer: 0, pending: false, busyUntil: 0, error: '', blocked: false, uis: [] });
  // goAt/timer: while counting down, when the request goes out. pending/busyUntil: bridge the gap until the snapshot says "collecting".
}

function buildCollectors() {
  for (const host of $$('[data-calib]')) {
    const c = COLLECTORS[host.dataset.calib];
    const el = document.importNode(E['tpl-multi'].content.firstElementChild, true);
    $$('[data-only]', el).forEach((n) => { if (!n.dataset.only.split(' ').includes(host.dataset.multi)) n.remove(); });
    const r = refs(el, 'r');
    if (r.hint) r.hint.textContent = $(`[data-calib-text="${c.id}"]`).textContent; // the instruction is written once, on the Calibrate tab
    r.btn.addEventListener('click', () => onCollectorBtn(c));
    if (c.live) r.btn.classList.remove('primary'); // a rarely needed button, not the main action
    c.uis.push({ r });
    host.append(el);
  }
}

function onCollectorBtn(c) {
  if (c.live) { // no countdown: the hub forgets what it learnt at once
    if (!c.pending) { c.error = ''; startCollector(c); }
    return;
  }
  if (c.timer) { // second press cancels the countdown
    clearTimeout(c.timer);
    c.timer = c.goAt = 0;
    return;
  }
  c.error = '';
  c.goAt = performance.now() + COUNT_MS;
  c.timer = setTimeout(() => startCollector(c), COUNT_MS);
}

async function startCollector(c) {
  c.timer = c.goAt = 0;
  if (!c.live && c.state()?.collecting) return; // someone else already started one
  c.pending = true;
  const r = await post(c.api);
  c.pending = false;
  if (r.ok) { c.total = r.seconds || c.total; c.busyUntil = performance.now() + 1500; return; }
  c.error = r.error || 'Unknown error.';
  c.blocked = c.id === 'headset' && /VR|headset orientation/i.test(c.error); // no VR app running: wait for one instead of letting the button fail again
}

function setRows(ul, rows) { // rows: [text, state?]; rebuilt only when they change
  const key = JSON.stringify(rows);
  if (ul._k === key) return;
  ul._k = key;
  ul.replaceChildren(...rows.map(([text, k]) => h('li', k ? { text, 'data-k': k } : { text })));
}

// The live status of a continuously fitted calibration (the headset camera): [text, state for the border, whether to show the instructions].
const capFirst = (t) => t.charAt(0).toUpperCase() + t.slice(1);
const ago = (s) => (s < 1 ? 'just now' : `${s < 120 ? Math.round(s) + ' s' : Math.round(s / 60) + ' min'} ago`);
function liveStatus(c, m) {
  const res = !m.collecting && m.result?.ok ? m.result : null;
  if (!res) { // no usable fit yet: still learning, or lost it
    const note = String(m.note || c.waiting).trim().replace(/\.+$/, ''), n = Number.isFinite(m.samples) ? m.samples : 0;
    return [`${capFirst(note)}. ${plural(n, 'sighting')} so far.`, 'wait', true];
  }
  const n = Number.isFinite, age = res.age_s, stale = n(age) && age > 60;
  const parts = [n(res.rotation_error_deg) && `camera and motion sensors agree to ${res.rotation_error_deg.toFixed(1)}°`,
    n(age) && !stale && `refreshed ${ago(age)}`, n(res.used) && `from ${plural(res.used, 'sighting')}`].filter(Boolean);
  const text = `In use${parts.length ? ': ' + parts.join(', ') : ''}.` + (stale ? ` Last refreshed ${ago(age)}: show a hand to the cameras to refresh it.` : '');
  return [text, stale ? 'wait' : 'ok', false];
}

function renderLiveCollector(c, m, off) {
  const [text, k, hint] = off || !m ? ['', '', false] : liveStatus(c, m); // when it cannot be used, only the reason is shown
  for (const { r } of c.uis) {
    r.btn.disabled = c.pending || !!off;
    setText(r.btn, 'Learn again');
    for (const el of [r.count, r.left, r.bar, r.pairs, r.cams]) el.hidden = true; // no countdown, no time left, no progress bar
    r.note.hidden = !off;
    if (off) setText(r.note, off);
    r.error.hidden = !c.error;
    setData(r.error, 'k', 'err');
    if (c.error) setText(r.error, 'Failed. ' + c.error);
    r.result.hidden = !text;
    setData(r.result, 'k', k);
    if (text) setText(r.result, text);
    if (r.hint) r.hint.hidden = !hint || !!c.error || !!off;
  }
}

function renderCollector(c) {
  const m = c.state();
  if (c.blocked && (snap?.hmd?.valid || !snap?.hmd)) { c.blocked = false; c.error = ''; } // a VR app is running now
  const off = c.off?.(c) ?? '';
  if (c.live) { renderLiveCollector(c, m, off); return; }
  const collecting = !!m?.collecting, counting = c.timer !== 0, starting = c.pending || performance.now() < c.busyUntil;
  if (off && counting) onCollectorBtn(c); // the camera was switched off during the countdown: cancel
  const left = Math.max(0, Math.ceil(m?.seconds_left ?? 0));
  if (collecting) c.total = Math.max(c.total, left);
  const res = !collecting && !starting && !c.timer && !c.error && m?.result; // a fresh error replaces the old result
  const rows = collecting ? c.rows(m) : [];
  const [resultText, resultRows] = res ? c.done(res) : ['', []];
  for (const { r } of c.uis) {
    r.btn.disabled = collecting || starting || !!off;
    setText(r.btn, c.timer ? 'Cancel' : 'Start');
    r.count.hidden = !counting;
    if (counting) setText(r.countNum, String(Math.max(1, Math.ceil((c.goAt - performance.now()) / 1000))));
    r.left.hidden = !(collecting || starting);
    r.leftNum.hidden = !collecting;
    setText(r.leftNum, String(left));
    setText(r.leftUnit, collecting ? 's left' : 'Starting...');
    r.bar.hidden = !collecting;
    if (collecting) { r.bar.max = c.total; r.bar.value = c.total - left; }
    r.pairs.hidden = !collecting;
    if (collecting) setRows(r.pairs, rows.length ? rows : [[c.waiting]]);
    r.note.hidden = !off || collecting || starting;
    if (off) setText(r.note, off);
    r.error.hidden = !c.error || collecting || starting;
    setData(r.error, 'k', 'err');
    if (c.error) setText(r.error, 'Failed. ' + c.error);
    if (r.hint) r.hint.hidden = collecting || !!res || !!c.error || !!off;
    r.result.hidden = !res;
    r.cams.hidden = !res || !resultRows.length;
    if (!res) continue;
    setData(r.result, 'k', res.ok ? 'ok' : 'err');
    setText(r.result, resultText);
    setRows(r.cams, resultRows);
  }
}

// Both calibrations are drawn on the Cameras tab (strips) and on the Calibrate tab (cards); the headset strip only while that camera is on.
function renderCollectors() {
  for (const c of Object.values(COLLECTORS)) renderCollector(c);
  const on = camViews.headset.visible;
  if (E.headStrip.hidden === on) E.headStrip.hidden = !on;
}

function renderCalibration() {
  setText(E.yawNow, snap ? toDeg(snap.yaw_offset) : '--');
  renderCollectors();
}

/* ===================================================================== main loop */

const RENDER = { cameras: renderCameras, view: render3d, gestures: renderGestures, smoothing: renderSmoothing, calibration: renderCalibration };
function frame() {
  requestAnimationFrame(frame);
  renderStatus();
  RENDER[activeTab]();
}

function init() {
  readColors();
  trackTopbar();
  buildCameras(); // before buildSliders: the settings sections hold the slider groups
  buildSliders();
  buildGestures();
  buildCollectors();
  buildView();
  $$('[data-path]').forEach((el) => { el.disabled = true; }); // enabled once the config has loaded
  bindControls();
  bindTabs();
  E.restBtn.addEventListener('click', () => countdownCalibration('rest'));
  E.headBtn.addEventListener('click', () => countdownCalibration('head'));
  E.levelBtn.addEventListener('click', () => countdownCalibration('level'));
  E.alignBtn.addEventListener('click', alignHeadset);
  setTab(store.get('handhub.tab'));
  loadConfig();
  loadCameras();
  connect();
  setInterval(() => { if (wsUp && performance.now() - lastMsg > 3000) ws.close(); }, 1000); // silent connection
  requestAnimationFrame(frame);
}
init();
