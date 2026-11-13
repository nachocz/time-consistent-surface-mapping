// Page 2 (Secs. 3.1-3.3): a deformation sequence of a simulated object, perceived anew at
// every frame, mapped to the target by ZoomOut from scratch (NTC, left) and by the
// time-consistent method (right).
import { MatrixMap } from '../lib/charts.js';
import { LinePlot } from '../lib/plot.js';
import { column, gripperColor, pullColors, referenceColors, valueColors } from '../lib/present.js';
import { initPage } from '../lib/shell.js';
import { onThemeChange, palette } from '../lib/theme.js';
import { button, card, control, fmt, h, label, readouts, segmented, select, seriesLegend, slider } from '../lib/ui.js';
import { MeshView } from '../lib/view3d.js';
import { Camera } from '../sim/camera.js';
import { DEFAULTS, TimeConsistentMapper, ntcStep } from '../sim/tcmap.js';
import { matchError } from '../sim/truth.js';
import { SCENARIOS, World } from '../sim/world.js';

await initPage();

const params = new URLSearchParams(location.search);
const PERIOD = 120, K_MAX = 1200;
const settings = {
    scenario: SCENARIOS[params.get('object')] ? params.get('object') : 'sheet',
    mode: ['map', 'eig', 'error'].includes(params.get('color')) ? params.get('color') : 'map',
    i: 5, amplitude: 1, points: 320, noise: 0.05, dots: false,
    ...DEFAULTS, step: 4,
};
const cache = {};
let world = null, camera = null, mapper = null, out = null, ntc = null, frame = null, history = null, running = false, speed = 1, lastWall = 0, budget = 0;
let refN = null, uvCache = null;

// ---------------------------------------------------------------- layout
const workspace = document.getElementById('workspace');
const runCard = card('', '');
const scenarioPick = select({ options: Object.entries(SCENARIOS).map(([value, s]) => ({ value, label: s.title })), value: settings.scenario,
    onChange: (v) => { settings.scenario = v; load(); } });
scenarioPick.el.style.width = 'auto';
const runButton = button('▶ Run', () => setRunning(!running), { primary: true });
const stepButton = button('Step', () => { setRunning(false); iterate(); render(true); });
const restartButton = button('↺ Restart', () => { setRunning(false); restart(); });
const speedPick = segmented({ options: [1, 2, 4].map((v) => ({ value: v, label: `${v}×` })), value: 1, onChange: (v) => { speed = v; } });
const clockEl = h('span', { class: 'clock' }, 'k = 1');
runCard.body.append(h('div', { class: 'toolbar' }, h('b', { style: 'font-size:0.8rem' }, 'Object'), scenarioPick.el, runButton, stepButton, restartButton,
    h('span', { class: 'toolbar-divider' }), speedPick.el, clockEl), h('p', { class: 'hint', style: 'margin:8px 0 0' }, h('span', { id: 'note' })));

const viewCard = card('Every frame: a new mesh', 'drag to orbit (both views move together)');
const pair = h('div', { class: 'view-pair' });
const leftEl = h('div', { class: 'view3d pair-fit' }), rightEl = h('div', { class: 'view3d pair-fit' });
const busy = h('div', { class: 'busy' }, 'Preparing the scenario…');
leftEl.append(h('span', { class: 'view-tag ntc' }, label('ZoomOut every frame: $\\Pi_{\\mathrm{NTC}}(k)$')), busy);
rightEl.append(h('span', { class: 'view-tag ours' }, label('Time-consistent: $\\Pi(k)$')));
pair.append(leftEl, rightEl);
const left = new MeshView(leftEl), right = new MeshView(rightEl);
left.link(right);
const modePick = segmented({ options: [{ value: 'map', label: 'surface map' }, { value: 'eig', label: 'eigenfunction $\\phi_i$' }, { value: 'error', label: 'error $\\|\\mathbf e_m\\|$' }],
    value: settings.mode, onChange: (v) => { settings.mode = v; render(true); } });
const iSlider = slider({ label: 'eigenfunction $i$', min: 1, max: 30, step: 1, value: settings.i, format: (v) => v.toFixed(0), onInput: (v) => { settings.i = Math.round(v); render(true); } });
const dotsToggle = h('label', { class: 'switch' }, h('input', { type: 'checkbox', onchange: (e) => { settings.dots = e.target.checked; render(true); } }), label('tracked points $\\mathbf X(k)$'));
const viewHint = h('p', { class: 'hint', style: 'margin:6px 0 0' });
viewCard.body.append(pair, h('div', { class: 'view-controls' }, control('color', modePick.el), iSlider.el, dotsToggle), viewHint);

const targetCard = card('Target $\\mathcal N$', 'reference colors');
const targetEl = h('div', { class: 'view3d thumb' });
targetCard.body.append(targetEl);
const targetView = new MeshView(targetEl);

const nowCard = card('Now', '');
const values = readouts([
    { key: 'k', label: 'frame $k$' }, { key: 'mret', label: '$M_{\\mathrm{ret}}(k)$', title: 'number of points of the mesh of this frame' },
    { key: 'distinct', label: 'distinct tracked points', title: 'how many different points of this frame the M tracked points select' },
    { key: 'e', label: '$\\|\\mathbf e\\|$', title: 'shape error (21) of the time-consistent map [cm]' },
    { key: 'entc', label: '$\\|\\mathbf e_{\\mathrm{NTC}}\\|$', title: 'shape error with the map of ZoomOut computed at this frame [cm]' },
    { key: 'true', label: 'error to true match [cm]', title: 'mean distance to the same material point on N (ground truth, ours / NTC)' },
]);
nowCard.body.append(values.el);

const errCard = card('Shape error', 'as in Fig. 8 (one axis for both)');
const errCanvas = h('canvas', { class: 'plot-canvas' });
const errPlot = new LinePlot(errCanvas, { xLabel: 'k', yLabel: '[cm]' });
const errLegend = h('div');
errCard.body.append(errCanvas, errLegend);

const cmCard = card('Tracking map $\\mathbf C^{\\mathcal M}(k)$', 'previous basis ← new basis');
const cmCanvas = h('canvas', { class: 'chart-canvas square' });
const cmMap = new MatrixMap(cmCanvas);
const signStrip = h('div', { class: 'sign-strip' });
cmCard.body.append(h('div', { class: 'matrix-small' }, cmCanvas, signStrip), h('p', { class: 'hint' }, label('Diagonal entries near $\\pm1$: the same eigenvector, possibly with its sign reversed (red in the strip, (13)). Blocks off the diagonal: eigenvectors of nearly equal eigenvalues that mix between frames.')));

const cCard = card('Functional maps to $\\mathcal N$', 'red box: the block reused at the next frame');
const cOursCanvas = h('canvas', { class: 'chart-canvas square' }), cNtcCanvas = h('canvas', { class: 'chart-canvas square' });
const cOurs = new MatrixMap(cOursCanvas), cNtc = new MatrixMap(cNtcCanvas);
cCard.body.append(h('div', { class: 'pair-charts' },
    h('div', {}, h('p', { class: 'chart-title' }, label('ours: $\\mathbf C(I_{\\max},k)$')), cOursCanvas),
    h('div', {}, h('p', { class: 'chart-title' }, label('NTC: $\\mathbf C_{\\mathrm{NTC}}(I_{\\max},k)$')), cNtcCanvas)));

const timeCard = card('Processing time per frame', 'in this browser, as in Fig. 9');
const timeCanvas = h('canvas', { class: 'plot-canvas short' });
const timePlot = new LinePlot(timeCanvas, { xLabel: 'k', yLabel: 'ms' });
const timeLegend = h('div');
timeCard.body.append(timeCanvas, timeLegend);

const setCard = card('Sequence and perception', 'restart to apply');
const ampSlider = slider({ label: 'motion amplitude', min: 0.2, max: 1.5, step: 0.05, value: settings.amplitude, onChange: (v) => { settings.amplitude = v; restart(); } });
const ptsSlider = slider({ label: 'points per frame (on the rest shape)', min: 150, max: 600, step: 10, value: settings.points, format: (v) => v.toFixed(0), onChange: (v) => { settings.points = v; restart(); } });
const noiseSlider = slider({ label: 'noise $\\sigma$ [cm]', min: 0, max: 0.3, step: 0.01, value: settings.noise, onChange: (v) => { settings.noise = v; restart(); } });
setCard.body.append(ampSlider.el, ptsSlider.el, noiseSlider.el);

const methodCard = card('Method', 'restart to apply');
const intSlider = (key, text, min, max) => slider({ label: text, min, max, step: 1, value: settings[key], format: (v) => v.toFixed(0), onChange: (v) => { settings[key] = Math.round(v); restart(); } }).el;
const trackPick = segmented({ options: [{ value: 'anchored', label: 'anchored (code)' }, { value: 'literal', label: '(13)–(15) as written' }, { value: 'none', label: 'off' }],
    value: settings.tracking, onChange: (v) => { settings.tracking = v; restart(); } });
const warmPick = segmented({ options: [{ value: true, label: 'on' }, { value: false, label: 'off (ICP every frame)' }], value: settings.warmStart, onChange: (v) => { settings.warmStart = v; restart(); } });
const initPick = segmented({ options: [{ value: 'icp', label: 'ICP (10)' }, { value: 'identity', label: '$\\mathbf C=\\mathbf I$' }], value: settings.init, onChange: (v) => { settings.init = v; restart(); } });
methodCard.body.append(intSlider('Iinit', '$I_{\\mathrm{initial}}$', 2, 10), intSlider('Iupdate', '$I_{\\mathrm{update}}$', 2, 20), intSlider('Imax', '$I_{\\max}$', 10, 50),
    intSlider('step', 'increment of $I$ per step', 1, 5),
    h('details', { class: 'variants', open: params.has('variants') }, h('summary', {}, 'Implementation variants'),
        control('tracking of the basis (Sec. 3.2)', trackPick.el), control('warm start (Sec. 3.3)', warmPick.el), control('initialization (Sec. 3.1)', initPick.el)));

workspace.append(runCard.el, h('div', { class: 'dash' },
    h('div', { class: 'dash-view' }, viewCard.el),
    h('div', { class: 'dash-side' }, nowCard.el, errCard.el, targetCard.el, cmCard.el, cCard.el, timeCard.el, setCard.el, methodCard.el)));

// ---------------------------------------------------------------- simulation
function load() {
    setRunning(false);
    busy.hidden = false;
    document.getElementById('note').textContent = `${SCENARIOS[settings.scenario].title}: ${SCENARIOS[settings.scenario].note}.`;
    setTimeout(() => {
        world = cache[settings.scenario] || (cache[settings.scenario] = new World(settings.scenario));
        restart();
        // ?frames=n: start n frames into the sequence (links to a given state)
        const skip = Math.min(Number(params.get('frames')) || 0, K_MAX - 1);
        for (let i = 0; i < skip; i++) iterate();
        if (skip) render(true);
        left.frame({ ...world.scenario.view, zoom: 0.85 }, [world.lattice.rest, ...motionSamples()]);
        targetView.frame({ ...world.scenario.view, zoom: 0.8 }, [world.target.P]);
        busy.hidden = true;
    }, 30);
}

// a few states of the motion, to frame the views
function motionSamples() {
    const s = world.load(motionSpec(1));
    return [s.P];
}
function motionSpec(a) {
    return world.scenario.motion.map((m) => ({ rot: m.rot.map((x) => a * settings.amplitude * x), move: m.move.map((x) => a * settings.amplitude * x) }));
}

function restart() {
    world.reset();
    camera = new Camera(world.lattice, { points: settings.points, noise: settings.noise, seed: 5 });
    const target = camera.capture(world.target.P);
    mapper = new TimeConsistentMapper(target, { Iinit: settings.Iinit, Iupdate: settings.Iupdate, Imax: settings.Imax, step: settings.step,
        tracking: settings.tracking, warmStart: settings.warmStart, init: settings.init, slanted: settings.slanted });
    refN = referenceColors(mapper.target.V);
    targetView.mesh('N', { V: mapper.target.V, F: mapper.target.F, colors: refN, wire: true });
    history = { k: [], e: [], eNtc: [], tOurs: [], tNtc: [], msOurs: [], msNtc: [] };
    iSlider.input.max = settings.Imax;
    perceive();
    render(true);
}

function perceive() {
    frame = camera.capture(world.P);
    out = mapper.step(frame);
    ntc = ntcStep(out.ret, mapper.target, { Iinit: settings.Iinit, Imax: settings.Imax, step: settings.step });
    // ground truth: the surface parameters of the tracked points
    const idx = out.k === 1 || settings.tracking === 'none' ? null : out.track.PiM;
    uvCache = idx ? Float64Array.from({ length: 2 * idx.length }, (_, i) => frame.uv[2 * idx[i >> 1] + (i & 1)]) : frame.uv;
    const chart = world.scenario.shape.chart;
    history.k.push(out.k); history.e.push(out.error.norm); history.eNtc.push(ntc.error.norm);
    history.tOurs.push(matchError({ uv: uvCache }, mapper.target, out.Pi, chart));
    history.tNtc.push(matchError(frame, mapper.target, ntc.Pi, chart));
    history.msOurs.push(out.ms.total); history.msNtc.push(ntc.ms);
}

function iterate() {
    if (history.k.length >= K_MAX) return;
    const k = out.k, a = Math.sin(Math.PI * k / PERIOD) ** 2;
    world.moveTowards(motionSpec(a), 1);
    perceive();
}

function setRunning(flag) {
    running = flag && history && history.k.length < K_MAX;
    runButton.replaceChildren(label(running ? '❚❚ Pause' : '▶ Run'));
    if (running) { lastWall = performance.now(); budget = 0; requestAnimationFrame(loop); }
}
let frameCount = 0;
function loop(now) {
    if (!running) return;
    budget += Math.min((now - lastWall) / 1000, 0.1) * 8 * speed;     // 1x: eight frames per second
    lastWall = now;
    const start = performance.now();
    while (budget >= 1 && performance.now() - start < 40) { iterate(); budget -= 1; }
    budget = Math.min(budget, 2);
    render(++frameCount % 4 === 0);
    if (history.k.length >= K_MAX) setRunning(false);
    else requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- views
function render(full = false) {
    if (!out) return;
    const p = palette(), K = out.ret.K;
    iSlider.el.hidden = settings.mode !== 'eig';
    let cl, cr;
    if (settings.mode === 'map') {
        cl = pullColors(refN, ntc.Pi); cr = pullColors(refN, out.Pi);
        viewHint.textContent = 'Each point takes the color of its image on the target N (top right).';
    } else if (settings.mode === 'eig') {
        const i = Math.min(settings.i, K) - 1;
        cl = valueColors(column(out.ret.evecs, out.ret.n, out.ret.K, i));
        cr = valueColors(column(out.basis.evecs, out.X.length / 3, out.basis.K, i));
        viewHint.textContent = `φ_${i + 1}: left, the eigenvector of this frame's mesh; right, the tracked basis.`;
    } else {
        const top = Math.max(1, ...ntc.error.per, ...out.error.per);
        cl = valueColors(ntc.error.per, 'viridis', top); cr = valueColors(out.error.per, 'viridis', top);
        viewHint.textContent = `‖e_m‖ from 0 (dark) to ${top.toFixed(1)} cm (yellow), same scale on both sides.`;
    }
    left.mesh('M', { V: out.ret.V, F: out.ret.F, colors: cl, wire: true });
    right.mesh('M', { V: out.X, F: out.F, colors: cr, wire: true });
    if (settings.dots) right.points('dots', { P: out.X, color: p.track, radius: 0.14 }); else right.hide('dots');
    const halves = world.grippers.map((g) => g.half), colors = world.grippers.map((_, k) => gripperColor(p, k));
    left.grippers(world.poses, halves, colors); right.grippers(world.poses, halves, colors);
    // readouts
    const k = out.k;
    clockEl.textContent = `k = ${k}`;
    const distinct = out.track ? new Set(out.track.PiM).size : out.X.length / 3;
    values.set('k', String(k)); values.set('mret', String(out.ret.n));
    values.set('distinct', `${distinct} / ${out.X.length / 3}`, distinct < 0.6 * out.X.length / 3 ? 'bad' : '');
    values.set('e', fmt(out.error.norm, 1)); values.set('entc', fmt(ntc.error.norm, 1));
    const n = history.k.length - 1;
    values.set('true', `${fmt(history.tOurs[n], 1)} / ${fmt(history.tNtc[n], 1)}`);
    // maps
    if (out.track) {
        const I = out.track.I;
        cmMap.set({ C: out.track.CM, I, J: I, size: settings.Imax });
        signStrip.replaceChildren(...Array.from(out.track.sgn, (s) => h('i', { class: s < 0 ? 'flip' : 'keep', title: s < 0 ? 'sign reversed' : 'sign kept' })));
    } else {
        cmMap.set({ C: null });
        signStrip.replaceChildren();
    }
    cOurs.set({ C: out.C, I: out.I, J: out.J, size: Math.max(settings.Imax, out.J), block: settings.warmStart ? settings.Iupdate : 0 });
    cNtc.set({ C: ntc.C, I: ntc.I, J: ntc.J, size: Math.max(settings.Imax, out.J) });
    if (!full && running) return;
    errPlot.set({ series: [
        { x: history.k, y: history.eNtc, color: p.ntc, width: 1.3 },
        { x: history.k, y: history.e, color: p.ours, width: 2.2 },
    ] });
    timePlot.set({ series: [
        { x: history.k, y: history.msNtc, color: p.ntc, width: 1.2, dash: [4, 3] },
        { x: history.k, y: history.msOurs, color: p.ours, width: 1.6 },
    ] });
}

function legends() {
    const p = palette();
    errLegend.replaceChildren(seriesLegend([{ label: 'time-consistent $\\|\\mathbf e\\|$', color: p.ours }, { label: 'ZoomOut every frame $\\|\\mathbf e_{\\mathrm{NTC}}\\|$', color: p.ntc }]));
    timeLegend.replaceChildren(seriesLegend([{ label: 'time-consistent', color: p.ours }, { label: 'ZoomOut every frame', color: p.ntc, dash: 'dashed' }]));
}
onThemeChange(() => { for (const v of [left, right, targetView]) v.refreshTheme(); legends(); render(true); });
legends();
load();
