// Page 4 (Sec. 4): the closed loop of the shape controller (js/sim/control.js) on the
// time-consistent maps (js/sim/tcmap.js) of an object simulated with ARAP and perceived
// by a depth camera (js/sim/session.js).
import { DistributionMap } from '../lib/charts.js';
import { LinePlot } from '../lib/plot.js';
import { gripperColor, pullColors, referenceColors, valueColors } from '../lib/present.js';
import { initPage } from '../lib/shell.js';
import { alpha, onThemeChange, palette } from '../lib/theme.js';
import { button, card, control, fmt, h, label, readouts, segmented, select, seriesLegend, slider, toggle } from '../lib/ui.js';
import { MeshView } from '../lib/view3d.js';
import { CONTROL_DEFAULTS } from '../sim/control.js';
import { ControlSession } from '../sim/session.js';
import { SCENARIOS, World } from '../sim/world.js';

await initPage();

const params = new URLSearchParams(location.search);
const K_MAX = 600, DEG = 180 / Math.PI;
const settings = {
    scenario: SCENARIOS[params.get('object')] ? params.get('object') : 'tee',
    colors: params.get('color') === 'error' ? 'error' : 'map',
    target: true, vectors: false, prediction: false, body: true,
    points: 320, noise: 0.05, ntc: true,
    ...CONTROL_DEFAULTS,
};
if (params.has('gamma')) settings.gamma = Number(params.get('gamma'));
const cache = {};
let world = null, session = null, refN = null, running = false, speed = 1, lastWall = 0, budget = 0, frames = 0;

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
const clockEl = h('span', { class: 'clock' }, 'k = 0');
runCard.body.append(h('div', { class: 'toolbar' }, h('b', { style: 'font-size:0.8rem' }, 'Object'), scenarioPick.el, runButton, stepButton, restartButton,
    h('span', { class: 'toolbar-divider' }), speedPick.el, clockEl), h('p', { class: 'hint', style: 'margin:8px 0 0' }, h('span', { id: 'note' })));

const viewCard = card('Simulated object, tracked mesh, and target', 'drag to orbit, scroll to zoom');
const viewEl = h('div', { class: 'view3d dash-fit' });
const busy = h('div', { class: 'busy' }, 'Preparing the scenario…');
const tools = h('div', { class: 'view-toolbar' });
viewEl.append(tools, busy);
const view = new MeshView(viewEl);
tools.append(button('Reset view', () => frameView(), { small: true }));
const colorPick = segmented({ options: [{ value: 'map', label: 'surface map $\\Pi(k)$' }, { value: 'error', label: 'error $\\|\\mathbf e_m\\|$' }], value: settings.colors,
    onChange: (v) => { settings.colors = v; render(true); } });
const layerToggles = [
    toggle({ label: 'target $\\mathcal N$', value: settings.target, onChange: (v) => { settings.target = v; render(true); } }),
    toggle({ label: 'error vectors', value: settings.vectors, onChange: (v) => { settings.vectors = v; render(true); } }),
    toggle({ label: 'model prediction $\\mathbf x+\\hat{\\mathbf L}_g\\mathbf u_g$', value: settings.prediction, onChange: (v) => { settings.prediction = v; render(true); } }),
    toggle({ label: 'object body', value: settings.body, onChange: (v) => { settings.body = v; render(true); } }),
];
const legendEl = h('div');
viewCard.body.append(viewEl, h('div', { class: 'toolbar', style: 'margin-top:10px' }, colorPick.el, ...layerToggles.map((t) => t.el)), legendEl);

const nowCard = card('Now', '');
const values = readouts([
    { key: 'k', label: 'iteration $k$' }, { key: 'phase', label: 'phase' }, { key: 'g', label: 'next action' },
    { key: 'e', label: '$\\|\\mathbf e\\|$ [cm]' }, { key: 'rms', label: 'RMS [cm]', title: '||e|| / sqrt(M): root mean square distance of a tracked point to its image on the target' },
    { key: 'entc', label: '$\\|\\mathbf e_{\\mathrm{NTC}}\\|$ [cm]' }, { key: 'mret', label: '$M_{\\mathrm{ret}}(k)$ / $M$' }, { key: 'ms', label: 'ms / iteration', title: 'mapping (ours and NTC) and ARAP in this browser' },
]);
nowCard.body.append(values.el);

const errCard = card('Shape error', 'as in Fig. 8 (one axis for both)');
const errCanvas = h('canvas', { class: 'plot-canvas' });
const errPlot = new LinePlot(errCanvas, { xLabel: 'k', yLabel: '[cm]' });
const errLegend = h('div');
errCard.body.append(errCanvas, errLegend);

const actCard = card('Actions', 'translation and rotation per iteration, as in Figs. 10-14');
const tCanvas = h('canvas', { class: 'plot-canvas short' }), rCanvas = h('canvas', { class: 'plot-canvas short' });
const tPlot = new LinePlot(tCanvas, { xLabel: 'k', yLabel: 'Δt [cm]' }), rPlot = new LinePlot(rCanvas, { xLabel: 'k', yLabel: 'Δr [°]' });
const actLegend = h('div');
actCard.body.append(tCanvas, rCanvas, actLegend);

const distCard = card('Distribution of the point errors $\\|\\mathbf e_m\\|$', 'as in Fig. 15');
const dOursCanvas = h('canvas', { class: 'plot-canvas short' }), dNtcCanvas = h('canvas', { class: 'plot-canvas short' });
const dOurs = new DistributionMap(dOursCanvas), dNtc = new DistributionMap(dNtcCanvas);
distCard.body.append(h('p', { class: 'chart-title' }, 'time-consistent'), dOursCanvas, h('p', { class: 'chart-title', style: 'margin-top:8px' }, 'ZoomOut every frame'), dNtcCanvas);

const fitCard = card('Model', 'how well (22)-(23) predicted each response');
const fitCanvas = h('canvas', { class: 'plot-canvas short' });
const fitPlot = new LinePlot(fitCanvas, { xLabel: 'k', yLabel: 'relative error', y: [0, 1.5] });
fitCard.body.append(fitCanvas, h('p', { class: 'hint' }, label('$\\|\\Delta\\mathbf x-\\hat{\\mathbf L}_g\\mathbf u_g\\|/\\|\\Delta\\mathbf x\\|$ after each controlled action (0: perfect prediction, 1: no better than predicting no motion). Near the target the actions are small and the responses mostly sampling noise.')));

const ctrlCard = card('Controller', 'applies at once');
const alphaSlider = slider({ label: 'gain $\\alpha$ (24)', min: 0.05, max: 1, step: 0.05, value: settings.alpha, onInput: (v) => { settings.alpha = v; session?.controller.set({ alpha: v }); } });
const gammaSlider = slider({ label: 'Tikhonov $\\gamma$ (23) [cm²]', min: 1e-6, max: 10, log: true, value: settings.gamma, format: (v) => (v < 0.01 ? v.toExponential(0) : v.toPrecision(2)),
    onInput: (v) => { settings.gamma = v; session?.controller.set({ gamma: v }); } });
const bSlider = slider({ label: 'buffer size $B$ (restarts)', min: 6, max: 30, step: 1, value: settings.B, format: (v) => v.toFixed(0), onChange: (v) => { settings.B = Math.round(v); restart(); } });
const mtSlider = slider({ label: 'largest translation per action [cm]', min: 0.1, max: 1.5, step: 0.05, value: settings.maxTranslation, onInput: (v) => { settings.maxTranslation = v; session?.controller.set({ maxTranslation: v }); } });
const mrSlider = slider({ label: 'largest rotation per action [°]', min: 1, max: 10, step: 0.5, value: settings.maxRotation * DEG, format: (v) => v.toFixed(1),
    onInput: (v) => { settings.maxRotation = v / DEG; session?.controller.set({ maxRotation: v / DEG }); } });
ctrlCard.body.append(alphaSlider.el, gammaSlider.el, bSlider.el, mtSlider.el, mrSlider.el);

const senseCard = card('Perception and mapping', 'restart to apply');
const ptsSlider = slider({ label: 'points per frame (on the rest shape)', min: 150, max: 600, step: 10, value: settings.points, format: (v) => v.toFixed(0), onChange: (v) => { settings.points = v; restart(); } });
const noiseSlider = slider({ label: 'noise $\\sigma$ [cm]', min: 0, max: 0.3, step: 0.01, value: settings.noise, onChange: (v) => { settings.noise = v; restart(); } });
const ntcToggle = toggle({ label: 'compute ZoomOut every frame (for $\\|\\mathbf e_{\\mathrm{NTC}}\\|$)', value: settings.ntc, onChange: (v) => { settings.ntc = v; if (session) session.settings.ntc = v; } });
senseCard.body.append(ptsSlider.el, noiseSlider.el, ntcToggle.el);

workspace.append(runCard.el, h('div', { class: 'dash' },
    h('div', { class: 'dash-view' }, viewCard.el),
    h('div', { class: 'dash-side' }, nowCard.el, errCard.el, actCard.el, distCard.el, fitCard.el, ctrlCard.el, senseCard.el)));

// ---------------------------------------------------------------- simulation
function load() {
    setRunning(false);
    busy.hidden = false;
    document.getElementById('note').textContent = `${SCENARIOS[settings.scenario].title}: ${SCENARIOS[settings.scenario].note}.`;
    setTimeout(() => {
        world = cache[settings.scenario] || (cache[settings.scenario] = new World(settings.scenario));
        restart();
        const skip = Math.min(Number(params.get('iterations')) || 0, K_MAX);
        for (let i = 0; i < skip; i++) iterate();
        frameView();
        render(true);
        busy.hidden = true;
    }, 30);
}
const frameView = () => view.frame({ ...world.scenario.view, zoom: 0.9 }, [world.lattice.rest, world.target.P]);

function restart() {
    session = new ControlSession(world, {
        points: settings.points, noise: settings.noise, ntc: settings.ntc,
        mapper: { step: 4 },
        control: { alpha: settings.alpha, B: settings.B, gamma: settings.gamma, maxTranslation: settings.maxTranslation, maxRotation: settings.maxRotation },
    });
    refN = referenceColors(session.mapper.target.V);
    render(true);
}

function iterate() {
    if (session.k >= K_MAX) return;
    const t0 = performance.now();
    session.step();
    session.lastMs = performance.now() - t0;
}

function setRunning(flag) {
    running = flag && session && session.k < K_MAX;
    runButton.replaceChildren(label(running ? '❚❚ Pause' : '▶ Run'));
    if (running) { lastWall = performance.now(); budget = 0; requestAnimationFrame(loop); }
}
function loop(now) {
    if (!running) return;
    budget += Math.min((now - lastWall) / 1000, 0.1) * 10 * speed;    // 1x: ten iterations per second
    lastWall = now;
    const start = performance.now();
    while (budget >= 1 && performance.now() - start < 45) { iterate(); budget -= 1; }
    budget = Math.min(budget, 2);
    render(++frames % 4 === 0);
    if (session.k >= K_MAX) setRunning(false);
    else requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- views
function render(full = false) {
    if (!session) return;
    const p = palette(), out = session.out, h_ = session.history, tgt = session.mapper.target;
    // tracked mesh
    const top = Math.max(2, h_.per[0] ? Math.max(...h_.per[0]) : 2);
    const colors = settings.colors === 'map' ? pullColors(refN, out.Pi) : valueColors(out.error.per, 'viridis', top);
    view.mesh('tracked', { V: out.X, F: out.F, colors, wire: true, order: 1 });
    if (settings.body) view.mesh('body', { V: world.P, F: world.lattice.body, color: p.object, offset: 2 }); else view.hide('body');
    if (settings.target) view.mesh('target', { V: tgt.V, F: tgt.F, color: p.target, opacity: 0.28, order: 3 }); else view.hide('target');
    if (settings.vectors) {
        const segs = new Float32Array(6 * out.Pi.length);
        for (let m = 0; m < out.Pi.length; m++) for (let e = 0; e < 3; e++) { segs[6 * m + e] = out.X[3 * m + e]; segs[6 * m + 3 + e] = tgt.V[3 * out.Pi[m] + e]; }
        view.lines('vectors', { segs, color: p.muted, opacity: 0.7 });
    } else view.hide('vectors');
    const pred = settings.prediction ? session.prediction() : null;
    if (pred) view.points('prediction', { P: pred, color: p.model, radius: 0.13 }); else view.hide('prediction');
    view.grippers(world.poses, world.grippers.map((g) => g.half), world.grippers.map((_, k) => gripperColor(p, k)));
    const pend = session.pending;
    view.setLabels(world.poses.map((T, k) => ({ p: T.t, text: `g${k + 1}${pend.g === k ? (pend.explore ? ' · explores next' : ' · acts next') : ''}`, color: gripperColor(p, k), dx: 12, dy: -24 })));
    // readouts
    const k = session.k, M = session.M;
    clockEl.textContent = `k = ${k}`;
    values.set('k', String(k));
    values.set('phase', pend.explore ? `exploring ${session.controller.k}/${session.controller.plan.length}` : 'control (24)');
    values.set('g', `g${pend.g + 1}`);
    values.set('e', fmt(out.error.norm, 1)); values.set('rms', fmt(out.error.norm / Math.sqrt(M), 2));
    values.set('entc', settings.ntc && session.ntc ? fmt(session.ntc.error.norm, 1) : '–');
    values.set('mret', `${out.ret.n} / ${M}`);
    values.set('ms', session.lastMs ? session.lastMs.toFixed(0) : '–');
    if (!full && running) return;
    errPlot.set({ series: [
        { x: h_.k, y: h_.eNtc, color: p.ntc, width: 1.2 },
        { x: h_.k, y: h_.e, color: p.ours, width: 2.2 },
    ], spans: explorationSpan(p) });
    const ks = h_.u.map((_, i) => i + 1), axis = [p.axisX, p.axisY, p.axisZ];
    // one series per gripper and axis, over the iterations at which that gripper acted
    const acted = world.grippers.map((_, g) => ks.filter((_, i) => h_.g[i] === g));
    const series = (offset, scale) => [0, 1, 2].flatMap((a) => world.grippers.map((_, g) => ({
        x: acted[g], y: h_.u.filter((_, i) => h_.g[i] === g).map((u) => u[offset + a] * scale), color: axis[a], width: 1.3, dash: g ? [4, 3] : [],
    })));
    tPlot.set({ series: series(0, 1), spans: explorationSpan(p) });
    rPlot.set({ series: series(3, DEG), spans: explorationSpan(p) });
    const yMax = Math.max(2, Math.max(...h_.per[0]) * 1.05);
    dOurs.set({ columns: h_.per, yMax, color: p.ours, kMax: Math.max(100, h_.k.length) });
    dNtc.set({ columns: h_.perNtc, yMax, color: p.ntc, kMax: Math.max(100, h_.k.length) });
    fitPlot.set({ series: [{ x: ks, y: h_.fit, color: p.model, width: 1.2 }], hlines: [{ y: 1, color: p.muted }] });
}

function explorationSpan(p) {
    const n = session.controller.plan.length;
    return [{ x0: 0, x1: Math.min(n, session.k), color: alpha(p.muted, 0.12) }];
}

function legends() {
    const p = palette();
    errLegend.replaceChildren(seriesLegend([{ label: 'time-consistent $\\|\\mathbf e\\|$ (fed to the controller)', color: p.ours }, { label: 'ZoomOut every frame $\\|\\mathbf e_{\\mathrm{NTC}}\\|$', color: p.ntc }, { label: 'exploration', color: p.muted, dash: 'band' }]));
    actLegend.replaceChildren(seriesLegend([{ label: 'x', color: p.axisX }, { label: 'y', color: p.axisY }, { label: 'z', color: p.axisZ }, { label: 'g1', color: p.muted }, { label: 'g2', color: p.muted, dash: 'dashed' }]));
    legendEl.replaceChildren(h('div', { class: 'legend' },
        ...[['target', p.target, 'band'], ['model prediction', p.model, 'dot'], ['g1', p.g1, 'line'], ['g2', p.g2, 'line']].map(([t, c, kind]) => {
            const item = h('span', { class: 'legend-item' }, h('i', { class: `swatch ${kind === 'line' ? '' : kind}` }), t);
            item.firstChild.style.color = c;
            return item;
        })));
}
onThemeChange(() => { view.refreshTheme(); legends(); render(true); });
legends();
load();
