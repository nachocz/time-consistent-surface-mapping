// Page 1 (Sec. 2): functional maps and ZoomOut between two perceived meshes of the same
// simulated object, flat (M) and deformed (N): the Laplace-Beltrami eigenfunctions, the
// functional map C(I) at every size of the refinement, the point map it gives, and the
// image of the selector of a point.
import { MatrixMap } from '../lib/charts.js';
import { LinePlot } from '../lib/plot.js';
import { column, pullColors, referenceColors, valueColors } from '../lib/present.js';
import { initPage } from '../lib/shell.js';
import { onThemeChange, palette } from '../lib/theme.js';
import { button, card, control, fmt, h, label, readouts, segmented, select, seriesLegend, slider } from '../lib/ui.js';
import { MeshView } from '../lib/view3d.js';
import { Camera } from '../sim/camera.js';
import { orthonormality } from '../sim/fmap.js';
import { classicZoomOut, initialMap, meshBasis } from '../sim/tcmap.js';
import { matchError } from '../sim/truth.js';
import { SCENARIOS, World } from '../sim/world.js';

await initPage();

const IMAX = 30;
const params = new URLSearchParams(location.search);
const settings = {
    scenario: SCENARIOS[params.get('object')] ? params.get('object') : 'sheet',
    init: params.get('init') === 'icp' ? 'icp' : 'identity',
    mode: ['map', 'eig', 'delta'].includes(params.get('mode')) ? params.get('mode') : 'map',
    I: IMAX, i: 2, picked: params.has('point') ? Number(params.get('point')) : null, seed: 1,
};
const cache = {};
let world = null, src = null, tgt = null, trace = null, ref = null, frames = null;

// ---------------------------------------------------------------- layout
const workspace = document.getElementById('workspace');
const topCard = card('', '');
const scenarioPick = select({
    options: Object.entries(SCENARIOS).map(([value, s]) => ({ value, label: s.title })), value: settings.scenario,
    onChange: (v) => { settings.scenario = v; settings.picked = null; load(); },
});
scenarioPick.el.style.width = 'auto';
const initPick = segmented({
    options: [{ value: 'identity', label: 'ZoomOut: $\\mathbf C(I_{\\mathrm{initial}})=\\mathbf I$' }, { value: 'icp', label: 'ICP (Sec. 3.1)' }],
    value: settings.init, onChange: (v) => { settings.init = v; compute(); },
});
const remesh = button('↻ Re-mesh $\\mathcal M$', () => { settings.seed += 1; settings.picked = null; perceive(); compute(); });
topCard.body.append(h('div', { class: 'toolbar' }, h('b', { style: 'font-size:0.8rem' }, 'Object'), scenarioPick.el, remesh,
    h('span', { class: 'toolbar-divider' }), h('b', { style: 'font-size:0.8rem' }, 'Initialization'), initPick.el),
h('p', { class: 'hint', style: 'margin:8px 0 0' }, h('span', { id: 'note' })));

const viewCard = card('Current shape $\\mathcal M$ and target shape $\\mathcal N$', 'drag to orbit (the views move together), scroll to zoom');
const pair = h('div', { class: 'view-pair' });
const leftEl = h('div', { class: 'view3d pair-fit' }), rightEl = h('div', { class: 'view3d pair-fit' });
const busy = h('div', { class: 'busy' }, 'Preparing the shapes…');
leftEl.append(h('span', { class: 'view-tag' }, label('$\\mathcal M$ (current)')), busy);
rightEl.append(h('span', { class: 'view-tag' }, label('$\\mathcal N$ (target)')));
pair.append(leftEl, rightEl);
const left = new MeshView(leftEl), right = new MeshView(rightEl);
left.link(right);
const modePick = segmented({
    options: [{ value: 'map', label: 'surface map $\\Pi$' }, { value: 'eig', label: 'eigenfunction $\\phi_i$' }, { value: 'delta', label: 'point → function' }],
    value: settings.mode, onChange: (v) => { settings.mode = v; render(); },
});
const ISlider = slider({ label: 'basis size $I$ (ZoomOut step)', min: 4, max: IMAX, step: 1, value: settings.I, format: (v) => v.toFixed(0),
    onInput: (v) => { settings.I = Math.round(v); render(); } });
const iSlider = slider({ label: 'eigenfunction $i$', min: 1, max: IMAX, step: 1, value: settings.i, format: (v) => v.toFixed(0),
    onInput: (v) => { settings.i = Math.round(v); render(); } });
const viewHint = h('p', { class: 'hint', style: 'margin:6px 0 0' });
viewCard.body.append(pair, h('div', { class: 'view-controls' }, control('color', modePick.el), ISlider.el, iSlider.el), viewHint);

const nowCard = card('Now', '');
const values = readouts([
    { key: 'M', label: 'points of $\\mathcal M$' }, { key: 'N', label: 'points of $\\mathcal N$' }, { key: 'I', label: 'size $I$' },
    { key: 'orth', label: '$\\|\\mathbf C^\\intercal\\mathbf C-\\mathbf I\\|_F$', title: 'orthonormality residual of C(I), the term of (6)' },
    { key: 'err', label: 'error to the true match [cm]', title: 'mean distance on the rest shape between the mapped point and the same material point (ground truth of the simulator)' },
    { key: 'cii', label: '$c_{i,i}$', title: 'diagonal entry of C(I) for the selected eigenfunction' },
]);
nowCard.body.append(values.el);

const cCard = card('Functional map $\\mathbf C(I)$', 'rows: $\\Phi^{\\mathcal M}$, columns: $\\Phi^{\\mathcal N}$');
const cCanvas = h('canvas', { class: 'chart-canvas square' });
const cMap = new MatrixMap(cCanvas);
cCard.body.append(h('div', { class: 'matrix-small' }, cCanvas), h('div', { class: 'colorbar signed' }, h('span', {}, '−1'), h('i'), h('span', {}, '+1')),
    h('p', { class: 'hint' }, 'Entries scaled by the largest magnitude. The near-diagonal structure says that eigenfunction i of M corresponds to eigenfunction i of N; a negative diagonal entry, that their signs differ.'));

const oCard = card('Refinement', 'every size $I$ of the ZoomOut trace');
const oCanvas = h('canvas', { class: 'plot-canvas short' });
const oPlot = new LinePlot(oCanvas, { xLabel: 'I', yIncludeZero: true });
const oLegend = h('div');
oCard.body.append(oCanvas, oLegend);

const eCard = card('Eigenvalues', '$\\lambda_i$ of both shapes');
const eCanvas = h('canvas', { class: 'plot-canvas short' });
const ePlot = new LinePlot(eCanvas, { xLabel: 'i', yIncludeZero: true });
const eLegend = h('div');
eCard.body.append(eCanvas, eLegend);

workspace.append(topCard.el, h('div', { class: 'dash' },
    h('div', { class: 'dash-view' }, viewCard.el),
    h('div', { class: 'dash-side' }, nowCard.el, cCard.el, oCard.el, eCard.el)));

// ---------------------------------------------------------------- computation
function load() {
    busy.hidden = false;
    document.getElementById('note').textContent = `${SCENARIOS[settings.scenario].title}: the flat piece (M) and its deformed shape (N), ${SCENARIOS[settings.scenario].note.replace(/ \(Fig.*\)$/, '')}.`;
    setTimeout(() => {
        world = cache[settings.scenario] || (cache[settings.scenario] = new World(settings.scenario));
        const camN = new Camera(world.lattice, { points: 320, noise: 0.02, seed: 4, offset: [0.21, 0.73] });
        const fN = camN.capture(world.target.P);
        tgt = meshBasis(fN, 2 * IMAX, 11);
        ref = referenceColors(tgt.V);
        perceive();
        compute();
        left.frame({ ...world.scenario.view, zoom: 0.82 }, [world.lattice.rest, world.target.P]);
        busy.hidden = true;
    }, 30);
}

function perceive() {
    const r = settings.seed;
    const cam = new Camera(world.lattice, { points: 300, noise: 0.02, seed: 10 + r, offset: [(0.37 * r) % 1, (0.61 * r + 0.2) % 1] });
    src = meshBasis(cam.capture(world.lattice.rest), IMAX, 100 + r);
}

function compute() {
    trace = [];
    if (settings.init === 'identity') classicZoomOut(src, tgt, { Iinit: 4, Imax: IMAX, step: 1, trace });
    else initialMap(src, tgt, { Iinit: 4, Imax: IMAX, step: 1, slanted: false, init: 'icp', trace });
    frames = trace.map((t) => ({ ...t, orth: orthonormality(t.C, t.I, t.J), err: matchError(src, tgt, t.Pi, world.scenario.shape.chart) }));
    const p = palette();
    oPlot.set({ series: [
        { x: frames.map((f) => f.I), y: frames.map((f) => f.err), color: p.ours, width: 2 },
        { x: frames.map((f) => f.I), y: frames.map((f) => f.orth / Math.sqrt(f.I)), color: p.accent, width: 1.6, dash: [5, 4] },
    ] });
    ePlot.set({ series: [
        { x: Array.from(src.evals, (_, i) => i + 1), y: Array.from(src.evals), color: p.ours, width: 1.8 },
        { x: Array.from(tgt.evals.slice(0, IMAX), (_, i) => i + 1), y: Array.from(tgt.evals.slice(0, IMAX)), color: p.target, width: 1.8, dash: [5, 4] },
    ] });
    render();
}

// ---------------------------------------------------------------- views
function render() {
    if (!frames) return;
    const p = palette(), f = frames.find((x) => x.I === settings.I) || frames[frames.length - 1], { C, I, J, Pi } = f;
    iSlider.el.hidden = settings.mode !== 'eig';
    ISlider.el.querySelector('label').style.opacity = settings.mode === 'eig' ? 0.6 : 1;
    let cl = null, cr = null, labels = [];
    left.hide('marker'); right.hide('marker');
    if (settings.mode === 'map') {
        cl = pullColors(ref, Pi); cr = ref;
        viewHint.textContent = 'Each point of M takes the color of its image on N. Smooth colors on M mean a continuous map.';
    } else if (settings.mode === 'eig') {
        const i = Math.min(settings.i, src.K) - 1;
        cl = valueColors(column(src.evecs, src.n, src.K, i)); cr = valueColors(column(tgt.evecs, tgt.n, tgt.K, i));
        viewHint.textContent = `φ_${i + 1} of each shape: λ = ${src.evals[i].toFixed(3)} on M, ${tgt.evals[i].toFixed(3)} on N.`;
    } else {
        const s = settings.picked;
        if (s === null || s >= src.n) {
            cl = new Float32Array(3 * src.n).fill(0.8); cr = new Float32Array(3 * tgt.n).fill(0.8);
            viewHint.textContent = 'Click a point of M.';
        } else {
            // the selector f_s in I eigenfunctions: coefficients a = (Phi^M_I)^T A f_s = a_s Phi^M_I[s, :]
            const a = new Float64Array(I);
            for (let k = 0; k < I; k++) a[k] = src.area[s] * src.evecs[s * src.K + k];
            const fs = new Float64Array(src.n);
            for (let m = 0; m < src.n; m++) { let v = 0; for (let k = 0; k < I; k++) v += src.evecs[m * src.K + k] * a[k]; fs[m] = v; }
            // its image: coefficients C^T a on N, g = Phi^N_J C^T a
            const b = new Float64Array(J);
            for (let j = 0; j < J; j++) { let v = 0; for (let k = 0; k < I; k++) v += C[k * J + j] * a[k]; b[j] = v; }
            const g = new Float64Array(tgt.n);
            for (let t = 0; t < tgt.n; t++) { let v = 0; for (let j = 0; j < J; j++) v += tgt.evecs[t * tgt.K + j] * b[j]; g[t] = v; }
            const pos = (x) => Math.max(0, x);
            cl = valueColors(fs.map(pos), 'heat'); cr = valueColors(g.map(pos), 'heat');
            left.points('marker', { P: src.V.slice(3 * s, 3 * s + 3), color: p.ink, radius: 0.45 });
            right.points('marker', { P: tgt.V.slice(3 * Pi[s], 3 * Pi[s] + 3), color: p.ink, radius: 0.45 });
            labels = [{ p: Array.from(tgt.V.slice(3 * Pi[s], 3 * Pi[s] + 3)), text: 'Π(s)', dx: 8, dy: -18 }];
            viewHint.textContent = 'Left: the selector of the clicked point s written with I eigenfunctions. Right: its image g_s = T_F(f_s) through C(I); the dot is the point that (8) assigns to s.';
        }
    }
    left.mesh('M', { V: src.V, F: src.F, colors: cl, wire: true, pickable: true });
    right.mesh('N', { V: tgt.V, F: tgt.F, colors: cr, wire: true });
    right.setLabels(labels);
    // readouts and charts
    values.set('M', String(src.n)); values.set('N', String(tgt.n)); values.set('I', `${I}`);
    values.set('orth', fmt(f.orth, 2)); values.set('err', fmt(f.err, 2), f.err < 2 ? 'good' : f.err > 5 ? 'bad' : '');
    const i = Math.min(settings.i, I) - 1;
    values.set('cii', settings.mode === 'eig' && settings.i <= I ? fmt(C[i * J + i], 2) : '–', settings.mode === 'eig' && C[i * J + i] < 0 ? 'bad' : '');
    cMap.set({ C, I, J, size: IMAX });
    oPlot.set({ vlines: [{ x: I, color: p.muted }] });
    ePlot.set({ vlines: settings.mode === 'eig' ? [{ x: settings.i, color: p.muted }] : [] });
}

left.onPick((name, vertex) => {
    if (name !== 'M') return;
    settings.picked = vertex;
    if (settings.mode !== 'delta') { settings.mode = 'delta'; modePick.set('delta'); }
    render();
});

function legends() {
    const p = palette();
    oLegend.replaceChildren(seriesLegend([{ label: 'error to the true match [cm]', color: p.ours }, { label: '$\\|\\mathbf C^\\intercal\\mathbf C-\\mathbf I\\|_F/\\sqrt I$', color: p.accent, dash: 'dashed' }]));
    eLegend.replaceChildren(seriesLegend([{ label: '$\\lambda^{\\mathcal M}_i$', color: p.ours }, { label: '$\\lambda^{\\mathcal N}_i$', color: p.target, dash: 'dashed' }]));
}
onThemeChange(() => { left.refreshTheme(); right.refreshTheme(); legends(); compute(); });
legends();
load();
