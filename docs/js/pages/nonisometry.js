// Page 3 (Sec. 3.4): the rule (18)-(19) for the size J of the target basis, on a flat
// beam M mapped to a stretched and bent beam N, against the square maps of ZoomOut.
import { MatrixMap } from '../lib/charts.js';
import { LinePlot } from '../lib/plot.js';
import { column, pullColors, referenceColors, valueColors } from '../lib/present.js';
import { initPage } from '../lib/shell.js';
import { onThemeChange, palette } from '../lib/theme.js';
import { button, card, control, fmt, h, label, readouts, segmented, seriesLegend, slider } from '../lib/ui.js';
import { MeshView } from '../lib/view3d.js';
import { targetSize } from '../sim/fmap.js';
import { makeLattice, rectangle } from '../sim/lattice.js';
import { initialMap, meshBasis } from '../sim/tcmap.js';
import { RECTANGLE_SYMMETRIES, matchError } from '../sim/truth.js';
import { rng } from '../sim/linalg.js';

await initPage();

const params = new URLSearchParams(location.search);
const L = 24, W = 6, IMAX = 30, KN = 60;
const settings = { s: Number(params.get('s')) || 0.7, bend: Number(params.get('bend') ?? 1), mode: 'map', I: 12 };
let src = null, tgt = null, ours = null, zo = null, ref = null, chartM = null, sweepToken = 0;
const sweep = { s: [], ours: [], zo: [] };

// ---------------------------------------------------------------- shapes
// A beam of length len and width W as a single sheet of nu x nv points (a regular mesh
// with a little noise), bent into a circular arc of `bend` rad.
function beam(len, bend, nu, nv, seed) {
    const lat = makeLattice({ nu, nv, nz: 1, h: 0, chart: rectangle(len, W) });
    const V = lat.rest.slice(), r = rng(seed);
    if (bend > 1e-6) {
        const R = len / bend;
        for (let i = 0; i < lat.n; i++) { const x = lat.rest[3 * i]; V[3 * i] = R * Math.sin(x / R); V[3 * i + 2] = R * (1 - Math.cos(x / R)); }
    }
    for (let i = 0; i < V.length; i++) V[i] += 0.01 * (r() - 0.5);
    return { V, F: lat.topTris, uv: lat.uv, n: lat.n };
}
function shapes(s, bend) {
    return { src: meshBasis(beam(L, 0, 33, 9, 3), IMAX, 3), tgt: meshBasis(beam(L * s, bend, Math.round(36 * s), 10, 4), KN, 5) };
}
// error on the rest shape of the flat beam: (u, v) scaled by its length and width
const chart = rectangle(L, W);
const maps = (S, T) => ({
    ours: initialMap(S, T, { Iinit: 4, Imax: IMAX, step: 1, slanted: true }),
    zo: initialMap(S, T, { Iinit: 4, Imax: IMAX, step: 1, slanted: false }),
});

// ---------------------------------------------------------------- layout
const workspace = document.getElementById('workspace');
const topCard = card('', '');
const sSlider = slider({ label: 'stretch $s$ of the target (length $s\\cdot$24 cm)', min: 0.6, max: 2, step: 0.05, value: settings.s, onChange: (v) => { settings.s = v; compute(); drawSweep(); } });
const bSlider = slider({ label: 'bend of the target [rad]', min: 0, max: 3.2, step: 0.1, value: settings.bend, format: (v) => v.toFixed(1), onChange: (v) => { settings.bend = v; compute(); startSweep(); } });
topCard.body.append(h('div', { class: 'jog-grid' }, sSlider.el, bSlider.el));

const viewCard = card('The flat beam $\\mathcal M$ mapped to the target $\\mathcal N$', 'drag to orbit (the views move together)');
const trio = h('div', { class: 'view-stack' });
const els = [0, 1, 2].map(() => h('div', { class: 'view3d stack-fit' }));
const tags = [h('span', { class: 'view-tag ntc' }), h('span', { class: 'view-tag ours' }), h('span', { class: 'view-tag' })];
els.forEach((el, i) => { el.append(tags[i]); trio.append(el); });
const busy = h('div', { class: 'busy' }, 'Computing…');
els[0].append(busy);
const views = els.map((el) => new MeshView(el));
views[0].link(views[1]); views[0].link(views[2]); views[1].link(views[2]);
const modePick = segmented({ options: [{ value: 'map', label: 'surface maps' }, { value: 'eig', label: 'eigenfunctions (Fig. 5)' }], value: settings.mode,
    onChange: (v) => { settings.mode = v; render(); } });
const ISlider = slider({ label: 'eigenfunction $I$', min: 2, max: IMAX, step: 1, value: settings.I, format: (v) => v.toFixed(0), onInput: (v) => { settings.I = Math.round(v); render(); } });
const viewHint = h('p', { class: 'hint', style: 'margin:6px 0 0' });
viewCard.body.append(trio, h('div', { class: 'view-controls' }, control('show', modePick.el), ISlider.el, h('span')), viewHint);

const nowCard = card('Comparison', 'same initialization, different J');
const values = readouts([
    { key: 'rho', label: '$\\rho$ (18)' }, { key: 'J', label: '$J$ at $I_{\\max}$', title: 'number of target eigenvectors of the last refinement step' },
    { key: 'ours', label: 'with (19): error [cm]', title: 'mean distance to the true match on the rest shape, up to the mirror symmetries of the beam' },
    { key: 'zo', label: 'ZoomOut $J=I$: error [cm]' },
]);
nowCard.body.append(values.el);

const eigCard = card('Eigenvalues and the choice of $J$', 'Fig. 5');
const eigCanvas = h('canvas', { class: 'plot-canvas' });
const eigPlot = new LinePlot(eigCanvas, { xLabel: 'index', yLabel: 'λ' });
const eigLegend = h('div');
eigCard.body.append(eigCanvas, eigLegend);

const jCard = card('$J(I)$', 'target eigenvectors per source size');
const jCanvas = h('canvas', { class: 'plot-canvas short' });
const jPlot = new LinePlot(jCanvas, { xLabel: 'I', yLabel: 'J' });
const jLegend = h('div');
jCard.body.append(jCanvas, jLegend);

const cCard = card('Functional maps', 'rows: $\\Phi^{\\mathcal M}$, columns: $\\Phi^{\\mathcal N}$');
const cZoCanvas = h('canvas', { class: 'chart-canvas square' }), cOursCanvas = h('canvas', { class: 'chart-canvas square' });
const cZo = new MatrixMap(cZoCanvas), cOurs = new MatrixMap(cOursCanvas);
cCard.body.append(h('div', { class: 'pair-charts' },
    h('div', {}, h('p', { class: 'chart-title' }, label('ZoomOut: $I\\times I$')), cZoCanvas),
    h('div', {}, h('p', { class: 'chart-title' }, label('with (19): $I\\times J$')), cOursCanvas)),
h('p', { class: 'hint' }, 'The dashed staircase marks J for every row size i (the columns each step uses).'));

const sweepCard = card('Sweep over the stretch', 'error to the true match, at the current bend');
const sweepCanvas = h('canvas', { class: 'plot-canvas' });
const sweepPlot = new LinePlot(sweepCanvas, { xLabel: 'stretch s', yLabel: '[cm]' });
const sweepLegend = h('div');
const sweepNote = h('p', { class: 'hint' });
sweepCard.body.append(sweepCanvas, sweepLegend, sweepNote);

workspace.append(topCard.el, h('div', { class: 'dash' },
    h('div', { class: 'dash-view' }, viewCard.el),
    h('div', { class: 'dash-side' }, nowCard.el, eigCard.el, jCard.el, cCard.el, sweepCard.el)));

// ---------------------------------------------------------------- computation
function compute() {
    busy.hidden = false;
    setTimeout(() => {
        ({ src, tgt } = shapes(settings.s, settings.bend));
        ({ ours, zo } = maps(src, tgt));
        ours.err = matchError(src, tgt, ours.Pi, chart, RECTANGLE_SYMMETRIES); zo.err = matchError(src, tgt, zo.Pi, chart, RECTANGLE_SYMMETRIES);
        ref = referenceColors(tgt.V);
        chartM = null;
        // lay the flat beam beside the target for the views: same camera for the three views
        views[0].frame({ from: [0, -40, 26], at: [0, 0, 2], zoom: 0.5 }, [src.V, tgt.V]);
        render();
        busy.hidden = true;
    }, 20);
}

function render() {
    if (!src) return;
    const p = palette(), I = settings.I;
    ISlider.el.hidden = settings.mode !== 'eig';
    if (settings.mode === 'map') {
        tags[0].replaceChildren(label('ZoomOut, $J=I$')); tags[1].replaceChildren(label('with (19), $J>I$ or $J<I$')); tags[2].replaceChildren(label('target $\\mathcal N$'));
        tags[0].className = 'view-tag ntc'; tags[1].className = 'view-tag ours';
        views[0].mesh('m', { V: src.V, F: src.F, colors: pullColors(ref, zo.Pi), wire: true });
        views[1].mesh('m', { V: src.V, F: src.F, colors: pullColors(ref, ours.Pi), wire: true });
        views[2].mesh('m', { V: tgt.V, F: tgt.F, colors: ref, wire: true });
        viewHint.textContent = 'Each point of the flat beam takes the color of its image on the target. A correct map reproduces the color sweep of the target along the beam, without jumps.';
    } else {
        const rho = Math.sqrt(src.total / tgt.total), J = targetSize(src, tgt, I, rho);
        tags[0].replaceChildren(label(`$\\phi^{\\mathcal M}_{${I}}$`)); tags[1].replaceChildren(label(`$\\phi^{\\mathcal N}_{${I}}$ (ZoomOut)`)); tags[2].replaceChildren(label(`$\\phi^{\\mathcal N}_{${J}}$ (19)`));
        tags[0].className = 'view-tag'; tags[1].className = 'view-tag ntc'; tags[2].className = 'view-tag ours';
        views[0].mesh('m', { V: src.V, F: src.F, colors: valueColors(column(src.evecs, src.n, src.K, I - 1)), wire: true });
        views[1].mesh('m', { V: tgt.V, F: tgt.F, colors: valueColors(column(tgt.evecs, tgt.n, tgt.K, I - 1)), wire: true });
        views[2].mesh('m', { V: tgt.V, F: tgt.F, colors: valueColors(column(tgt.evecs, tgt.n, tgt.K, J - 1)), wire: true });
        viewHint.textContent = `λ^M_${I} = ${src.evals[I - 1].toFixed(3)}, ρ λ^M_${I} = ${(rho * src.evals[I - 1]).toFixed(3)}; λ^N_${I} = ${tgt.evals[I - 1].toFixed(3)}, λ^N_${J} = ${tgt.evals[J - 1].toFixed(3)}. Signs are arbitrary: compare the patterns.`;
    }
    const rho = Math.sqrt(src.total / tgt.total);
    values.set('rho', fmt(rho, 2)); values.set('J', String(ours.J));
    values.set('ours', fmt(ours.err, 2), ours.err < zo.err - 0.2 ? 'good' : ours.err > zo.err + 0.2 ? 'bad' : '');
    values.set('zo', fmt(zo.err, 2), zo.err < ours.err - 0.2 ? 'good' : zo.err > ours.err + 0.2 ? 'bad' : '');
    // eigenvalues: lambda^N_j (staircase), lambda^M_i and rho lambda^M_i, the selected I and its J
    const idx = (n) => Array.from({ length: n }, (_, i) => i + 1);
    const J = targetSize(src, tgt, I, rho);
    eigPlot.set({
        series: [
            { x: idx(KN), y: Array.from(tgt.evals), color: p.target, width: 1.8 },
            { x: idx(IMAX), y: Array.from(src.evals), color: p.ink, width: 1.4, dash: [4, 3] },
            { x: idx(IMAX), y: Array.from(src.evals, (v) => v * rho), color: p.ours, width: 1.8 },
        ],
        hlines: settings.mode === 'eig' ? [{ y: src.evals[I - 1] * rho, color: p.ours, label: `ρ λ^M_${I}` }] : [],
        vlines: settings.mode === 'eig' ? [{ x: I, color: p.muted }, { x: J, color: p.ours }] : [],
        markers: settings.mode === 'eig' ? [{ x: J, y: tgt.evals[J - 1], color: p.ours }, { x: I, y: tgt.evals[I - 1], color: p.ntc }] : [],
    });
    const guide = idx(IMAX).map((i) => targetSize(src, tgt, i, rho));
    jPlot.set({ series: [
        { x: idx(IMAX), y: idx(IMAX), color: p.ntc, width: 1.4, dash: [4, 3] },
        { x: idx(IMAX), y: guide, color: p.ours, width: 2 },
    ] });
    const size = Math.max(IMAX, ours.J);
    cZo.set({ C: zo.C, I: zo.I, J: zo.J, size });
    cOurs.set({ C: ours.C, I: ours.I, J: ours.J, size, guide });
}

// sweep over the stretch at the current bend, in slices so that the page stays responsive
function startSweep() {
    const token = ++sweepToken, list = [];
    for (let s = 0.6; s <= 2.0001; s += 0.1) list.push(Math.round(s * 100) / 100);
    sweep.s = []; sweep.ours = []; sweep.zo = [];
    sweepNote.textContent = 'computing…';
    let i = 0;
    const slice = () => {
        if (token !== sweepToken) return;
        const { src: S, tgt: T } = shapes(list[i], settings.bend), r = maps(S, T);
        sweep.s.push(list[i]); sweep.ours.push(matchError(S, T, r.ours.Pi, chart, RECTANGLE_SYMMETRIES)); sweep.zo.push(matchError(S, T, r.zo.Pi, chart, RECTANGLE_SYMMETRIES));
        drawSweep();
        if (++i < list.length) setTimeout(slice, 0);
        else sweepNote.textContent = `bend ${settings.bend.toFixed(1)} rad; error up to the mirror symmetries of the beam. Smaller is better.`;
    };
    setTimeout(slice, 300);
}
function drawSweep() {
    const p = palette();
    sweepPlot.set({ series: [{ x: sweep.s, y: sweep.zo, color: p.ntc, width: 1.6, dash: [4, 3] }, { x: sweep.s, y: sweep.ours, color: p.ours, width: 2 }],
        markers: [...sweep.s.map((s, i) => ({ x: s, y: sweep.ours[i], color: p.ours, r: 2.5 })), ...sweep.s.map((s, i) => ({ x: s, y: sweep.zo[i], color: p.ntc, r: 2.5 }))],
        vlines: [{ x: settings.s, color: p.muted }] });
}

function legends() {
    const p = palette();
    eigLegend.replaceChildren(seriesLegend([{ label: '$\\lambda^{\\mathcal N}_j$', color: p.target }, { label: '$\\lambda^{\\mathcal M}_i$', color: p.ink, dash: 'dashed' }, { label: '$\\rho\\,\\lambda^{\\mathcal M}_i$', color: p.ours }]));
    jLegend.replaceChildren(seriesLegend([{ label: 'ZoomOut $J=I$', color: p.ntc, dash: 'dashed' }, { label: '(19)', color: p.ours }]));
    sweepLegend.replaceChildren(seriesLegend([{ label: 'with (19)', color: p.ours }, { label: 'ZoomOut $J=I$', color: p.ntc, dash: 'dashed' }]));
}
onThemeChange(() => { views.forEach((v) => v.refreshTheme()); legends(); render(); drawSweep(); });
legends();
compute();
startSweep();
