// Overview page: previews of the four interactive pages, computed with the simulation
// modules after the first paint (a functional map, a short sequence with the errors of
// both maps, a slanted functional map, and the shape control problem seen from above).
import { signedColor } from '../lib/charts.js';
import { pullColors, referenceColors } from '../lib/present.js';
import { initPage } from '../lib/shell.js';
import { alpha, onThemeChange, palette } from '../lib/theme.js';
import { Camera } from '../sim/camera.js';
import { makeLattice, rectangle } from '../sim/lattice.js';
import { TimeConsistentMapper, initialMap, meshBasis, ntcStep } from '../sim/tcmap.js';
import { World } from '../sim/world.js';

await initPage();

const font = () => getComputedStyle(document.body).fontFamily;
const css = ([r, g, b]) => `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
function preview(id, draw) {
    const canvas = document.getElementById(id);
    const paint = () => {
        const dpr = Math.min(window.devicePixelRatio || 1, 2), w = canvas.clientWidth, h = canvas.clientHeight;
        if (!w || !h) return;
        canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const p = palette();
        ctx.fillStyle = p.canvas; ctx.fillRect(0, 0, w, h);
        ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        draw(ctx, w, h, p);
    };
    new ResizeObserver(paint).observe(canvas);
    onThemeChange(paint);
    return paint;
}
const caption = (ctx, text, p) => { ctx.fillStyle = p.muted; ctx.font = `600 10px ${font()}`; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillText(text, 12, 16); };
const waiting = (ctx, w, h, p) => { ctx.fillStyle = p.muted; ctx.font = `11px ${font()}`; ctx.textAlign = 'center'; ctx.fillText('computing…', w / 2, h / 2 + 6); };

// a matrix drawn in a square at (x, y) of side s
function matrix(ctx, C, I, J, x, y, s, size = Math.max(I, J)) {
    let m = 0;
    for (const v of C) m = Math.max(m, Math.abs(v));
    const c = s / size;
    ctx.fillStyle = css(signedColor(0)); ctx.fillRect(x, y, s, s);
    for (let i = 0; i < I; i++) for (let j = 0; j < J; j++) { ctx.fillStyle = css(signedColor(C[i * J + j] / (m || 1))); ctx.fillRect(x + j * c, y + i * c, c + 0.4, c + 0.4); }
}

// a mesh seen from above, colored per point (triangles filled with the mean color)
function topView(ctx, V, F, colors, box, color = null) {
    for (let t = 0; t < F.length; t += 3) {
        const a = F[t], b = F[t + 1], c = F[t + 2];
        ctx.fillStyle = color || css([0, 1, 2].map((e) => (colors[3 * a + e] + colors[3 * b + e] + colors[3 * c + e]) / 3));
        ctx.strokeStyle = ctx.fillStyle; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(...box(V, a)); ctx.lineTo(...box(V, b)); ctx.lineTo(...box(V, c)); ctx.closePath(); ctx.fill(); ctx.stroke();
    }
}
function fitBox(sets, w, h, top = 26, pad = 14) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const V of sets) for (let i = 0; i < V.length; i += 3) { x0 = Math.min(x0, V[i]); x1 = Math.max(x1, V[i]); y0 = Math.min(y0, V[i + 1]); y1 = Math.max(y1, V[i + 1]); }
    const s = Math.min((w - 2 * pad) / (x1 - x0), (h - top - pad) / (y1 - y0)), ox = (w - s * (x0 + x1)) / 2, oy = top + (h - top - pad) / 2 + s * (y0 + y1) / 2;
    return (V, i) => [ox + s * V[3 * i], oy - s * V[3 * i + 1]];
}

const state = {};

// 0: a functional map and the colors of a surface map (sheet, flat to folded)
const paint0 = preview('preview0', (ctx, w, h, p) => {
    caption(ctx, 'C(I) and the surface map, flat sheet → folded', p);
    const s0 = state.fmap;
    if (!s0) return waiting(ctx, w, h, p);
    const side = Math.min(h - 40, w * 0.36);
    matrix(ctx, s0.C, s0.I, s0.J, 14, 28, side);
    const box = fitBox([s0.src.V], w - side - 30, h);
    ctx.save(); ctx.translate(side + 24, 0);
    topView(ctx, s0.src.V, s0.src.F, s0.colors, box);
    ctx.restore();
});

// 1: errors of both maps along a short deformation sequence
const paint1 = preview('preview1', (ctx, w, h, p) => {
    caption(ctx, '‖e‖ (time-consistent) and ‖e_NTC‖ (ZoomOut every frame)', p);
    const s1 = state.seq;
    if (!s1 || s1.e.length < 2) return waiting(ctx, w, h, p);
    const L = 14, R = w - 14, T = 28, B = h - 14, n = 60, top = Math.max(...s1.e, ...s1.n) * 1.05;
    ctx.strokeStyle = p.grid; ctx.lineWidth = 1;
    for (let k = 0; k <= 3; k++) { const y = T + k * (B - T) / 3; ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(R, y); ctx.stroke(); }
    for (const [ys, color, width] of [[s1.n, p.ntc, 1.3], [s1.e, p.ours, 2.4]]) {
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
        ys.forEach((y, k) => { const X = L + k / (n - 1) * (R - L), Y = B - y / top * (B - T); if (k) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); });
        ctx.stroke();
    }
});

// 2: the slanted functional map of a compressed, bent beam
const paint2 = preview('preview2', (ctx, w, h, p) => {
    caption(ctx, 'ZoomOut (J = I) and the slanted map of (19)', p);
    const s2 = state.slant;
    if (!s2) return waiting(ctx, w, h, p);
    const side = Math.min(h - 40, (w - 50) / 2), size = Math.max(s2.zo.I, s2.zo.J, s2.ours.J);
    matrix(ctx, s2.zo.C, s2.zo.I, s2.zo.J, w / 2 - side - 8, 28, side, size);
    matrix(ctx, s2.ours.C, s2.ours.I, s2.ours.J, w / 2 + 8, 28, side, size);
});

// 3: the T-shaped piece from above: rest shape, target, and grippers
const paint3 = preview('preview3', (ctx, w, h, p) => {
    caption(ctx, 'T-shaped piece: current (gray), target (orange), grippers', p);
    const s3 = state.tee;
    if (!s3) return waiting(ctx, w, h, p);
    const box = fitBox([s3.rest, s3.target], w, h);
    topView(ctx, s3.rest, s3.F, null, box, alpha(p.object, 0.9));
    topView(ctx, s3.target, s3.F, null, box, alpha(p.target, 0.55));
    for (const [k, c] of s3.grippers.entries()) {
        const [x, y] = box([c[0], c[1], 0], 0);
        ctx.strokeStyle = k ? p.g2 : p.g1; ctx.lineWidth = 2; ctx.strokeRect(x - 9, y - 9, 18, 18);
    }
});

// computations after the first paint
setTimeout(() => {
    const world = new World('sheet');
    const tgt = meshBasis(new Camera(world.lattice, { points: 220, seed: 4 }).capture(world.target.P), 40, 2);
    const src = meshBasis(new Camera(world.lattice, { points: 200, seed: 5, offset: [0.6, 0.3] }).capture(world.lattice.rest), 20, 3);
    const r = initialMap(src, tgt, { Iinit: 4, Imax: 20, step: 2, slanted: false });
    state.fmap = { C: r.C, I: r.I, J: r.J, src, colors: pullColors(referenceColors(tgt.V), r.Pi) };
    paint0();

    const tee = new World('tee', { goal: true }), lat = tee.lattice, topOf = (P) => Float64Array.from({ length: 3 * lat.M }, (_, i) => P[3 * lat.top[Math.floor(i / 3)] + (i % 3)]);
    state.tee = { rest: topOf(lat.rest), target: topOf(tee.target.P), F: lat.topTris, grippers: tee.grippers.map((g) => g.restPose.t) };
    paint3();

    const beam = (len, bend, nu, nv) => {
        const b = makeLattice({ nu, nv, nz: 1, h: 0, chart: rectangle(len, 6) }), V = b.rest.slice(), R = len / bend;
        for (let i = 0; i < b.n; i++) { const x = b.rest[3 * i]; V[3 * i] = R * Math.sin(x / R); V[3 * i + 2] = R * (1 - Math.cos(x / R)); }
        return { V, F: b.topTris, n: b.n };
    };
    const flat = makeLattice({ nu: 33, nv: 9, nz: 1, h: 0, chart: rectangle(24, 6) });
    const S = meshBasis({ V: flat.rest, F: flat.topTris, n: flat.n }, 30, 3), T = meshBasis(beam(24 * 0.7, 1, 25, 10), 60, 5);
    state.slant = { zo: initialMap(S, T, { Iinit: 4, Imax: 30, step: 1, slanted: false }), ours: initialMap(S, T, { Iinit: 4, Imax: 30, step: 1, slanted: true }) };
    paint2();

    // a short fold-and-return sequence, in slices
    const seqWorld = new World('sheet'), cam = new Camera(seqWorld.lattice, { points: 220, seed: 5 });
    const mapper = new TimeConsistentMapper(cam.capture(seqWorld.target.P), { step: 4, Imax: 24 });
    state.seq = { e: [], n: [] };
    let k = 0;
    const slice = () => {
        const a = Math.sin(Math.PI * k / 60) ** 2;
        seqWorld.moveTowards(seqWorld.scenario.motion.map((m) => ({ rot: m.rot.map((x) => a * x), move: m.move.map((x) => a * x) })), 1);
        const out = mapper.step(cam.capture(seqWorld.P));
        state.seq.e.push(out.error.norm); state.seq.n.push(ntcStep(out.ret, mapper.target, { Imax: 24, step: 4 }).error.norm);
        if (++k % 6 === 0) paint1();
        if (k < 60) setTimeout(slice, 0); else paint1();
    };
    setTimeout(slice, 0);
}, 80);
