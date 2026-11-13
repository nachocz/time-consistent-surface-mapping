// Charts on canvases: the functional map matrix C (rows i of the source basis, columns j
// of the target basis), drawn with the parula-like map of the figures of the paper
// (negative blue, zero teal, positive yellow); and the distribution of the point errors
// over time (Fig. 15 of the paper).
import { alpha, onThemeChange, palette } from './theme.js';

const PARULA = [[53, 42, 135], [15, 92, 221], [18, 125, 216], [7, 156, 207], [21, 177, 180], [89, 189, 140], [165, 190, 107], [225, 185, 82], [249, 251, 14]];
function ramp(stops, t) {
    const x = Math.max(0, Math.min(1, t)) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
    return stops[i].map((v, k) => (v + (stops[i + 1][k] - v) * f) / 255);
}
export const parula = (t) => ramp(PARULA, t);
export const signedColor = (v) => parula(0.5 + 0.5 * v);
const css = ([r, g, b]) => `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;

function setup(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2), w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return null;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h, p: palette(), font: getComputedStyle(document.body).fontFamily };
}

// The matrix C (I x J, row-major), scaled by max |c|, inside a frame of `size` x `size`
// cells (so that maps of different sizes keep their cell size). Options of set():
// {C, I, J, size, block: n (outline of the top-left n x n block), slope: rho (a line
// j = i / rho... drawn as the diagonal of a slanted map), title}.
export class MatrixMap {
    constructor(canvas, { rowLabel = 'i', colLabel = 'j' } = {}) {
        this.canvas = canvas; this.rowLabel = rowLabel; this.colLabel = colLabel;
        this.data = null;
        new ResizeObserver(() => this.draw()).observe(canvas);
        onThemeChange(() => this.draw());
    }
    set(data) { this.data = data; this.draw(); }
    draw() {
        const s = setup(this.canvas);
        if (!s) return;
        const { ctx, w, h, p, font } = s;
        const L = 26, T = 8, R = w - 8, B = h - 22;
        ctx.fillStyle = p.canvas; ctx.fillRect(L, T, R - L, B - T);
        if (!this.data || !this.data.C) return;
        const { C, I, J, size = Math.max(I, J), block = 0, guide = null } = this.data;
        const cell = Math.min((R - L) / size, (B - T) / size), x0 = L, y0 = T;
        let m = 0;
        for (let k = 0; k < I * J; k++) m = Math.max(m, Math.abs(C[k]));
        m = m || 1;
        ctx.fillStyle = css(signedColor(0));
        ctx.fillRect(x0, y0, cell * size, cell * size);
        for (let i = 0; i < I; i++) for (let j = 0; j < J; j++) {
            ctx.fillStyle = css(signedColor(C[i * J + j] / m));
            ctx.fillRect(x0 + j * cell, y0 + i * cell, cell + 0.4, cell + 0.4);
        }
        // the part of the frame outside C(I, J)
        ctx.fillStyle = alpha(p.canvas, 0.82);
        ctx.fillRect(x0 + J * cell, y0, (size - J) * cell, I * cell);
        ctx.fillRect(x0, y0 + I * cell, size * cell, (size - I) * cell);
        ctx.strokeStyle = p.lineStrong; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, size * cell, size * cell);
        ctx.strokeStyle = p.ink; ctx.lineWidth = 1.5; ctx.strokeRect(x0, y0, J * cell, I * cell);
        if (block > 0) {
            ctx.strokeStyle = p.ours; ctx.lineWidth = 2;
            ctx.strokeRect(x0 + 1, y0 + 1, Math.min(block, J) * cell - 2, Math.min(block, I) * cell - 2);
        }
        if (guide) {                                    // J(i): the number of target columns per row size
            ctx.strokeStyle = p.dark ? '#ffffff' : '#111111'; ctx.lineWidth = 1.4; ctx.setLineDash([3, 3]);
            ctx.beginPath();
            guide.forEach((j, i) => { const x = x0 + j * cell, y = y0 + (i + 1) * cell; if (i) ctx.lineTo(x, y0 + i * cell); else ctx.moveTo(x, y0); ctx.lineTo(x, y); });
            ctx.stroke(); ctx.setLineDash([]);
        }
        ctx.fillStyle = p.muted; ctx.font = `10px ${font}`; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        const step = size > 40 ? 20 : size > 20 ? 10 : 5;
        for (let j = 0; j <= size; j += step) ctx.fillText(String(j), x0 + j * cell, y0 + size * cell + 3);
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        for (let i = 0; i <= size; i += step) ctx.fillText(String(i), x0 - 4, y0 + i * cell);
        ctx.font = `600 10px ${font}`; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(`${this.colLabel} →`, x0 + size * cell * 0.5, y0 + size * cell + 3);
        ctx.save(); ctx.translate(8, y0 + size * cell * 0.5); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(`← ${this.rowLabel}`, 0, 0); ctx.restore();
        ctx.fillStyle = p.ink; ctx.font = `700 11px ${font}`; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
        ctx.fillText(`${I} × ${J}`, x0 + size * cell - 4, y0 + size * cell - 16);
    }
}

// Distribution of the point errors ||e_m(k)|| over the iterations k: one column per
// iteration, the fraction of points in each bin of error as a shade of the series color.
export class DistributionMap {
    constructor(canvas, { yLabel = '‖e_m‖ [cm]', xLabel = 'k' } = {}) {
        this.canvas = canvas; this.yLabel = yLabel; this.xLabel = xLabel;
        this.data = null;
        new ResizeObserver(() => this.draw()).observe(canvas);
        onThemeChange(() => this.draw());
    }
    // data: {columns: [Float64Array | null], yMax, color, kMax}
    set(data) { this.data = data; this.draw(); }
    draw() {
        const s = setup(this.canvas);
        if (!s || !this.data) return;
        const { ctx, w, h, p, font } = s, { columns, yMax, color, kMax = columns.length } = this.data;
        const L = 40, R = w - 8, T = 6, B = h - 22, bins = 40;
        ctx.fillStyle = p.canvas; ctx.fillRect(L, T, R - L, B - T);
        const cw = (R - L) / Math.max(kMax, 1), bh = (B - T) / bins;
        const toRgb = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
        const rgb = color.startsWith('#') ? toRgb(color) : [200, 40, 40];
        columns.forEach((col, k) => {
            if (!col) return;
            const counts = new Float64Array(bins);
            for (const v of col) counts[Math.min(bins - 1, Math.floor(v / yMax * bins))] += 1;
            for (let b = 0; b < bins; b++) {
                if (!counts[b]) continue;
                const f = Math.min(1, 6 * counts[b] / col.length);
                ctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${(0.12 + 0.88 * f).toFixed(3)})`;
                ctx.fillRect(L + k * cw, B - (b + 1) * bh, Math.max(cw, 1) + 0.3, bh + 0.3);
            }
        });
        ctx.strokeStyle = p.lineStrong; ctx.lineWidth = 1; ctx.strokeRect(L, T, R - L, B - T);
        ctx.fillStyle = p.muted; ctx.font = `10px ${font}`; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        for (const f of [0, 0.5, 1]) ctx.fillText((yMax * f).toFixed(yMax * f >= 10 ? 0 : 1), L - 4, B - f * (B - T));
        ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        const step = kMax > 300 ? 100 : kMax > 120 ? 50 : kMax > 40 ? 20 : 10;
        for (let k = 0; k <= kMax; k += step) ctx.fillText(String(k), L + k * cw, B + 3);
        ctx.font = `600 10px ${font}`; ctx.textBaseline = 'alphabetic'; ctx.fillText(this.xLabel, (L + R) / 2, h - 1);
        ctx.save(); ctx.translate(10, (T + B) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.fillText(this.yLabel, 0, 0); ctx.restore();
    }
}
