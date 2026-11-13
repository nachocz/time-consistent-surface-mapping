// Colors of the surface maps and of the functions drawn on the meshes.
//
// Surface map: as in the figures of the paper, the target N carries a smooth coloring of
// its points (here from their coordinates along the principal axes of N), and every
// point m of the current shape takes the color of its image Pi[m]. Where the map is
// continuous, the colors of M vary smoothly; a flip or a jump of the map shows at once.

const clamp01 = (t) => Math.max(0, Math.min(1, t));

function hsl(h, s, l) {
    const f = (n) => { const k = (n + h * 12) % 12, a = s * Math.min(l, 1 - l); return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
    return [f(0), f(8), f(4)];
}

// Principal axes of a point set (power iterations on the covariance), the first two.
function principalAxes(V) {
    const n = V.length / 3, c = [0, 0, 0];
    for (let m = 0; m < n; m++) for (let e = 0; e < 3; e++) c[e] += V[3 * m + e] / n;
    const S = new Array(9).fill(0);
    for (let m = 0; m < n; m++) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) S[3 * a + b] += (V[3 * m + a] - c[a]) * (V[3 * m + b] - c[b]);
    const power = (deflate) => {
        let v = [0.83, 0.47, 0.31];
        for (let it = 0; it < 60; it++) {
            let w = [0, 1, 2].map((a) => S[3 * a] * v[0] + S[3 * a + 1] * v[1] + S[3 * a + 2] * v[2]);
            if (deflate) { const d = w[0] * deflate[0] + w[1] * deflate[1] + w[2] * deflate[2]; w = w.map((x, i) => x - d * deflate[i]); }
            const l = Math.hypot(...w) || 1;
            v = w.map((x) => x / l);
        }
        return v;
    };
    const a1 = power(null), a2 = power(a1);
    // fixed orientation: the axes point towards +x and +y as much as possible
    if (a1[0] + 0.01 * a1[1] < 0) a1.forEach((x, i) => { a1[i] = -x; });
    if (a2[1] + 0.01 * a2[0] < 0) a2.forEach((x, i) => { a2[i] = -x; });
    return { c, a1, a2 };
}

// Reference colors of the target points (3n, 0..1).
export function referenceColors(V) {
    const n = V.length / 3, { c, a1, a2 } = principalAxes(V), s = new Float64Array(2 * n);
    let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    for (let m = 0; m < n; m++) {
        const d = [V[3 * m] - c[0], V[3 * m + 1] - c[1], V[3 * m + 2] - c[2]];
        s[2 * m] = d[0] * a1[0] + d[1] * a1[1] + d[2] * a1[2];
        s[2 * m + 1] = d[0] * a2[0] + d[1] * a2[1] + d[2] * a2[2];
        for (let k = 0; k < 2; k++) { lo[k] = Math.min(lo[k], s[2 * m + k]); hi[k] = Math.max(hi[k], s[2 * m + k]); }
    }
    const out = new Float32Array(3 * n);
    for (let m = 0; m < n; m++) {
        const u = (s[2 * m] - lo[0]) / (hi[0] - lo[0] || 1), v = (s[2 * m + 1] - lo[1]) / (hi[1] - lo[1] || 1);
        const col = hsl((0.98 + 0.78 * u) % 1, 0.7, 0.36 + 0.3 * v);
        out.set(col, 3 * m);
    }
    return out;
}

// Colors of the source points through the map Pi (source m takes the color of Pi[m]).
export function pullColors(ref, Pi) {
    const out = new Float32Array(3 * Pi.length);
    for (let m = 0; m < Pi.length; m++) { out[3 * m] = ref[3 * Pi[m]]; out[3 * m + 1] = ref[3 * Pi[m] + 1]; out[3 * m + 2] = ref[3 * Pi[m] + 2]; }
    return out;
}

// Diverging map for signed functions (eigenfunctions): blue - light - red.
const DIV = [[33, 102, 172], [103, 169, 207], [209, 229, 240], [247, 247, 247], [253, 219, 199], [239, 138, 98], [178, 24, 43]];
const HEAT = [[252, 253, 191], [254, 176, 120], [241, 96, 93], [183, 55, 121], [114, 31, 129], [44, 17, 95], [0, 0, 4]].reverse();
const VIRIDIS = [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]];
function ramp(stops, t) {
    const x = clamp01(t) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
    return stops[i].map((v, k) => (v + (stops[i + 1][k] - v) * f) / 255);
}
export const diverging = (t) => ramp(DIV, 0.5 + 0.5 * t);           // t in [-1, 1]
export const heat = (t) => ramp(HEAT, t);                            // t in [0, 1]
export const viridis = (t) => ramp(VIRIDIS, t);

// Per-point colors of values f (n) with a color map; scale: max |f| (signed) or max f.
export function valueColors(f, map = 'diverging', scale = null) {
    const n = f.length, out = new Float32Array(3 * n);
    let s = scale;
    if (!s) { s = 0; for (let i = 0; i < n; i++) s = Math.max(s, Math.abs(f[i])); s = s || 1; }
    const fn = map === 'diverging' ? diverging : map === 'heat' ? heat : viridis;
    for (let i = 0; i < n; i++) out.set(fn(map === 'diverging' ? f[i] / s : f[i] / s), 3 * i);
    return out;
}

// Column i of a row-major basis (n x K) as a function.
export function column(evecs, n, K, i) {
    const out = new Float64Array(n);
    for (let m = 0; m < n; m++) out[m] = evecs[m * K + i];
    return out;
}

export const css = ([r, g, b]) => `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
export const gripperColor = (p, k) => p[`g${k + 1}`];
