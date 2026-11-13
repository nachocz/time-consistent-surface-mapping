// Rigid iterative closest point (Besl and McKay, 1992) between two point sets, used to
// initialize the first surface map (Sec. 3.1): X is aligned to Y, and the nearest
// neighbors of the aligned points X_ICP in Y give the point map Pi of eq. (10).
import { hornRotation, mulVec3 } from './linalg.js';
import { nearestRows } from './fmap.js';

// Rigid transform {R, t} that minimizes sum || R x_m + t - y_m ||^2 (paired points).
export function kabsch(X, Y, pairs = null) {
    const n = X.length / 3, cx = [0, 0, 0], cy = [0, 0, 0];
    for (let m = 0; m < n; m++) {
        const k = pairs ? pairs[m] : m;
        for (let e = 0; e < 3; e++) { cx[e] += X[3 * m + e] / n; cy[e] += Y[3 * k + e] / n; }
    }
    const S = new Array(9).fill(0);
    for (let m = 0; m < n; m++) {
        const k = pairs ? pairs[m] : m;
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) S[3 * a + b] += (X[3 * m + a] - cx[a]) * (Y[3 * k + b] - cy[b]);
    }
    const R = hornRotation(S), Rc = mulVec3(R, cx);
    return { R, t: [cy[0] - Rc[0], cy[1] - Rc[1], cy[2] - Rc[2]] };
}

function transform(T, X, out) {
    const { R, t } = T;
    for (let m = 0; m < X.length; m += 3) {
        const x = X[m], y = X[m + 1], z = X[m + 2];
        out[m] = R[0] * x + R[1] * y + R[2] * z + t[0];
        out[m + 1] = R[3] * x + R[4] * y + R[5] * z + t[1];
        out[m + 2] = R[6] * x + R[7] * y + R[8] * z + t[2];
    }
    return out;
}

// ICP from the alignment of the centroids. Returns {T, X (aligned), pairs (nearest point
// of Y for every aligned point), rms}.
export function icp(X, Y, { iterations = 40, tolerance = 1e-7 } = {}) {
    const n = X.length / 3, N = Y.length / 3, cx = [0, 0, 0], cy = [0, 0, 0];
    for (let m = 0; m < n; m++) for (let e = 0; e < 3; e++) cx[e] += X[3 * m + e] / n;
    for (let m = 0; m < N; m++) for (let e = 0; e < 3; e++) cy[e] += Y[3 * m + e] / N;
    let T = { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [cy[0] - cx[0], cy[1] - cx[1], cy[2] - cx[2]] };
    const Xa = new Float64Array(X.length), pairs = new Int32Array(n);
    let prev = Infinity, rms = Infinity;
    for (let it = 0; it < iterations; it++) {
        transform(T, X, Xa);
        nearestRows(Xa, n, 3, Y, N, 3, 3, pairs);
        let s = 0;
        for (let m = 0; m < n; m++) for (let e = 0; e < 3; e++) { const d = Xa[3 * m + e] - Y[3 * pairs[m] + e]; s += d * d; }
        rms = Math.sqrt(s / n);
        if (prev - rms < tolerance * Math.max(1, rms)) break;
        prev = rms;
        T = kabsch(X, Y, pairs);
    }
    transform(T, X, Xa);
    nearestRows(Xa, n, 3, Y, N, 3, 3, pairs);
    return { T, X: Xa, pairs, rms };
}
