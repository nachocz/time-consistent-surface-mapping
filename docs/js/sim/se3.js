// Rigid transforms T = (R, t) in SE(3), their logarithm and exponential (closed forms,
// e.g. Murray et al., Appendix A), and the geodesic path H = exp(s log T) of eq. (3).
// A transform maps a point p to R p + t.
import { I3, add3, mul3, mulVec3, norm3, skew, transpose3 } from './linalg.js';

export const identity = () => ({ R: I3(), t: [0, 0, 0] });
export const apply = (T, p) => add3(mulVec3(T.R, p), T.t);
export const compose = (A, B) => ({ R: mul3(A.R, B.R), t: add3(mulVec3(A.R, B.t), A.t) });   // A after B
export function inverse(T) {
    const Rt = transpose3(T.R), t = mulVec3(Rt, T.t);
    return { R: Rt, t: [-t[0], -t[1], -t[2]] };
}

// Rotation angle theta in [0, pi] of R (atan2 keeps the precision near 0 and pi).
export const rotationAngle = (R) => Math.atan2(0.5 * Math.hypot(R[7] - R[5], R[2] - R[6], R[3] - R[1]), (R[0] + R[4] + R[8] - 1) / 2);

// Rotation vector w (axis times angle) of R, robust near 0 and near pi.
export function logSO3(R) {
    const th = rotationAngle(R);
    const vee = [R[7] - R[5], R[2] - R[6], R[3] - R[1]];
    if (th < 1e-7) return vee.map((v) => v / 2);
    if (Math.PI - th > 1e-5) {
        const f = th / (2 * Math.sin(th));
        return vee.map((v) => v * f);
    }
    // near pi: axis from the diagonal of (R + I) / 2 = k k^T
    const d = [R[0], R[4], R[8]];
    const i = d.indexOf(Math.max(...d));
    const k = [0, 0, 0];
    k[i] = Math.sqrt(Math.max(0, (d[i] + 1) / 2));
    const j = (i + 1) % 3, l = (i + 2) % 3;
    k[j] = (R[3 * i + j] + R[3 * j + i]) / (4 * k[i]);
    k[l] = (R[3 * i + l] + R[3 * l + i]) / (4 * k[i]);
    const n = norm3(k), sign = (vee[0] * k[0] + vee[1] * k[1] + vee[2] * k[2]) < 0 ? -1 : 1;
    return k.map((v) => sign * v * th / n);
}

// Coefficients of exp / log of SE(3), with series expansions for small angles.
function coefficients(th) {
    if (th < 1e-4) {
        const t2 = th * th;
        return { A: 1 - t2 / 6, B: 0.5 - t2 / 24, C: 1 / 6 - t2 / 120, D: 1 / 12 + t2 / 720 };
    }
    const s = Math.sin(th), c = Math.cos(th);
    return { A: s / th, B: (1 - c) / (th * th), C: (th - s) / (th * th * th), D: (1 - th * (1 + c) / (2 * s)) / (th * th) };
}

// log T = (w, v): rotation vector w and translational part v = V^{-1} t, where
// V^{-1} = I - W/2 + (1/theta^2)(1 - theta (1 + cos theta) / (2 sin theta)) W^2 is the
// matrix V_g of eq. (17) of the paper (W = log R).
export function log(T) {
    const w = logSO3(T.R), th = norm3(w), W = skew(w), W2 = mul3(W, W), { D } = coefficients(th);
    const Vinv = I3().map((v, i) => v - 0.5 * W[i] + D * W2[i]);
    return { w, v: mulVec3(Vinv, T.t) };
}

export function exp({ w, v }) {
    const th = norm3(w), W = skew(w), W2 = mul3(W, W), { A, B, C } = coefficients(th);
    const R = I3().map((x, i) => x + A * W[i] + B * W2[i]);
    const V = I3().map((x, i) => x + B * W[i] + C * W2[i]);
    return { R, t: mulVec3(V, v) };
}

// Geodesic path from the identity towards T: H(s) = exp(s log T), s in [0, 1] (eq. 3).
export function power(T, s) {
    const { w, v } = log(T);
    return exp({ w: w.map((x) => x * s), v: v.map((x) => x * s) });
}

// The nearest proper rotation of T.R (Gram-Schmidt on the columns). Poses that are
// composed many times are kept orthonormal: with increments U = T_target T^{-1}, where
// the inverse uses the transpose, a rounding error of R would double at every step.
export function orthonormalize(T) {
    const R = T.R;
    let a = [R[0], R[3], R[6]], b = [R[1], R[4], R[7]];
    const na = Math.hypot(...a);
    a = a.map((v) => v / na);
    const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    b = b.map((v, i) => v - d * a[i]);
    const nb = Math.hypot(...b);
    b = b.map((v) => v / nb);
    const c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    return { R: [a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]], t: T.t.slice() };
}

// Roll, pitch, yaw (R = Rz(yaw) Ry(pitch) Rx(roll)).
export function eulerZYX(R) {
    const pitch = Math.asin(Math.max(-1, Math.min(1, -R[6])));
    return [Math.atan2(R[7], R[8]), pitch, Math.atan2(R[3], R[0])];
}

// Rigid transform that rotates by R about the point c, then translates by d.
export function aboutPoint(R, c, d = [0, 0, 0]) {
    const Rc = mulVec3(R, c);
    return { R, t: [c[0] - Rc[0] + d[0], c[1] - Rc[1] + d[1], c[2] - Rc[2] + d[2]] };
}

// Apply T to all points of P (flat array), writing into out (may equal P).
export function applyAll(T, P, out = new Float64Array(P.length)) {
    const [a, b, c, d, e, f, g, h, i] = T.R, [tx, ty, tz] = T.t;
    for (let k = 0; k < P.length; k += 3) {
        const x = P[k], y = P[k + 1], z = P[k + 2];
        out[k] = a * x + b * y + c * z + tx;
        out[k + 1] = d * x + e * y + f * z + ty;
        out[k + 2] = g * x + h * y + i * z + tz;
    }
    return out;
}
