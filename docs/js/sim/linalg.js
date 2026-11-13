// Small dense linear algebra for 3D geometry. Points are stored in flat Float64Arrays
// (x0, y0, z0, x1, ...); 3x3 matrices are flat row-major arrays of length 9.
// These modules also run under Node.js (tests/test_sim.mjs).

export const I3 = () => [1, 0, 0, 0, 1, 0, 0, 0, 1];

export function mul3(A, B) {
    const C = new Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
        C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
    }
    return C;
}
export const transpose3 = (A) => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
export const mulVec3 = (A, v) => [A[0] * v[0] + A[1] * v[1] + A[2] * v[2], A[3] * v[0] + A[4] * v[1] + A[5] * v[2], A[6] * v[0] + A[7] * v[1] + A[8] * v[2]];
export const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale3 = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm3 = (a) => Math.hypot(a[0], a[1], a[2]);
export const det3 = (A) => A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
export const skew = (w) => [0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0];
export const point = (P, i) => [P[3 * i], P[3 * i + 1], P[3 * i + 2]];

// Rotation about a unit-free axis vector w (angle |w|), Rodrigues' formula.
export function rotationFromVector(w) {
    const th = norm3(w);
    if (th < 1e-14) return I3();
    const k = scale3(w, 1 / th), K = skew(k), K2 = mul3(K, K), s = Math.sin(th), c = 1 - Math.cos(th);
    return I3().map((v, i) => v + s * K[i] + c * K2[i]);
}

// ---------------------------------------------------------------- quaternions (w, x, y, z)
export function quatToMat(q) {
    const [w, x, y, z] = q;
    return [
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ];
}

// Jacobi eigen-decomposition of a symmetric 4x4 matrix (flat, row-major).
// Returns the unit eigenvector of the largest eigenvalue.
function topEigenvector4(N) {
    const a = N.slice(), v = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    for (let sweep = 0; sweep < 30; sweep++) {
        let off = 0;
        for (let p = 0; p < 3; p++) for (let q = p + 1; q < 4; q++) off += a[4 * p + q] * a[4 * p + q];
        if (off < 1e-30) break;
        for (let p = 0; p < 3; p++) for (let q = p + 1; q < 4; q++) {
            const apq = a[4 * p + q];
            if (Math.abs(apq) < 1e-300) continue;
            const theta = (a[4 * q + q] - a[4 * p + p]) / (2 * apq);
            const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
            const c = 1 / Math.sqrt(t * t + 1), s = t * c;
            for (let k = 0; k < 4; k++) {            // A <- A J (columns p, q)
                const akp = a[4 * k + p], akq = a[4 * k + q];
                a[4 * k + p] = c * akp - s * akq;
                a[4 * k + q] = s * akp + c * akq;
            }
            for (let k = 0; k < 4; k++) {            // A <- J^T A (rows p, q)
                const apk = a[4 * p + k], aqk = a[4 * q + k];
                a[4 * p + k] = c * apk - s * aqk;
                a[4 * q + k] = s * apk + c * aqk;
            }
            for (let k = 0; k < 4; k++) {            // V <- V J
                const vkp = v[4 * k + p], vkq = v[4 * k + q];
                v[4 * k + p] = c * vkp - s * vkq;
                v[4 * k + q] = s * vkp + c * vkq;
            }
        }
    }
    let best = 0;
    for (let k = 1; k < 4; k++) if (a[5 * k] > a[5 * best]) best = k;
    const e = [v[best], v[4 + best], v[8 + best], v[12 + best]];
    const n = Math.hypot(...e);
    return e.map((x) => x / n);
}

// Optimal rotation of the orthogonal Procrustes problem (Horn, 1987). With the
// cross-covariance S = sum_m a_m b_m^T of centered source points a_m and target points
// b_m (S[3*i+j] = sum a_i b_j), returns R in SO(3) that minimizes sum ||R a_m - b_m||^2.
// The quaternion formulation always yields a proper rotation, also for coplanar sets.
export function hornRotation(S) {
    const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = S;
    const N = [
        xx + yy + zz, yz - zy, zx - xz, xy - yx,
        yz - zy, xx - yy - zz, xy + yx, zx + xz,
        zx - xz, xy + yx, -xx + yy - zz, yz + zy,
        xy - yx, zx + xz, yz + zy, -xx - yy + zz,
    ];
    return quatToMat(topEigenvector4(N));
}

// Rotational part of A (the R in SO(3) that maximizes tr(R^T A)), by the iteration of
// Mueller et al. (2016) warm-started at the unit quaternion q, which is updated in place.
// Used by the local step of ARAP, where A = sum_j w_ij e'_ij e_ij^T.
export function polarRotation(A, q, iterations = 4) {
    for (let it = 0; it < iterations; it++) {
        const R = quatToMat(q);
        let ox = 0, oy = 0, oz = 0, d = 0;
        for (let c = 0; c < 3; c++) {
            const rx = R[c], ry = R[3 + c], rz = R[6 + c], ax = A[c], ay = A[3 + c], az = A[6 + c];
            ox += ry * az - rz * ay; oy += rz * ax - rx * az; oz += rx * ay - ry * ax;
            d += rx * ax + ry * ay + rz * az;
        }
        const inv = 1 / (Math.abs(d) + 1e-12);
        ox *= inv; oy *= inv; oz *= inv;
        const w = Math.hypot(ox, oy, oz);
        if (w < 1e-12) break;
        const h = 0.5 * w, s = Math.sin(h) / w;
        const p = [Math.cos(h), ox * s, oy * s, oz * s];   // q <- p * q
        const [a0, a1, a2, a3] = p, [b0, b1, b2, b3] = q;
        q[0] = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3;
        q[1] = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2;
        q[2] = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1;
        q[3] = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0;
        const n = Math.hypot(q[0], q[1], q[2], q[3]);
        q[0] /= n; q[1] /= n; q[2] /= n; q[3] /= n;
        if (w < 1e-9) break;
    }
    return quatToMat(q);
}

// Deterministic pseudo-random numbers (mulberry32), for reproducible tests and noise.
export function rng(seed = 1) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
