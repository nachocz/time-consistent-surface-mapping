// Shape control law (Sec. 4.1). The grippers act one at a time (cyclically), so that the
// change of the tracked points Delta x(k) = x(k) - x(k-1) can be attributed to the action
// u_g = (Delta t_g, Delta r_g) of a single gripper. Each gripper keeps two buffers with its
// last B actions U_g (6 x B) and the changes they produced Delta X_g (3M x B), from which
// the interaction matrix of (22), Delta x = L_g u_g, is estimated (23),
//     L_g = Delta X_g U_g^T (U_g U_g^T + gamma I_6)^(-1),
// and the action is u_g = -alpha L_g^+ e (24). Translations in cm, rotations in rad
// (Rodrigues vector). The buffers are initialized with B small exploratory actions per
// gripper, along each axis and back.

export const CONTROL_DEFAULTS = { alpha: 0.3, B: 12, gamma: 0.2, lever: 10, maxTranslation: 0.5, maxRotation: 4 * Math.PI / 180, explore: [0.3, 2.5 * Math.PI / 180] };

// Solve the symmetric positive definite system A x = b (n x n, row-major) by Cholesky.
function solveSpd(A, b, n) {
    const L = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
        let s = A[i * n + j];
        for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
        if (i === j) L[i * n + i] = Math.sqrt(Math.max(s, 1e-300)); else L[i * n + j] = s / L[j * n + j];
    }
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k]; y[i] = s / L[i * n + i]; }
    for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * y[k]; y[i] = s / L[i * n + i]; }
    return y;
}

export class SequentialShapeController {
    // G grippers (`active`: which of them may move), M tracked points
    constructor(G, M, options = {}, active = null) {
        this.options = { ...CONTROL_DEFAULTS, ...options };
        this.G = G; this.M = M;
        this.active = active || Array.from({ length: G }, () => true);
        this.buffers = Array.from({ length: G }, () => ({ U: [], dX: [] }));
        // exploration schedule: +/- each axis, gripper after gripper
        const [dt, dr] = this.options.explore, plan = [];
        for (let a = 0; a < 6; a++) for (const s of [1, -1]) {
            for (let g = 0; g < G; g++) {
                if (!this.active[g]) continue;
                const u = [0, 0, 0, 0, 0, 0];
                u[a] = s * (a < 3 ? dt : dr);
                plan.push({ g, u });
            }
        }
        this.plan = plan;
        this.k = 0;
        this.turn = this.active.indexOf(true);
    }

    set(options) { Object.assign(this.options, options); }

    get exploring() { return this.k < this.plan.length; }

    // Store the action u of gripper g and the change dx of the tracked points it produced.
    record(g, u, dx) {
        const b = this.buffers[g], l = this.options.lever;
        b.U.unshift(Float64Array.from(u, (v, i) => (i < 3 ? v : l * v))); b.dX.unshift(dx);
        while (b.U.length > this.options.B) { b.U.pop(); b.dX.pop(); }
    }

    // Estimate (23) of L_g (3M x 6, row-major), or null while the buffer has fewer than 6 entries.
    estimate(g) {
        const { U, dX } = this.buffers[g], B = U.length, M3 = 3 * this.M;
        if (B < 6) return null;
        const UUt = new Float64Array(36);
        for (let b = 0; b < B; b++) for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) UUt[i * 6 + j] += U[b][i] * U[b][j];
        for (let i = 0; i < 6; i++) UUt[i * 6 + i] += this.options.gamma;
        // L = dX U^T (U U^T + gamma I)^-1: rows of L solve (U U^T + gamma I) l = U dx_row
        const L = new Float64Array(M3 * 6), rhs = new Float64Array(6);
        const inv = Array.from({ length: 6 }, (_, c) => { const e = new Float64Array(6); e[c] = 1; return solveSpd(UUt, e, 6); });
        for (let r = 0; r < M3; r++) {
            rhs.fill(0);
            for (let b = 0; b < B; b++) { const x = dX[b][r]; for (let i = 0; i < 6; i++) rhs[i] += x * U[b][i]; }
            for (let c = 0; c < 6; c++) { let s = 0; for (let i = 0; i < 6; i++) s += inv[c][i] * rhs[i]; L[r * 6 + c] = s; }
        }
        return L;
    }

    // Action (24) of gripper g for the error e (3M), with the step limits of the robots.
    // Returns {u, L, raw} (raw: before the limits).
    action(g, e) {
        const L = this.estimate(g);
        if (!L) return { u: [0, 0, 0, 0, 0, 0], L: null, raw: null };
        const M3 = 3 * this.M, LtL = new Float64Array(36), Lte = new Float64Array(6);
        for (let r = 0; r < M3; r++) {
            const o = r * 6;
            for (let i = 0; i < 6; i++) { Lte[i] += L[o + i] * e[r]; for (let j = 0; j < 6; j++) LtL[i * 6 + j] += L[o + i] * L[o + j]; }
        }
        let tr = 0;
        for (let i = 0; i < 6; i++) tr += LtL[i * 7];
        for (let i = 0; i < 6; i++) LtL[i * 7] += 1e-9 * tr + 1e-300;
        const l = this.options.lever, x = solveSpd(LtL, Lte, 6);
        const raw = Array.from(x, (v, i) => -this.options.alpha * (i < 3 ? v : v / l)), u = raw.slice();
        const nt = Math.hypot(u[0], u[1], u[2]), nr = Math.hypot(u[3], u[4], u[5]);
        const { maxTranslation: mt, maxRotation: mr } = this.options;
        if (nt > mt) for (let i = 0; i < 3; i++) u[i] *= mt / nt;
        if (nr > mr) for (let i = 3; i < 6; i++) u[i] *= mr / nr;
        return { u, L, raw };
    }

    // The next action: exploration first, then the control law for the gripper whose turn it is.
    next(e) {
        let out;
        if (this.exploring) out = { ...this.plan[this.k], L: null, raw: null, explore: true };
        else {
            const g = this.turn;
            out = { g, ...this.action(g, e), explore: false };
            do { this.turn = (this.turn + 1) % this.G; } while (!this.active[this.turn]);
        }
        this.k++;
        return out;
    }
}

// Prediction of the tracked points after the action u of a gripper with the estimate L:
// x + L u (the deformation model the controller uses).
export function predict(x, L, u) {
    const out = Float64Array.from(x);
    if (!L) return out;
    for (let r = 0; r < x.length; r++) { let s = 0; for (let i = 0; i < 6; i++) s += L[r * 6 + i] * u[i]; out[r] += s; }
    return out;
}
