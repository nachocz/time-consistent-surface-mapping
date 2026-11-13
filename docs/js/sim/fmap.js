// Functional maps (Sec. 2) and their refinement (Secs. 2.2 and 3.3-3.4).
//
// A basis is {evecs (row-major n x K), evals (K), area (n), n, K}: row m of evecs is the
// spectral embedding of point m, column i the eigenvector phi_i. A point-to-point map
// Pi: M -> N is stored as an Int32Array of length M (Pi[m] = n means Pi_{m,n} = 1, eq. 5).
// A functional map C(I) is stored row-major with I rows and J columns.

// Index of the nearest row of A (na x dim, row stride sa) for every row of Q (nq x dim,
// row stride sq), by exhaustive search with early termination. `hint` (optional) gives a
// likely answer per row (e.g., the previous map), whose distance bounds the search.
export function nearestRows(Q, nq, sq, A, na, sa, dim, out = new Int32Array(nq), hint = null) {
    for (let q = 0; q < nq; q++) {
        const oq = q * sq;
        let best = Infinity, arg = 0;
        if (hint) {
            const oa = hint[q] * sa;
            let d = 0;
            for (let k = 0; k < dim; k++) { const t = Q[oq + k] - A[oa + k]; d += t * t; }
            best = d * (1 + 1e-12) + 1e-300; arg = hint[q];
        }
        for (let a = 0; a < na; a++) {
            const oa = a * sa;
            let d = 0;
            for (let k = 0; k < dim; k++) {
                const t = Q[oq + k] - A[oa + k];
                d += t * t;
                if (d >= best) break;
            }
            if (d < best) { best = d; arg = a; }
        }
        out[q] = arg;
    }
    return out;
}

// C(I, J) = (Phi^M_I)^T A^M Pi Phi^N_J, eqs. (9), (10), (12), (17).
export function fmapFromMap(src, I, tgt, J, Pi) {
    const C = new Float64Array(I * J), { evecs: P, K: sk, area } = src, { evecs: Q, K: tk } = tgt;
    for (let m = 0; m < src.n; m++) {
        const om = m * sk, on = Pi[m] * tk, a = area[m];
        for (let i = 0; i < I; i++) {
            const pi = a * P[om + i];
            if (pi === 0) continue;
            const oc = i * J;
            for (let j = 0; j < J; j++) C[oc + j] += pi * Q[on + j];
        }
    }
    return C;
}

// Pi = argmin || Pi Phi^N_J C^T - Phi^M_I ||_F, eqs. (8), (11), (16): every point m of the
// source goes to the target point whose row of Phi^N_J C^T is nearest to row m of Phi^M_I.
export function mapFromFmap(src, I, tgt, J, C, hint = null) {
    const { evecs: Q, K: tk, n: N } = tgt, E = new Float64Array(N * I);
    for (let t = 0; t < N; t++) {
        const ot = t * tk, oe = t * I;
        for (let i = 0; i < I; i++) {
            let s = 0;
            const oc = i * J;
            for (let j = 0; j < J; j++) s += Q[ot + j] * C[oc + j];
            E[oe + i] = s;
        }
    }
    return nearestRows(src.evecs, src.n, src.K, E, N, I, I, new Int32Array(src.n), hint);
}

// Number of target eigenvectors J for I source eigenvectors, eq. (19):
// J = max{ r | lambda^N_r <= lambda^M_I rho }, with rho = sqrt(A(M) / A(N)) (18).
// Without rho (null), J = I: the square maps of ZoomOut.
export function targetSize(src, tgt, I, rho) {
    if (rho === null || rho === undefined) return Math.min(I, tgt.K);
    const limit = src.evals[I - 1] * rho;
    let J = 0;
    while (J < tgt.K && tgt.evals[J] <= limit) J++;
    return Math.max(1, J);
}

// Top-left I x J block of C (rows i < I, columns j < J; zero outside C).
export function block(C, rows, cols, I, J) {
    const out = new Float64Array(I * J);
    for (let i = 0; i < Math.min(I, rows); i++) for (let j = 0; j < Math.min(J, cols); j++) out[i * J + j] = C[i * cols + j];
    return out;
}

// Coarse-to-fine refinement (ZoomOut, Melzi et al. 2019; Algorithm 1 of the paper):
// from C(I0) (I0 x J0), alternate the point map (16) and the larger functional map (17)
// until I = Imax. `step` is the increment of I (1 in the paper, 4 in the reference code);
// rho enables the non-isometry rule (19). Every size I is recorded in `trace` (optional):
// {I, J, C, Pi}, with Pi the point map computed from that C(I).
export function zoomOut(src, tgt, { C0, I0, J0, Imax, step = 1, rho = null, trace = null, hint = null }) {
    let I = I0, J = J0, C = C0, Pi = hint;
    for (;;) {
        Pi = mapFromFmap(src, I, tgt, J, C, Pi);
        trace?.push({ I, J, C, Pi });
        if (I >= Imax) break;
        I = Math.min(I + step, Imax);
        J = targetSize(src, tgt, I, rho);
        C = fmapFromMap(src, I, tgt, J, Pi);
    }
    return { Pi, C, I, J };
}

// Orthonormality residual || C^T C - I ||_F of a square block (the term of eq. 6).
export function orthonormality(C, I, J) {
    const n = Math.min(I, J);
    let s = 0;
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
        let v = 0;
        for (let i = 0; i < I; i++) v += C[i * J + a] * C[i * J + b];
        if (a === b) v -= 1;
        s += v * v;
    }
    return Math.sqrt(s);
}
