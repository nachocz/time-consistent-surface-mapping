// Laplace-Beltrami eigenbasis of a triangle mesh (Sec. 2.1): the cotangent Laplacian L
// (Pinkall and Polthier, 1993) and the lumped area matrix A (a third of the area of the
// triangles around each node), and the eigenpairs of L phi = lambda A phi with the
// smallest eigenvalues, ordered by increasing eigenvalue. The eigenvectors are
// A-orthonormal (Phi^T A Phi = I), so that (Phi)^+ = Phi^T A, as used in (9) and (17).
//
// The basis of every perceived mesh is computed in the browser at every frame, so the
// solver is a shift-and-invert Lanczos iteration with full reorthogonalization: the
// operator A^(1/2) (L + mu A)^(-1) A^(1/2) is applied through a banded Cholesky factor of
// L + mu A, with the nodes in reverse Cuthill-McKee order to keep the band narrow.
// The eigenvectors come with arbitrary signs, which depend on the random start vector:
// as with any eigensolver, nothing fixes the sign of phi_i from one mesh to the next.
import { rng } from './linalg.js';

// Cotangent Laplacian in CSR form (positive semi-definite: L_ii = sum_j w_ij, L_ij = -w_ij,
// w_ij = (cot alpha + cot beta) / 2) and lumped areas.
export function cotanLaplacian(V, F, n) {
    const area = new Float64Array(n), weights = new Map();
    const add = (i, j, w) => {
        const key = i < j ? i * n + j : j * n + i;
        weights.set(key, (weights.get(key) || 0) + w);
    };
    let total = 0;
    for (let t = 0; t < F.length; t += 3) {
        const id = [F[t], F[t + 1], F[t + 2]];
        const P = id.map((x) => [V[3 * x], V[3 * x + 1], V[3 * x + 2]]);
        const e = (a, b) => [P[b][0] - P[a][0], P[b][1] - P[a][1], P[b][2] - P[a][2]];
        const u = e(0, 1), v = e(0, 2);
        const cx = u[1] * v[2] - u[2] * v[1], cy = u[2] * v[0] - u[0] * v[2], cz = u[0] * v[1] - u[1] * v[0];
        const twice = Math.hypot(cx, cy, cz);
        total += twice / 2;
        for (const x of id) area[x] += twice / 6;
        if (twice < 1e-14) continue;
        for (let c = 0; c < 3; c++) {                  // angle at corner c, opposite edge (a, b)
            const a = (c + 1) % 3, b = (c + 2) % 3, p = e(c, a), q = e(c, b);
            const cot = (p[0] * q[0] + p[1] * q[1] + p[2] * q[2]) / twice;
            add(id[a], id[b], 0.5 * Math.max(-1e3, Math.min(1e3, cot)));
        }
    }
    // CSR with the diagonal first in every row
    const rows = Array.from({ length: n }, () => []), diag = new Float64Array(n);
    for (const [key, w] of weights) {
        const i = Math.floor(key / n), j = key - i * n;
        rows[i].push([j, -w]); rows[j].push([i, -w]);
        diag[i] += w; diag[j] += w;
    }
    const start = new Int32Array(n + 1);
    for (let i = 0; i < n; i++) start[i + 1] = start[i] + rows[i].length + 1;
    const col = new Int32Array(start[n]), val = new Float64Array(start[n]);
    for (let i = 0; i < n; i++) {
        let k = start[i];
        col[k] = i; val[k++] = diag[i];
        for (const [j, w] of rows[i]) { col[k] = j; val[k++] = w; }
    }
    return { n, start, col, val, area, total };
}

// Reverse Cuthill-McKee ordering of the graph of a CSR matrix: order[k] = node.
export function reverseCuthillMcKee({ n, start, col }) {
    const degree = Int32Array.from({ length: n }, (_, i) => start[i + 1] - start[i]);
    const seen = new Uint8Array(n), order = [];
    while (order.length < n) {
        let root = -1;                                 // a node of least degree, per component
        for (let i = 0; i < n; i++) if (!seen[i] && (root < 0 || degree[i] < degree[root])) root = i;
        // move to a pseudo-peripheral node: the last node of a breadth-first search, twice
        for (let pass = 0; pass < 2; pass++) {
            const mark = new Uint8Array(n), queue = [root];
            mark[root] = 1;
            for (let q = 0; q < queue.length; q++) {
                const v = queue[q];
                for (let k = start[v]; k < start[v + 1]; k++) { const u = col[k]; if (!mark[u] && !seen[u]) { mark[u] = 1; queue.push(u); } }
            }
            root = queue[queue.length - 1];
        }
        const head = order.length;
        order.push(root); seen[root] = 1;
        for (let q = head; q < order.length; q++) {
            const v = order[q], next = [];
            for (let k = start[v]; k < start[v + 1]; k++) { const u = col[k]; if (!seen[u]) { seen[u] = 1; next.push(u); } }
            next.sort((a, b) => degree[a] - degree[b]);
            order.push(...next);
        }
    }
    return order.reverse();
}

// Banded Cholesky factor of the symmetric positive definite matrix L + mu diag(area),
// in the order `order`; solve(b) solves (L + mu A) x = b in place (original numbering).
function bandedCholesky(Lap, mu, order) {
    const { n, start, col, val, area } = Lap, pos = new Int32Array(n);
    order.forEach((v, k) => { pos[v] = k; });
    let bw = 0;
    for (let i = 0; i < n; i++) for (let k = start[i]; k < start[i + 1]; k++) bw = Math.max(bw, Math.abs(pos[i] - pos[col[k]]));
    const W = bw + 1, C = new Float64Array(n * W);       // C[i*W + (i - j)], j <= i (permuted)
    for (let i = 0; i < n; i++) {
        const pi = pos[i];
        for (let k = start[i]; k < start[i + 1]; k++) {
            const pj = pos[col[k]];
            if (pj <= pi) C[pi * W + (pi - pj)] += val[k] + (col[k] === i ? mu * area[i] : 0);
        }
    }
    for (let i = 0; i < n; i++) {
        const j0 = Math.max(0, i - bw);
        for (let j = j0; j <= i; j++) {
            let s = C[i * W + (i - j)];
            for (let k = Math.max(j0, j - bw); k < j; k++) s -= C[i * W + (i - k)] * C[j * W + (j - k)];
            if (i === j) {
                if (!(s > 0)) throw new Error('spectral: L + mu A is not positive definite');
                C[i * W] = Math.sqrt(s);
            } else C[i * W + (i - j)] = s / C[j * W];
        }
    }
    const y = new Float64Array(n);
    return function solve(b) {
        for (let i = 0; i < n; i++) y[i] = b[order[i]];
        for (let i = 0; i < n; i++) {
            let s = y[i];
            for (let k = Math.max(0, i - bw); k < i; k++) s -= C[i * W + (i - k)] * y[k];
            y[i] = s / C[i * W];
        }
        for (let i = n - 1; i >= 0; i--) {
            let s = y[i];
            for (let k = i + 1; k <= Math.min(n - 1, i + bw); k++) s -= C[k * W + (k - i)] * y[k];
            y[i] = s / C[i * W];
        }
        for (let i = 0; i < n; i++) b[order[i]] = y[i];
        return b;
    };
}

// Eigenvalues and eigenvectors of a symmetric tridiagonal matrix (diagonal d, off-diagonal
// e[i] between rows i-1 and i, e[0] unused): the QL algorithm with implicit shifts
// (tql2 of EISPACK, as in JAMA). Returns {values (ascending), vectors (row-major n x n,
// column k = eigenvector k)}.
export function tridiagonalEigen(dIn, eIn) {
    const n = dIn.length, d = Float64Array.from(dIn), e = new Float64Array(n), Vm = new Float64Array(n * n);
    for (let i = 1; i < n; i++) e[i - 1] = eIn[i];
    for (let i = 0; i < n; i++) Vm[i * n + i] = 1;
    let f = 0, tst1 = 0;
    const eps = 2 ** -52;
    for (let l = 0; l < n; l++) {
        tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
        let m = l;
        while (m < n - 1 && Math.abs(e[m]) > eps * tst1) m++;
        if (m > l) {
            let iter = 0;
            do {
                if (++iter > 60) break;
                let g = d[l], p = (d[l + 1] - g) / (2 * e[l]), r = Math.hypot(p, 1);
                if (p < 0) r = -r;
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                const dl1 = d[l + 1];
                let h = g - d[l];
                for (let i = l + 2; i < n; i++) d[i] -= h;
                f += h;
                p = d[m];
                let c = 1, c2 = 1, c3 = 1, s = 0, s2 = 0;
                const el1 = e[l + 1];
                for (let i = m - 1; i >= l; i--) {
                    c3 = c2; c2 = c; s2 = s;
                    g = c * e[i]; h = c * p;
                    r = Math.hypot(p, e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r; c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    for (let k = 0; k < n; k++) {
                        const a = k * n + i;
                        h = Vm[a + 1];
                        Vm[a + 1] = s * Vm[a] + c * h;
                        Vm[a] = c * Vm[a] - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
            } while (Math.abs(e[l]) > eps * tst1);
        }
        d[l] += f;
        e[l] = 0;
    }
    const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => d[a] - d[b]);
    const values = Float64Array.from(idx, (i) => d[i]), vectors = new Float64Array(n * n);
    for (let r = 0; r < n; r++) for (let k = 0; k < n; k++) vectors[r * n + k] = Vm[r * n + idx[k]];
    return { values, vectors };
}

// The K eigenpairs of L phi = lambda A phi with the smallest eigenvalues.
// Returns {evals (K), evecs (row-major n x K: row m is the spectral embedding of point m),
// K, n, area (lumped, n), total (surface area), lap}.
export function eigenbasis(V, F, n, K, { seed = 1, steps = null } = {}) {
    const lap = cotanLaplacian(V, F, n);
    K = Math.min(K, n);
    // shift below the spectrum (~ lambda_2 / 10); a sliver triangle can make the clamped
    // cotangent matrix slightly indefinite, then a larger shift is used
    const order = reverseCuthillMcKee(lap);
    let mu = 1 / Math.max(lap.total, 1e-12), solve = null;
    for (let attempt = 0; !solve; attempt++) {
        try { solve = bandedCholesky(lap, mu, order); } catch (error) { if (attempt === 4) throw error; mu *= 100; }
    }
    const sq = Float64Array.from(lap.area, (a) => Math.sqrt(Math.max(a, 1e-300)));
    const m = Math.min(n, steps || 2 * K + 20);
    const random = rng(seed), Q = [], alpha = new Float64Array(m), beta = new Float64Array(m);
    const unit = () => {                            // random vector orthogonal to Q
        const v = Float64Array.from({ length: n }, () => random() - 0.5);
        for (let pass = 0; pass < 2; pass++) for (const q of Q) { let s = 0; for (let i = 0; i < n; i++) s += q[i] * v[i]; for (let i = 0; i < n; i++) v[i] -= s * q[i]; }
        let l = 0;
        for (let i = 0; i < n; i++) l += v[i] * v[i];
        l = Math.sqrt(l);
        for (let i = 0; i < n; i++) v[i] /= l;
        return v;
    };
    Q.push(unit());
    const w = new Float64Array(n);
    let steps_ = m;
    for (let j = 0; j < m; j++) {
        const q = Q[j];
        for (let i = 0; i < n; i++) w[i] = sq[i] * q[i];
        solve(w);
        for (let i = 0; i < n; i++) w[i] *= sq[i];
        let a = 0;
        for (let i = 0; i < n; i++) a += q[i] * w[i];
        alpha[j] = a;
        // full reorthogonalization (twice is enough)
        for (let pass = 0; pass < 2; pass++) {
            for (const p of Q) {
                let s = 0;
                for (let i = 0; i < n; i++) s += p[i] * w[i];
                for (let i = 0; i < n; i++) w[i] -= s * p[i];
            }
        }
        if (j === m - 1) break;
        let b = 0;
        for (let i = 0; i < n; i++) b += w[i] * w[i];
        b = Math.sqrt(b);
        if (b < 1e-12 * Math.abs(a)) {               // invariant subspace: continue in a new direction
            beta[j + 1] = 0;
            Q.push(unit());
        } else {
            beta[j + 1] = b;
            Q.push(Float64Array.from(w, (x) => x / b));
        }
        steps_ = j + 2;
    }
    const T = tridiagonalEigen(alpha.subarray(0, steps_), beta.subarray(0, steps_));
    const ms = steps_, evals = new Float64Array(K), evecs = new Float64Array(n * K);
    for (let k = 0; k < K; k++) {
        const c = ms - 1 - k;                        // largest theta = 1 / (lambda + mu) first
        evals[k] = Math.max(0, 1 / T.values[c] - mu);
        for (let j = 0; j < ms; j++) {
            const s = T.vectors[j * ms + c], q = Q[j];
            if (s === 0) continue;
            for (let i = 0; i < n; i++) evecs[i * K + k] += s * q[i];
        }
        for (let i = 0; i < n; i++) evecs[i * K + k] /= sq[i];
    }
    return { evals, evecs, K, n, area: lap.area, total: lap.total, lap };
}

// Lumped areas only (no basis): a third of the area of the triangles around each node.
export function lumpedArea(V, F, n) {
    const area = new Float64Array(n);
    let total = 0;
    for (let t = 0; t < F.length; t += 3) {
        const a = F[t], b = F[t + 1], c = F[t + 2];
        const u0 = V[3 * b] - V[3 * a], u1 = V[3 * b + 1] - V[3 * a + 1], u2 = V[3 * b + 2] - V[3 * a + 2];
        const v0 = V[3 * c] - V[3 * a], v1 = V[3 * c + 1] - V[3 * a + 1], v2 = V[3 * c + 2] - V[3 * a + 2];
        const s = 0.5 * Math.hypot(u1 * v2 - u2 * v1, u2 * v0 - u0 * v2, u0 * v1 - u1 * v0);
        total += s;
        area[a] += s / 3; area[b] += s / 3; area[c] += s / 3;
    }
    return { area, total };
}
