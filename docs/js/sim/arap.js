// As-rigid-as-possible deformation (Sorkine and Alexa, 2007) of the volumetric lattice,
// the deformation model of the simulations of the paper (Sec. IV-A). The nodes held by
// the grippers are fixed at the positions the grippers impose (handles). The energy
//     E = sum_i sum_{j in N(i)} w_ij || (p_i - p_j) - R_i (r_i - r_j) ||^2
// (rest positions r) is minimized by alternating the local step (each R_i is the
// rotational part of the deformed 1-ring) and the global step (a Laplacian system,
// factorized once since the handles do not change). The weights are uniform, or
// w_ij = (s_i + s_j) / 2 for a stiffness s per node: a stiffer region deviates less from
// a rigid motion. The inner loops run on flat arrays (neighbors in CSR form, rest edge
// vectors precomputed) and do not allocate, which keeps them fast in every engine.

export class Arap {
    constructor(rest, neighbors, handles, stiffness = null) {
        const n = rest.length / 3;
        this.n = n; this.rest = rest; this.neighbors = neighbors;
        // neighbors in CSR form, the rest edge vectors r_i - r_j and the weight of every entry
        const start = new Int32Array(n + 1);
        for (let i = 0; i < n; i++) start[i + 1] = start[i] + neighbors[i].length;
        const list = new Int32Array(start[n]), edge = new Float64Array(3 * start[n]), weight = new Float64Array(start[n]);
        for (let i = 0; i < n; i++) {
            for (let a = 0; a < neighbors[i].length; a++) {
                const j = neighbors[i][a], e = start[i] + a;
                list[e] = j;
                weight[e] = stiffness ? 0.5 * (stiffness[i] + stiffness[j]) : 1;
                edge[3 * e] = rest[3 * i] - rest[3 * j]; edge[3 * e + 1] = rest[3 * i + 1] - rest[3 * j + 1]; edge[3 * e + 2] = rest[3 * i + 2] - rest[3 * j + 2];
            }
        }
        this.start = start; this.list = list; this.edge = edge; this.weight = weight;
        this.isHandle = new Uint8Array(n);
        for (const h of handles) this.isHandle[h] = 1;
        this.free = new Int32Array(n).fill(-1);
        let nf = 0;
        for (let i = 0; i < n; i++) if (!this.isHandle[i]) this.free[i] = nf++;
        this.nf = nf;
        this.quats = new Float64Array(4 * n);
        for (let i = 0; i < n; i++) this.quats[4 * i] = 1;
        this.rotations = new Float64Array(9 * n);
        this.factorize();
        this.rhs = [new Float64Array(nf), new Float64Array(nf), new Float64Array(nf)];
    }

    // Banded Cholesky factor of the Laplacian restricted to the free nodes; the lattice
    // numbering keeps the band narrow (it is set by the nodes of one cross-section).
    factorize() {
        const { n, nf, free, start, list, weight } = this;
        let bw = 0;
        for (let i = 0; i < n; i++) {
            if (free[i] < 0) continue;
            for (let e = start[i]; e < start[i + 1]; e++) if (free[list[e]] >= 0) bw = Math.max(bw, Math.abs(free[i] - free[list[e]]));
        }
        const W = bw + 1, L = new Float64Array(nf * W);           // L[i*W + (i - j)], j <= i
        for (let i = 0; i < n; i++) {
            const fi = free[i];
            if (fi < 0) continue;
            for (let e = start[i]; e < start[i + 1]; e++) {
                L[fi * W] += weight[e];
                const fj = free[list[e]];
                if (fj >= 0 && fj < fi) L[fi * W + (fi - fj)] = -weight[e];
            }
        }
        for (let i = 0; i < nf; i++) {
            const j0 = Math.max(0, i - bw);
            for (let j = j0; j <= i; j++) {
                let s = L[i * W + (i - j)];
                const k0 = Math.max(j0, j - bw);
                for (let k = k0; k < j; k++) s -= L[i * W + (i - k)] * L[j * W + (j - k)];
                if (i === j) {
                    if (s <= 0) throw new Error('ARAP: Laplacian not positive definite (a part of the object is not held)');
                    L[i * W] = Math.sqrt(s);
                } else L[i * W + (i - j)] = s / L[j * W];
            }
        }
        this.band = bw; this.chol = L;
    }

    solveFactor(b) {                    // in place: L L^T x = b
        const { nf, band: bw, chol: L } = this, W = bw + 1;
        for (let i = 0; i < nf; i++) {
            let s = b[i];
            for (let k = Math.max(0, i - bw); k < i; k++) s -= L[i * W + (i - k)] * b[k];
            b[i] = s / L[i * W];
        }
        for (let i = nf - 1; i >= 0; i--) {
            let s = b[i];
            for (let k = i + 1; k <= Math.min(nf - 1, i + bw); k++) s -= L[k * W + (k - i)] * b[k];
            b[i] = s / L[i * W];
        }
    }

    // Local step: R_i from the deformed positions P. R_i is the rotational part of
    // A_i = sum_j (p_i - p_j)(r_i - r_j)^T, by the iteration of Mueller et al. (2016)
    // warm-started at the previous rotation (see polarRotation in js/sim/linalg.js).
    localStep(P, iterations = 3) {
        const { n, start, list, edge, weight, quats: Qt, rotations } = this;
        for (let i = 0; i < n; i++) {
            let a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0, a8 = 0;
            const pi0 = P[3 * i], pi1 = P[3 * i + 1], pi2 = P[3 * i + 2];
            for (let e = start[i]; e < start[i + 1]; e++) {
                const j = list[e], we = weight[e];
                const e0 = we * (pi0 - P[3 * j]), e1 = we * (pi1 - P[3 * j + 1]), e2 = we * (pi2 - P[3 * j + 2]);
                const f0 = edge[3 * e], f1 = edge[3 * e + 1], f2 = edge[3 * e + 2];
                a0 += e0 * f0; a1 += e0 * f1; a2 += e0 * f2;
                a3 += e1 * f0; a4 += e1 * f1; a5 += e1 * f2;
                a6 += e2 * f0; a7 += e2 * f1; a8 += e2 * f2;
            }
            let w = Qt[4 * i], x = Qt[4 * i + 1], y = Qt[4 * i + 2], z = Qt[4 * i + 3];
            let r0 = 1, r1 = 0, r2 = 0, r3 = 0, r4 = 1, r5 = 0, r6 = 0, r7 = 0, r8 = 1;
            for (let it = 0; it <= iterations; it++) {
                r0 = 1 - 2 * (y * y + z * z); r1 = 2 * (x * y - w * z); r2 = 2 * (x * z + w * y);
                r3 = 2 * (x * y + w * z); r4 = 1 - 2 * (x * x + z * z); r5 = 2 * (y * z - w * x);
                r6 = 2 * (x * z - w * y); r7 = 2 * (y * z + w * x); r8 = 1 - 2 * (x * x + y * y);
                if (it === iterations) break;
                // omega = sum_c (column c of R) x (column c of A), divided by |sum_c R_c . A_c|
                const ox = (r3 * a6 - r6 * a3) + (r4 * a7 - r7 * a4) + (r5 * a8 - r8 * a5);
                const oy = (r6 * a0 - r0 * a6) + (r7 * a1 - r1 * a7) + (r8 * a2 - r2 * a8);
                const oz = (r0 * a3 - r3 * a0) + (r1 * a4 - r4 * a1) + (r2 * a5 - r5 * a2);
                const d = Math.abs(r0 * a0 + r3 * a3 + r6 * a6 + r1 * a1 + r4 * a4 + r7 * a7 + r2 * a2 + r5 * a5 + r8 * a8) + 1e-12;
                const vx = ox / d, vy = oy / d, vz = oz / d, ang = Math.sqrt(vx * vx + vy * vy + vz * vz);
                if (ang < 1e-12) break;
                const hc = Math.cos(0.5 * ang), hs = Math.sin(0.5 * ang) / ang;
                const pw = hc, px = vx * hs, py = vy * hs, pz = vz * hs;     // q <- p * q
                const nw = pw * w - px * x - py * y - pz * z, nx = pw * x + px * w + py * z - pz * y;
                const ny = pw * y - px * z + py * w + pz * x, nz = pw * z + px * y - py * x + pz * w;
                const l = Math.sqrt(nw * nw + nx * nx + ny * ny + nz * nz);
                w = nw / l; x = nx / l; y = ny / l; z = nz / l;
            }
            Qt[4 * i] = w; Qt[4 * i + 1] = x; Qt[4 * i + 2] = y; Qt[4 * i + 3] = z;
            const o = 9 * i;
            rotations[o] = r0; rotations[o + 1] = r1; rotations[o + 2] = r2; rotations[o + 3] = r3; rotations[o + 4] = r4;
            rotations[o + 5] = r5; rotations[o + 6] = r6; rotations[o + 7] = r7; rotations[o + 8] = r8;
        }
    }

    // Global step: free positions from the rotations and the handle positions in P.
    globalStep(P) {
        const { n, start, list, edge, weight, rotations: Q, free, rhs } = this;
        const [bx, by, bz] = rhs;
        for (let i = 0; i < n; i++) {
            const fi = free[i];
            if (fi < 0) continue;
            let b0 = 0, b1 = 0, b2 = 0;
            const a = 9 * i;
            for (let e = start[i]; e < start[i + 1]; e++) {
                const j = list[e], c = 9 * j, h = 0.5 * weight[e];
                const f0 = edge[3 * e], f1 = edge[3 * e + 1], f2 = edge[3 * e + 2];
                b0 += h * ((Q[a] + Q[c]) * f0 + (Q[a + 1] + Q[c + 1]) * f1 + (Q[a + 2] + Q[c + 2]) * f2);
                b1 += h * ((Q[a + 3] + Q[c + 3]) * f0 + (Q[a + 4] + Q[c + 4]) * f1 + (Q[a + 5] + Q[c + 5]) * f2);
                b2 += h * ((Q[a + 6] + Q[c + 6]) * f0 + (Q[a + 7] + Q[c + 7]) * f1 + (Q[a + 8] + Q[c + 8]) * f2);
                if (free[j] < 0) { b0 += weight[e] * P[3 * j]; b1 += weight[e] * P[3 * j + 1]; b2 += weight[e] * P[3 * j + 2]; }
            }
            bx[fi] = b0; by[fi] = b1; bz[fi] = b2;
        }
        for (let d = 0; d < 3; d++) {
            this.solveFactor(rhs[d]);
            for (let i = 0; i < n; i++) if (free[i] >= 0) P[3 * i + d] = rhs[d][free[i]];
        }
    }

    // Iterations from the current positions P (in place); the handles must already hold
    // their imposed positions. The local-global iteration converges slowly for the
    // bending and twisting of long, thin parts; Anderson acceleration (Peng et al., 2018)
    // extrapolates the free positions from the last iterates, and an accelerated point
    // that increases the ARAP energy is replaced by the plain iterate.
    solve(P, iterations = 8, { accelerate = true, window = 6 } = {}) {
        if (!accelerate) {
            for (let it = 0; it < iterations; it++) { this.localStep(P); this.globalStep(P); }
            return P;
        }
        const { n, nf, free } = this, n3 = 3 * nf;
        if (!this.aa || this.aa.window !== window) {
            const vec = () => new Float64Array(n3);
            this.aa = { window, x: vec(), g: vec(), f: vec(), gPrev: vec(), fPrev: vec(), gSafe: vec(),
                dG: Array.from({ length: window }, vec), dF: Array.from({ length: window }, vec) };
        }
        const A = this.aa, { x, g, f, gPrev, fPrev, gSafe, dG, dF } = A;
        const gather = (dst) => { for (let i = 0; i < n; i++) if (free[i] >= 0) { const a = 3 * free[i]; dst[a] = P[3 * i]; dst[a + 1] = P[3 * i + 1]; dst[a + 2] = P[3 * i + 2]; } };
        const scatter = (src) => { for (let i = 0; i < n; i++) if (free[i] >= 0) { const a = 3 * free[i]; P[3 * i] = src[a]; P[3 * i + 1] = src[a + 1]; P[3 * i + 2] = src[a + 2]; } };
        gather(x);
        let count = 0, head = 0, Eprev = Infinity, havePrev = false, haveSafe = false;
        for (let it = 0; it < iterations; it++) {
            this.localStep(P);
            let E = this.energy(P);
            if (E > Eprev && haveSafe) {               // safeguard: back to the plain iterate
                scatter(gSafe); x.set(gSafe);
                this.localStep(P); E = this.energy(P);
                count = 0; havePrev = false;
            }
            Eprev = E;
            this.globalStep(P);
            gather(g);
            for (let a = 0; a < n3; a++) f[a] = g[a] - x[a];
            gSafe.set(g); haveSafe = true;
            if (havePrev) {
                const slot = head; head = (head + 1) % window; count = Math.min(count + 1, window);
                for (let a = 0; a < n3; a++) { dG[slot][a] = g[a] - gPrev[a]; dF[slot][a] = f[a] - fPrev[a]; }
            }
            gPrev.set(g); fPrev.set(f); havePrev = true;
            if (count > 0) {
                // theta = argmin || f - dF theta ||  (normal equations, lightly regularized)
                const m = count, slots = Array.from({ length: m }, (_, k) => (head - m + k + window) % window);
                const N = new Float64Array(m * m), b = new Float64Array(m);
                for (let p = 0; p < m; p++) {
                    const Fp = dF[slots[p]];
                    let bp = 0;
                    for (let a = 0; a < n3; a++) bp += Fp[a] * f[a];
                    b[p] = bp;
                    for (let q = p; q < m; q++) {
                        const Fq = dF[slots[q]];
                        let s = 0;
                        for (let a = 0; a < n3; a++) s += Fp[a] * Fq[a];
                        N[p * m + q] = N[q * m + p] = s;
                    }
                }
                let trace = 0;
                for (let p = 0; p < m; p++) trace += N[p * m + p];
                for (let p = 0; p < m; p++) N[p * m + p] += 1e-10 * trace + 1e-300;
                const theta = solveSmall(N, b, m);
                x.set(g);
                if (theta) {
                    for (let p = 0; p < m; p++) {
                        const Gp = dG[slots[p]], t = theta[p];
                        for (let a = 0; a < n3; a++) x[a] -= t * Gp[a];
                    }
                }
            } else x.set(g);
            scatter(x);
        }
        return P;
    }

    energy(P) {
        const { n, start, list, edge, weight, rotations: Q } = this;
        let E = 0;
        for (let i = 0; i < n; i++) {
            const a = 9 * i;
            for (let e = start[i]; e < start[i + 1]; e++) {
                const j = list[e], f0 = edge[3 * e], f1 = edge[3 * e + 1], f2 = edge[3 * e + 2];
                const d0 = P[3 * i] - P[3 * j] - (Q[a] * f0 + Q[a + 1] * f1 + Q[a + 2] * f2);
                const d1 = P[3 * i + 1] - P[3 * j + 1] - (Q[a + 3] * f0 + Q[a + 4] * f1 + Q[a + 5] * f2);
                const d2 = P[3 * i + 2] - P[3 * j + 2] - (Q[a + 6] * f0 + Q[a + 7] * f1 + Q[a + 8] * f2);
                E += weight[e] * (d0 * d0 + d1 * d1 + d2 * d2);
            }
        }
        return E;
    }

    snapshot() { return { quats: this.quats.slice() }; }
    restore(s) { this.quats.set(s.quats); }
}

// Gaussian elimination with partial pivoting for the small Anderson system (null if singular).
function solveSmall(N, b, m) {
    const A = Array.from({ length: m }, (_, i) => [...N.slice(i * m, i * m + m), b[i]]);
    for (let c = 0; c < m; c++) {
        let p = c;
        for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
        if (Math.abs(A[p][c]) < 1e-300) return null;
        [A[c], A[p]] = [A[p], A[c]];
        for (let r = c + 1; r < m; r++) {
            const f = A[r][c] / A[c][c];
            for (let k = c; k <= m; k++) A[r][k] -= f * A[c][k];
        }
    }
    const x = new Float64Array(m);
    for (let r = m - 1; r >= 0; r--) {
        let s = A[r][m];
        for (let k = r + 1; k < m; k++) s -= A[r][k] * x[k];
        x[r] = s / A[r][r];
    }
    return x.every(Number.isFinite) ? x : null;
}
