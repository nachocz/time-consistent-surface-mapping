// Time-consistent and non-isometry robust surface mapping (Sec. 3, Algorithm 1), and the
// non-time-consistent baseline: ZoomOut computed anew at every frame (Sec. 4.2).
//
// At every frame k the method receives a new mesh M_ret(k) (points X_ret(k), in any order
// and number) and returns
//   - the tracked points X(k) (M x 3): the same M object points at every frame (15),
//   - the point map Pi(k): tracked point m -> target point Pi[m] (16),
//   - the functional map C(I_max, k) (I_max x J),
// from which the shape error E(k) = X(k) - Pi(k) Y (20) is defined.
//
// Tracking of the basis and of the points (Sec. 3.2), three ways:
//   'anchored' as in the reference MATLAB code (previousstatefmap.m): the map Pi^M(k)
//              between consecutive meshes is initialized by nearest neighbors and refined
//              coarse-to-fine as in (11)-(12); the basis of the new mesh is aligned with the
//              whole map C^M(k), which corrects the signs (13) and also rotations inside
//              eigenspaces of nearly equal eigenvalues; and the tracked points (15) are the
//              retrieved points whose aligned embedding is nearest to the embedding of each
//              tracked point at k = 1. The tracked points stay anchored to that embedding.
//   'literal'  (13)-(15) as written: the basis Pi^M(k) Phi_ret(k) C^M_sgn(k) and the points
//              Pi^M(k) X_ret(k) on the rows that Pi^M(k) selects. Pi^M(k) is not one-to-one,
//              so tracked points that select the same retrieved point keep the same
//              embedding from then on: they merge, and the tracking degrades over time.
//   'none'     no tracking: the basis of every new mesh is used as it comes.
import { block, fmapFromMap, mapFromFmap, nearestRows, targetSize, zoomOut } from './fmap.js';
import { icp } from './icp.js';
import { eigenbasis, lumpedArea } from './spectral.js';

export const DEFAULTS = {
    Iinit: 4,          // I_initial: size of C(I, k = 1) before the refinement
    Iupdate: 5,        // I_update: re-used block of C(I_max, k - 1) (Sec. 3.3)
    Imax: 30,          // I_max: largest basis size
    step: 1,           // increment of I in the refinements (1 in the paper, 4 in the code)
    trackFrom: 15,     // first I of the tracking refinement (11)-(12) (15 in the code)
    tracking: 'anchored', // Sec. 3.2: 'anchored' (reference code) | 'literal' (13)-(15) | 'none'
    warmStart: true,   // initialize C(I_update, k) with the block of C(I_max, k - 1)
    slanted: true,     // non-isometry rule (18)-(19) for J
    init: 'icp',       // initialization at k = 1 (Sec. 3.1): 'icp' | 'identity'
};

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const identity = (I) => { const C = new Float64Array(I * I); for (let i = 0; i < I; i++) C[i * I + i] = 1; return C; };
const centered = (V) => {
    const n = V.length / 3, c = [0, 0, 0], out = new Float64Array(V.length);
    for (let m = 0; m < n; m++) for (let e = 0; e < 3; e++) c[e] += V[3 * m + e] / n;
    for (let m = 0; m < n; m++) for (let e = 0; e < 3; e++) out[3 * m + e] = V[3 * m + e] - c[e];
    return out;
};
const rows = (V, idx, dim = 3) => {
    const out = new Float64Array(idx.length * dim);
    for (let m = 0; m < idx.length; m++) for (let e = 0; e < dim; e++) out[dim * m + e] = V[dim * idx[m] + e];
    return out;
};

// Shape error (20): E = X - Pi Y, with ||e|| = sqrt(sum_m ||e_m||^2) (21).
export function shapeError(X, Y, Pi) {
    const M = Pi.length, E = new Float64Array(3 * M), per = new Float64Array(M);
    let s = 0;
    for (let m = 0; m < M; m++) {
        let d = 0;
        for (let e = 0; e < 3; e++) { const v = X[3 * m + e] - Y[3 * Pi[m] + e]; E[3 * m + e] = v; d += v * v; }
        per[m] = Math.sqrt(d); s += d;
    }
    return { E, per, norm: Math.sqrt(s) };
}

// Basis of a mesh {V, F, n} with K eigenvectors (seed: start vector of the eigensolver,
// which fixes the arbitrary signs of the eigenvectors).
export function meshBasis(mesh, K, seed = 1) {
    return { ...mesh, ...eigenbasis(mesh.V, mesh.F, mesh.n, K, { seed }) };
}

// ZoomOut (Melzi et al., 2019) from C(I_initial) = identity, square maps (J = I): the
// non-time-consistent maps Pi_NTC of the paper.
export function classicZoomOut(src, tgt, { Iinit = 4, Imax = 30, step = 1, trace = null } = {}) {
    return zoomOut(src, tgt, { C0: identity(Iinit), I0: Iinit, J0: Iinit, Imax, step, rho: null, trace });
}

// First map (Sec. 3.1): ICP between X and Y, nearest neighbors for Pi, and (10).
export function initialMap(src, tgt, { Iinit, Imax, step, slanted, init = 'icp', trace = null }) {
    const rho = slanted ? Math.sqrt(src.total / tgt.total) : null;
    if (init === 'identity') {
        return { ...zoomOut(src, tgt, { C0: identity(Iinit), I0: Iinit, J0: Iinit, Imax, step, rho, trace }), rho };
    }
    const aligned = icp(src.V, tgt.V);
    const J0 = targetSize(src, tgt, Iinit, rho);
    const C0 = fmapFromMap(src, Iinit, tgt, J0, aligned.pairs);
    return { ...zoomOut(src, tgt, { C0, I0: Iinit, J0, Imax, step, rho, trace }), rho, icp: aligned };
}

export class TimeConsistentMapper {
    // target: {V, F, n} (the target mesh N, points Y)
    constructor(target, options = {}) {
        this.options = { ...DEFAULTS, ...options };
        const Kn = Math.min(target.n, Math.max(2 * this.options.Imax, 40));
        this.target = meshBasis(target, Kn, 11);
        this.seed = 101;
        this.reset();
    }

    reset() { this.k = 0; this.state = null; }

    set(options) { Object.assign(this.options, options); }

    // One frame: the perceived mesh {V, F, n}. Returns the result of Algorithm 1 at k.
    step(frame) {
        const o = this.options, tgt = this.target, k = ++this.k, t0 = now();
        const ret = meshBasis(frame, Math.min(o.Imax, frame.n), this.seed++);
        const t1 = now();
        let src, X, track = null, idx = null;
        if (k === 1) {
            // X(1) = X_ret(1): the first retrieved points are the points to track
            src = ret; X = ret.V;
            this.ref = { F: ret.F, evecs: ret.evecs, K: ret.K, n: ret.n };
        } else if (o.tracking === 'none') {
            src = ret; X = ret.V;
        } else if (o.tracking === 'anchored') {
            track = this.trackAnchored(ret);
            src = track.aligned; X = track.X; idx = track.idx;
        } else {
            track = this.trackLiteral(ret);
            src = track.basis; X = track.X;
        }
        const t2 = now();
        // the map to the target (Sec. 3.3 with the rule of Sec. 3.4)
        const rho = o.slanted ? Math.sqrt(src.total / tgt.total) : null;
        const trace = [];
        let res;
        if (k === 1 || !o.warmStart || !this.state) {
            res = initialMap(src, tgt, { ...o, trace });
        } else {
            const { C: Cp, I: Ip, J: Jp } = this.state;
            const Iu = Math.min(o.Iupdate, Ip, Jp, src.K);
            res = zoomOut(src, tgt, { C0: block(Cp, Ip, Jp, Iu, Iu), I0: Iu, J0: Iu, Imax: o.Imax, step: o.step, rho, trace });
        }
        const t3 = now();
        // map of the tracked points (with the anchored tracking, the map is on the retrieved mesh)
        const Pi = idx ? Int32Array.from(idx, (r) => res.Pi[r]) : res.Pi;
        const error = shapeError(X, tgt.V, Pi);
        this.state = { C: res.C, I: res.I, J: res.J, src, X };
        // the basis on the tracked points (rows of the aligned basis for the anchored tracking)
        const basis = idx ? { evecs: rows(src.evecs, idx, src.K), K: src.K } : { evecs: src.evecs, K: src.K };
        return {
            k, ret, src, X, Pi, PiSrc: res.Pi, C: res.C, I: res.I, J: res.J, rho: rho ?? 1, trace, track, error, basis,
            F: (k === 1 || o.tracking !== 'none') ? this.ref.F : ret.F,
            ms: { basis: t1 - t0, tracking: t2 - t1, mapping: t3 - t2, total: t3 - t0 },
        };
    }

    // Sec. 3.2 as written: (11)-(15) on the tracked points.
    trackLiteral(ret) {
        const o = this.options, prev = this.state.src, M = prev.n, I0 = Math.min(o.trackFrom, o.Imax, ret.K, prev.K);
        // Pi^M(k) initialized by the nearest retrieved point of every tracked point (centered clouds)
        const PiM0 = nearestRows(centered(this.state.X), M, 3, centered(ret.V), ret.n, 3, 3);
        const C0 = fmapFromMap(prev, I0, ret, I0, PiM0);
        const Imax = Math.min(o.Imax, ret.K, prev.K);
        const r = zoomOut(prev, ret, { C0, I0, J0: I0, Imax, step: o.step, rho: null });
        const PiM = r.Pi;
        const CM = fmapFromMap(prev, Imax, ret, Imax, PiM);                     // (12) at I_max
        const sgn = Float64Array.from({ length: Imax }, (_, i) => (CM[i * Imax + i] < 0 ? -1 : 1));   // (13)
        const K = Imax, evecs = new Float64Array(M * K);
        for (let m = 0; m < M; m++) {
            const om = m * K, or = PiM[m] * ret.K;
            for (let i = 0; i < K; i++) evecs[om + i] = ret.evecs[or + i] * sgn[i];   // (14)
        }
        const X = rows(ret.V, PiM);                                               // (15)
        const { area, total } = lumpedArea(X, this.ref.F, M);
        const basis = { V: X, F: this.ref.F, n: M, K, evecs, evals: ret.evals.slice(0, K), area, total };
        return { PiM, CM, sgn, basis, X, I: Imax };
    }

    // Sec. 3.2 as in the reference code: align the basis of the retrieved mesh with the
    // previous (aligned) one, and find the tracked points by their embedding at k = 1.
    trackAnchored(ret) {
        const o = this.options, prev = this.state.src, I0 = Math.min(o.trackFrom, o.Imax, ret.K, prev.K);
        const Imax = Math.min(o.Imax, ret.K, prev.K);
        // map from the retrieved points to the previous retrieved points (nearest neighbors of
        // the centered clouds, then the coarse-to-fine refinement)
        const T0 = nearestRows(centered(ret.V), ret.n, 3, centered(prev.V), prev.n, 3, 3);
        const C0 = fmapFromMap(ret, I0, prev, I0, T0);
        const r = zoomOut(ret, prev, { C0, I0, J0: I0, Imax, step: o.step, rho: null });
        const CM = fmapFromMap(ret, Imax, prev, Imax, r.Pi);                    // Phi_ret^T A_ret T Phi_prev
        const K = Imax, evecs = new Float64Array(ret.n * K);
        for (let p = 0; p < ret.n; p++) {
            for (let j = 0; j < K; j++) { let s = 0; for (let i = 0; i < K; i++) s += ret.evecs[p * ret.K + i] * CM[i * K + j]; evecs[p * K + j] = s; }
        }
        const aligned = { ...ret, evecs, K };
        // tracked points: nearest aligned embedding to the embedding of each point at k = 1
        const idx = nearestRows(this.ref.evecs, this.ref.n, this.ref.K, evecs, ret.n, K, Math.min(K, this.ref.K));
        const X = rows(ret.V, idx);
        const sgn = Float64Array.from({ length: K }, (_, i) => (CM[i * K + i] < 0 ? -1 : 1));
        return { aligned, idx, X, CM, sgn, I: K, PiM: idx };
    }
}

// The non-time-consistent baseline at one frame: ZoomOut from the identity on the retrieved
// mesh (its own basis), and the error of the retrieved points.
export function ntcStep(ret, target, { Iinit = 4, Imax = 30, step = 1 } = {}) {
    const t0 = now();
    const r = classicZoomOut(ret, target, { Iinit, Imax: Math.min(Imax, ret.K), step });
    return { ...r, error: shapeError(ret.V, target.V, r.Pi), ms: now() - t0 };
}
