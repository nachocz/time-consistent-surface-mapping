// The simulated scene: a foam-like piece (js/sim/lattice.js) held by grippers and
// deformed with ARAP (js/sim/arap.js), as in the simulations of Sec. 4.2. Lengths are in
// cm. The grippers hold the lattice nodes inside their boxes; a gripper pose is a rigid
// transform {R, t} of its box. Targets are produced by the same simulator, by moving the
// grippers from the rest shape to prescribed poses in small increments (quasi-static
// loading), so that every target is reachable.
import { Arap } from './arap.js';
import { mulVec3, rotationFromVector, transpose3 } from './linalg.js';
import { disc, graspedNodes, makeLattice, rectangle } from './lattice.js';
import { aboutPoint, apply, compose, inverse, orthonormalize, power } from './se3.js';

const DEG = Math.PI / 180;

// Pieces of the experiments (Sec. 4.3) and of the simulations (Fig. 8). Gripper specs are
// lattice nodes (i, j) with half-sizes (di, dj) in cells; poses relative to the rest shape:
// rotation vector `rot` (rad) about the gripper center, then translation `move` (cm).
// `target`: the target shape of the mapping pages; `goal`: the farther (more deformed)
// target shape of the shape-control page, the same deformation about 1.35 times larger.
export const SCENARIOS = {
    sheet: {
        title: 'Foam sheet',
        note: 'a rectangular sheet with two symmetry axes, folded and twisted (Fig. 13)',
        shape: { nu: 31, nv: 21, nz: 2, h: 1.0, chart: rectangle(24, 16) },
        grippers: [{ i: 1, j: 10, di: 1, dj: 2 }, { i: 29, j: 10, di: 1, dj: 2 }],
        target: [{ rot: [10 * DEG, -40 * DEG, 0], move: [2.5, 0, 1.5] }, { rot: [-10 * DEG, 40 * DEG, 0], move: [-2.5, 0, 1.5] }],
        goal: [{ rot: [14 * DEG, -54 * DEG, 0], move: [3.4, 0, 2] }, { rot: [-14 * DEG, 54 * DEG, 0], move: [-3.4, 0, 2] }],
        motion: [{ rot: [25 * DEG, -35 * DEG, 0], move: [2, 1, 1] }, { rot: [-25 * DEG, 35 * DEG, 0], move: [-2, -1, 1] }],
        view: { from: [8, -34, 26], at: [0, 0, 1] },
    },
    hat: {
        title: 'Mexican hat',
        note: 'a disc with radial symmetry, bent by one gripper while the other holds (Fig. 10)',
        shape: { nu: 27, nv: 27, nz: 2, h: 1.0, chart: disc(10) },
        grippers: [{ i: 1, j: 13, di: 1, dj: 2 }, { i: 25, j: 13, di: 1, dj: 2 }],
        fixed: [true, false],
        target: [{ rot: [0, 0, 0], move: [0, 0, 0] }, { rot: [0, 55 * DEG, 0], move: [-4, 0, 3] }],
        goal: [{ rot: [0, 0, 0], move: [0, 0, 0] }, { rot: [0, 74 * DEG, 0], move: [-5.4, 0, 4] }],
        motion: [{ rot: [0, 0, 0], move: [0, 0, 0] }, { rot: [30 * DEG, 50 * DEG, 0], move: [-3.5, 1.5, 2.5] }],
        view: { from: [6, -30, 24], at: [0, 0, 1] },
    },
    star: {
        title: 'Sea star',
        note: 'a five-armed star bent and twisted by two arms (Fig. 8, last simulation)',
        shape: { nu: 29, nv: 29, nz: 2, h: 1.0, chart: disc(10, (th) => 0.78 + 0.32 * Math.cos(5 * (th - Math.PI / 2))) },
        grippers: [{ i: 3, j: 6, di: 1, dj: 1 }, { i: 25, j: 6, di: 1, dj: 1 }],
        target: [{ rot: [-20 * DEG, -35 * DEG, 0], move: [1.5, 0.5, 2] }, { rot: [-20 * DEG, 35 * DEG, 0], move: [-1.5, 0.5, 2] }],
        goal: [{ rot: [-27 * DEG, -47 * DEG, 0], move: [2, 0.7, 2.7] }, { rot: [-27 * DEG, 47 * DEG, 0], move: [-2, 0.7, 2.7] }],
        motion: [{ rot: [20 * DEG, -30 * DEG, 0], move: [1.5, 1, 1.5] }, { rot: [-20 * DEG, 30 * DEG, 0], move: [-1.5, -1, 1.5] }],
        view: { from: [4, -30, 28], at: [0, 0, 0] },
    },
    tee: {
        title: 'T-shaped piece',
        note: 'a T cut-out (like the T-shaped pool noodle), twisted and bent (Fig. 11)',
        shape: {
            nu: 31, nv: 25, nz: 2, h: 1.2, chart: rectangle(24, 20),
            mask: (u, v) => v > 0.62 || (u > 0.36 && u < 0.64),
        },
        grippers: [{ i: 15, j: 1, di: 2, dj: 1 }, { i: 1, j: 21, di: 1, dj: 2 }],
        target: [{ rot: [40 * DEG, 0, 0], move: [0, 2, 2.5] }, { rot: [0, -40 * DEG, 0], move: [2, 0, 2.5] }],
        goal: [{ rot: [54 * DEG, 0, 0], move: [0, 2.7, 3.4] }, { rot: [0, -54 * DEG, 0], move: [2.7, 0, 3.4] }],
        motion: [{ rot: [40 * DEG, 0, 20 * DEG], move: [0, 2, 2.5] }, { rot: [10 * DEG, -45 * DEG, 0], move: [2, 0, 2] }],
        view: { from: [10, -30, 28], at: [0, 1, 1] },
    },
};

const relative = (spec, center) => aboutPoint(rotationFromVector(spec.rot || [0, 0, 0]), center, spec.move || [0, 0, 0]);

export class World {
    // `goal`: use the scenario's goal (shape control) instead of its target as the target shape
    constructor(id, { arapIterations = 8, goal = false } = {}) {
        const sc = SCENARIOS[id];
        this.id = id; this.scenario = sc;
        this.arapIterations = arapIterations;
        const lat = makeLattice(sc.shape);
        this.lattice = lat;
        this.grippers = sc.grippers.map((spec, k) => {
            const nodes = graspedNodes(lat, spec);
            const c = [0, 1, 2].map((e) => nodes.reduce((s, x) => s + lat.rest[3 * x + e], 0) / nodes.length);
            const R = [1, 0, 0, 0, 1, 0, 0, 0, 1];
            const Rt = transpose3(R);
            const local = Array.from(nodes, (x) => mulVec3(Rt, [lat.rest[3 * x] - c[0], lat.rest[3 * x + 1] - c[1], lat.rest[3 * x + 2] - c[2]]));
            const half = [0, 1, 2].map((e) => Math.max(...local.map((p) => Math.abs(p[e]))) + (e === 2 ? 0.45 : 0.4));
            return { ...spec, nodes, local, restPose: { R, t: c }, half, fixed: !!(sc.fixed && sc.fixed[k]) };
        });
        const handles = new Set();
        for (const g of this.grippers) for (const x of g.nodes) handles.add(x);
        this.arap = new Arap(lat.rest, lat.neighbors, handles);
        this.blend = this.blendWeights();
        this.restSpec = this.grippers.map(() => ({ rot: [0, 0, 0], move: [0, 0, 0] }));
        this.initial = { P: lat.rest.slice(), poses: this.grippers.map((g) => g.restPose), quats: this.arap.snapshot() };
        this.setTarget(goal && sc.goal ? sc.goal : sc.target);
        this.reset();
    }

    get G() { return this.grippers.length; }

    placeHandles(P, poses) {
        this.grippers.forEach((g, k) => {
            g.nodes.forEach((x, a) => {
                const p = apply(poses[k], g.local[a]);
                P[3 * x] = p[0]; P[3 * x + 1] = p[1]; P[3 * x + 2] = p[2];
            });
        });
    }

    // Weights of the predictor: each node follows the grippers with weights that decay
    // with the distance along the lattice edges (rest shape) to the grasped nodes.
    blendWeights() {
        const { n, rest, neighbors } = this.lattice, G = this.grippers.length, W = new Float64Array(n * G);
        this.grippers.forEach((g, k) => {
            const d = new Float64Array(n).fill(Infinity), done = new Uint8Array(n), heap = [];
            const push = (v) => { heap.push(v); let i = heap.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (d[heap[p]] <= d[heap[i]]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; } };
            const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let s = i; if (l < heap.length && d[heap[l]] < d[heap[s]]) s = l; if (r < heap.length && d[heap[r]] < d[heap[s]]) s = r; if (s === i) break; [heap[s], heap[i]] = [heap[i], heap[s]]; i = s; } } return top; };
            for (const x of g.nodes) { d[x] = 0; push(x); }
            while (heap.length) {
                const v = pop();
                if (done[v]) continue;
                done[v] = 1;
                for (const u of neighbors[v]) {
                    const l = Math.hypot(rest[3 * u] - rest[3 * v], rest[3 * u + 1] - rest[3 * v + 1], rest[3 * u + 2] - rest[3 * v + 2]);
                    if (d[v] + l < d[u]) { d[u] = d[v] + l; push(u); }
                }
            }
            for (let i = 0; i < n; i++) W[i * G + k] = 1 / (d[i] + 0.5) ** 2;
        });
        for (let i = 0; i < n; i++) {
            let s = 0;
            for (let k = 0; k < G; k++) s += W[i * G + k];
            for (let k = 0; k < G; k++) W[i * G + k] /= s;
        }
        return W;
    }

    // Initial guess after the grippers move by U (world frame): every free node follows
    // the blend of the gripper motions, and the rotation of every ARAP cell (the warm
    // start of the local step) turns with the gripper of largest weight (both exact for
    // rigid motions).
    predict(P, U) {
        const { n } = this.lattice, G = U.length, W = this.blend, isHandle = this.arap.isHandle, Q = this.arap.quats;
        const q = U.map(({ R }) => {                    // unit quaternion (w, x, y, z) of R
            const tr = R[0] + R[4] + R[8];
            let w, x, y, z;
            if (tr > 0) { const s = 2 * Math.sqrt(tr + 1); w = s / 4; x = (R[7] - R[5]) / s; y = (R[2] - R[6]) / s; z = (R[3] - R[1]) / s; }
            else if (R[0] > R[4] && R[0] > R[8]) { const s = 2 * Math.sqrt(1 + R[0] - R[4] - R[8]); w = (R[7] - R[5]) / s; x = s / 4; y = (R[1] + R[3]) / s; z = (R[2] + R[6]) / s; }
            else if (R[4] > R[8]) { const s = 2 * Math.sqrt(1 + R[4] - R[0] - R[8]); w = (R[2] - R[6]) / s; x = (R[1] + R[3]) / s; y = s / 4; z = (R[5] + R[7]) / s; }
            else { const s = 2 * Math.sqrt(1 + R[8] - R[0] - R[4]); w = (R[3] - R[1]) / s; x = (R[2] + R[6]) / s; y = (R[5] + R[7]) / s; z = s / 4; }
            return [w, x, y, z];
        });
        for (let i = 0; i < n; i++) {
            let k = 0;
            for (let g = 1; g < G; g++) if (W[i * G + g] > W[i * G + k]) k = g;
            const [a0, a1, a2, a3] = q[k], o = 4 * i, b0 = Q[o], b1 = Q[o + 1], b2 = Q[o + 2], b3 = Q[o + 3];
            Q[o] = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3; Q[o + 1] = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2;
            Q[o + 2] = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1; Q[o + 3] = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0;
        }
        for (let i = 0; i < n; i++) {
            if (isHandle[i]) continue;
            const p = [P[3 * i], P[3 * i + 1], P[3 * i + 2]], q = [0, 0, 0];
            for (let k = 0; k < G; k++) {
                const a = apply(U[k], p), w = W[i * G + k];
                q[0] += w * a[0]; q[1] += w * a[1]; q[2] += w * a[2];
            }
            P[3 * i] = q[0]; P[3 * i + 1] = q[1]; P[3 * i + 2] = q[2];
        }
    }

    // Quasi-static loading from the rest shape to the gripper poses of `specs`
    // (or from the state `from`, if given). Returns {P, poses, quats}.
    load(specs, { steps = 24, iterations = 12, from = null } = {}) {
        const base = from || { P: this.lattice.rest, poses: this.grippers.map((g) => g.restPose), quats: null };
        const P = base.P.slice();
        const goal = specs.map((s, k) => compose(relative(s, this.grippers[k].restPose.t), this.grippers[k].restPose));
        const moves = goal.map((T, k) => compose(T, inverse(base.poses[k])));
        const saved = this.arap.snapshot();
        if (base.quats) this.arap.restore(base.quats);
        else { this.arap.quats.fill(0); for (let i = 0; i < this.arap.n; i++) this.arap.quats[4 * i] = 1; }
        let poses = base.poses;
        for (let s = 1; s <= steps; s++) {
            const next = moves.map((D, k) => orthonormalize(compose(power(D, s / steps), base.poses[k])));
            this.predict(P, next.map((T, k) => compose(T, inverse(poses[k]))));
            poses = next;
            this.placeHandles(P, poses);
            this.arap.solve(P, s === steps ? 4 * iterations : iterations);
        }
        const state = { P, poses, quats: this.arap.snapshot() };
        this.arap.restore(saved);
        return state;
    }

    setTarget(specs) {
        this.targetSpec = specs;
        this.target = this.load(specs);
    }

    reset() {
        this.P = this.initial.P.slice();
        this.poses = this.initial.poses.map((T) => ({ R: T.R.slice(), t: T.t.slice() }));
        this.arap.restore(this.initial.quats);
    }

    // Move every gripper by its rigid transform U_g (world frame) and let the object settle.
    move(U, iterations = this.arapIterations) {
        this.poses = this.poses.map((T, k) => orthonormalize(compose(U[k], T)));
        this.predict(this.P, U);
        this.placeHandles(this.P, this.poses);
        this.arap.solve(this.P, iterations);
    }

    // Drive the grippers towards the poses `specs` (relative to the rest shape) by the
    // fraction s of the remaining way along the SE(3) geodesic.
    moveTowards(specs, s, iterations = this.arapIterations) {
        const U = specs.map((spec, k) => {
            const goal = compose(relative(spec, this.grippers[k].restPose.t), this.grippers[k].restPose);
            return power(compose(goal, inverse(this.poses[k])), s);
        });
        this.move(U, iterations);
    }

    // The simulated response to the gripper motions U (without changing the state).
    response(U, iterations = 30) {
        const P = this.P.slice(), saved = this.arap.snapshot();
        this.predict(P, U);
        this.placeHandles(P, this.poses.map((T, k) => compose(U[k], T)));
        this.arap.solve(P, iterations);
        this.arap.restore(saved);
        return P;
    }

    // Noise-free top surface (lattice top nodes) of the positions P.
    topSurface(P = this.P) {
        const { top } = this.lattice, out = new Float64Array(3 * top.length);
        for (let m = 0; m < top.length; m++) for (let e = 0; e < 3; e++) out[3 * m + e] = P[3 * top[m] + e];
        return out;
    }
}

// Gripper action u_g = (dt, dr) (translation in cm, Rodrigues vector in rad) as the rigid
// transform that rotates by dr about the gripper center c and translates by dt.
export const actionTransform = (u, c) => aboutPoint(rotationFromVector([u[3], u[4], u[5]]), c, [u[0], u[1], u[2]]);
