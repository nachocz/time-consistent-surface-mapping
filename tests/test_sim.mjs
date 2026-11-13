// Checks of the simulation modules of the demos (docs/js/sim), with Node.js:
//     node tests/test_sim.mjs            (about a minute)
// The eigensolver against the analytic spectrum of a rectangle, ZoomOut on two samplings
// of the same surface, the camera, ARAP under rigid gripper motions, the time-consistent
// maps along a deformation sequence (against the ground truth of the simulator and
// against ZoomOut at every frame), the merging of tracked points with (13)-(15) read
// literally, the rule (19) on a compressed beam, and the closed loop of Sec. 4.
import assert from 'node:assert/strict';
import { Camera } from '../docs/js/sim/camera.js';
import { makeLattice, rectangle } from '../docs/js/sim/lattice.js';
import { rotationFromVector } from '../docs/js/sim/linalg.js';
import { ControlSession } from '../docs/js/sim/session.js';
import { eigenbasis, tridiagonalEigen } from '../docs/js/sim/spectral.js';
import { TimeConsistentMapper, initialMap, meshBasis, ntcStep } from '../docs/js/sim/tcmap.js';
import { RECTANGLE_SYMMETRIES, matchError } from '../docs/js/sim/truth.js';
import { SCENARIOS, World } from '../docs/js/sim/world.js';
import { aboutPoint } from '../docs/js/sim/se3.js';

let failures = 0;
function check(name, fn) {
    const t0 = performance.now();
    try { const note = fn(); console.log(`ok    ${name} (${(performance.now() - t0).toFixed(0)} ms)${note ? `  ${note}` : ''}`); }
    catch (e) { failures++; console.log(`FAIL  ${name}\n      ${e.stack.split('\n').slice(0, 3).join('\n      ')}`); }
}
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;

check('tridiagonal QL: eigenpairs of a known matrix', () => {
    // tridiag(-1, 2, -1) of size n: lambda_k = 2 - 2 cos(k pi / (n + 1))
    const n = 12, d = new Float64Array(n).fill(2), e = new Float64Array(n).fill(-1);
    const { values } = tridiagonalEigen(d, e);
    for (let k = 1; k <= n; k++) assert.ok(Math.abs(values[k - 1] - (2 - 2 * Math.cos(k * Math.PI / (n + 1)))) < 1e-12);
});

check('Laplace-Beltrami basis of a rectangle: analytic eigenvalues, A-orthonormal', () => {
    const lat = makeLattice({ nu: 41, nv: 21, nz: 1, h: 0, chart: rectangle(20, 10) });
    const B = eigenbasis(lat.rest, lat.topTris, lat.n, 30, { seed: 3 });
    const exact = [];
    for (let p = 0; p < 12; p++) for (let q = 0; q < 8; q++) exact.push(Math.PI ** 2 * (p * p / 400 + q * q / 100));
    exact.sort((a, b) => a - b);
    for (let k = 1; k < 12; k++) assert.ok(Math.abs(B.evals[k] / exact[k] - 1) < 0.03, `lambda_${k}: ${B.evals[k]} vs ${exact[k]}`);
    let worst = 0;
    for (let a = 0; a < B.K; a++) for (let b = 0; b < B.K; b++) {
        let s = 0;
        for (let i = 0; i < B.n; i++) s += B.evecs[i * B.K + a] * B.area[i] * B.evecs[i * B.K + b];
        worst = Math.max(worst, Math.abs(s - (a === b ? 1 : 0)));
    }
    assert.ok(worst < 1e-9, `orthonormality ${worst}`);
});

check('camera: a new mesh per frame (shuffled), the same samples for a static object', () => {
    const w = new World('sheet'), cam = new Camera(w.lattice, { points: 300, noise: 0 });
    const a = cam.capture(w.P), b = cam.capture(w.P);
    assert.equal(a.n, b.n);
    const key = (f) => Array.from({ length: f.n }, (_, i) => `${f.V[3 * i].toFixed(6)},${f.V[3 * i + 1].toFixed(6)}`).sort().join(';');
    assert.equal(key(a), key(b), 'same samples');
    assert.notDeepEqual(Array.from(a.V.slice(0, 30)), Array.from(b.V.slice(0, 30)), 'different order');
    const folded = cam.capture(w.target.P);
    assert.notEqual(folded.n, a.n, 'the number of points changes with the shape');
    return `M_ret: ${a.n} flat, ${folded.n} folded`;
});

check('ARAP: a rigid motion of both grippers moves the object rigidly', () => {
    const w = new World('tee'), T = aboutPoint(rotationFromVector([0.2, -0.3, 0.5]), [1, 2, 0], [3, -1, 2]);
    w.move(w.grippers.map(() => T));
    let worst = 0;
    const { rest } = w.lattice;
    for (let i = 0; i < w.lattice.n; i++) {
        const p = [rest[3 * i], rest[3 * i + 1], rest[3 * i + 2]], R = T.R;
        const q = [0, 1, 2].map((r) => R[3 * r] * p[0] + R[3 * r + 1] * p[1] + R[3 * r + 2] * p[2] + T.t[r]);
        worst = Math.max(worst, Math.hypot(q[0] - w.P[3 * i], q[1] - w.P[3 * i + 1], q[2] - w.P[3 * i + 2]));
    }
    assert.ok(worst < 1e-9, `deviation ${worst}`);
    return `largest deviation from the rigid motion ${worst.toExponential(1)} cm`;
});

check('ZoomOut with the ICP initialization recovers the map between two samplings of a shape', () => {
    const w = new World('star'), chart = SCENARIOS.star.shape.chart;
    const a = meshBasis(new Camera(w.lattice, { points: 300, seed: 1, offset: [0.1, 0.2] }).capture(w.target.P), 30, 1);
    const b = meshBasis(new Camera(w.lattice, { points: 340, seed: 2, offset: [0.6, 0.7] }).capture(w.target.P), 60, 2);
    const r = initialMap(a, b, { Iinit: 4, Imax: 30, step: 1, slanted: true });
    const err = matchError(a, b, r.Pi, chart);
    assert.ok(err < 0.8, `error to the true match ${err.toFixed(2)} cm`);
    return `error ${err.toFixed(2)} cm`;
});

function sequence(tracking, frames = 80, scenario = 'sheet') {
    const w = new World(scenario), cam = new Camera(w.lattice, { points: 320, noise: 0.05, seed: 5 }), chart = SCENARIOS[scenario].shape.chart;
    const mapper = new TimeConsistentMapper(cam.capture(w.target.P), { step: 4, tracking });
    const out = { truth: [], truthNtc: [], e: [], eNtc: [], distinct: [] };
    for (let k = 0; k < frames; k++) {
        const a = Math.sin(Math.PI * k / frames) ** 2;
        w.moveTowards(SCENARIOS[scenario].motion.map((s) => ({ rot: s.rot.map((x) => a * x), move: s.move.map((x) => a * x) })), 1);
        const f = cam.capture(w.P), r = mapper.step(f), idx = r.track?.PiM;
        const uv = idx ? Float64Array.from({ length: 2 * idx.length }, (_, i) => f.uv[2 * idx[i >> 1] + (i & 1)]) : f.uv;
        out.truth.push(matchError({ uv }, mapper.target, r.Pi, chart)); out.e.push(r.error.norm);
        out.distinct.push(idx ? new Set(idx).size / idx.length : 1);
        if (tracking === 'anchored') {
            const n = ntcStep(r.ret, mapper.target, { step: 4 });
            out.truthNtc.push(matchError(f, mapper.target, n.Pi, chart)); out.eNtc.push(n.error.norm);
        }
    }
    return out;
}
const maxJump = (a) => Math.max(...a.slice(1).map((x, i) => Math.abs(x - a[i])));

check('time-consistent maps along a deformation: accurate and smooth; ZoomOut every frame jumps', () => {
    const s = sequence('anchored');
    assert.ok(mean(s.truth) < 1 && Math.max(...s.truth) < 1.5, `ours: mean ${mean(s.truth)}, max ${Math.max(...s.truth)}`);
    assert.ok(mean(s.truthNtc) > 3, `NTC mean ${mean(s.truthNtc)}`);
    assert.ok(maxJump(s.e) < 10 && maxJump(s.eNtc) > 50, `jumps of ||e||: ${maxJump(s.e)} vs ${maxJump(s.eNtc)}`);
    return `true-match error ${mean(s.truth).toFixed(2)} vs ${mean(s.truthNtc).toFixed(2)} cm; largest jump of the error ${maxJump(s.e).toFixed(1)} vs ${maxJump(s.eNtc).toFixed(1)}`;
});

check('(13)-(15) read literally: tracked points merge', () => {
    const s = sequence('literal', 80), d = s.distinct;
    assert.ok(d[79] < 0.7 && d[79] < d[20] && d[20] < d[1], `distinct fraction ${d[1]} ${d[20]} ${d[79]}`);
    return `distinct tracked points: ${(100 * d[1]).toFixed(0)} % at k = 2, ${(100 * d[20]).toFixed(0)} % at k = 21, ${(100 * d[79]).toFixed(0)} % at k = 80`;
});

check('rule (19) on a compressed, bent beam: closer to the true match than J = I', () => {
    const beam = (len, bend, nu, nv) => {
        const lat = makeLattice({ nu, nv, nz: 1, h: 0, chart: rectangle(len, 6) }), V = lat.rest.slice(), R = len / bend;
        for (let i = 0; i < lat.n; i++) { const x = lat.rest[3 * i]; V[3 * i] = R * Math.sin(x / R); V[3 * i + 2] = R * (1 - Math.cos(x / R)); }
        return { V, F: lat.topTris, uv: lat.uv, n: lat.n };
    };
    const flat = makeLattice({ nu: 33, nv: 9, nz: 1, h: 0, chart: rectangle(24, 6) });
    const src = meshBasis({ V: flat.rest, F: flat.topTris, uv: flat.uv, n: flat.n }, 30, 3), tgt = meshBasis(beam(24 * 0.7, 1, 25, 10), 60, 5);
    const err = (slanted) => matchError(src, tgt, initialMap(src, tgt, { Iinit: 4, Imax: 30, step: 1, slanted }).Pi, rectangle(24, 6), RECTANGLE_SYMMETRIES);
    const ours = err(true), zo = err(false);
    assert.ok(ours < zo - 0.5, `${ours} vs ${zo}`);
    return `${ours.toFixed(2)} vs ${zo.toFixed(2)} cm`;
});

check('closed loop (Sec. 4): the shape error decreases on the T-shaped piece', () => {
    const s = new ControlSession(new World('tee', { goal: true }), { mapper: { step: 4 }, ntc: false });
    for (let k = 0; k < 140; k++) s.step();
    const e = s.history.e, last = mean(e.slice(-20));
    assert.ok(last < 0.3 * e[0], `${e[0]} -> ${last}`);
    return `||e|| ${e[0].toFixed(1)} -> ${last.toFixed(1)} cm (RMS ${(last / Math.sqrt(s.M)).toFixed(2)} cm)`;
});

if (failures) { console.log(`\n${failures} check(s) failed`); process.exit(1); }
console.log('\nall checks passed');
