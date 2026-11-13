// One closed-loop run of Sec. 4: the simulated object (ARAP), the camera that perceives a
// new mesh at every iteration, the time-consistent surface mapping, and the sequential
// shape controller. An iteration executes the pending gripper action, perceives the new
// mesh, maps it, stores the response in the buffer of the gripper that acted, and
// computes the next action, so that the state and its analysis always correspond. The
// maps of ZoomOut computed anew at every iteration give the non-time-consistent error
// e_NTC, which is recorded but never fed to the controller, as in the paper.
import { Camera } from './camera.js';
import { SequentialShapeController, predict } from './control.js';
import { identity } from './se3.js';
import { TimeConsistentMapper, ntcStep } from './tcmap.js';
import { actionTransform } from './world.js';

export class ControlSession {
    constructor(world, { points = 320, noise = 0.05, seed = 3, mapper = {}, control = {}, ntc = true } = {}) {
        this.world = world;
        this.settings = { points, noise, seed, mapper: { ...mapper }, control: { ...control }, ntc };
        this.restart();
    }

    restart() {
        const { world, settings } = this;
        world.reset();
        this.camera = new Camera(world.lattice, { points: settings.points, noise: settings.noise, seed: settings.seed });
        // the target mesh, acquired once by the same camera (pre-acquired in the experiments)
        this.targetFrame = this.camera.capture(world.target.P);
        this.mapper = new TimeConsistentMapper(this.targetFrame, settings.mapper);
        this.k = 0;
        this.out = this.perceive();
        this.M = this.out.X.length / 3;
        this.controller = new SequentialShapeController(world.G, this.M, settings.control, world.grippers.map((g) => !g.fixed));
        this.pending = this.controller.next(this.out.error.E);
        this.history = { k: [0], e: [this.out.error.norm], eNtc: [this.ntc?.error.norm ?? NaN], per: [this.out.error.per], perNtc: [this.ntc?.error.per ?? null],
            u: [], g: [], explore: [], fit: [], mret: [this.out.ret.n], ms: [this.out.ms.total], msNtc: [this.ntc?.ms ?? NaN] };
    }

    perceive() {
        const frame = this.camera.capture(this.world.P, { points: this.settings.points, noise: this.settings.noise });
        const out = this.mapper.step(frame);
        this.ntc = this.settings.ntc ? ntcStep(out.ret, this.mapper.target, { Iinit: this.mapper.options.Iinit, Imax: this.mapper.options.Imax, step: this.mapper.options.step }) : null;
        return out;
    }

    // Prediction of the controller's model for the pending action (x + L_g u_g), or null.
    prediction() {
        const { pending, out } = this;
        return pending?.L ? predict(out.X, pending.L, pending.u) : null;
    }

    step() {
        const { world, pending } = this, prev = this.out;
        const U = world.grippers.map((_, i) => (i === pending.g ? actionTransform(pending.u, world.poses[i].t) : identity()));
        world.move(U);
        this.k += 1;
        this.out = this.perceive();
        const X = this.out.X, dx = new Float64Array(X.length);
        for (let i = 0; i < X.length; i++) dx[i] = X[i] - prev.X[i];
        // how well the model predicted the response: || dx - L u || / || dx ||
        let fit = NaN;
        if (pending.L) {
            const p = predict(prev.X, pending.L, pending.u);
            let num = 0, den = 0;
            for (let i = 0; i < X.length; i++) { num += (X[i] - p[i]) ** 2; den += dx[i] ** 2; }
            fit = Math.sqrt(num / Math.max(den, 1e-12));
        }
        this.controller.record(pending.g, pending.u, dx);
        const h = this.history;
        h.k.push(this.k); h.e.push(this.out.error.norm); h.eNtc.push(this.ntc?.error.norm ?? NaN);
        h.per.push(this.out.error.per); h.perNtc.push(this.ntc?.error.per ?? null);
        h.u.push(pending.u.slice()); h.g.push(pending.g); h.explore.push(!!pending.explore); h.fit.push(fit);
        h.mret.push(this.out.ret.n); h.ms.push(this.out.ms.total); h.msNtc.push(this.ntc?.ms ?? NaN);
        this.pending = this.controller.next(this.out.error.E);
    }
}
