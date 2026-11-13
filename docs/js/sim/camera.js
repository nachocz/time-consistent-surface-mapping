// Perception: the mesh M_ret(k) that the method receives at every frame. In the
// experiments of the paper, an RGB-D camera above the robots sees the object, which is
// segmented by color, downsampled on a grid of 2.5 cm, and meshed
// (objectsurfacemeshing.m). Every frame brings a new set of points: their number M_ret(k)
// changes, their order is arbitrary, and they carry measurement noise.
//
// The camera of the demos is a depth camera looking down (orthographic): it samples the
// top surface of the simulated object on a pixel grid fixed in the world, keeps the
// nearest surface along each pixel ray, and joins neighboring pixels into triangles, as
// depth images are meshed. As the object moves, the samples slide over its surface and
// the pixels that see it change; the points are then shuffled, and Gaussian noise of
// standard deviation sigma is added to every coordinate.
import { rng } from './linalg.js';

export class Camera {
    // options: points (number of points on the rest shape, which sets the pixel size),
    // noise (cm), seed, offset ([x, y] shift of the pixel grid, in pixels)
    constructor(lattice, { points = 320, noise = 0.05, seed = 7, offset = [0.37, 0.61] } = {}) {
        this.lattice = lattice;
        this.noise = noise;
        this.random = rng(seed);
        this.offset = offset;
        // pixel size: `points` pixels on the area of the rest shape
        const { rest, top, topTris } = lattice;
        let area = 0;
        for (let t = 0; t < topTris.length; t += 3) {
            const [a, b, c] = [topTris[t], topTris[t + 1], topTris[t + 2]].map((m) => top[m]);
            area += 0.5 * Math.abs((rest[3 * b] - rest[3 * a]) * (rest[3 * c + 1] - rest[3 * a + 1]) - (rest[3 * c] - rest[3 * a]) * (rest[3 * b + 1] - rest[3 * a + 1]));
        }
        this.area = area;
        this.setPoints(points);
    }

    setPoints(points) { this.points = points; this.pixel = Math.sqrt(this.area / points); }

    gauss() {
        const r = this.random;
        return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
    }

    // The mesh perceived from the lattice positions P: {V (3n), F (3f), uv (2n), n}.
    // uv are the surface parameters of the points, used only to draw and to evaluate
    // the maps against the ground truth, never by the method.
    capture(P, { noise = this.noise, shuffle = true } = {}) {
        const { lattice, pixel: h } = this, { top, topTris, uv: UV } = lattice;
        const ox = this.offset[0] * h, oy = this.offset[1] * h;
        let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
        for (const p of top) { x0 = Math.min(x0, P[3 * p]); x1 = Math.max(x1, P[3 * p]); y0 = Math.min(y0, P[3 * p + 1]); y1 = Math.max(y1, P[3 * p + 1]); }
        const i0 = Math.floor((x0 - ox) / h) - 1, j0 = Math.floor((y0 - oy) / h) - 1;
        const W = Math.ceil((x1 - ox) / h) + 2 - i0, H = Math.ceil((y1 - oy) / h) + 2 - j0;
        const depth = new Float64Array(W * H).fill(-Infinity), su = new Float64Array(W * H), sv = new Float64Array(W * H);
        // rasterize the triangles of the top surface (nearest = highest z)
        for (let t = 0; t < topTris.length; t += 3) {
            const m = [topTris[t], topTris[t + 1], topTris[t + 2]], p = m.map((x) => top[x]);
            const ax = P[3 * p[0]], ay = P[3 * p[0] + 1], bx = P[3 * p[1]], by = P[3 * p[1] + 1], cx = P[3 * p[2]], cy = P[3 * p[2] + 1];
            const det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
            if (Math.abs(det) < 1e-12) continue;
            const ia = Math.ceil((Math.min(ax, bx, cx) - ox) / h), ib = Math.floor((Math.max(ax, bx, cx) - ox) / h);
            const ja = Math.ceil((Math.min(ay, by, cy) - oy) / h), jb = Math.floor((Math.max(ay, by, cy) - oy) / h);
            for (let i = ia; i <= ib; i++) for (let j = ja; j <= jb; j++) {
                const x = ox + i * h, y = oy + j * h;
                const l1 = ((x - ax) * (cy - ay) - (cx - ax) * (y - ay)) / det, l2 = ((bx - ax) * (y - ay) - (x - ax) * (by - ay)) / det, l0 = 1 - l1 - l2;
                if (l0 < -1e-9 || l1 < -1e-9 || l2 < -1e-9) continue;
                const z = l0 * P[3 * p[0] + 2] + l1 * P[3 * p[1] + 2] + l2 * P[3 * p[2] + 2], k = (i - i0) * H + (j - j0);
                if (z > depth[k]) {
                    depth[k] = z;
                    su[k] = l0 * UV[2 * m[0]] + l1 * UV[2 * m[1]] + l2 * UV[2 * m[2]];
                    sv[k] = l0 * UV[2 * m[0] + 1] + l1 * UV[2 * m[1] + 1] + l2 * UV[2 * m[2] + 1];
                }
            }
        }
        // pixels that see the object, and the triangles between neighboring pixels
        const ids = new Int32Array(W * H).fill(-1), V = [], uv = [];
        for (let k = 0; k < W * H; k++) {
            if (depth[k] === -Infinity) continue;
            ids[k] = V.length / 3;
            V.push(ox + (Math.floor(k / H) + i0) * h, oy + ((k % H) + j0) * h, depth[k]);
            uv.push(su[k], sv[k]);
        }
        const F = [], used = new Uint8Array(V.length / 3), jump = 3 * h;
        const d2 = (a, b) => (V[3 * a] - V[3 * b]) ** 2 + (V[3 * a + 1] - V[3 * b + 1]) ** 2 + (V[3 * a + 2] - V[3 * b + 2]) ** 2;
        for (let i = 0; i < W - 1; i++) for (let j = 0; j < H - 1; j++) {
            const q = [ids[i * H + j], ids[(i + 1) * H + j], ids[(i + 1) * H + j + 1], ids[i * H + j + 1]];
            const tris = [];
            if (q.every((x) => x >= 0)) {
                // split along the shorter diagonal
                if (d2(q[0], q[2]) < d2(q[1], q[3])) tris.push([q[0], q[1], q[2]], [q[0], q[2], q[3]]);
                else tris.push([q[0], q[1], q[3]], [q[1], q[2], q[3]]);
            } else if (q.filter((x) => x >= 0).length === 3) tris.push(q.filter((x) => x >= 0));
            for (const t of tris) {
                // no triangles across depth discontinuities (occlusion boundaries)
                if (Math.abs(V[3 * t[0] + 2] - V[3 * t[1] + 2]) > jump || Math.abs(V[3 * t[1] + 2] - V[3 * t[2] + 2]) > jump || Math.abs(V[3 * t[0] + 2] - V[3 * t[2] + 2]) > jump) continue;
                F.push(t[0], t[1], t[2]);
                for (const x of t) used[x] = 1;
            }
        }
        // keep the points of some triangle, in a random order
        const keep = [];
        for (let i = 0; i < used.length; i++) if (used[i]) keep.push(i);
        const r = this.random;
        if (shuffle) for (let i = keep.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [keep[i], keep[j]] = [keep[j], keep[i]]; }
        const newId = new Int32Array(used.length).fill(-1);
        keep.forEach((old, i) => { newId[old] = i; });
        const n = keep.length, Vo = new Float64Array(3 * n), UVo = new Float64Array(2 * n);
        keep.forEach((old, i) => {
            for (let e = 0; e < 3; e++) Vo[3 * i + e] = V[3 * old + e] + (noise > 0 ? noise * this.gauss() : 0);
            UVo[2 * i] = uv[2 * old]; UVo[2 * i + 1] = uv[2 * old + 1];
        });
        // orient the triangles upwards (towards the camera) on average
        const Fo = Int32Array.from(F, (x) => newId[x]);
        for (let t = 0; t < Fo.length; t += 3) {
            const a = Fo[t], b = Fo[t + 1], c = Fo[t + 2];
            const nz = (Vo[3 * b] - Vo[3 * a]) * (Vo[3 * c + 1] - Vo[3 * a + 1]) - (Vo[3 * c] - Vo[3 * a]) * (Vo[3 * b + 1] - Vo[3 * a + 1]);
            if (nz < 0) { Fo[t + 1] = c; Fo[t + 2] = b; }
        }
        return { V: Vo, F: Fo, uv: UVo, n };
    }
}
