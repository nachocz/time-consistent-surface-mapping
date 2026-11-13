// Objects of the demos: flat foam-like pieces of thickness h, discretized as a structured
// lattice of nu x nv x nz nodes. Node (i, j, k) lies at chart(u, v) in the plane
// (u = i / (nu - 1), v = j / (nv - 1)) and at height z = -h/2 + h k / (nz - 1). The chart
// maps the unit square onto the piece (a rectangle, a disc, a star, ...), and an
// optional mask removes lattice cells (a T, an L, ...).
//
// The lattice is the volumetric model simulated with ARAP (js/sim/arap.js). Its top
// layer is the surface a camera above the object sees; js/sim/camera.js samples that
// surface anew at every frame, as the RGB-D segmentation of the experiments does.

export function makeLattice(spec) {
    const { nu, nv, nz = 2, h = 1, chart } = spec;
    const mask = spec.mask || (() => true);
    // cells (i, j) between nodes i..i+1, j..j+1, kept if their center is inside the mask
    const cell = new Uint8Array((nu - 1) * (nv - 1));
    for (let i = 0; i < nu - 1; i++) for (let j = 0; j < nv - 1; j++) {
        cell[i * (nv - 1) + j] = mask((i + 0.5) / (nu - 1), (j + 0.5) / (nv - 1)) ? 1 : 0;
    }
    const hasCell = (i, j) => i >= 0 && j >= 0 && i < nu - 1 && j < nv - 1 && cell[i * (nv - 1) + j] === 1;
    const hasNode = (i, j) => hasCell(i - 1, j - 1) || hasCell(i - 1, j) || hasCell(i, j - 1) || hasCell(i, j);
    // compact numbering of the existing nodes, in (i, j, k) order (narrow band for ARAP)
    const id = new Int32Array(nu * nv * nz).fill(-1);
    const nodeU = [], nodeV = [], nodeK = [];
    for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
        if (!hasNode(i, j)) continue;
        for (let k = 0; k < nz; k++) {
            id[(i * nv + j) * nz + k] = nodeU.length;
            nodeU.push(i); nodeV.push(j); nodeK.push(k);
        }
    }
    const n = nodeU.length;
    const index = (i, j, k) => (i < 0 || j < 0 || i >= nu || j >= nv ? -1 : id[(i * nv + j) * nz + k]);
    const rest = new Float64Array(3 * n);
    for (let a = 0; a < n; a++) {
        const [x, y] = chart(nodeU[a] / (nu - 1), nodeV[a] / (nv - 1));
        rest[3 * a] = x; rest[3 * a + 1] = y;
        rest[3 * a + 2] = nz > 1 ? -h / 2 + h * nodeK[a] / (nz - 1) : 0;
    }

    // ARAP cells: the 26-neighborhood, restricted to nodes that share a kept cell
    const shareCell = (i, j, a, b) => {
        const i0 = Math.min(i, a), j0 = Math.min(j, b);
        if (i !== a && j !== b) return hasCell(i0, j0);
        if (i !== a) return hasCell(i0, j - 1) || hasCell(i0, j);
        if (j !== b) return hasCell(i - 1, j0) || hasCell(i, j0);
        return true;
    };
    const neighbors = new Array(n);
    for (let p = 0; p < n; p++) {
        const i = nodeU[p], j = nodeV[p], k = nodeK[p], list = [];
        for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) for (let dk = -1; dk <= 1; dk++) {
            if (!di && !dj && !dk) continue;
            const c = k + dk;
            if (c < 0 || c >= nz) continue;
            const q = index(i + di, j + dj, c);
            if (q >= 0 && shareCell(i, j, i + di, j + dj)) list.push(q);
        }
        neighbors[p] = Int32Array.from(list);
    }

    // top layer: the perceived surface, with its own numbering (one point per column)
    const top = [], column = new Int32Array(n), topId = new Int32Array(nu * nv).fill(-1);
    for (let p = 0; p < n; p++) {
        if (nodeK[p] === nz - 1) { topId[nodeU[p] * nv + nodeV[p]] = top.length; top.push(p); }
    }
    for (let p = 0; p < n; p++) column[p] = topId[nodeU[p] * nv + nodeV[p]];
    const uv = new Float64Array(2 * top.length);
    top.forEach((p, m) => { uv[2 * m] = nodeU[p] / (nu - 1); uv[2 * m + 1] = nodeV[p] / (nv - 1); });

    // closed surface for drawing, oriented outwards in the rest shape: the top faces,
    // the bottom faces, and the side walls along the boundary edges of the kept cells
    const surface = [], wallTris = [];
    const pushTri = (list, a, b, c, out) => {
        const P = (x) => [rest[3 * x], rest[3 * x + 1], rest[3 * x + 2]];
        const A = P(a), B = P(b), C = P(c);
        const nx = (B[1] - A[1]) * (C[2] - A[2]) - (B[2] - A[2]) * (C[1] - A[1]);
        const ny = (B[2] - A[2]) * (C[0] - A[0]) - (B[0] - A[0]) * (C[2] - A[2]);
        const nzz = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]);
        if (nx * out[0] + ny * out[1] + nzz * out[2] < 0) list.push(a, c, b); else list.push(a, b, c);
    };
    const quad = (list, q, out) => { pushTri(list, q[0], q[1], q[2], out); pushTri(list, q[0], q[2], q[3], out); };
    const center = (i, j) => {             // rest center of cell (i, j) in the plane
        const [x, y] = chart((i + 0.5) / (nu - 1), (j + 0.5) / (nv - 1));
        return [x, y];
    };
    for (let i = 0; i < nu - 1; i++) for (let j = 0; j < nv - 1; j++) {
        if (!hasCell(i, j)) continue;
        if (nz > 1) {
            quad(surface, [index(i, j, nz - 1), index(i + 1, j, nz - 1), index(i + 1, j + 1, nz - 1), index(i, j + 1, nz - 1)], [0, 0, 1]);
            quad(surface, [index(i, j, 0), index(i + 1, j, 0), index(i + 1, j + 1, 0), index(i, j + 1, 0)], [0, 0, -1]);
        }
        // side walls on the edges of this cell without a kept neighbor cell
        const c = center(i, j);
        const edges = [[i, j, i + 1, j, i, j - 1], [i + 1, j, i + 1, j + 1, i + 1, j], [i + 1, j + 1, i, j + 1, i, j + 1], [i, j + 1, i, j, i - 1, j]];
        for (const [a0, b0, a1, b1, ni, nj] of edges) {
            if (hasCell(ni, nj)) continue;
            for (let k = 0; k < nz - 1; k++) {
                const q = [index(a0, b0, k), index(a1, b1, k), index(a1, b1, k + 1), index(a0, b0, k + 1)];
                const mid = [0.5 * (rest[3 * q[0]] + rest[3 * q[1]]), 0.5 * (rest[3 * q[0] + 1] + rest[3 * q[1] + 1])];
                const out = [mid[0] - c[0], mid[1] - c[1], 0];
                quad(surface, q, out); quad(wallTris, q, out);
            }
        }
    }
    // top-surface triangles (top numbering), for the noise-free perceived surface
    const topTris = [];
    for (let i = 0; i < nu - 1; i++) for (let j = 0; j < nv - 1; j++) {
        if (!hasCell(i, j)) continue;
        const a = topId[i * nv + j], b = topId[(i + 1) * nv + j], c = topId[(i + 1) * nv + j + 1], d = topId[i * nv + j + 1];
        const d2 = (x, y) => (rest[3 * top[x]] - rest[3 * top[y]]) ** 2 + (rest[3 * top[x] + 1] - rest[3 * top[y] + 1]) ** 2;
        if (d2(a, c) < d2(b, d)) topTris.push(a, b, c, a, c, d); else topTris.push(a, b, d, b, c, d);
    }
    // bottom faces and walls only: the body under the perceived surface (drawn with it)
    const body = [];
    for (let i = 0; i < nu - 1; i++) for (let j = 0; j < nv - 1; j++) {
        if (hasCell(i, j) && nz > 1) quad(body, [index(i, j, 0), index(i + 1, j, 0), index(i + 1, j + 1, 0), index(i, j + 1, 0)], [0, 0, -1]);
    }
    body.push(...wallTris);

    return {
        spec, nu, nv, nz, n, M: top.length, rest, neighbors, index, hasCell,
        nodeU: Int32Array.from(nodeU), nodeV: Int32Array.from(nodeV), nodeK: Int32Array.from(nodeK),
        top: Int32Array.from(top), column, uv, topId,
        surface: Uint32Array.from(surface), body: Uint32Array.from(body), topTris: Int32Array.from(topTris),
    };
}

// Position on the top surface of the lattice positions P at the parameter (u, v), by
// bilinear interpolation in the cell that contains it (null if that cell is not kept).
export function surfacePoint(lat, P, u, v, out = [0, 0, 0]) {
    const { nu, nv, nz } = lat;
    const x = Math.min(Math.max(u, 0), 1) * (nu - 1), y = Math.min(Math.max(v, 0), 1) * (nv - 1);
    const i = Math.min(nu - 2, Math.floor(x)), j = Math.min(nv - 2, Math.floor(y));
    if (!lat.hasCell(i, j)) return null;
    const fx = x - i, fy = y - j, k = nz - 1;
    const a = lat.index(i, j, k), b = lat.index(i + 1, j, k), c = lat.index(i + 1, j + 1, k), d = lat.index(i, j + 1, k);
    const wa = (1 - fx) * (1 - fy), wb = fx * (1 - fy), wc = fx * fy, wd = (1 - fx) * fy;
    for (let e = 0; e < 3; e++) out[e] = wa * P[3 * a + e] + wb * P[3 * b + e] + wc * P[3 * c + e] + wd * P[3 * d + e];
    return out;
}

// Lattice nodes held by a gripper centered at the node (i, j): |i' - i| <= di,
// |j' - j| <= dj, through the whole thickness.
export function graspedNodes(lat, { i, j, di, dj }) {
    const out = [];
    for (let a = i - di; a <= i + di; a++) for (let b = j - dj; b <= j + dj; b++) {
        for (let k = 0; k < lat.nz; k++) { const p = lat.index(a, b, k); if (p >= 0) out.push(p); }
    }
    return Int32Array.from(out);
}

// ---------------------------------------------------------------- charts of the unit square
// Rectangle of size w x l centered at the origin (u along x).
export const rectangle = (l, w) => (u, v) => [l * (u - 0.5), w * (v - 0.5)];

// Disc of radius R: the elliptical grid mapping of the square [-1, 1]^2 onto the disc,
// with the radius modulated by f(theta) (a star for f = 1 + a cos(n theta)).
export function disc(R, f = () => 1) {
    return (u, v) => {
        const s = 2 * u - 1, t = 2 * v - 1;
        const x = s * Math.sqrt(Math.max(0, 1 - t * t / 2)), y = t * Math.sqrt(Math.max(0, 1 - s * s / 2));
        const r = R * f(Math.atan2(y, x));
        return [r * x, r * y];
    };
}
