// Ground truth of the simulations, for evaluation only (the method never uses it): every
// perceived point carries the surface parameters (u, v) of the material point it samples,
// so the true match of a point of one mesh is the point of the other mesh with the same
// (u, v), and the error of a map is measured on the rest shape of the object.

// Mean distance on the rest shape (chart) between the true and the mapped point. With
// `symmetries` (maps of the surface parameters (u, v) that are isometries of both shapes),
// the smallest error over them: a map onto a symmetric copy is then not counted as wrong.
export function matchError(src, tgt, Pi, chart, symmetries = null) {
    let best = Infinity;
    for (const sym of symmetries || [(u, v) => [u, v]]) {
        let s = 0;
        for (let m = 0; m < Pi.length; m++) {
            const a = chart(src.uv[2 * m], src.uv[2 * m + 1]), [u, v] = sym(tgt.uv[2 * Pi[m]], tgt.uv[2 * Pi[m] + 1]), b = chart(u, v);
            s += Math.hypot(a[0] - b[0], a[1] - b[1]);
        }
        best = Math.min(best, s / Math.max(Pi.length, 1));
    }
    return best;
}

// The four mirror symmetries of a rectangle in its parameters.
export const RECTANGLE_SYMMETRIES = [(u, v) => [u, v], (u, v) => [1 - u, v], (u, v) => [u, 1 - v], (u, v) => [1 - u, 1 - v]];

// Per-point version (cm), for coloring.
export function matchErrors(src, tgt, Pi, chart) {
    const out = new Float64Array(Pi.length);
    for (let m = 0; m < Pi.length; m++) {
        const a = chart(src.uv[2 * m], src.uv[2 * m + 1]), b = chart(tgt.uv[2 * Pi[m]], tgt.uv[2 * Pi[m] + 1]);
        out[m] = Math.hypot(a[0] - b[0], a[1] - b[1]);
    }
    return out;
}
