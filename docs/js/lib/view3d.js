// 3D view (three.js) of meshes whose size and connectivity change from frame to frame:
// the perceived meshes M_ret(k), the tracked mesh, the target N, the body of the
// simulated object, point sets (tracked points), segments (error vectors), and the
// grippers. Layers are addressed by name and rebuilt only when their size grows.
// Two views can share their camera (link), e.g., the current shape and the target side
// by side. Labels are HTML placed at projected points. A click on a pickable layer
// reports the nearest vertex.
import * as THREE from '../../vendor/three/three.module.min.js';
import { OrbitControls } from '../../vendor/three/OrbitControls.js';
import { palette } from './theme.js';

export class MeshView {
    constructor(container, { fov = 30 } = {}) {
        this.container = container;
        try {
            this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
        } catch {
            container.append(Object.assign(document.createElement('p'), { className: 'view3d-hint', textContent: 'WebGL is not available in this browser.' }));
            this.failed = true;
            return;
        }
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        container.prepend(this.renderer.domElement);
        this.labelLayer = Object.assign(document.createElement('div'), { className: 'view3d-labels' });
        container.append(this.labelLayer);
        this.labels = [];
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(fov, 1, 0.5, 800);
        this.camera.up.set(0, 0, 1);
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.12;
        this.scene.add(new THREE.HemisphereLight(0xffffff, 0x4a4f58, 1.8));
        const sun = new THREE.DirectionalLight(0xffffff, 1.5);
        sun.position.set(-20, -30, 50);
        this.scene.add(sun);
        const fill = new THREE.DirectionalLight(0xffffff, 0.45);
        fill.position.set(30, 20, -10);
        this.scene.add(fill);
        this.layers = new Map();
        this.linked = [];
        this.dirty = true;
        this.scene.background = new THREE.Color(palette().scene);
        this.grid = new THREE.GridHelper(120, 48, 0x8a929c, 0x8a929c);
        this.grid.rotation.x = Math.PI / 2;
        this.grid.material.transparent = true;
        this.grid.material.opacity = palette().dark ? 0.1 : 0.14;
        this.scene.add(this.grid);
        new ResizeObserver(() => this.resize()).observe(container);
        this.controls.addEventListener('change', () => {
            this.dirty = true;
            if (!this.syncing) for (const other of this.linked) other.copyCamera(this);
        });
        this.setupPicking();
        const loop = () => {
            if (this.disposed) return;
            if (this.controls.update() || this.dirty) { this.renderer.render(this.scene, this.camera); this.placeLabels(); this.dirty = false; }
            requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
    }

    resize() {
        if (this.failed) return;
        const w = this.container.clientWidth, h = this.container.clientHeight;
        if (!w || !h) return;
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        if (!this.sized && this.pendingFrame) { this.sized = true; this.frame(...this.pendingFrame); }
        this.sized = true;
        this.dirty = true;
    }

    // ---------------------------------------------------------------- camera
    // Look from the direction `from - at` at the bounding sphere of the point sets.
    frame(view, pointSets) {
        if (this.failed) return;
        this.pendingFrame = [view, pointSets];
        if (!this.sized) return;
        const dir = new THREE.Vector3(...view.from).sub(new THREE.Vector3(...view.at)).normalize();
        const box = new THREE.Box3(), v = new THREE.Vector3();
        for (const P of pointSets || []) for (let i = 0; i + 2 < P.length; i += 3) box.expandByPoint(v.set(P[i], P[i + 1], P[i + 2]));
        if (box.isEmpty()) box.set(new THREE.Vector3(...view.at).subScalar(10), new THREE.Vector3(...view.at).addScalar(10));
        const center = box.getCenter(new THREE.Vector3()), radius = box.getSize(new THREE.Vector3()).length() / 2;
        const vfov = this.camera.fov * Math.PI / 180, hfov = 2 * Math.atan(Math.tan(vfov / 2) * Math.max(this.camera.aspect || 1, 0.5));
        const dist = (view.zoom || 1.05) * radius / Math.sin(Math.min(vfov, hfov) / 2);
        this.controls.target.copy(center);
        this.camera.position.copy(center).addScaledVector(dir, dist);
        this.controls.update();
        this.grid.position.set(center.x, center.y, box.min.z - 1.5);
        this.dirty = true;
        for (const other of this.linked) other.copyCamera(this);
    }

    // Share the camera with another view (both directions).
    link(other) {
        if (this.failed || other.failed) return;
        this.linked.push(other); other.linked.push(this);
        other.copyCamera(this);
    }
    copyCamera(src) {
        this.syncing = true;
        this.camera.position.copy(src.camera.position);
        this.camera.quaternion.copy(src.camera.quaternion);
        this.controls.target.copy(src.controls.target);
        this.controls.update();
        this.grid.position.copy(src.grid.position);
        this.syncing = false;
        this.dirty = true;
    }

    // ---------------------------------------------------------------- layers
    layer(name, make) {
        let entry = this.layers.get(name);
        if (!entry) { entry = make(); this.layers.set(name, entry); this.scene.add(entry.object); }
        return entry;
    }

    hide(name) {
        const entry = this.layers.get(name);
        if (entry) { entry.object.visible = false; this.dirty = true; }
    }
    show(name, flag = true) {
        const entry = this.layers.get(name);
        if (entry) { entry.object.visible = flag; this.dirty = true; }
    }

    // Triangle mesh: V (3n), F (3f); colors (3n, 0..1) or color (CSS); opacity; wire
    // (draw the edges); offset (polygon offset, pushes the mesh back in depth).
    mesh(name, { V, F, colors = null, color = '#cccccc', opacity = 1, wire = false, offset = 0, side = 'double', order = 0, pickable = false }) {
        if (this.failed) return;
        const entry = this.layer(name, () => {
            const geometry = new THREE.BufferGeometry();
            const material = new THREE.MeshLambertMaterial({ vertexColors: false, side: THREE.DoubleSide });
            const object = new THREE.Mesh(geometry, material);
            object.frustumCulled = false;
            const edges = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ transparent: true, opacity: 0.35 }));
            edges.frustumCulled = false;
            object.add(edges);
            return { object, geometry, material, edges, capV: 0, capF: 0 };
        });
        const n = V.length / 3, nf = F.length / 3, g = entry.geometry;
        if (n > entry.capV) {
            entry.capV = Math.ceil(n * 1.3);
            g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3 * entry.capV), 3));
            g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(3 * entry.capV), 3));
            // computeVertexNormals reuses an existing normal attribute: drop the old, smaller one,
            // or the new vertices get no normals (dark, unlit triangles)
            g.deleteAttribute('normal');
        }
        if (nf > entry.capF) {
            entry.capF = Math.ceil(nf * 1.3);
            g.setIndex(new THREE.BufferAttribute(new Uint32Array(3 * entry.capF), 1));
        }
        g.attributes.position.array.set(V); g.attributes.position.needsUpdate = true;
        // unused index entries become degenerate triangles: computeVertexNormals reads the
        // whole index buffer, not only the draw range
        g.index.array.set(F); g.index.array.fill(0, F.length); g.index.needsUpdate = true;
        g.setDrawRange(0, F.length);
        if (colors) { g.attributes.color.array.set(colors); g.attributes.color.needsUpdate = true; }
        g.computeVertexNormals();
        const m = entry.material;
        m.vertexColors = !!colors;
        m.color.set(colors ? '#ffffff' : color);
        m.transparent = opacity < 1; m.opacity = opacity; m.depthWrite = opacity >= 1;
        m.side = side === 'front' ? THREE.FrontSide : THREE.DoubleSide;
        m.polygonOffset = offset !== 0; m.polygonOffsetFactor = offset; m.polygonOffsetUnits = offset;
        m.needsUpdate = true;
        entry.object.renderOrder = order;
        entry.object.visible = true;
        entry.n = n; entry.V = V; entry.F = F; entry.pickable = pickable;
        // edges
        entry.edges.visible = !!wire;
        if (wire) {
            const seg = new Float32Array(6 * 3 * nf);
            for (let t = 0; t < nf; t++) for (let e = 0; e < 3; e++) {
                const a = F[3 * t + e], b = F[3 * t + (e + 1) % 3], o = 18 * t + 6 * e;
                seg[o] = V[3 * a]; seg[o + 1] = V[3 * a + 1]; seg[o + 2] = V[3 * a + 2];
                seg[o + 3] = V[3 * b]; seg[o + 4] = V[3 * b + 1]; seg[o + 5] = V[3 * b + 2];
            }
            entry.edges.geometry.setAttribute('position', new THREE.BufferAttribute(seg, 3));
            entry.edges.material.color.set(typeof wire === 'string' ? wire : (palette().dark ? '#ffffff' : '#1b1e22'));
            entry.edges.material.opacity = typeof wire === 'string' ? 0.6 : 0.22;
        }
        this.dirty = true;
    }

    // Points drawn as small spheres: P (3n), colors (3n) or color, radius.
    points(name, { P, colors = null, color = '#222222', radius = 0.18, order = 5 }) {
        if (this.failed) return;
        const n = P.length / 3;
        let entry = this.layers.get(name);
        if (!entry || entry.cap < n) {
            if (entry) { this.scene.remove(entry.object); entry.object.dispose?.(); }
            const cap = Math.ceil(n * 1.3) + 8;
            const object = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 10, 8), new THREE.MeshLambertMaterial({ color: '#ffffff' }), cap);
            object.frustumCulled = false;
            object.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(3 * cap), 3);
            entry = { object, cap };
            this.layers.set(name, entry);
            this.scene.add(object);
        }
        const { object } = entry, mtx = new THREE.Matrix4(), c = new THREE.Color(color);
        for (let i = 0; i < n; i++) {
            mtx.makeScale(radius, radius, radius).setPosition(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
            object.setMatrixAt(i, mtx);
            if (colors) object.instanceColor.setXYZ(i, colors[3 * i], colors[3 * i + 1], colors[3 * i + 2]);
            else object.instanceColor.setXYZ(i, c.r, c.g, c.b);
        }
        object.count = n;
        object.instanceMatrix.needsUpdate = true;
        object.instanceColor.needsUpdate = true;
        object.renderOrder = order;
        object.visible = true;
        this.dirty = true;
    }

    // Line segments: segs (6 per segment), color, opacity.
    lines(name, { segs, color = '#444444', opacity = 0.85, order = 6 }) {
        if (this.failed) return;
        const entry = this.layer(name, () => {
            const object = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ transparent: true }));
            object.frustumCulled = false;
            return { object };
        });
        entry.object.geometry.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(segs), 3));
        entry.object.material.color.set(color);
        entry.object.material.opacity = opacity;
        entry.object.renderOrder = order;
        entry.object.visible = true;
        this.dirty = true;
    }

    // Gripper boxes: poses [{R, t}], half sizes [[hx, hy, hz]], colors (outline).
    grippers(poses, halves, colors) {
        if (this.failed) return;
        if (!this.gripperObjects || this.gripperObjects.length !== poses.length) {
            for (const g of this.gripperObjects || []) { this.scene.remove(g.solid); this.scene.remove(g.outline); }
            const box = new THREE.BoxGeometry(2, 2, 2), edges = new THREE.EdgesGeometry(box);
            this.gripperObjects = poses.map(() => {
                const solid = new THREE.Mesh(box, new THREE.MeshLambertMaterial({ color: palette().gripper, transparent: true, opacity: 0.3, depthWrite: false }));
                const outline = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: '#000000' }));
                solid.renderOrder = 7;
                for (const m of [solid, outline]) { m.matrixAutoUpdate = false; this.scene.add(m); }
                return { solid, outline };
            });
        }
        this.gripperObjects.forEach((g, k) => {
            const { R, t } = poses[k], h = halves[k];
            for (const obj of [g.solid, g.outline]) {
                obj.matrix.set(R[0] * h[0], R[1] * h[1], R[2] * h[2], t[0], R[3] * h[0], R[4] * h[1], R[5] * h[2], t[1], R[6] * h[0], R[7] * h[1], R[8] * h[2], t[2], 0, 0, 0, 1);
                obj.visible = true;
            }
            g.solid.material.color.set(palette().gripper);
            g.outline.material.color.set(colors[k]);
        });
        this.dirty = true;
    }
    hideGrippers() { for (const g of this.gripperObjects || []) { g.solid.visible = false; g.outline.visible = false; } this.dirty = true; }

    // ---------------------------------------------------------------- picking
    // fn(name, vertex, point) on a click (not a drag) on a pickable mesh layer.
    onPick(fn) { this.pickFn = fn; }
    setupPicking() {
        const el = this.renderer.domElement, ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
        let down = null;
        el.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
        el.addEventListener('pointerup', (e) => {
            if (!down || !this.pickFn || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
            const r = el.getBoundingClientRect();
            ndc.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
            ray.setFromCamera(ndc, this.camera);
            const targets = [...this.layers.entries()].filter(([, l]) => l.pickable && l.object.visible);
            let best = null;
            for (const [name, l] of targets) {
                const hit = ray.intersectObject(l.object, false)[0];
                if (hit && (!best || hit.distance < best.hit.distance)) best = { name, l, hit };
            }
            if (!best) return;
            const { face, point } = best.hit, V = best.l.V;
            let vtx = face.a, bd = Infinity;
            for (const x of [face.a, face.b, face.c]) {
                const d = Math.hypot(V[3 * x] - point.x, V[3 * x + 1] - point.y, V[3 * x + 2] - point.z);
                if (d < bd) { bd = d; vtx = x; }
            }
            this.pickFn(best.name, vtx, [point.x, point.y, point.z]);
        });
    }

    // ---------------------------------------------------------------- labels
    // items: [{p: [x, y, z], text, color, cls, dx, dy}]
    setLabels(items) { this.extraLabels = items; this.dirty = true; }
    placeLabels() {
        const items = this.extraLabels || [];
        while (this.labels.length < items.length) {
            const el = document.createElement('span');
            this.labelLayer.append(el);
            this.labels.push(el);
        }
        const w = this.container.clientWidth, h = this.container.clientHeight, v = new THREE.Vector3();
        this.labels.forEach((el, k) => {
            const item = items[k];
            if (!item) { el.style.display = 'none'; return; }
            v.set(item.p[0], item.p[1], item.p[2]).project(this.camera);
            if (v.z > 1) { el.style.display = 'none'; return; }
            el.style.display = '';
            el.className = `label3d ${item.cls || ''}`;
            el.textContent = item.text;
            if (item.color) el.style.color = item.color; else el.style.removeProperty('color');
            el.style.transform = `translate(${((v.x + 1) / 2 * w + (item.dx ?? 6)).toFixed(1)}px, ${((1 - v.y) / 2 * h + (item.dy ?? -8)).toFixed(1)}px)`;
        });
    }

    refreshTheme() {
        if (this.failed) return;
        const p = palette();
        this.scene.background = new THREE.Color(p.scene);
        this.grid.material.opacity = p.dark ? 0.1 : 0.14;
        this.dirty = true;
    }
}
