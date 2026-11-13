// Equations with KaTeX (vendored, offline), in the notation of the paper.

export const MACROS = {
    '\\R': '\\mathbb{R}',
    '\\Mc': '\\mathcal{M}', '\\Nc': '\\mathcal{N}', '\\F': '\\mathcal{F}',
    '\\X': '\\mathbf{X}', '\\Y': '\\mathbf{Y}', '\\E': '\\mathbf{E}', '\\C': '\\mathbf{C}', '\\A': '\\mathbf{A}',
    '\\Lm': '\\mathbf{L}', '\\Lh': '\\hat{\\mathbf{L}}', '\\Um': '\\mathbf{U}', '\\I': '\\mathbf{I}',
    '\\x': '\\mathbf{x}', '\\y': '\\mathbf{y}', '\\e': '\\mathbf{e}', '\\uv': '\\mathbf{u}', '\\dX': '\\Delta\\mathbf{X}',
    '\\Xret': '\\mathbf{X}_{\\mathrm{ret}}', '\\Mret': 'M_{\\mathrm{ret}}',
    '\\PhiM': '\\Phi^{\\mathcal{M}}', '\\PhiN': '\\Phi^{\\mathcal{N}}', '\\PiM': '\\Pi^{\\mathcal{M}}',
    '\\CM': '\\mathbf{C}^{\\mathcal{M}}', '\\Csgn': '\\mathbf{C}^{\\mathcal{M}}_{\\mathrm{sgn}}',
    '\\Ii': 'I_{\\mathrm{initial}}', '\\Iu': 'I_{\\mathrm{update}}', '\\Imax': 'I_{\\max}',
    '\\NTC': '\\mathrm{NTC}', '\\T': '\\intercal', '\\sgn': '\\operatorname{sgn}',
};

export function tex(source, display = false) {
    return window.katex.renderToString(source, {
        displayMode: display, macros: { ...MACROS }, throwOnError: false, strict: 'ignore',
    });
}

// Render every <span class="tex"> (inline) and <div class="tex-block"> (display) below root, once.
export function renderMath(root = document) {
    for (const el of root.querySelectorAll('.tex, .tex-block')) {
        if (el.dataset.rendered) continue;
        const ref = el.dataset.eq ? `<span class="eq-ref">(${el.dataset.eq})</span>` : '';
        el.innerHTML = tex(el.textContent, el.classList.contains('tex-block')) + ref;
        el.dataset.rendered = '1';
    }
}
