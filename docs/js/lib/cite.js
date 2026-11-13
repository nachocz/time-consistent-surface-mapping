// Citation dialog.
import { h } from './ui.js';

const DOI = '10.1109/TASE.2025.3529180';
const CITATION = 'I. Cuiral-Zueco and G. López-Nicolás, "Time Consistent Surface Mapping for Deformable Object Shape Control," '
    + 'IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 11099-11111, 2025. doi: 10.1109/TASE.2025.3529180.';
const BIBTEX = [
    '@article{cuiral2025time,',
    '  author  = {Cuiral-Zueco, Ignacio and L{\\\'o}pez-Nicol{\\\'a}s, Gonzalo},',
    '  title   = {Time Consistent Surface Mapping for Deformable Object Shape Control},',
    '  journal = {IEEE Transactions on Automation Science and Engineering},',
    '  year    = {2025},',
    '  volume  = {22},',
    '  pages   = {11099--11111},',
    '  doi     = {10.1109/TASE.2025.3529180},',
    '}',
].join('\n');

async function copy(text, button) {
    try {
        await navigator.clipboard.writeText(text);
    } catch {
        const area = h('textarea', { style: 'position:fixed;opacity:0' });
        area.value = text;
        document.body.append(area);
        area.select();
        document.execCommand('copy');
        area.remove();
    }
    button.textContent = 'Copied ✓';
    setTimeout(() => { button.textContent = 'Copy'; }, 1400);
}

export function citeDialog() {
    const block = (title, text) => {
        const copyButton = h('button', { class: 'button small', type: 'button' }, 'Copy');
        copyButton.addEventListener('click', () => copy(text, copyButton));
        return h('div', { class: 'cite-block' }, h('div', { class: 'cite-block-head' }, h('span', {}, title), copyButton), h('pre', {}, text));
    };
    const close = h('button', { class: 'button', type: 'button' }, 'Close');
    const dialog = h('dialog', { class: 'cite-dialog', 'aria-labelledby': 'citeTitle' },
        h('p', { class: 'kicker' }, 'Citation'),
        h('h2', { id: 'citeTitle' }, 'Cite the paper'),
        h('p', { class: 'cite-note' }, h('a', { href: `https://doi.org/${DOI}`, target: '_blank', rel: 'noopener' }, `doi.org/${DOI}`)),
        block('Reference', CITATION),
        block('BibTeX', BIBTEX),
        h('div', { class: 'cite-actions' }, close));
    close.addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', (event) => {           // a click on the backdrop closes
        const r = dialog.getBoundingClientRect();
        if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close();
    });
    dialog.addEventListener('close', () => dialog.remove());
    document.body.append(dialog);
    dialog.showModal();
}
