/* ═══════════════════════════════════════════════════════════
   paper_to_web — JavaScript
   Navigation, TOC highlighting, search, and interactions
   ═══════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

    // ─── Navbar scroll effect ──────────────────────────────
    const nav = document.getElementById('topNav');
    const backToTop = document.getElementById('backToTop');

    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            nav.classList.add('scrolled');
        } else {
            nav.classList.remove('scrolled');
        }

        if (window.scrollY > 400) {
            backToTop.classList.add('visible');
        } else {
            backToTop.classList.remove('visible');
        }
    });

    // ─── Back to top button ────────────────────────────────
    if (backToTop) {
        backToTop.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    // ─── Mobile nav toggle ─────────────────────────────────
    const navToggle = document.getElementById('navToggle');
    const sidebar = document.getElementById('sidebar');

    if (navToggle && sidebar) {
        navToggle.addEventListener('click', () => {
            sidebar.classList.toggle('mobile-open');
        });
    }

    // ─── TOC active section highlighting ───────────────────
    // Only highlight in-page anchors (href starts with "#")
    const tocLinks = document.querySelectorAll('.toc-list a[href^="#"]');
    const sectionHeadings = document.querySelectorAll('.section-heading[id]');

    function updateActiveToc() {
        if (tocLinks.length === 0 || sectionHeadings.length === 0) return;

        const scrollPos = window.scrollY + 120;
        let currentId = '';

        sectionHeadings.forEach(heading => {
            if (heading.offsetTop <= scrollPos) {
                currentId = heading.id;
            }
        });

        tocLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href === '#' + currentId) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }

    let ticking = false;
    window.addEventListener('scroll', () => {
        if (!ticking) {
            requestAnimationFrame(() => {
                updateActiveToc();
                ticking = false;
            });
            ticking = true;
        }
    });

    // Initial highlight
    updateActiveToc();

    // ─── Reference search ──────────────────────────────────
    const refSearch = document.getElementById('refSearch');
    if (refSearch) {
        refSearch.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            const refs = document.querySelectorAll('.reference-entry');

            refs.forEach(ref => {
                if (!query || ref.textContent.toLowerCase().includes(query)) {
                    ref.style.display = '';
                } else {
                    ref.style.display = 'none';
                }
            });
        });
    }

    // ─── Citation tooltip on hover ─────────────────────────
    const citations = document.querySelectorAll('.citation[title]');
    citations.forEach(cite => {
        cite.addEventListener('mouseenter', (e) => {
            const title = cite.getAttribute('title');
            if (!title) return;

            let tooltip = document.createElement('div');
            tooltip.className = 'citation-tooltip';
            tooltip.textContent = title;
            tooltip.style.cssText = `
                position: fixed;
                background: #1a1a2e;
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-family: var(--font-sans);
                font-size: 13px;
                line-height: 1.5;
                max-width: 350px;
                z-index: 10000;
                pointer-events: none;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            `;

            document.body.appendChild(tooltip);

            const rect = cite.getBoundingClientRect();
            tooltip.style.left = Math.min(rect.left, window.innerWidth - tooltip.offsetWidth - 20) + 'px';
            tooltip.style.top = (rect.top - tooltip.offsetHeight - 8) + 'px';

            cite._tooltip = tooltip;
        });

        cite.addEventListener('mouseleave', () => {
            if (cite._tooltip) {
                cite._tooltip.remove();
                cite._tooltip = null;
            }
        });
    });

    // ─── Smooth scroll for in-page links ───────────────────
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (!href || href === '#') return;
            const target = document.querySelector(href);
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // ─── Footnote tooltip on hover ─────────────────────────
    const footnotes = document.querySelectorAll('.footnote[data-note]');
    footnotes.forEach(fn => {
        fn.addEventListener('mouseenter', (e) => {
            const note = fn.getAttribute('data-note');
            if (!note) return;

            let tooltip = document.createElement('div');
            tooltip.className = 'footnote-tooltip';
            tooltip.textContent = note;
            tooltip.style.cssText = `
                position: fixed;
                background: #333;
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-family: var(--font-sans);
                font-size: 12px;
                line-height: 1.5;
                max-width: 300px;
                z-index: 10000;
                pointer-events: none;
            `;

            document.body.appendChild(tooltip);
            const rect = fn.getBoundingClientRect();
            tooltip.style.left = rect.left + 'px';
            tooltip.style.top = (rect.bottom + 4) + 'px';

            fn._tooltip = tooltip;
        });

        fn.addEventListener('mouseleave', () => {
            if (fn._tooltip) {
                fn._tooltip.remove();
                fn._tooltip = null;
            }
        });
    });

    // ─── Lazy figure loading animation ─────────────────────
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.survey-figure, .table-container').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        observer.observe(el);
    });

    // ─── Citation block: tab switching & copy ──────────────
    document.querySelectorAll('.cite-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const block = tab.closest('.cite-block');
            block.querySelectorAll('.cite-tab').forEach(t => t.classList.remove('active'));
            block.querySelectorAll('.cite-panel').forEach(p => p.style.display = 'none');
            tab.classList.add('active');
            const target = block.querySelector('#' + tab.dataset.target);
            if (target) target.style.display = '';
        });
    });

    document.querySelectorAll('.cite-copy-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const panel = btn.closest('.cite-panel');
            const el = panel.querySelector('.' + btn.dataset.clip);
            if (!el) return;
            const text = el.textContent;
            navigator.clipboard.writeText(text).then(() => {
                btn.textContent = 'Copied!';
                btn.classList.add('copied');
                setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
            });
        });
    });
});
