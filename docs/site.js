
const menuButton = document.querySelector('.menu-button');
const nav = document.querySelector('.site-nav');

if (menuButton && nav) {
  menuButton.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    menuButton.setAttribute('aria-expanded', String(open));
  });

  nav.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      nav.classList.remove('open');
      menuButton.setAttribute('aria-expanded', 'false');
    });
  });
}

const copyButton = document.querySelector('#copy-citation');
const citation = document.querySelector('#citation-text');

if (copyButton && citation) {
  copyButton.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(citation.textContent);
      const original = copyButton.textContent;
      copyButton.textContent = 'Copied';
      setTimeout(() => copyButton.textContent = original, 1500);
    } catch {
      copyButton.textContent = 'Select text to copy';
    }
  });
}
