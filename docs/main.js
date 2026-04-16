/* main.js — HTDemucsWithSpatial demo */

/* ── Copy BibTeX ── */
function copyBibtex() {
  const text = document.getElementById('bibtex').innerText;
  navigator.clipboard.writeText(text).then(() => {
    const label = document.getElementById('copy-label');
    label.textContent = 'Copied!';
    setTimeout(() => (label.textContent = 'Copy'), 2000);
  }).catch(() => {
    /* fallback for older browsers */
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    const label = document.getElementById('copy-label');
    label.textContent = 'Copied!';
    setTimeout(() => (label.textContent = 'Copy'), 2000);
  });
}

/* ── Stop all other players when one starts ── */
document.addEventListener('DOMContentLoaded', () => {
  const audios = Array.from(document.querySelectorAll('audio'));

  audios.forEach(a => {
    a.addEventListener('play', () => {
      audios.forEach(other => {
        if (other !== a && !other.paused) other.pause();
      });
    });
  });
});
