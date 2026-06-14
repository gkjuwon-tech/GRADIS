/* GRADIS — scroll animations (GSAP + ScrollTrigger) */
(function () {
  function init() {
    /* ---------- nav: solid on scroll ---------- */
    const nav = document.querySelector('.nav');
    if (nav) {
      const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 40);
      window.addEventListener('scroll', onScroll, { passive: true });
      onScroll();
    }
  }

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
