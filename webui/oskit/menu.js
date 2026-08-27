/* ============================================================================
   oskit/menu.js — context menus
   ----------------------------------------------------------------------------
   The gesture every windowing system answers and this one did not: right-click.
   Window verbs (float, close, focus) existed only as 20px glyphs in a 26px title
   bar, and the tiling could be changed only from the shell or the palette. Both
   are now where the hand already is.

   The menu owns no vocabulary either — callers pass items. It positions itself
   inside the viewport, closes on the next click, Escape, scroll or resize, and
   never leaves a stray listener behind, which is the usual way a hand-rolled
   context menu turns into a leak.
   ============================================================================ */
(function (global) {
  "use strict";

  const esc = (s) => String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  let el = null, teardown = null;

  function close() {
    if (teardown) { teardown(); teardown = null; }
    if (el) { el.remove(); el = null; }
  }

  /** items: [{label, hint, run, disabled} | {sep:true}] */
  function open(x, y, items) {
    close();
    el = document.createElement("div");
    el.className = "cm";
    el.setAttribute("role", "menu");
    el.innerHTML = (items || []).map((it, i) => it.sep
      ? '<div class="cm-sep"></div>'
      : '<button class="cm-item" data-i="' + i + '"' + (it.disabled ? " disabled" : "") + ">"
        + '<span class="cm-label">' + esc(it.label) + "</span>"
        + (it.hint ? '<span class="cm-hint">' + esc(it.hint) + "</span>" : "")
        + "</button>").join("");
    document.body.appendChild(el);

    // Position after insertion: the size is unknown until it is in the document,
    // and a menu that opens off the bottom of a 720px viewport is unusable.
    const r = el.getBoundingClientRect();
    el.style.left = Math.max(4, Math.min(x, window.innerWidth - r.width - 4)) + "px";
    el.style.top = Math.max(4, Math.min(y, window.innerHeight - r.height - 4)) + "px";

    el.addEventListener("click", (e) => {
      const b = e.target.closest(".cm-item");
      if (!b) return;
      const it = items[Number(b.dataset.i)];
      close();
      if (it && it.run && !it.disabled) setTimeout(() => { try { it.run(); } catch (er) {} }, 0);
    });

    const onDown = (e) => { if (!el || !el.contains(e.target)) close(); };
    const onKey = (e) => { if (e.key === "Escape") close(); };
    // Capture phase: a menu must close even when the click lands on something that
    // stops propagation, which several panels in this app do.
    document.addEventListener("pointerdown", onDown, true);
    document.addEventListener("keydown", onKey, true);
    window.addEventListener("resize", close);
    window.addEventListener("scroll", close, true);
    teardown = () => {
      document.removeEventListener("pointerdown", onDown, true);
      document.removeEventListener("keydown", onKey, true);
      window.removeEventListener("resize", close);
      window.removeEventListener("scroll", close, true);
    };
    return el;
  }

  global.Menu = { open, close };
})(window);
