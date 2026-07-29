(function () {
  "use strict";
  var STORAGE_KEY = "built-theme";
  var toggle = document.getElementById("theme-toggle");
  if (!toggle) return;

  var systemPrefersDark = window.matchMedia("(prefers-color-scheme: dark)");

  function effectiveTheme() {
    var explicit = localStorage.getItem(STORAGE_KEY);
    if (explicit === "light" || explicit === "dark") return explicit;
    return systemPrefersDark.matches ? "dark" : "light";
  }

  function render() {
    toggle.textContent = effectiveTheme() === "dark" ? "☀" : "☾";
  }

  toggle.addEventListener("click", function () {
    var next = effectiveTheme() === "dark" ? "light" : "dark";
    localStorage.setItem(STORAGE_KEY, next);
    document.documentElement.setAttribute("data-theme", next);
    render();
  });

  // Live-update the icon (not the page, which already tracks the system via the
  // CSS media query) if the OS theme changes while no explicit choice is stored.
  systemPrefersDark.addEventListener("change", function () {
    if (!localStorage.getItem(STORAGE_KEY)) render();
  });

  render();
})();

// --- Transcript details state preservation ---
// HTMX replaces the entire <div id="transcript"> on each 2s poll, which clears
// native <details> open/closed state. This handler remembers which details were
// expanded before swap and re-opens them on the new fragment.
//
// Listens on `document` rather than on #transcript itself: an outerHTML swap
// detaches the original #transcript node and inserts a fresh one, so a listener
// attached directly to that original node stops receiving anything after the
// first swap. Also: the real event names are htmx:beforeSwap/htmx:afterSwap —
// there is no htmx:aroundSwap in this htmx version (a prior version of this
// handler used that name and silently never fired at all).
(function () {
  "use strict";
  var openKeys = null;

  document.addEventListener("htmx:beforeSwap", function () {
    var transcript = document.getElementById("transcript");
    if (!transcript) return;
    var keys = [];
    transcript.querySelectorAll("details[open]").forEach(function (det) {
      var eventDiv = det.closest(".event");
      if (!eventDiv) return;
      var head = eventDiv.querySelector(".event-head span:first-child");
      if (head) keys.push(head.textContent.trim());
    });
    openKeys = keys;
  });

  document.addEventListener("htmx:afterSwap", function () {
    if (!openKeys) return;
    var transcript = document.getElementById("transcript");
    if (!transcript) return;
    transcript.querySelectorAll(".event").forEach(function (eventDiv) {
      var head = eventDiv.querySelector(".event-head span:first-child");
      if (!head) return;
      if (openKeys.indexOf(head.textContent.trim()) !== -1) {
        var det = eventDiv.querySelector("details");
        if (det) det.setAttribute("open", "");
      }
    });
  });
})();

// --- Board "N done" collapse state preservation ---
// Same problem and same fix as the transcript above: HTMX replaces the entire
// #board-wrap on each 3s poll, which clears native <details> open/closed state on
// the per-column "N done" sections. Each one has a stable data-column attribute
// (one per pm/developer/tester/deployer column), so — unlike the transcript,
// whose events have no stable identity across polls — matching by that attribute
// is enough.
(function () {
  "use strict";
  var openColumns = null;

  document.addEventListener("htmx:beforeSwap", function () {
    var boardWrap = document.getElementById("board-wrap");
    if (!boardWrap) return;
    var cols = [];
    boardWrap.querySelectorAll("details[data-column][open]").forEach(function (det) {
      cols.push(det.dataset.column);
    });
    openColumns = cols;
  });

  document.addEventListener("htmx:afterSwap", function () {
    if (!openColumns) return;
    var boardWrap = document.getElementById("board-wrap");
    if (!boardWrap) return;
    boardWrap.querySelectorAll("details[data-column]").forEach(function (det) {
      if (openColumns.indexOf(det.dataset.column) !== -1) {
        det.setAttribute("open", "");
      }
    });
  });
})();
