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
// HTMX replaces the entire <div id="transcript"> on each 2s poll, which clears native
// <details> open/closed state. This handler remembers which details were expanded before
// swap and re-opens them on the new fragment.
(function () {
  "use strict";
  var transcript = document.getElementById("transcript");
  if (!transcript) return;

  transcript.addEventListener("htmx:aroundSwap", function (evt) {
    // Collect identifiers of open details before swap wipes the DOM.
    var openKeys = [];
    transcript.querySelectorAll("details[open]").forEach(function (det) {
      var eventDiv = det.closest(".event");
      if (!eventDiv) return;
      var head = eventDiv.querySelector(".event-head span:first-child");
      if (!head) return;
      openKeys.push(head.textContent.trim());
    });
    this._openKeys = openKeys;
  }.bind(transcript));

  transcript.addEventListener("htmx:afterSwap", function (evt) {
    if (!this._openKeys) return;
    this.querySelectorAll(".event").forEach(function (eventDiv) {
      var head = eventDiv.querySelector(".event-head span:first-child");
      if (!head) return;
      var key = head.textContent.trim();
      if (this._openKeys.indexOf(key) !== -1) {
        var det = eventDiv.querySelector("details");
        if (det) det.setAttribute("open", "");
      }
    }.bind(this));
  });
})();
