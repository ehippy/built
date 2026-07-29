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
