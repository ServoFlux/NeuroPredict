// Small, dependency-free UI helpers: drag-and-drop file picker + submit spinner.
(function () {
  "use strict";

  // --- Drag-and-drop file picker ---------------------------------------
  // Each .dropzone wraps a real <input type="file">. We just show the chosen
  // file name and add a highlight while a file is dragged over it.
  document.querySelectorAll(".dropzone").forEach(function (zone) {
    var input = zone.querySelector('input[type="file"]');
    var nameEl = zone.querySelector(".dropzone-file");
    if (!input) return;

    function showName() {
      if (!nameEl) return;
      if (input.files && input.files.length > 0) {
        nameEl.textContent = input.files[0].name;
        zone.classList.add("has-file");
      } else {
        nameEl.textContent = nameEl.dataset.empty || "";
        zone.classList.remove("has-file");
      }
    }

    input.addEventListener("change", showName);

    ["dragenter", "dragover"].forEach(function (ev) {
      zone.addEventListener(ev, function (e) {
        e.preventDefault();
        zone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      zone.addEventListener(ev, function (e) {
        e.preventDefault();
        zone.classList.remove("dragover");
      });
    });
    zone.addEventListener("drop", function (e) {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        showName();
      }
    });

    showName();
  });

  // --- Submit spinner --------------------------------------------------
  // When a form with [data-loading] is submitted, swap the button into a
  // "working" state so the user knows the model is running.
  document.querySelectorAll("form[data-loading]").forEach(function (form) {
    form.addEventListener("submit", function () {
      var btn = form.querySelector("button[type='submit']");
      if (!btn) return;
      btn.disabled = true;
      btn.classList.add("is-loading");
      var label = btn.querySelector(".btn-label");
      if (label) label.textContent = btn.dataset.loadingLabel || "Working…";
    });
  });
})();
