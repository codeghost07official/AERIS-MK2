(() => {
  const form = document.getElementById("telemetry-upload-form");
  if (!form) return;
  form.addEventListener("submit", () => {
    const button = form.querySelector("button[type='submit']");
    if (!button) return;
    button.disabled = true;
    button.textContent = "Analyzing telemetry…";
    form.setAttribute("aria-busy", "true");
  });
})();
