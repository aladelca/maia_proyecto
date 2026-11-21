import { test, expect } from "@playwright/test";
const result = { model_version: "stack-five-a64415fc2bd6", rows: 4, eligible: 3, epochs: [
  { epoch_id: "a", onset_s: 0, prediction: "W", eligible: true },
  { epoch_id: "b", onset_s: 30, prediction: null, eligible: false },
  { epoch_id: "c", onset_s: 60, prediction: "N2", eligible: true },
  { epoch_id: "d", onset_s: 90, prediction: "REM", eligible: true },
] };
test.beforeEach(async ({ page }) => {
  await page.route("**/config.json", r => r.fulfill({ json: { api: "" } }));
  await page.route("**/public/v1/examples/default", r => r.fulfill({ json: { model_version: "stack-five-a64415fc2bd6" } }));
});
test("demo offline, keyboard exploration, zoom and mobile layout", async ({ page }) => {
  let requests = 0;
  await page.route("**/public/v1/predictions**", r => { requests++; return r.abort(); });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByRole("button", { name: "Explorar modo demo" }).click();
  await expect(page.getByText("MODO DEMO · DATOS SINTÉTICOS", { exact: true })).toBeVisible();
  const slider = page.getByRole("slider");
  await slider.focus(); await slider.press("ArrowRight");
  await expect(slider).toHaveValue("1");
  await expect(page.locator(".epoch-detail")).toContainText("Época 2 de 840");
  await page.getByRole("button", { name: "Ampliar selección" }).click();
  await expect(page.getByRole("button", { name: "Ver noche completa" })).toHaveAttribute("aria-pressed", "true");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await expect(page.getByRole("button", { name: /Descargar probabilidades/ })).toHaveCount(0);
  expect(requests).toBe(0);
  await page.getByRole("button", { name: "Salir del modo demo" }).click();
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toHaveCount(0);
});
test("demo remains available when configuration fails", async ({ page }) => {
  await page.route("**/config.json", r => r.fulfill({ status: 500, body: "failure" }));
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("configuración");
  await expect(page.getByRole("button", { name: "Probar ejemplo público" })).toBeDisabled();
  await page.getByRole("button", { name: "Explorar modo demo" }).click();
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toBeVisible();
});
test("HTTP 500 explains failure without silently switching to demo", async ({ page }) => {
  await page.route("**/public/v1/predictions", r => r.fulfill({ status: 500, body: "Internal Server Error" }));
  await page.goto("/");
  await page.getByRole("button", { name: "Probar ejemplo público" }).click();
  await expect(page.getByRole("alert")).toContainText("HTTP 500");
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Probar ejemplo público" })).toBeEnabled();
});
test("invalid EDF rejected before upload", async ({ page }) => {
  await page.goto("/");
  for (const file of [{ name: "night.csv", buffer: Buffer.from("data") }, { name: "empty.edf", buffer: Buffer.alloc(0) }]) {
    await page.locator("#night").setInputFiles({ ...file, mimeType: "application/octet-stream" });
    await page.getByRole("button", { name: "Generar predicción" }).click();
    await expect(page.getByRole("alert")).toContainText("EDF de hasta 128 MiB");
  }
});
test("example contract, processing, result gaps and reload", async ({ page }) => {
  let polls = 0;
  await page.route("**/public/v1/predictions", async r => {
    expect(r.request().postDataJSON()).toEqual({ example: "default", model: "stack_five" });
    expect(r.request().headers()["x-guest-token"]).toBeTruthy();
    expect(r.request().headers()["idempotency-key"]).toBeTruthy();
    await r.fulfill({ json: { job_id: "job-1" } });
  });
  await page.route("**/public/v1/predictions/job-1", r => r.fulfill({ json: ++polls === 1 ? { state: "processing" } : { state: "succeeded", files: { json: "/result.json", csv: "/result.csv" } } }));
  await page.route("**/result.json", r => r.fulfill({ json: result }));
  await page.goto("/");
  await page.getByRole("button", { name: "Probar ejemplo público" }).click();
  await expect(page.getByRole("status")).toContainText("Procesando");
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toBeVisible();
  await expect(page.getByText("3 épocas analizadas · 1 excluidas")).toBeVisible();
  await page.getByRole("slider").focus(); await page.getByRole("slider").press("ArrowRight");
  await expect(page.locator(".epoch-detail")).toContainText("Sin predicción");
  expect((await page.locator("svg path").getAttribute("d"))?.match(/ M/g)).toHaveLength(2);
  await page.reload();
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toBeVisible();
});
test("upload keeps signed form and version contract", async ({ page }) => {
  await page.route("**/public/v1/uploads", async r => {
    expect(r.request().postDataJSON()).toEqual({ size: 4, sha256: expect.stringMatching(/^[a-f0-9]{64}$/), format: "edf", model: "stack_five" });
    await r.fulfill({ json: { url: "/signed-upload", fields: { key: "night" }, upload_id: "upload-1" } });
  });
  await page.route("**/signed-upload", r => r.fulfill({ status: 204, headers: { "x-amz-version-id": "version-1" } }));
  await page.route("**/public/v1/predictions", async r => {
    expect(r.request().postDataJSON()).toEqual({ upload_id: "upload-1", version_id: "version-1", format: "edf", model: "stack_five" });
    await r.fulfill({ json: { job_id: "upload-job" } });
  });
  await page.route("**/public/v1/predictions/upload-job", r => r.fulfill({ json: { state: "failed", reason: "Archivo incompatible" } }));
  await page.goto("/");
  await page.locator("#night").setInputFiles({ name: "night.edf", mimeType: "application/octet-stream", buffer: Buffer.from("PAR1") });
  await page.getByRole("button", { name: "Generar predicción" }).click();
  await expect(page.getByRole("alert")).toContainText("Archivo incompatible");
  await expect(page.getByRole("button", { name: "Reintentar consulta" })).toBeVisible();
});
test("failed status can be retried and CSV link refreshed", async ({ page }) => {
  let polls = 0;
  await page.route("**/public/v1/predictions/retry-job", r => ++polls === 1
    ? r.fulfill({ status: 503, json: { message: "unavailable" } })
    : r.fulfill({ json: { state: "succeeded", files: { json: "/result.json", csv: "/fresh.csv" } } }));
  await page.route("**/result.json", r => r.fulfill({ json: result }));
  await page.route("**/fresh.csv", r => r.fulfill({ contentType: "text/csv", headers: { "content-disposition": "attachment; filename=probabilities.csv" }, body: "epoch_id,prediction\na,W" }));
  await page.goto("/?job=retry-job");
  await expect(page.getByRole("alert")).toContainText("HTTP 503");
  await page.getByRole("button", { name: "Reintentar consulta" }).click();
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toBeVisible();
  const downloaded = page.waitForEvent("download");
  await page.getByRole("button", { name: "Descargar probabilidades en CSV" }).click();
  expect((await downloaded).suggestedFilename()).toBe("probabilities.csv");
  expect(polls).toBe(3);
});
test("missing upload version never starts prediction", async ({ page }) => {
  let predictions = 0;
  await page.route("**/public/v1/uploads", r => r.fulfill({ json: { url: "/signed-upload", fields: {}, upload_id: "upload-1" } }));
  await page.route("**/signed-upload", r => r.fulfill({ status: 204 }));
  await page.route("**/public/v1/predictions", r => { predictions++; return r.abort(); });
  await page.goto("/");
  await page.locator("#night").setInputFiles({ name: "night.edf", mimeType: "application/octet-stream", buffer: Buffer.from("PAR1") });
  await page.getByRole("button", { name: "Generar predicción" }).click();
  await expect(page.getByRole("alert")).toContainText("confirmar la versión");
  expect(predictions).toBe(0);
});
test("empty results and malformed results do not crash", async ({ page }) => {
  await page.route("**/public/v1/predictions/empty", r => r.fulfill({ json: { state: "succeeded", files: { json: "/result.json", csv: "/result.csv" } } }));
  await page.route("**/result.json", r => r.fulfill({ json: { ...result, epochs: [], rows: 0, eligible: 0 } }));
  await page.goto("/?job=empty");
  await expect(page.getByText("No hay épocas disponibles para visualizar.")).toBeVisible();
  await page.route("**/result.json", r => r.fulfill({ json: { epochs: "invalid" } }));
  await page.reload();
  await expect(page.getByRole("alert")).toContainText("formato válido");
});
test("model tab supports keyboard navigation and preserves analysis state", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Explorar modo demo" }).click();
  await page.getByRole("slider").focus();
  await page.getByRole("slider").press("ArrowRight");
  const analysis = page.getByRole("tab", { name: "Analizar una noche" });
  await analysis.focus(); await analysis.press("ArrowRight");
  await expect(page.getByRole("tab", { name: "Sobre el proyecto" })).toBeFocused();
  await expect(page.getByRole("heading", { name: "Cómo predice el modelo" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Por qué elegimos este modelo" })).toBeVisible();
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toHaveCount(0);
  await page.getByRole("tab", { name: "Sobre el proyecto" }).press("ArrowLeft");
  await expect(page.getByRole("slider")).toHaveValue("1");
  await page.getByRole("tab", { name: "Sobre el proyecto" }).click();
  await page.reload();
  await expect(page.getByRole("tab", { name: "Sobre el proyecto" })).toHaveAttribute("aria-selected", "true");
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
test("wrong release disables live submission but allows demo and model explanation", async ({ page }) => {
  await page.route("**/public/v1/examples/default", r => r.fulfill({ json: { model_version: "old-release" } }));
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Probar ejemplo público" })).toBeDisabled();
  await expect(page.getByText("El servicio aún no confirma", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: "Explorar modo demo" }).click();
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toBeVisible();
  await page.route("**/public/v1/examples/default", r => r.fulfill({ json: { model_version: "stack-five-a64415fc2bd6" } }));
  await page.getByRole("button", { name: "Comprobar de nuevo" }).click();
  await expect(page.getByRole("button", { name: "Probar ejemplo público" })).toBeEnabled();
});
test("old results are not attributed to the new model", async ({ page }) => {
  await page.route("**/public/v1/predictions/old", r => r.fulfill({ json: { state: "succeeded", files: { json: "/result.json", csv: "/result.csv" } } }));
  await page.route("**/result.json", r => r.fulfill({ json: { ...result, model_version: "old" } }));
  await page.goto("/?job=old");
  await expect(page.getByRole("alert")).toContainText("versión anterior");
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toHaveCount(0);
});
test("switching tabs keeps an active job and query parameters", async ({ page }) => {
  let polls = 0;
  await page.route("**/public/v1/predictions/active", r => r.fulfill({ json: ++polls === 1 ? { state: "processing" } : { state: "succeeded", files: { json: "/result.json", csv: "/result.csv" } } }));
  await page.route("**/result.json", r => r.fulfill({ json: result }));
  await page.goto("/?job=active");
  await page.getByRole("tab", { name: "Sobre el proyecto" }).click();
  await expect(page.getByRole("heading", { name: "Cómo predice el modelo" })).toBeVisible();
  await expect(page.locator("#panel-analysis .epoch-detail")).toContainText("Época 1 de 4");
  await page.getByRole("tab", { name: "Analizar una noche" }).click();
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toBeVisible();
  await expect(page).toHaveURL(/job=active/);
});
test("demo summary is illustrative and has no personal assessment", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Explorar modo demo" }).click();
  await expect(page.getByRole("heading", { name: "Tu resumen de sueño" })).toBeVisible();
  await expect(page.locator(".sleep-summary")).toContainText("5 h 57 min");
  await expect(page.locator(".sleep-summary")).toContainText("datos sintéticos");
  await expect(page.locator(".sleep-summary input")).toHaveCount(0);
});
for (const hours of [6, 7]) {
  test(`complete ${hours} hour recording requires context before guidance`, async ({ page }) => {
    await page.addInitScript(() => sessionStorage.setItem("maia-source-summary", "upload"));
    const epochs = Array.from({ length: hours * 120 }, (_, i) => ({ epoch_id: String(i), onset_s: i * 30, prediction: "N2", eligible: true }));
    await page.route("**/public/v1/predictions/summary", r => r.fulfill({ json: { state: "succeeded", files: { json: "/summary.json", csv: "/summary.csv" } } }));
    await page.route("**/summary.json", r => r.fulfill({ json: { ...result, rows: epochs.length, eligible: epochs.length, epochs } }));
    await page.goto("/?job=summary");
    await expect(page.locator(".sleep-summary")).toContainText(`${hours} h 0 min`);
    await expect(page.locator(".sleep-guidance")).toHaveCount(0);
    await page.getByRole("checkbox").check();
    await expect(page.locator(".sleep-guidance")).toContainText(hours < 7 ? "por debajo" : "alcanza la referencia mínima");
    await page.getByRole("checkbox").uncheck();
    await expect(page.locator(".sleep-guidance")).toHaveCount(0);
  });
}
test("gaps and excluded epochs prevent a full-night assessment", async ({ page }) => {
  await page.addInitScript(() => sessionStorage.setItem("maia-source-incomplete", "upload"));
  await page.route("**/public/v1/predictions/incomplete", r => r.fulfill({ json: { state: "succeeded", files: { json: "/summary.json", csv: "/summary.csv" } } }));
  await page.route("**/summary.json", r => r.fulfill({ json: { ...result, epochs: [result.epochs[0], result.epochs[1], result.epochs[3]] } }));
  await page.goto("/?job=incomplete");
  await expect(page.locator(".sleep-summary")).toContainText("No comparamos");
  await expect(page.getByRole("checkbox")).toHaveCount(0);
});
test("progress groups server work and surfaces the EMG validation reason", async ({ page }) => {
  let polls = 0;
  await page.route("**/public/v1/predictions", r => r.fulfill({ json: { job_id: "emg-validation" } }));
  await page.route("**/public/v1/predictions/emg-validation", r => r.fulfill({ json: ++polls === 1 ? { state: "processing" } : { state: "failed", reason: "LightGBM requiere EMG submental a 1 Hz." } }));
  await page.goto("/");
  await page.getByRole("button", { name: "Probar ejemplo público" }).click();
  await expect(page.locator('.analysis-progress [aria-current="step"]')).toContainText("Calcular variables y predecir");
  await expect(page.locator('.analysis-progress .skipped')).toContainText("No aplica");
  await expect(page.getByRole("alert")).toContainText("LightGBM requiere EMG submental a 1 Hz.");
  await expect(page.locator('.analysis-progress .failed')).toContainText("Requiere atención");
  await expect(page.locator('.analysis-progress li').last()).toContainText("Pendiente");
});
test("new analysis clears the result, selected file and saved job across reload", async ({ page }) => {
  let polls = 0;
  await page.addInitScript(() => { if (!sessionStorage.getItem("reset-test-initialized")) { sessionStorage.setItem("maia-stack-five-job", "reset-job"); sessionStorage.setItem("maia-source-reset-job", "upload"); sessionStorage.setItem("reset-test-initialized", "yes"); } });
  await page.route("**/public/v1/predictions/reset-job", r => { polls++; return r.fulfill({ json: { state: "succeeded", files: { json: "/result.json", csv: "/result.csv" } } }); });
  await page.route("**/result.json", r => r.fulfill({ json: result }));
  await page.goto("/?job=reset-job#analizar");
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toBeVisible();
  await page.locator("#night").setInputFiles({ name: "night.edf", mimeType: "application/octet-stream", buffer: Buffer.from("test") });
  await page.getByRole("button", { name: "Nuevo análisis" }).click();
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toHaveCount(0);
  await expect(page.locator(".analysis-progress")).toHaveCount(0);
  expect(await page.locator("#night").evaluate((input: HTMLInputElement) => input.files?.length)).toBe(0);
  await expect(page).not.toHaveURL(/job=/);
  await expect(page.getByRole("button", { name: "Generar predicción" })).toBeDisabled();
  await page.reload();
  await expect(page.getByRole("heading", { name: "Analizar una noche", exact: true })).toBeVisible();
  expect(polls).toBe(1);
  await page.getByRole("button", { name: "Explorar modo demo" }).click();
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toBeVisible();
  await page.getByRole("button", { name: "Nuevo análisis" }).click();
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Salir del modo demo" })).toHaveCount(0);
});
for (const mode of ["local", "local-stack"]) {
  test(`upstream ${mode} configuration works without an API field`, async ({ page }) => {
    await page.route("**/config.json", r => r.fulfill({ json: { mode } }));
    await page.route("**/v1/edf", r => r.fulfill({ json: { ...result, classes: ["W", "N1", "N2", "N3", "REM", "M"], profile: "eeg_pz_oz" } }));
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Analizar un registro EDF" })).toBeVisible();
    await page.locator("#night").setInputFiles({ name: "test.edf", mimeType: "application/octet-stream", buffer: Buffer.from("fixture") });
    await page.getByRole("button", { name: "Generar predicción" }).click();
    await expect(page.getByRole("img", { name: /Hipnograma/ })).toBeVisible();
    await expect(page.getByText("Movimiento", { exact: true })).toBeVisible();
  });
}
