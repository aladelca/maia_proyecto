import { test, expect } from "@playwright/test";

test("la interfaz publica una sola arquitectura y envía EDF al híbrido", async ({
  page,
}) => {
  await page.route("**/config.json", (route) =>
    route.fulfill({ json: { api: "" } }),
  );
  await page.route("**/public/v1/examples/default", (route) =>
    route.fulfill({ json: { model_version: "stack-five-a64415fc2bd6" } }),
  );
  await page.route("**/public/v1/uploads", (route) => {
    expect(route.request().headers()["x-guest-token"]).toMatch(
      /^[a-f0-9]{64}$/,
    );
    expect(route.request().postDataJSON()).toMatchObject({
      format: "edf",
      model: "stack_five",
    });
    return route.fulfill({
      json: { fields: {}, upload_id: "test", url: "/signed-edf" },
    });
  });
  await page.route("**/signed-edf", (route) =>
    route.fulfill({ status: 201, headers: { "x-amz-version-id": "v1" } }),
  );
  await page.route("**/public/v1/predictions", (route) => {
    expect(route.request().postDataJSON()).toEqual({
      format: "edf",
      model: "stack_five",
      upload_id: "test",
      version_id: "v1",
    });
    return route.fulfill({ json: { job_id: "stack-test" } });
  });
  await page.route("**/public/v1/predictions/stack-test", (route) =>
    route.fulfill({
      json: {
        state: "succeeded",
        files: { json: "/night.json", csv: "/night.csv" },
      },
    }),
  );
  await page.route("**/night.json", (route) =>
    route.fulfill({
      json: {
        model_version: "stack-five-a64415fc2bd6",
        classes: ["W", "N1", "N2", "N3", "REM"],
        rows: 1,
        eligible: 1,
        epochs: [
          { epoch_id: "0", onset_s: 0, prediction: "N2", eligible: true },
        ],
      },
    }),
  );
  await page.goto("/");
  await expect(page.locator("#night")).toHaveAttribute("accept", ".edf");
  await expect(page.getByRole("combobox")).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "Cómo predice el modelo" }),
  ).toBeVisible();
  await page.locator("#night").setInputFiles({
    name: "demo.edf",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("fixture"),
  });
  await page.getByRole("button", { name: "Generar predicción" }).click();
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toBeVisible();
  await expect(
    page.getByText("stack-five-a64415fc2bd6", { exact: true }),
  ).toBeVisible();
});

test("un EDF sin hipnograma produce seis probabilidades y un CSV", async ({
  page,
}) => {
  const file = process.env.MAIA_LOCAL_EDF;
  test.skip(!file, "Requiere un EDF de desarrollo y el servicio local.");
  await page.goto("/");
  await page.locator("#night").setInputFiles(file!);
  await page.getByLabel("Canales de entrada").selectOption("eeg_pz_oz");
  await page.getByRole("button", { name: "Generar predicción" }).click();
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toBeVisible({
    timeout: 60000,
  });
  await expect(page.getByText("Canales utilizados · Pz-Oz")).toBeVisible();
  await expect(page.getByText("Movimiento", { exact: true })).toBeVisible();
  const downloaded = page.waitForEvent("download");
  await page.getByRole("button", { name: /Descargar probabilidades/ }).click();
  expect((await downloaded).suggestedFilename()).toBe("hipnograma.csv");
});

test("los 12 perfiles devuelven probabilidades por época del mismo EDF", async ({
  page,
  request,
}) => {
  const file = process.env.MAIA_LOCAL_EDF;
  test.skip(
    !file,
    "Requiere el servicio local y una noche con los cuatro canales.",
  );
  await page.goto("/");
  const profiles = await page
    .getByLabel("Canales de entrada")
    .locator("option")
    .evaluateAll((options) =>
      options
        .map((e) => (e as HTMLOptionElement).value)
        .filter((v) => v !== "auto"),
    );
  expect(new Set(profiles).size).toBe(12);
  const buffer = await readFile(file!);
  const classes = ["W", "N1", "N2", "N3", "REM", "M"];
  let epochs: number | undefined;
  for (const profile of profiles) {
    const res = await request.post("/v1/edf", {
      multipart: {
        profile,
        file: {
          name: "night.edf",
          mimeType: "application/octet-stream",
          buffer,
        },
      },
    });
    expect(res.ok(), profile).toBeTruthy();
    const body = await res.json();
    expect(body.profile).toBe(profile);
    expect(body.classes).toEqual(classes);
    epochs ??= body.rows;
    expect(body.epochs).toHaveLength(epochs!);
    expect(body.eligible).toBeGreaterThan(0);
    let maxError = 0;
    for (const row of body.epochs) {
      if (row.eligible) {
        const sum = classes.reduce((v, c) => v + row["prob_" + c], 0);
        maxError = Math.max(maxError, Math.abs(sum - 1));
      } else {
        expect(row.prediction).toBeNull();
      }
    }
    expect(maxError, profile).toBeLessThan(1e-5);
    await res.dispose();
  }
});

test("el laboratorio espera la release elegida antes de admitir predicciones", async ({
  page,
}) => {
  await page.route("**/config.json", (route) =>
    route.fulfill({ json: { api: "" } }),
  );
  await page.route("**/public/v1/examples/default", (route) =>
    route.fulfill({ json: { example: "default" } }),
  );
  await page.goto("/");
  await expect(page.getByRole("status")).toContainText(
    "La API aún no confirma",
  );
  await expect(
    page.getByRole("button", { name: "Probar ejemplo público" }),
  ).toBeDisabled();
});

test("la release local procesa la muestra con las cinco etapas y exporta el CSV", async ({
  page,
}) => {
  test.skip(
    !process.env.MAIA_LOCAL_STACK,
    "Requiere el servidor local del híbrido.",
  );
  await page.goto("/");
  await page.getByRole("button", { name: "Probar ejemplo público" }).click();
  await expect(
    page.getByText("20 épocas analizadas · 0 excluidas"),
  ).toBeVisible({ timeout: 60000 });
  await expect(
    page.getByText("stack-five-a64415fc2bd6", { exact: true }),
  ).toBeVisible();
  const event = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Descargar probabilidades en CSV" })
    .click();
  expect((await event).suggestedFilename()).toBe("hipnograma.csv");
});
