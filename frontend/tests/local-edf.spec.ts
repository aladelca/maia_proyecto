import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";

test("la carga pública de EDF depende del modelo publicado y conserva la selección de canales", async ({
  page,
}) => {
  let edfEnabled = false;
  await page.route("**/config.json", (route) =>
    route.fulfill({ json: { api: "", edfEnabled } }),
  );
  await page.goto("/");
  await expect(page.locator("#night")).toHaveAttribute("accept", ".parquet");
  edfEnabled = true;
  await page.reload();
  await expect(page.locator("#night")).toHaveAttribute(
    "accept",
    ".parquet,.edf",
  );

  let token: string | undefined;
  await page.route("**/public/v1/uploads", (route) => {
    const req = route.request();
    token = req.headers()["x-guest-token"];
    expect(token).toMatch(/^[a-f0-9]{64}$/);
    expect(req.postDataJSON()).toMatchObject({ format: "edf", size: 5 });
    return route.fulfill({
      json: { fields: {}, upload_id: "night", url: "/signed-upload" },
    });
  });
  await page.route("**/signed-upload", (route) =>
    route.fulfill({
      status: 201,
      headers: { "x-amz-version-id": "v1" },
      body: "",
    }),
  );
  await page.route("**/public/v1/predictions", (route) => {
    expect(route.request().headers()["x-guest-token"]).toBe(token);
    expect(route.request().postDataJSON()).toEqual({
      format: "edf",
      profile: "eeg_pz_oz",
      upload_id: "night",
      version_id: "v1",
    });
    return route.fulfill({ json: { job_id: "job-publico" } });
  });
  await page.route("**/public/v1/predictions/job-publico", (route) =>
    route.fulfill({
      json: {
        state: "succeeded",
        files: { json: "/result.json", csv: "/result.csv" },
      },
    }),
  );
  await page.route("**/result.json", (route) =>
    route.fulfill({
      json: {
        model_version: "edf",
        profile: "eeg_pz_oz",
        classes: ["W", "N1", "N2", "N3", "REM", "M"],
        rows: 1,
        eligible: 1,
        epochs: [{ onset_s: 0, prediction: "N2", eligible: true }],
      },
    }),
  );
  await page.locator("#night").setInputFiles({
    name: "noche.edf",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("demo\n"),
  });
  await page.getByLabel("Canales de entrada").selectOption("eeg_pz_oz");
  await page.getByRole("button", { name: "Generar predicción" }).click();
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toBeVisible();
  await expect(page.getByText("Canales utilizados · Pz-Oz")).toBeVisible();
  await expect(page.getByText("Movimiento", { exact: true })).toBeVisible();
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

for (const neural of [true, false]) {
  test(`LightGBM con EDF y disponibilidad neuronal ${neural}`, async ({
    page,
  }) => {
    await page.route("**/config.json", (route) =>
      route.fulfill({
        json: { api: "", edfEnabled: neural, edfLgbmEnabled: true },
      }),
    );
    await page.route("**/public/v1/uploads", async (route) => {
      expect(route.request().postDataJSON()).toMatchObject({
        format: "edf",
        model: "lightgbm",
      });
      await route.fulfill({
        json: { fields: {}, upload_id: "test", url: "/signed-edf" },
      });
    });
    await page.route("**/signed-edf", (route) =>
      route.fulfill({ status: 201, headers: { "x-amz-version-id": "1" } }),
    );
    await page.route("**/public/v1/predictions", async (route) => {
      expect(route.request().postDataJSON()).toMatchObject({
        format: "edf",
        model: "lightgbm",
        profile: "auto",
      });
      await route.fulfill({ json: { job_id: "lgbm-test" } });
    });
    await page.route("**/public/v1/predictions/lgbm-test", (route) =>
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
          model_version: "lgbm-test",
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
    await page
      .locator("#night")
      .setInputFiles({
        name: "demo.edf",
        mimeType: "application/octet-stream",
        buffer: Buffer.from("fixture"),
      });
    const choice = page.getByLabel("Modelo para EDF");
    await expect(choice).toHaveValue(neural ? "fullnight" : "lightgbm");
    await choice.selectOption("lightgbm");
    await expect(page.getByLabel("Canales de entrada")).toHaveCount(0);
    await page.getByRole("button", { name: "Generar predicción" }).click();
    await expect(page.getByText("lgbm-test", { exact: true })).toBeVisible();
    await expect(page.getByText("Movimiento", { exact: true })).toHaveCount(0);
  });
}

test("el EDF sintético se procesa con el LightGBM publicado", async ({
  page,
}) => {
  const file = process.env.MAIA_LGBM_DEMO_EDF;
  test.skip(
    !file,
    "Requiere el EDF sintético de diez minutos y la API con LightGBM habilitado",
  );
  await page.goto("/");
  await page.locator("#night").setInputFiles(file!);
  await page.getByLabel("Modelo para EDF").selectOption("lightgbm");
  await page.getByRole("button", { name: "Generar predicción" }).click();
  await expect(
    page.getByText("20 épocas analizadas · 0 excluidas"),
  ).toBeVisible({ timeout: 120000 });
  const resultPanel = page
    .locator("section")
    .filter({
      has: page.getByRole("heading", { name: "Hipnograma predicho" }),
    });
  await expect(resultPanel.getByText(/^lgbm-/)).toBeVisible();
  const download = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Descargar probabilidades en CSV" })
    .click();
  const path = await (await download).path();
  const lines = (await readFile(path!, "utf8")).trim().split(/\r?\n/);
  expect(lines).toHaveLength(21);
  expect(lines[0].split(",").filter((c) => c.startsWith("prob_"))).toEqual([
    "prob_W",
    "prob_N1",
    "prob_N2",
    "prob_N3",
    "prob_REM",
  ]);
});
