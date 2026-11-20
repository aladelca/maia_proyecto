import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";

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
