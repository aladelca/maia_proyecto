import { test, expect } from "@playwright/test";
import { resolve } from "node:path";

test("acceso público, recarga, aislamiento entre visitantes y carga de una noche", async ({
  page,
  browser,
}) => {
  if (!process.env.DEMO_ORIGIN) throw Error("Configura DEMO_ORIGIN.");
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Analizar una noche" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Iniciar sesión" }),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "Probar ejemplo público" }).click();
  await expect(page).toHaveURL(/job=/);
  await page.reload();
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toBeVisible({
    timeout: 90000,
  });
  const first = page.url();
  const guest = await browser.newContext();
  try {
    const other = await guest.newPage();
    await other.goto(first);
    await expect(other.getByRole("alert")).toContainText(
      "Recurso no disponible",
    );
    await expect(other.getByRole("img", { name: /Hipnograma/ })).toHaveCount(0);
  } finally {
    await guest.close();
  }
  await page
    .locator("#night")
    .setInputFiles(
      process.env.EXAMPLE_PARQUET ??
        resolve("../var/cloud/bundle/example.parquet"),
    );
  await page.getByRole("button", { name: "Generar predicción" }).click();
  await expect(page).not.toHaveURL(first);
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toBeVisible({
    timeout: 60000,
  });
  await expect(page.getByText(/841 épocas analizadas/)).toBeVisible();
  const downloadEvent = page.waitForEvent("download");
  await page.getByRole("button", { name: /Descargar probabilidades/ }).click();
  expect((await downloadEvent).suggestedFilename()).toMatch(/\.csv$/);
});
