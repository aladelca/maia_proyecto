import { test, expect } from "@playwright/test";
import { resolve } from "node:path";

test("Cognito, ejemplo público, recarga y carga de una noche", async ({
  page,
}) => {
  const user = process.env.DEMO_USERNAME;
  const password = process.env.DEMO_PASSWORD;
  if (!user || !password || !process.env.DEMO_ORIGIN)
    throw Error("Configura DEMO_ORIGIN, DEMO_USERNAME y DEMO_PASSWORD.");
  await page.goto("/");
  await page.getByRole("button", { name: "Iniciar sesión" }).click();
  const name = page.locator('input[name="username"]:visible').first();
  await name.fill(user);
  const pass = page.locator('input[name="password"]:visible').first();
  if (!(await pass.isVisible()))
    await page.getByRole("button", { name: /next|siguiente/i }).click();
  await pass.fill(password);
  await page
    .locator("input[type=submit]:visible, button[type=submit]:visible")
    .last()
    .click();
  await page.getByRole("button", { name: "Probar ejemplo público" }).click();
  await expect(page).toHaveURL(/job=/);
  await page.reload();
  await expect(page.getByRole("img", { name: /Hipnograma/ })).toBeVisible({
    timeout: 90000,
  });
  const first = page.url();
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
