import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  timeout: 150000,
  workers: 1,
  retries: 0,
  use: {
    baseURL: process.env.DEMO_ORIGIN,
    channel: process.env.BROWSER_CHANNEL ?? "chromium",
    trace: "off",
    screenshot: "off",
    video: "off",
  },
});
