import { defineConfig } from "@playwright/test";
const remote = !!process.env.DEMO_ORIGIN;
export default defineConfig({
  testDir: "./tests",
  testMatch: remote ? "predictions.spec.ts" : "ux.spec.ts",
  timeout: remote ? 150000 : 30000,
  workers: 1,
  retries: 0,
  use: {
    baseURL: process.env.DEMO_ORIGIN ?? "http://127.0.0.1:4173",
    channel: process.env.BROWSER_CHANNEL ?? "chromium",
    trace: "off", screenshot: "off", video: "off",
  },
  webServer: remote ? undefined : {
    command: "npm run dev -- --host 127.0.0.1 --port 4173 --strictPort",
    url: "http://127.0.0.1:4173", reuseExistingServer: false,
  },
});
