import { defineConfig } from "@playwright/test";
const remote = !!process.env.DEMO_ORIGIN;
export default defineConfig({
  testDir: "./tests",
  testMatch: process.env.MAIA_LOCAL_EDF || process.env.MAIA_LOCAL_STACK ? "local-edf.spec.ts" : remote ? "predictions.spec.ts" : ["ux.spec.ts", "local-edf.spec.ts"],
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
