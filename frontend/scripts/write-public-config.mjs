import { writeFileSync } from "node:fs";

const input = process.env.MAIA_PUBLIC_CONFIG;
if (!input) throw new Error("Configura MAIA_PUBLIC_CONFIG en Vercel.");

const publicStrings = ["api", "origin", "issuer", "clientId", "login"];
let config;
try {
  config = JSON.parse(input);
  if (publicStrings.some((key) => key in config && typeof config[key] !== "string")) {
    throw new Error();
  }
  const api = new URL(config.api);
  if (
    api.protocol !== "https:" || api.username || api.password || api.search || api.hash
  ) {
    throw new Error();
  }
  if ("edfEnabled" in config && typeof config.edfEnabled !== "boolean") {
    throw new Error();
  }
  config.api = api.href.replace(/\/+$/, "");
} catch {
  throw new Error("MAIA_PUBLIC_CONFIG requiere JSON válido y una URL HTTPS pública para api.");
}

// These are the browser-facing fields in Terraform's public_config output.
const publicJson = JSON.stringify(config, [...publicStrings, "edfEnabled"]);
writeFileSync(new URL("../dist/config.json", import.meta.url), publicJson + "\n");
