import type { Epoch } from "./Hypnogram";
// Deterministic synthetic night; never presented as a model prediction.
const cycle = ["W", "N1", "N2", "N3", "N2", "REM", "N2"];
export function demoResult() {
  const epochs: Epoch[] = Array.from({ length: 840 }, (_, i) => ({
    epoch_id: `demo-${i + 1}`, onset_s: i * 30,
    eligible: i % 113 !== 0,
    prediction: i % 113 === 0 ? null : cycle[Math.floor(i / 20) % cycle.length],
  }));
  return { model_version: "DEMO · Datos sintéticos", epochs, eligible: epochs.filter(e => e.eligible).length, rows: epochs.length };
}
