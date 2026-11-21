import { useState } from "react";
import { Project } from "./project";

type Config = {
  api: string;
  mode?: string;
};
type Epoch = {
  epoch_id: string;
  onset_s: number;
  prediction: string | null;
  eligible: boolean;
};
type Result = {
  model_version: string;
  classes?: string[];
  profile?: string;
  epochs: Epoch[];
  eligible: number;
  rows: number;
};
type Job = {
  state: "processing" | "succeeded" | "failed";
  reason?: string;
  files?: { json: string; csv: string };
};
const releaseVersion = "stack-five-a64415fc2bd6";
const baseStages = ["W", "REM", "N1", "N2", "N3"];
const sleepNames: Record<string, string> = {
  W: "Vigilia",
  REM: "REM",
  M: "Movimiento",
  N1: "Sueño N1",
  N2: "Sueño N2",
  N3: "Sueño N3",
};

function Hypnogram({ rows, stages }: { rows: Epoch[]; stages: string[] }) {
  const first = rows[0]?.onset_s ?? 0;
  const minutes = Math.max(
    1,
    ((rows.at(-1)?.onset_s ?? first) + 30 - first) / 60,
  );
  const px = (sec: number) => 70 + ((sec - first) / 60 / minutes) * 800;
  const py = (s: string) => 30 + stages.indexOf(s) * 36;
  let line = "",
    last = -60;
  for (const row of rows) {
    if (!row.prediction) {
      last = -60;
      continue;
    }
    const y = py(row.prediction);
    line +=
      (last + 30 === row.onset_s ? " L" : " M") +
      px(row.onset_s) +
      " " +
      y +
      " H" +
      px(row.onset_s + 30);
    last = row.onset_s;
  }
  return (
    <svg
      viewBox={`0 0 900 ${stages.length * 36 + 43}`}
      role="img"
      aria-label="Hipnograma. Eje horizontal en minutos, eje vertical por estadio."
    >
      {stages.map((s, i) => (
        <g key={s}>
          <text x="14" y={35 + i * 36}>
            {s}
          </text>
          <line x1="65" x2="870" y1={30 + i * 36} y2={30 + i * 36} />
        </g>
      ))}
      <path d={line} fill="none" stroke="#14776e" strokeWidth="2.5" />
      {[0, 0.25, 0.5, 0.75, 1].map((x) => (
        <text
          key={x}
          x={70 + x * 800}
          y={stages.length * 36 + 30}
          textAnchor="middle"
        >
          {Math.round(x * minutes)} min
        </text>
      ))}
    </svg>
  );
}
const channelNames: Record<string, string> = {
  eeg_fpz_cz: "Fpz-Cz",
  eeg_pz_oz: "Pz-Oz",
  eog: "EOG",
  emg: "EMG",
};
function prettyProfile(value: string) {
  return value
    .split("+")
    .map((c) => channelNames[c] ?? c)
    .join(" + ");
}
const profiles = [
  ["eeg_fpz_cz"],
  ["eeg_pz_oz"],
  ["eeg_fpz_cz", "eeg_pz_oz"],
].flatMap((eeg) =>
  [[], ["eog"], ["emg"], ["eog", "emg"]].map((extra) =>
    [...eeg, ...extra].join("+"),
  ),
);
function ProfilePicker({
  value,
  change,
}: {
  value: string;
  change: (profile: string) => void;
}) {
  return (
    <label>
      Canales de entrada
      <select value={value} onChange={(e) => change(e.target.value)}>
        <option value="auto">Todos los canales disponibles</option>
        {profiles.map((p) => (
          <option key={p} value={p}>
            {prettyProfile(p)}
          </option>
        ))}
      </select>
    </label>
  );
}
function ResultPanel({
  result,
  download,
}: {
  result: Result;
  download: () => void;
}) {
  const stages = result.classes?.includes("M")
    ? [...baseStages, "M"]
    : baseStages;
  return (
    <section className="panel">
      <p className="eyebrow">{result.model_version}</p>
      <h2>Hipnograma predicho</h2>
      <p>
        {result.eligible} épocas analizadas · {result.rows - result.eligible}{" "}
        excluidas
      </p>
      {result.profile && (
        <p>Canales utilizados · {prettyProfile(result.profile)}</p>
      )}
      <Hypnogram rows={result.epochs} stages={stages} />
      <div className="stages">
        {stages.map((stage) => (
          <div key={stage}>
            <span>{sleepNames[stage]}</span>
            <b>
              {(
                result.epochs.filter((r) => r.prediction === stage).length / 2
              ).toLocaleString("es")}{" "}
              min
            </b>
          </div>
        ))}
      </div>
      <button className="secondary" onClick={download}>
        Descargar probabilidades en CSV
      </button>
    </section>
  );
}
export function LocalLab({ stack = false }: { stack?: boolean }) {
  const [file, setFile] = useState<File | null>(null);
  const [profile, setProfile] = useState("auto");
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function infer(example = false) {
    if (!file && !example) return;
    setError("");
    setResult(null);
    setBusy(true);
    try {
      const body = new FormData();
      if (!example && file) {
        if (!file.size || file.size > 128 * 1024 ** 2)
          throw Error("Selecciona un EDF de hasta 128 MiB.");
        body.append("file", file);
        if (!stack) body.append("profile", profile);
      }
      const response = await fetch(example ? "/v1/edf/example" : "/v1/edf", {
        method: "POST",
        body,
      });
      const payload = await response.json();
      if (!response.ok)
        throw Error(payload.error ?? "No se pudo procesar el registro.");
      setResult(payload);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  function download() {
    if (!result?.epochs.length) return;
    const keys = Object.keys(result.epochs[0]);
    const rows = result.epochs.map((row) =>
      keys.map((k) => row[k as keyof Epoch] ?? "").join(","),
    );
    const url = URL.createObjectURL(
      new Blob([[keys.join(","), ...rows].join("\n")], { type: "text/csv" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "hipnograma.csv";
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <>
      <header>
        <strong>MAIA</strong>
        <span>Laboratorio del sueño</span>
      </header>
      <main>
        <h1>Análisis del sueño</h1>
        <section className="panel">
          <h2>Analizar un registro EDF</h2>
          <label htmlFor="night">Polisomnografía · EDF, hasta 128 MiB</label>
          <input
            id="night"
            type="file"
            accept=".edf"
            onChange={(e) => {
              setFile(e.target.files?.[0] ?? null);
              setResult(null);
            }}
          />
          {stack ? (
            <p>
              Se requieren ambos EEG, EOG horizontal a 100 Hz y la envolvente
              EMG submental a 1 Hz.
            </p>
          ) : (
            <ProfilePicker value={profile} change={setProfile} />
          )}
          <div className="actions">
            <button disabled={busy || !file} onClick={() => infer()}>
              Generar predicción
            </button>
            {stack && (
              <button
                className="secondary"
                disabled={busy}
                onClick={() => infer(true)}
              >
                Probar ejemplo público
              </button>
            )}
          </div>
          {busy && <p role="status">Procesando el registro completo…</p>}
          {error && <p role="alert">{error}</p>}
        </section>
        {result && <ResultPanel result={result} download={download} />}
        {stack && <Project />}
      </main>
    </>
  );
}
