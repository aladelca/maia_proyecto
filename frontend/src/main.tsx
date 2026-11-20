import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Project } from "./project";
import "./style.css";

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
function LocalLab({ stack = false }: { stack?: boolean }) {
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

function guestToken() {
  let token = sessionStorage.getItem("maia-guest-token");
  if (!token) {
    token = Array.from(crypto.getRandomValues(new Uint8Array(32)), (byte) =>
      byte.toString(16).padStart(2, "0"),
    ).join("");
    sessionStorage.setItem("maia-guest-token", token);
  }
  return token;
}

function App({ cfg }: { cfg: Config }) {
  const [available, setAvailable] = useState<boolean | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    fetch(cfg.api + "/public/v1/examples/default", {
      signal: controller.signal,
      cache: "no-store",
    })
      .then(
        async (response) =>
          response.ok &&
          (await response.json()).model_version === releaseVersion,
      )
      .then((active) => {
        if (!controller.signal.aborted) setAvailable(active);
      })
      .catch(() => {
        if (!controller.signal.aborted) setAvailable(false);
      });
    return () => controller.abort();
  }, [cfg.api]);
  const [busy, setBusy] = useState(""),
    [error, setError] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState(
    () =>
      new URLSearchParams(location.search).get("job") ??
      sessionStorage.getItem("maia-stack-five-job") ??
      "",
  );
  const [result, setResult] = useState<Result | null>(null);
  async function api(path: string, opts: RequestInit = {}) {
    const r = await fetch(cfg.api + "/public" + path, {
      ...opts,
      headers: {
        "X-Guest-Token": guestToken(),
        ...(opts.body ? { "Content-Type": "application/json" } : {}),
        ...opts.headers,
      },
    });
    const data = await r.json();
    if (!r.ok) {
      throw Error(data.error ?? "Solicitud rechazada (" + r.status + ")");
    }
    return data;
  }
  useEffect(() => {
    if (!job) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const status: Job = await api(
          "/v1/predictions/" + encodeURIComponent(job),
        );
        if (!active) return;
        if (status.state === "processing") {
          setBusy("Procesando la noche…");
          timer = setTimeout(poll, 1800);
          return;
        }
        if (status.state === "failed" || !status.files)
          throw Error(status.reason ?? "No se pudo procesar la noche.");
        const f = await fetch(status.files.json);
        if (!f.ok) throw Error("No se pudo descargar el resultado.");
        const data = await f.json();
        if (data.model_version !== releaseVersion)
          throw Error(
            "Este resultado pertenece a una versión anterior. Vuelve a analizar el EDF con el modelo publicado.",
          );
        if (active) {
          setResult(data);
          setBusy("");
        }
      } catch (e) {
        if (active) {
          setError(String((e as Error).message));
          setBusy("");
        }
      }
    };
    poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [job, cfg.api]);
  async function run(example: boolean) {
    setError("");
    setBusy(example ? "Enviando ejemplo…" : "Verificando archivo…");
    setResult(null);
    setJob("");
    try {
      let input: Record<string, string> = {
        example: "default",
        model: "stack_five",
      };
      if (!example) {
        if (
          !file ||
          !file.size ||
          file.size > 128 * 1024 ** 2 ||
          !/\.edf$/i.test(file.name)
        )
          throw Error("Selecciona una polisomnografía EDF de hasta 128 MiB.");
        const digest = Array.from(
          new Uint8Array(
            await crypto.subtle.digest("SHA-256", await file.arrayBuffer()),
          ),
          (x) => x.toString(16).padStart(2, "0"),
        ).join("");
        const form = await api("/v1/uploads", {
          method: "POST",
          body: JSON.stringify({
            size: file.size,
            sha256: digest,
            format: "edf",
            model: "stack_five",
          }),
        });
        const body = new FormData();
        for (const [k, v] of Object.entries(form.fields))
          body.append(k, String(v));
        body.append("file", file);
        setBusy("Subiendo la noche a S3…");
        const r = await fetch(form.url, { method: "POST", body });
        const version = r.headers.get("x-amz-version-id");
        if (!r.ok || !version)
          throw Error("No se pudo confirmar la versión del archivo subido.");
        input = {
          upload_id: form.upload_id,
          version_id: version,
          format: "edf",
          model: "stack_five",
        };
      }
      const key = crypto.randomUUID();
      const opts = {
        method: "POST",
        headers: { "Idempotency-Key": key },
        body: JSON.stringify(input),
      };
      let accepted;
      try {
        accepted = await api("/v1/predictions", opts);
      } catch (e) {
        if (!(e instanceof TypeError)) throw e;
        accepted = await api("/v1/predictions", opts);
      }
      history.replaceState(null, "", "?job=" + accepted.job_id);
      sessionStorage.setItem("maia-stack-five-job", accepted.job_id);
      setJob(accepted.job_id);
      setBusy("Procesando la noche…");
    } catch (e) {
      setBusy("");
      setError((e as Error).message);
    }
  }
  async function download() {
    try {
      const j = await api("/v1/predictions/" + job);
      location.assign(j.files.csv);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return (
    <>
      <header>
        <strong>MAIA</strong>
        <span>Laboratorio del sueño</span>
        <nav aria-label="Secciones">
          <a href="#analizar">Analizar</a>
          <a href="#proyecto">Proyecto</a>
          <a href="#modelo">Modelo</a>
          <a href="#experimentos">Resultados</a>
        </nav>
      </header>
      <main>
        <div className="intro">
          <p className="eyebrow">SLEEP CASSETTE · 30 S</p>
          <h1>Análisis del sueño</h1>
          <p>
            Sube una polisomnografía para obtener el hipnograma y la
            probabilidad de cada etapa del sueño en intervalos de 30 segundos.
          </p>
        </div>
        <section className="panel" id="analizar">
          <h2>Analizar una noche</h2>
          <label htmlFor="night">
            Polisomnografía en formato EDF · hasta 128 MiB
          </label>
          <input
            id="night"
            type="file"
            accept=".edf"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <p className="caption">
            Se requieren EEG Fpz-Cz, EEG Pz-Oz y EOG horizontal a 100 Hz, y la
            envolvente de EMG submental a 1 Hz. El ejemplo utiliza diez minutos
            de señal sintética.
          </p>
          {available === false && (
            <p role="status">
              La API aún no confirma la disponibilidad de esta versión. Vuelve a
              cargar la página para comprobarla.
            </p>
          )}
          <div className="actions">
            <button
              disabled={!!busy || !file || !available}
              onClick={() => run(false)}
            >
              Generar predicción
            </button>
            <button
              className="secondary"
              disabled={!!busy || !available}
              onClick={() => run(true)}
            >
              Probar ejemplo público
            </button>
          </div>
          {busy && <p role="status">{busy}</p>}
          {error && <p role="alert">{error}</p>}
        </section>
        {result && <ResultPanel result={result} download={download} />}
        <Project />
      </main>
    </>
  );
}
const root = createRoot(document.getElementById("root")!);
(async () => {
  const response = await fetch("/config.json", { cache: "no-store" });
  if (!response.ok) throw Error();
  const cfg: Config = await response.json();
  root.render(
    cfg.mode === "local-stack" ? (
      <LocalLab stack />
    ) : cfg.mode === "local" ? (
      <LocalLab />
    ) : (
      <App cfg={cfg} />
    ),
  );
})().catch(() =>
  root.render(
    <main role="alert">
      No se pudo cargar la configuración del laboratorio.
    </main>,
  ),
);
