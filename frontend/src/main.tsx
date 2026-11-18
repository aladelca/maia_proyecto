import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { UserManager, WebStorageStateStore, User } from "oidc-client-ts";
import "./style.css";

type Config = {
  issuer: string;
  clientId: string;
  api: string;
  login: string;
  origin: string;
};
type Epoch = {
  epoch_id: string;
  onset_s: number;
  prediction: string | null;
  eligible: boolean;
};
type Result = {
  model_version: string;
  epochs: Epoch[];
  eligible: number;
  rows: number;
};
type Job = {
  state: "processing" | "succeeded" | "failed";
  reason?: string;
  files?: { json: string; csv: string };
};
const stages = ["W", "REM", "N1", "N2", "N3"];
const sleepNames: Record<string, string> = {
  W: "Vigilia",
  REM: "REM",
  N1: "Sueño N1",
  N2: "Sueño N2",
  N3: "Sueño N3",
};

function Hypnogram({ rows }: { rows: Epoch[] }) {
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
      viewBox="0 0 900 223"
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
        <text key={x} x={70 + x * 800} y="210" textAnchor="middle">
          {Math.round(x * minutes)} min
        </text>
      ))}
    </svg>
  );
}
function App({ cfg, auth }: { cfg: Config; auth: UserManager }) {
  const [user, setUser] = useState<User | null>(null),
    [ready, setReady] = useState(false),
    [busy, setBusy] = useState(""),
    [error, setError] = useState("");
  const [file, setFile] = useState<File | null>(null),
    [job, setJob] = useState(""),
    [result, setResult] = useState<Result | null>(null),
    [csv, setCsv] = useState("");
  useEffect(() => {
    (async () => {
      try {
        const q = new URLSearchParams(location.search);
        let u: User | null;
        if (q.has("code") || q.has("error")) {
          u = await auth.signinRedirectCallback();
          history.replaceState(null, "", location.pathname);
        } else u = await auth.getUser();
        if (u && !u.expired) {
          setUser(u);
          setJob(
            new URLSearchParams(location.search).get("job") ??
              sessionStorage.getItem("maia-job-" + u.profile.sub) ??
              "",
          );
        }
      } catch {
        setError("La sesión no pudo completarse. Vuelve a iniciar sesión.");
      } finally {
        setReady(true);
      }
    })();
  }, [auth]);
  async function api(path: string, opts: RequestInit = {}) {
    const r = await fetch(cfg.api + path, {
      ...opts,
      headers: {
        Authorization: "Bearer " + user?.access_token,
        ...(opts.body ? { "Content-Type": "application/json" } : {}),
        ...opts.headers,
      },
    });
    const data = await r.json();
    if (!r.ok) {
      if (r.status === 401) {
        setUser(null);
        setJob("");
        setResult(null);
        setCsv("");
        setBusy("");
        await auth.removeUser();
        throw Error(
          "Tu sesión venció. Vuelve a iniciar sesión para consultar el resultado.",
        );
      }
      throw Error(data.error ?? "Solicitud rechazada (" + r.status + ")");
    }
    return data;
  }
  useEffect(() => {
    if (!job || !user) return;
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
        if (active) {
          setResult(data);
          setCsv(status.files.csv);
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
  }, [job, user]);
  async function run(example: boolean) {
    setError("");
    setBusy(example ? "Enviando ejemplo…" : "Verificando archivo…");
    setResult(null);
    setCsv("");
    setJob("");
    try {
      let input: Record<string, string> = { example: "default" };
      if (!example) {
        if (!file || file.size > 32 * 1024 ** 2 || file.size === 0)
          throw Error("Selecciona un archivo Parquet de hasta 32 MiB.");
        const digest = Array.from(
          new Uint8Array(
            await crypto.subtle.digest("SHA-256", await file.arrayBuffer()),
          ),
          (x) => x.toString(16).padStart(2, "0"),
        ).join("");
        const form = await api("/v1/uploads", {
          method: "POST",
          body: JSON.stringify({ size: file.size, sha256: digest }),
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
        input = { upload_id: form.upload_id, version_id: version };
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
      sessionStorage.setItem("maia-job-" + user?.profile.sub, accepted.job_id);
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
        {user && (
          <button
            className="link"
            onClick={async () => {
              await auth.removeUser();
              location.assign(
                cfg.login +
                  "/logout?client_id=" +
                  encodeURIComponent(cfg.clientId) +
                  "&logout_uri=" +
                  encodeURIComponent(cfg.origin + "/"),
              );
            }}
          >
            Cerrar sesión
          </button>
        )}
      </header>
      <main>
        <div className="intro">
          <p className="eyebrow">SLEEP CASSETTE · 30 S</p>
          <h1>Análisis del sueño</h1>
          <p>
            Explora la secuencia de estadios de una noche completa a partir de
            sus características fisiológicas.
          </p>
        </div>
        <section className="panel">
          <h2>{user ? "Analizar una noche" : "Acceso al laboratorio"}</h2>
          {!user ? (
            <>
              <p>
                Inicia sesión para procesar una noche preparada o probar el
                modelo con un registro público.
              </p>
              <button
                disabled={!ready}
                onClick={() =>
                  auth
                    .signinRedirect()
                    .catch(() =>
                      setError("No se pudo abrir el inicio de sesión."),
                    )
                }
              >
                Iniciar sesión
              </button>
            </>
          ) : (
            <>
              <label htmlFor="night">
                Archivo de características · Parquet, hasta 32 MiB
              </label>
              <input
                id="night"
                type="file"
                accept=".parquet"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
              <div className="actions">
                <button disabled={!!busy || !file} onClick={() => run(false)}>
                  Generar predicción
                </button>
                <button
                  className="secondary"
                  disabled={!!busy}
                  onClick={() => run(true)}
                >
                  Probar ejemplo público
                </button>
              </div>
            </>
          )}
          {busy && <p role="status">{busy}</p>}
          {error && <p role="alert">{error}</p>}
        </section>
        {result && (
          <section className="panel">
            <p className="eyebrow">{result.model_version}</p>
            <h2>Hipnograma predicho</h2>
            <p>
              {result.eligible} épocas analizadas ·{" "}
              {result.rows - result.eligible} excluidas
            </p>
            <Hypnogram rows={result.epochs} />
            <div className="stages">
              {stages.map((s) => (
                <div key={s}>
                  <span>{sleepNames[s]}</span>
                  <b>
                    {(
                      result.epochs.filter((r) => r.prediction === s).length / 2
                    ).toLocaleString("es")}{" "}
                    min
                  </b>
                </div>
              ))}
            </div>
            {csv && (
              <button className="secondary" onClick={download}>
                Descargar probabilidades en CSV
              </button>
            )}
          </section>
        )}
      </main>
    </>
  );
}
const root = createRoot(document.getElementById("root")!);
(async () => {
  const response = await fetch("/config.json", { cache: "no-store" });
  if (!response.ok) throw Error();
  const cfg: Config = await response.json();
  const auth = new UserManager({
    authority: cfg.issuer,
    client_id: cfg.clientId,
    redirect_uri: cfg.origin + "/",
    response_type: "code",
    scope: "openid aws.cognito.signin.user.admin",
    userStore: new WebStorageStateStore({ store: sessionStorage }),
    automaticSilentRenew: false,
    loadUserInfo: false,
  });
  root.render(<App cfg={cfg} auth={auth} />);
})().catch(() =>
  root.render(
    <main role="alert">
      No se pudo cargar la configuración del laboratorio.
    </main>,
  ),
);
