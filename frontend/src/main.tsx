import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
import maiaBanner from "./assets/maia-banner.png";
import { Hypnogram, stages, sleepNames, stageOf, type Epoch } from "./Hypnogram";
import { demoResult } from "./demo";
import { Project } from "./project";
import { Team } from "./Team";
import { AnalysisProgress } from "./AnalysisProgress";
import { SleepSummary } from "./SleepSummary";
const releaseVersion = "stack-five-a64415fc2bd6";

type Config = { api: string; resultProxy?: string; uploadProxy?: string; storageHost?: string };
type Result = {
  model_version: string;
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
// Bound each network operation so a stalled service cannot lock the UI forever.
async function request(url: string, options: RequestInit = {}, timeout = 30000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (error) {
    if (controller.signal.aborted) throw Error("El servicio tardó demasiado en responder. Vuelve a intentarlo o explora el modo demo.");
    throw error;
  } finally {
    clearTimeout(timer);
  }
}
function errorMessage(error: unknown) {
  return error instanceof TypeError
    ? "No se pudo conectar con el servicio. Revisa tu conexión y vuelve a intentarlo, o explora el modo demo."
    : error instanceof Error ? error.message : "No se pudo completar la solicitud.";
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

function App({ cfg }: { cfg: Config | null }) {
  const tabFromHash = () => location.hash === "#nosotros" ? "team" : ["#modelo", "#experimentos", "#resumen-proyecto", "#arquitectura", "#repositorio"].includes(location.hash) ? "model" : "analysis";
  const [tab, setTab] = useState(tabFromHash);
  useEffect(() => {
    const sync = () => setTab(tabFromHash());
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);
  function selectTab(next: string) {
    setTab(next);
    const url = new URL(location.href);
    url.hash = next === "team" ? "nosotros" : next === "model" ? "modelo" : "analizar";
    history.pushState(null, "", url);
  }
  const [available, setAvailable] = useState<boolean | null>(null);
  const [availabilityRetry, setAvailabilityRetry] = useState(0);
  useEffect(() => {
    if (!cfg) return;
    let active = true;
    setAvailable(null);
    request(cfg.api + "/public/v1/examples/default", { cache: "no-store" })
      .then(async response => response.ok && (await response.json()).model_version === releaseVersion)
      .then(value => { if (active) setAvailable(value); })
      .catch(() => { if (active) setAvailable(false); });
    return () => { active = false; };
  }, [cfg?.api, availabilityRetry]);
  const [demo, setDemo] = useState(false);
  const [retry, setRetry] = useState(0);
  const [downloading, setDownloading] = useState(false);
  const [busy, setBusy] = useState(""),
    [error, setError] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [job, setJob] = useState(
    () =>
      new URLSearchParams(location.search).get("job") ??
      sessionStorage.getItem("maia-stack-five-job") ??
      "",
  );
  const [step, setStep] = useState(job ? 3 : -1);
  const [progressExample, setProgressExample] = useState(() => sessionStorage.getItem("maia-source-" + job) === "example");
  const [result, setResult] = useState<Result | null>(null),
    [csv, setCsv] = useState("");
  function resultUrl(value: string) {
    const url = new URL(value, location.origin);
    return cfg?.resultProxy && cfg.storageHost === url.hostname && url.protocol === "https:"
      ? cfg.resultProxy + "?url=" + encodeURIComponent(value)
      : value;
  }
  async function api(path: string, opts: RequestInit = {}) {
    if (!cfg) throw Error("La configuración del servicio no está disponible. Puedes explorar el modo demo.");
    const r = await request(cfg.api + "/public" + path, {
      ...opts,
      headers: {
        "X-Guest-Token": guestToken(),
        ...(opts.body ? { "Content-Type": "application/json" } : {}),
        ...opts.headers,
      },
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      throw Error(r.status >= 500 ? `El servicio no pudo completar la solicitud (HTTP ${r.status}). Puedes reintentar o explorar el modo demo.` : data.error ?? data.message ?? `Solicitud rechazada (${r.status})`);
    }
    return data;
  }
  useEffect(() => {
    if (!job || demo || !cfg) return;
    setError("");
    setBusy("Consultando el estado del análisis…");
    setStep(3);
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
        setStep(4);
        setBusy("Cargando el hipnograma…");
        const f = await request(resultUrl(status.files.json));
        if (!f.ok) throw Error("No se pudo descargar el resultado.");
        const data = await f.json();
        if (!Array.isArray(data.epochs) || !data.epochs.every((e: Epoch) => typeof e.epoch_id === "string" && Number.isFinite(e.onset_s) && typeof e.eligible === "boolean") || !Number.isFinite(data.rows) || !Number.isFinite(data.eligible)) throw Error("El resultado recibido no tiene un formato válido.");
        if (data.model_version !== releaseVersion) throw Error("Este resultado pertenece a una versión anterior. Vuelve a analizar el EDF con el modelo publicado.");
        if (active) {
          setStep(5);
          setResult(data);
          setCsv(status.files.csv);
          setBusy("");
        }
      } catch (e) {
        if (active) {
          setError(errorMessage(e));
          setBusy("");
        }
      }
    };
    poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [job, cfg?.api, demo, retry]);
  async function run(example: boolean) {
    if (!available || !cfg) return;
    setDemo(false);
    setStep(0);
    setProgressExample(example);
    setError("");
    sessionStorage.removeItem("maia-stack-five-job");
    const url = new URL(location.href);
    url.searchParams.delete("job");
    history.replaceState(null, "", url);
    setBusy(example ? "Enviando ejemplo…" : "Verificando archivo…");
    setResult(null);
    setCsv("");
    setJob("");
    try {
      let input: Record<string, string> = { example: "default", model: "stack_five" };
      if (!example) {
        if (!file || !file.name.toLowerCase().endsWith(".edf") || file.size > 128 * 1024 ** 2 || file.size === 0)
          throw Error("Selecciona una polisomnografía EDF de hasta 128 MiB.");
        const digest = Array.from(
          new Uint8Array(
            await crypto.subtle.digest("SHA-256", await file.arrayBuffer()),
          ),
          (x) => x.toString(16).padStart(2, "0"),
        ).join("");
        const form = await api("/v1/uploads", {
          method: "POST",
          body: JSON.stringify({ size: file.size, sha256: digest, format: "edf", model: "stack_five" }),
        });
        const body = new FormData();
        for (const [k, v] of Object.entries(form.fields))
          body.append(k, String(v));
        body.append("file", file);
        setStep(1);
        setBusy("Subiendo el archivo de la noche…");
        const destination = new URL(form.url, location.origin);
        const uploadUrl = cfg.uploadProxy && destination.protocol === "https:" && destination.hostname === cfg.storageHost
          ? cfg.uploadProxy + "?url=" + encodeURIComponent(form.url) : form.url;
        const r = await request(uploadUrl, { method: "POST", body }, 120000);
        const version = r.headers.get("x-amz-version-id");
        if (!r.ok || !version)
          throw Error("No se pudo confirmar la versión del archivo subido.");
        input = { upload_id: form.upload_id, version_id: version, format: "edf", model: "stack_five" };
      }
      setStep(2);
      setBusy("Enviando el análisis…");
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
      if (typeof accepted.job_id !== "string" || !accepted.job_id) throw Error("El servicio no devolvió un identificador de análisis válido.");
      url.hash = location.hash;
      url.searchParams.set("job", accepted.job_id);
      history.replaceState(null, "", url);
      sessionStorage.setItem("maia-stack-five-job", accepted.job_id);
      sessionStorage.setItem("maia-source-" + accepted.job_id, example ? "example" : "upload");
      setJob(accepted.job_id);
      setBusy("Procesando la noche…");
    } catch (e) {
      setBusy("");
      setError(errorMessage(e));
    }
  }
  function resetAnalysis() {
    if (busy || downloading) return;
    sessionStorage.removeItem("maia-stack-five-job");
    sessionStorage.removeItem("maia-guest-job");
    if (job) sessionStorage.removeItem("maia-source-" + job);
    const url = new URL(location.href);
    url.searchParams.delete("job");
    url.hash = "analizar";
    history.replaceState(null, "", url);
    setJob("");
    setResult(null);
    setCsv("");
    setFile(null);
    setFileInputKey(key => key + 1);
    setDemo(false);
    setError("");
    setStep(-1);
    setProgressExample(false);
    setTab("analysis");
    document.getElementById("analysis-title")?.focus({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: "instant" });
  }
  function showDemo() {
    setDemo(true);
    setStep(-1);
    setError("");
    setBusy("");
    setCsv("");
    setResult(demoResult());
  }
  async function download() {
    setDownloading(true);
    setError("");
    try {
      const j = await api("/v1/predictions/" + encodeURIComponent(job));
      if (!j.files?.csv) throw Error("El CSV aún no está disponible.");
      location.assign(resultUrl(j.files.csv));
    } catch (e) {
      setError(errorMessage(e));
    } finally { setDownloading(false); }
  }
  return (
    <>
      <a className="skip-link" href="#main">Saltar al contenido</a>
      <header className="site-header">
        <div className="header-inner">
          <div className="brand-logo"><img src={maiaBanner} alt="Maestría en Inteligencia Artificial" /></div>
          <div className="header-title"><span>Laboratorio del sueño</span><small>Aprendizaje de máquina aplicado a señales EEG</small></div>
          <a className="project-link" href="#proyecto">Acerca del proyecto <span aria-hidden="true">↗</span></a>
        </div>
      </header>
      <main id="main">
        <div className="intro">
          <p className="section-label">Análisis de señales EEG</p>
          <h1 id="analysis-title" tabIndex={-1}>Análisis del sueño</h1>
          <p>
            Explora los estadios del sueño y su evolución a lo largo de una noche, a partir de sus características fisiológicas.
          </p>
        </div>
        <div className="lab-tabs" role="tablist" aria-label="Secciones del laboratorio" onKeyDown={event => {
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          const tabs = ["analysis", "model", "team"];
          const index = tabs.indexOf(tab);
          const next = event.key === "Home" ? tabs[0] : event.key === "End" ? tabs[2] : tabs[(index + (event.key === "ArrowRight" ? 1 : 2)) % 3];
          selectTab(next);
          document.getElementById(`tab-${next}`)?.focus();
        }}>
          <button id="tab-analysis" role="tab" aria-selected={tab === "analysis"} aria-controls="panel-analysis" tabIndex={tab === "analysis" ? 0 : -1} onClick={() => selectTab("analysis")}>Analizar una noche{busy && <span className="tab-progress"> · En proceso</span>}</button>
          <button id="tab-model" role="tab" aria-selected={tab === "model"} aria-controls="panel-model" tabIndex={tab === "model" ? 0 : -1} onClick={() => selectTab("model")}>Sobre el proyecto</button>
          <button id="tab-team" role="tab" aria-selected={tab === "team"} aria-controls="panel-team" tabIndex={tab === "team" ? 0 : -1} onClick={() => selectTab("team")}>Sobre nosotros</button>
        </div>
        <div id="panel-analysis" role="tabpanel" aria-labelledby="tab-analysis" hidden={tab !== "analysis"}>
        {(result || error || job || file) && <div className="reset-analysis"><button className="secondary" disabled={!!busy || downloading} onClick={resetAnalysis}>Nuevo análisis</button><span>Volver al inicio sin resultados</span></div>}
        {!cfg && <p role="alert">No se pudo cargar la configuración del servicio. El modo demo sigue disponible. <button className="secondary" onClick={() => location.reload()}>Recargar configuración</button></p>}
        <section className="panel demo-panel" aria-label="Modo demo">
          <div><h2>Demostración</h2><p>Prueba el hipnograma con datos ficticios, sin subir un archivo.</p></div>
          <button className="secondary" disabled={!!busy} onClick={showDemo}>Explorar modo demo</button>
          {demo && <button className="secondary" onClick={() => { setDemo(false); setResult(null); setRetry(n => n + 1); }}>Salir del modo demo</button>}
        </section>
        <section className="panel" aria-busy={!!busy}>
          <h2>Analizar una noche</h2>
          <p id="edf-help">Carga una polisomnografía EDF con EEG Fpz-Cz, EEG Pz-Oz y EOG horizontal a 100 Hz, y envolvente EMG submental a 1 Hz. El análisis utiliza el registro completo en intervalos de 30 segundos.</p>
          <label htmlFor="night">
            Polisomnografía · EDF, hasta 128 MiB
          </label>
          <input
            key={fileInputKey}
            id="night"
            type="file"
            accept=".edf"
            disabled={!!busy}
            aria-describedby="edf-help file-info"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <p id="file-info">{file ? `${file.name} · ${(file.size / 1024 ** 2).toFixed(2)} MiB` : "Ningún archivo seleccionado. Máximo: 128 MiB."}</p>
          {cfg && available === null && <p role="status">Comprobando disponibilidad del modelo…</p>}
          {cfg && available === false && <div className="notice" role="status"><p>El servicio aún no confirma la versión del modelo EDF. Puedes explorar el modo demo o consultar cómo funciona el modelo.</p><button className="secondary" onClick={() => setAvailabilityRetry(n => n + 1)}>Comprobar de nuevo</button></div>}
          <p className="caption">El ejemplo público procesa diez minutos de señal sintética en el servidor. El modo demo solo muestra datos ficticios locales.</p>
          <div className="actions">
            <button disabled={!!busy || !file || !cfg || !available} onClick={() => run(false)}>
              Generar predicción
            </button>
            <button
              className="secondary"
              disabled={!!busy || !cfg || !available}
              onClick={() => run(true)}
            >
              Probar ejemplo público
            </button>
          </div>
          {!demo && <AnalysisProgress step={step} example={progressExample} failed={!!error && !result} />}
          {busy && <p className="notice" role="status">{busy} Puedes mantener esta página abierta; al aceptar el análisis, podrás recargarla para consultar su estado.</p>}
          {error && <div className="notice error" role="alert"><p>{error}</p>{error.includes("EMG") && <p>El EDF no cumple el requisito de entrada: se necesita la envolvente EMG submental a 1 Hz. Un canal EMG crudo a otra frecuencia no es equivalente; comprueba la exportación y el preprocesamiento del registro.</p>}{job && !demo && <button className="secondary" disabled={!!busy} onClick={() => setRetry(n => n + 1)}>Reintentar consulta</button>}</div>}
        </section>
        {result && (
          <section className="panel">
            <p className="eyebrow">{result.model_version}</p>
            <h2>{demo ? "Hipnograma de demostración" : "Hipnograma predicho"}</h2>
            {demo && <p className="notice demo-notice"><strong>MODO DEMO · DATOS SINTÉTICOS</strong><br />Esta noche es ficticia. No se ha analizado un archivo ni ejecutado el modelo.</p>}
            <p>
              {result.eligible} épocas analizadas ·{" "}
              {result.rows - result.eligible} excluidas
            </p>
            {result.profile && <p>Canales utilizados · {result.profile.split("+").map(channel => ({ eeg_fpz_cz: "EEG Fpz-Cz", eeg_pz_oz: "EEG Pz-Oz", eog: "EOG", emg: "EMG" }[channel] ?? channel)).join(" + ")}</p>}
            <Hypnogram key={demo ? "demo" : job} rows={result.epochs} />
            <div className="stages">
              {stages.map((s) => (
                <div key={s}>
                  <span>{sleepNames[s]}</span>
                  <b>
                    {(
                      result.epochs.filter((r) => stageOf(r) === s).length / 2
                    ).toLocaleString("es")}{" "}
                    min
                  </b>
                </div>
              ))}
            </div>
            <SleepSummary key={"summary-" + (demo ? "demo" : job)} rows={result.epochs} source={demo ? "demo" : sessionStorage.getItem("maia-source-" + job) ?? "unknown"} />
            {csv && (
              <button className="secondary" disabled={downloading} onClick={download}>
                {downloading ? "Preparando descarga…" : "Descargar probabilidades en CSV"}
              </button>
            )}
          </section>
        )}
        </div>
        <div id="panel-model" role="tabpanel" aria-labelledby="tab-model" hidden={tab !== "model"} tabIndex={0}>
          <Project />
        </div>
        <div id="panel-team" role="tabpanel" aria-labelledby="tab-team" hidden={tab !== "team"} tabIndex={0}><Team /></div>
      </main>
      <footer id="proyecto" className="project-footer">
        <div className="footer-inner">
          <div className="project-description">
            <p className="section-label">Sobre el proyecto</p>
            <h2>Clasificación automática de estadios del sueño mediante aprendizaje de máquina a partir de señales EEG</h2>
            <p>Esta solución forma parte del proyecto de la Maestría en Inteligencia Artificial.</p>
          </div>
          <div className="project-team"><h3>Conoce el equipo</h3><p>Las personas detrás de esta solución.</p><a href="#nosotros">Sobre nosotros →</a></div>
        </div>
      </footer>
    </>
  );
}
const root = createRoot(document.getElementById("root")!);
root.render(<main><p role="status">Cargando el laboratorio…</p></main>);
(async () => {
  const response = await request("/config.json", { cache: "no-store" }, 10000);
  if (!response.ok) throw Error();
  const cfg: Config = await response.json();
  if (typeof cfg.api !== "string") throw Error("Configuración inválida");
  root.render(<App cfg={cfg} />);
})().catch(() => root.render(<App cfg={null} />));
