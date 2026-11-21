import { useMemo, useState } from "react";
export type Epoch = { epoch_id: string; onset_s: number; prediction: string | null; eligible: boolean };
export const stages = ["W", "REM", "N1", "N2", "N3"];
export const sleepNames: Record<string, string> = { W: "Vigilia", REM: "REM", N1: "Sueño N1", N2: "Sueño N2", N3: "Sueño N3" };
export const stageOf = (row: Epoch) => row.eligible && stages.includes(row.prediction ?? "") ? row.prediction! : null;
export function Hypnogram({ rows }: { rows: Epoch[] }) {
  const sorted = useMemo(() => [...rows].sort((a, b) => a.onset_s - b.onset_s), [rows]);
  const [selected, setSelected] = useState(0);
  const [zoom, setZoom] = useState(false);
  if (!sorted.length) return <p>No hay épocas disponibles para visualizar.</p>;
  const index = Math.min(selected, sorted.length - 1);
  const row = sorted[index];
  const visible = zoom ? sorted.slice(Math.max(0, index - 30), Math.min(sorted.length, index + 31)) : sorted;
  const first = visible[0].onset_s;
  const duration = visible.at(-1)!.onset_s + 30 - first;
  const px = (sec: number) => 70 + (sec - first) / duration * 800;
  const py = (stage: string) => 30 + stages.indexOf(stage) * 36;
  let line = "", last = -Infinity;
  for (const item of visible) {
    const stage = stageOf(item);
    if (!stage) { last = -Infinity; continue; }
    line += `${last + 30 === item.onset_s ? " L" : " M"}${px(item.onset_s)} ${py(stage)} H${px(item.onset_s + 30)}`;
    last = item.onset_s;
  }
  return <>
    <div className="chart-toolbar"><p>Épocas de 30 s · Los espacios indican datos sin predicción.</p><button className="secondary" aria-pressed={zoom} onClick={() => setZoom(!zoom)}>{zoom ? "Ver noche completa" : "Ampliar selección"}</button></div>
    <div className="chart-scroll">
      <svg viewBox="0 0 900 223" role="img" aria-label="Hipnograma. Eje horizontal en minutos desde el inicio del registro, eje vertical por estadio. Selecciona una época con el control inferior." onClick={e => {
        const rect = e.currentTarget.getBoundingClientRect();
        const seconds = first + Math.max(0, Math.min(1, ((e.clientX - rect.left) / rect.width * 900 - 70) / 800)) * duration;
        const nearest = sorted.reduce((best, item, i) => Math.abs(item.onset_s + 15 - seconds) < Math.abs(sorted[best].onset_s + 15 - seconds) ? i : best, 0);
        setSelected(nearest);
      }}>
        {stages.map((stage, i) => <g key={stage}><text x="14" y={35 + i * 36}>{stage}</text><line x1="65" x2="870" y1={30 + i * 36} y2={30 + i * 36} /></g>)}
        <rect x={px(row.onset_s)} y="15" width={Math.max(2, 800 * 30 / duration)} height="175" fill="#e8a522" opacity="0.25" />
        <path d={line} fill="none" stroke="#14776e" strokeWidth="2.5" />
        {[0, .25, .5, .75, 1].map(x => <text key={x} x={70 + x * 800} y="210" textAnchor="middle">{((first + x * duration) / 60).toLocaleString("es", { maximumFractionDigits: 1 })} min</text>)}
      </svg>
    </div>
    <label htmlFor="epoch">Explorar época (usa las flechas del teclado)</label>
    <input id="epoch" type="range" min="0" max={sorted.length - 1} value={index} onChange={e => setSelected(Number(e.target.value))} aria-valuetext={`Época ${index + 1}, ${row.onset_s / 60} minutos, ${sleepNames[stageOf(row) ?? ""] ?? "Sin predicción"}`} />
    <div className="epoch-detail" role="status">Época {index + 1} de {sorted.length} · {row.onset_s / 60}–{(row.onset_s + 30) / 60} min · <strong>{sleepNames[stageOf(row) ?? ""] ?? "Sin predicción"}</strong><span>ID: {row.epoch_id} · {row.eligible ? "Elegible" : "Excluida"}</span></div>
  </>;
}
