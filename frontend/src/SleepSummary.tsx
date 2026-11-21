import { useState } from "react";
import { stageOf, type Epoch } from "./Hypnogram";

function duration(seconds: number) {
  const minutes = seconds / 60;
  return `${Math.floor(minutes / 60)} h ${(minutes % 60).toLocaleString("es")} min`;
}

export function SleepSummary({ rows, source }: { rows: Epoch[]; source: string }) {
  const [confirmed, setConfirmed] = useState(false);
  const sorted = [...rows].sort((a, b) => a.onset_s - b.onset_s);
  const sleep = rows.filter(row => { const stage = stageOf(row); return stage && stage !== "W"; }).length * 30;
  const wake = rows.filter(row => stageOf(row) === "W").length * 30;
  const unknown = rows.filter(row => !stageOf(row)).length * 30;
  const gaps = sorted.some((row, i) => i > 0 && row.onset_s - sorted[i - 1].onset_s !== 30);
  const complete = rows.length > 0 && !unknown && !gaps;
  const synthetic = source === "demo" || source === "example";
  return <section className="sleep-summary" aria-labelledby="sleep-summary-title">
    <p className="eyebrow">{synthetic ? "EJEMPLO ILUSTRATIVO" : "LECTURA DEL REGISTRO"}</p>
    <h3 id="sleep-summary-title">Tu resumen de sueño</h3>
    <div className="sleep-totals">
      <div><span>Sueño estimado</span><strong>{duration(sleep)}</strong><small>N1 + N2 + N3 + REM</small></div>
      <div><span>Vigilia estimada</span><strong>{duration(wake)}</strong><small>Dentro del registro</small></div>
      
    </div>
    {synthetic ? <p>Estos tiempos proceden de datos sintéticos. Sirven para explorar el resumen, no describen el descanso de una persona.</p> : <p>El modelo estima {duration(sleep)} de sueño en las épocas disponibles. Un registro puede ser un fragmento de la noche; este tiempo no mide por sí solo la calidad del descanso.</p>}
    <details className="record-details"><summary>Detalles del registro</summary><p>Tiempo sin clasificar: {duration(unknown)}. {gaps ? "También hay huecos o tiempos solapados." : "Las épocas están ordenadas en intervalos de 30 segundos."}</p>{!complete && <p>{rows.length ? "No comparamos esta estimación con una noche completa porque faltan datos clasificados o hay discontinuidades." : "No hay épocas disponibles para interpretar."}</p>}</details>
    {source === "upload" && complete && <label className="sleep-context"><input type="checkbox" checked={confirmed} onChange={e => setConfirmed(e.target.checked)} />El archivo es un registro real de una noche completa de una persona de 18 a 60 años.</label>}
    {source === "upload" && complete && confirmed && <div className="sleep-guidance" role="status">
      <h4>{sleep < 7 * 3600 ? "Una duración estimada por debajo de la referencia" : "La duración estimada alcanza la referencia mínima"}</h4>
      <p>{sleep < 7 * 3600 ? `El modelo estima ${duration(sleep)} de sueño. Para adultos de 18 a 60 años, la referencia general es dormir al menos 7 horas regularmente. Si esto refleja tu rutina, considera reservar más tiempo para dormir y revisar tus horarios.` : "La referencia general para adultos de 18 a 60 años es al menos 7 horas regularmente. Alcanzarla no garantiza un descanso reparador: también importa cómo te sientes durante el día."}</p>
    </div>}
    <h4>Hábitos que puedes probar</h4>
    <p className="caption">Orientación general, no una recomendación clínica personalizada.</p>
    <ul className="sleep-habits">
      <li><strong>Un horario constante.</strong> Intenta acostarte y levantarte a horas similares cada día.</li>
      <li><strong>Prepara el descanso.</strong> Apaga las pantallas al menos 30 minutos antes y mantén el dormitorio tranquilo y fresco.</li>
      <li><strong>Revisa la cafeína.</strong> Evítala por la tarde y la noche.</li>
    </ul>
    <p className="caption">Si los problemas para dormir o el cansancio persisten, consulta con un profesional de salud. Esta herramienta de investigación no diagnostica trastornos del sueño. <a href="https://www.cdc.gov/sleep/about/" target="_blank" rel="noreferrer">Fuente: recomendaciones de sueño de los CDC</a>.</p>
  </section>;
}
