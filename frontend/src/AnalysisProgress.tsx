const steps = ["Preparar entrada", "Subir archivo", "Enviar análisis", "Calcular variables y predecir", "Cargar hipnograma"];
export function AnalysisProgress({ step, example, failed }: { step: number; example: boolean; failed: boolean }) {
  if (step < 0) return null;
  return <section className="analysis-progress" aria-label="Progreso del análisis">
    <h3>{failed ? "El análisis se detuvo" : step === 5 ? "Análisis completado" : "Tu análisis, paso a paso"}</h3>
    <ol>{steps.map((label, index) => {
      const skipped = example && index === 1;
      const done = index < step;
      const current = index === step;
      return <li key={label} className={skipped ? "skipped" : current && failed ? "failed" : done ? "done" : current ? "current" : "pending"} aria-current={current ? "step" : undefined}>
        <span className="step-icon" aria-hidden="true">{skipped ? "–" : done ? "✓" : current && failed ? "!" : index + 1}</span>
        <span>{label}<small>{skipped ? "No aplica: ejemplo del servidor" : current && failed ? "Requiere atención" : done ? "Completado" : current ? "En curso" : "Pendiente"}</small></span>
      </li>;
    })}</ol>
    {step === 3 && !failed && <p className="caption">El servidor está procesando el registro.</p>}
  </section>;
}
