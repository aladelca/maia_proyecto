const repo = "https://github.com/aladelca/maia_proyecto";
const fixed = (x: number) =>
  x.toLocaleString("es", {
    minimumFractionDigits: 3,
    maximumFractionDigits: 3,
  });

const pipeline = [
  [
    "Datos",
    "Sleep-EDF Expanded: 153 noches de 78 sujetos de Sleep Cassette, 62 para desarrollo y 16 reservados para la prueba, más 44 noches de Sleep Telemetry para medir la transferencia.",
  ],
  [
    "Variables",
    "145 descriptores por época a partir de dos canales de EEG (Fpz-Cz y Pz-Oz), EOG y EMG, más 290 variables de contexto temporal calculadas con las épocas vecinas.",
  ],
  [
    "Modelos",
    "LightGBM ajustado con validación cruzada anidada por sujeto para noches preparadas. Para registros EDF completos, una red neuronal de registro completo con seis clases: los cinco estadios y movimiento.",
  ],
  [
    "Despliegue",
    "Interfaz React en Vercel; API, orquestación e inferencia sin servidor en AWS (API Gateway, Lambda, Step Functions y S3). Infraestructura en Terraform y entrega continua con GitHub Actions.",
  ],
];

// Solo cifras verificadas en MLflow (sc-w30-v2 e ibio-fullnight) y en los resultados de cada
// campaña, con SHA-256. Actualizar al promover otra versión.
const release = "lgbm-49f74e944ce5457b";
const headline: [string, number][] = [
  ["F1 macro", 0.789],
  ["Exactitud", 0.83],
  ["Kappa de Cohen", 0.769],
];
const stageF1: [string, number][] = [
  ["W · Vigilia", 0.924],
  ["N1", 0.516],
  ["N2", 0.849],
  ["N3", 0.808],
  ["REM", 0.854],
];
const live = "LightGBM ajustado";
// Modelo, variables, F1 macro, DE del F1 entre folds, exactitud y kappa.
const models: [string, number, number, number, number, number][] = [
  [live, 435, 0.789, 0.018, 0.83, 0.769],
  ["LightGBM · contexto", 435, 0.782, 0.016, 0.817, 0.754],
  ["BiLSTM ajustada · 15 épocas centradas", 435, 0.78, 0.017, 0.819, 0.755],
  ["CatBoost · contexto ajustado", 435, 0.775, 0.019, 0.821, 0.756],
  ["CatBoost · contexto", 435, 0.769, 0.02, 0.799, 0.733],
  ["LSTM · 15 épocas pasadas", 145, 0.767, 0.018, 0.805, 0.739],
  ["CatBoost · multicanal ampliado", 145, 0.731, 0.033, 0.764, 0.688],
  ["CatBoost · multicanal", 88, 0.727, 0.032, 0.76, 0.682],
  ["Random Forest · EEG Fpz-Cz", 25, 0.654, 0.021, 0.713, 0.612],
  ["CatBoost · EEG ampliado", 42, 0.649, 0.028, 0.692, 0.595],
  ["CatBoost · EEG Fpz-Cz", 25, 0.644, 0.029, 0.687, 0.588],
];
const fullNight: [string, number, number, number][] = [
  ["Cassette, prueba", 0.71, 0.74, 0.752],
  ["Telemetry", 0.608, 0.66, 0.697],
];

function Bar({ value }: { value: number }) {
  return (
    <div className="bar">
      <i aria-hidden="true" style={{ width: `${value * 80}%` }} />
      {fixed(value)}
    </div>
  );
}

export function Project() {
  return (
    <>
      <section className="panel" id="proyecto">
        <p className="eyebrow">PROYECTO DE GRADO · MAIA</p>
        <h2>Sobre el proyecto</h2>
        <p>
          Para estadificar el sueño, un especialista revisa la polisomnografía
          (EEG, EOG y EMG) y asigna un estadio a cada época de 30 segundos. Una
          noche tiene cerca de 900 épocas, así que el proceso toma horas, y la
          concordancia entre evaluadores ronda el 83 %. En el estadio de
          transición N1 es aún menor.
        </p>
        <p>
          Este proyecto entrena modelos que clasifican cada época en cinco
          estadios (W, N1, N2, N3 y REM) y los publica en esta aplicación, que
          dibuja el hipnograma estimado de la noche.
        </p>
        <div className="facts">
          {pipeline.map(([title, text]) => (
            <div key={title}>
              <h3>{title}</h3>
              <p>{text}</p>
            </div>
          ))}
        </div>
        <p>
          Proyecto de grado de la Maestría en Inteligencia Artificial de la
          Universidad de los Andes, desarrollado por Carlos Adrián Alarcón
          Delgado, Andrés Enrique Bautista Pabón y John Janer Buila Ferrin.{" "}
          <a href={repo}>Código y documentación en GitHub</a>.
        </p>
      </section>
      <section className="panel" id="modelo">
        <p className="eyebrow">{release}</p>
        <h2>Modelo en servicio</h2>
        <p>
          LightGBM multiclase dentro de un pipeline de scikit-learn, entrenado
          con los 62 sujetos de desarrollo: 122 noches y 152.052 épocas
          etiquetadas. Procesa noches preparadas en Parquet. Los registros EDF
          completos usan la red directa de seis clases de la campaña de registro
          completo (run 681a52f2), con los canales que elija el visitante.
        </p>
        <div className="kpis">
          {headline.map(([label, value]) => (
            <div key={label}>
              <span>{label}</span>
              <b>{fixed(value)}</b>
              {label === "F1 macro" && (
                <small>± {fixed(0.018)} entre folds</small>
              )}
            </div>
          ))}
        </div>
        <p className="caption">
          Validación cruzada de cinco folds por sujeto. MLflow:
          sleep-cassette-lgbm-bilstm-refinement-v1, run 6e742213; el modelo
          publicado es su reajuste final, run 99f262e3.
        </p>
        <div className="split">
          <dl>
            <div>
              <dt>Entrada</dt>
              <dd>
                145 variables por época y 290 de contexto temporal, 435 en
                total.
              </dd>
            </div>
            <div>
              <dt>Hiperparámetros</dt>
              <dd>
                261 iteraciones, 63 hojas, mínimo de 200 muestras por hoja, tasa
                de aprendizaje 0,05, L2 = 5, 80 % de filas y columnas y pesos
                inversos de clase.
              </dd>
            </div>
            <div>
              <dt>Inferencia</dt>
              <dd>En CPU, con AWS Lambda y Step Functions.</dd>
            </div>
          </dl>
          <table>
            <caption>F1 por estadio, predicciones fuera de muestra</caption>
            <tbody>
              {stageF1.map(([stage, f1]) => (
                <tr key={stage}>
                  <th scope="row">{stage}</th>
                  <td>
                    <Bar value={f1} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          N1, el estadio de transición, es el más difícil para el modelo y
          también aquel en el que más discrepan los especialistas.
        </p>
      </section>
      <section className="panel" id="experimentos">
        <p className="eyebrow">VALIDACIÓN CRUZADA · 5 FOLDS POR SUJETO</p>
        <h2>Experimentación y métricas</h2>
        <p>
          Todos los modelos usan los mismos 152.052 segmentos de desarrollo y
          cinco folds fijos por sujeto, de modo que las noches de una persona
          nunca se usan a la vez para entrenar y evaluar. Los hiperparámetros se
          eligieron con un 20 % interno de los sujetos de cada entrenamiento. La
          métrica principal es el F1 macro, que pondera por igual los cinco
          estadios.
        </p>
        <div className="scroll">
          <table className="compare">
            <caption>
              Noches preparadas. Media de cinco folds; DE: desviación del F1
              macro entre folds.
            </caption>
            <thead>
              <tr>
                <th>Modelo</th>
                <th className="num wide">Variables</th>
                <th>F1 macro</th>
                <th className="num wide">DE</th>
                <th className="num">Exactitud</th>
                <th className="num">Kappa</th>
              </tr>
            </thead>
            <tbody>
              {models.map(([model, inputs, f1, sd, acc, kappa]) => (
                <tr key={model} className={model === live ? "live" : undefined}>
                  <th scope="row">
                    {model} {model === live && <em>en servicio</em>}
                  </th>
                  <td className="num wide">{inputs}</td>
                  <td>
                    <Bar value={f1} />
                  </td>
                  <td className="num wide">{fixed(sd)}</td>
                  <td className="num">{fixed(acc)}</td>
                  <td className="num">{fixed(kappa)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="caption">
          MLflow: sleep-cassette-literature-v2, sleep-cassette-lstm-mps-v1 y
          sleep-cassette-lgbm-bilstm-refinement-v1.
        </p>
        <ul className="findings">
          <li>
            Combinar los dos canales de EEG con EOG y EMG fue el cambio con
            mayor aporte: +0,083 de F1 macro frente a usar solo Fpz-Cz.
          </li>
          <li>
            El contexto temporal de las épocas vecinas sumó +0,038. Ampliar los
            descriptores de un solo canal aportó poco (+0,006).
          </li>
          <li>
            El ajuste anidado llevó LightGBM de 0,782 a 0,789. El intervalo de
            confianza del 95 % de esa mejora, de 0,003 a 0,011, excluye el cero.
          </li>
          <li>
            Las redes LSTM y BiLSTM quedaron cerca, pero no superaron a
            LightGBM.
          </li>
        </ul>
        <h3>Registro completo (EDF)</h3>
        <p>
          Para procesar una polisomnografía entera, sin recortar la vigilia, se
          entrenó una mezcla de LightGBM con redes neuronales sobre las ondas y
          el espectrograma. Se evaluó en los 16 sujetos reservados de Sleep
          Cassette (31 noches) y en 22 personas de Sleep Telemetry (44 noches),
          una cohorte con otra adquisición y otra población.
        </p>
        <div className="scroll">
          <table className="compare">
            <caption>
              F1 macro de cinco clases en el registro completo, con ambos EEG y
              EOG. Doble: LightGBM y ondas. Triple: además, el espectrograma.
            </caption>
            <thead>
              <tr>
                <th>Muestra</th>
                <th className="num">LightGBM</th>
                <th className="num">Doble</th>
                <th className="num">Triple</th>
              </tr>
            </thead>
            <tbody>
              {fullNight.map(([sample, lgbm, double, triple]) => (
                <tr key={sample}>
                  <th scope="row">{sample}</th>
                  <td className="num">{fixed(lgbm)}</td>
                  <td className="num">{fixed(double)}</td>
                  <td className="num">
                    <b>{fixed(triple)}</b>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="caption">
          Campaña ibio-fullnight 681a52f2, MLflow run 8e5ba3bf. Resultados
          verificados por SHA-256.
        </p>
        <ul className="findings">
          <li>
            Añadir el espectrograma mejoró el F1 macro en 1,23 puntos en
            Cassette (IC 95 %: 0,65 a 1,82) y en 3,74 en Telemetry (2,42 a
            4,86).
          </li>
          <li>
            N1 sigue siendo el estadio más difícil: su F1 en Cassette prueba fue
            0,423.
          </li>
        </ul>
        <p className="note">
          Las dos campañas no son comparables entre sí. Las noches preparadas
          conservan solo 30 minutos de vigilia alrededor del sueño, recortados
          con las anotaciones de referencia, por lo que su desempeño corresponde
          a un análisis retrospectivo. El registro completo evalúa todas las
          épocas de la noche.
        </p>
        <p>
          <a href={`${repo}/blob/main/docs/refinement-results.md`}>
            Resultados de noches preparadas
          </a>{" "}
          ·{" "}
          <a href={`${repo}/blob/main/docs/ibio-registro-completo.md`}>
            Registro completo
          </a>{" "}
          · <a href={`${repo}/blob/main/modelos/README.md`}>Ficha del modelo</a>
        </p>
      </section>
    </>
  );
}
