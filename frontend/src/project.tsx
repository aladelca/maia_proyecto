import architecture from "./assets/arquitectura-solucion.png";
const repo = "https://github.com/aladelca/maia_proyecto";
const report = repo + "/blob/main/docs/comparacion-fullnight-lightgbm.md";

export function Project() {
  return (
    <>
      <section className="panel" id="resumen-proyecto">
        <h2>Clasificación automática de estadios del sueño mediante aprendizaje de máquina a partir de señales EEG</h2>
        <p>
          Esta solución forma parte del proyecto de grado de la Maestría en Inteligencia Artificial (MAIA) de la Universidad de los Andes. Busca estimar la secuencia de vigilia, N1, N2, N3 y REM de una noche a
          partir de dos derivaciones de electroencefalografía, una señal de
          movimientos oculares y una envolvente de actividad muscular. Cada
          punto del hipnograma representa 30 segundos.
        </p>
      </section>
      <section className="panel repository-panel" id="repositorio">
        <p className="eyebrow">CÓDIGO Y DOCUMENTACIÓN</p>
        <h2>Repositorio del proyecto</h2>
        <p>Consulta el código fuente, la documentación técnica, los experimentos y el historial de desarrollo de la solución.</p>
        <a className="repository-link" href={repo} target="_blank" rel="noreferrer">Ver repositorio en GitHub <span aria-hidden="true">↗</span></a>
      </section>
      <section className="panel" id="arquitectura">
        <h2>Arquitectura de la solución</h2>
        <p>Del archivo de una noche a una visualización de sus estadios: así se conectan la carga, el procesamiento y los resultados.</p>
        <figure className="architecture-figure">
          <a href={architecture} target="_blank" rel="noreferrer" aria-label="Ampliar el diagrama de arquitectura de la solución">
            <img src={architecture} width="1536" height="1024" alt="Flujo: cargar EDF, guardar en S3, preparar intervalos de 30 segundos, combinar 435 características fisiológicas con 5 probabilidades de FullNightNet, clasificar con LightGBM y mostrar hipnograma, CSV y resumen." />
          </a>
          <figcaption>Diagrama ilustrativo del análisis de un registro. El modelo ya está entrenado. Pulsa la imagen para ampliarla.</figcaption>
        </figure>
        <details className="record-details"><summary>Leer el flujo paso a paso</summary>
          <ol className="architecture-steps">
            <li><strong>Cargar:</strong> se recibe un EDF con los canales EEG, EOG y EMG requeridos.</li>
            <li><strong>Guardar y procesar:</strong> el archivo se carga a S3 y se inicia un trabajo en AWS. La interfaz consulta su estado.</li>
            <li><strong>Preparar:</strong> se validan las señales y se organizan en intervalos de 30 segundos.</li>
            <li><strong>Combinar:</strong> se obtienen 435 características con contexto temporal y cinco probabilidades de FullNightNet.</li>
            <li><strong>Clasificar:</strong> LightGBM utiliza las 440 entradas para estimar W, N1, N2, N3 o REM.</li>
            <li><strong>Explorar:</strong> se muestran el hipnograma, las probabilidades descargables y un resumen orientativo.</li>
          </ol>
        </details>
      </section>
      <section className="panel" id="modelo">
        <h2>Cómo predice el modelo</h2>
        <p>
          El mismo EDF alimenta dos ramas. LightGBM combina sus salidas para
          asignar una de las cinco etapas a cada intervalo.
        </p>
        <div className="facts">
          <div>
            <h3>435 características de las señales</h3>
            <p>
              La primera rama calcula 145 descriptores por época, como la
              potencia del EEG en distintas bandas y la amplitud del EMG. Añade
              otros 290 que resumen las épocas vecinas. El contexto utiliza
              también señales posteriores, por lo que el análisis se realiza con
              la noche grabada.
            </p>
          </div>
          <div>
            <h3>5 probabilidades de FullNightNet</h3>
            <p>
              Una red convolucional aprende patrones directamente de las ondas
              EEG y EOG, incorpora el EMG y observa la evolución temporal. Para
              cada época tomamos las probabilidades de W, N1, N2, N3 y REM y las
              normalizamos para que sumen uno.
            </p>
          </div>
          <div>
            <h3>Una decisión con LightGBM</h3>
            <p>
              El clasificador recibe las 440 entradas. Sus árboles aprenden
              cuándo las probabilidades de la red aportan información adicional
              a las características fisiológicas. El CSV contiene las cinco
              probabilidades finales y la etapa más probable.
            </p>
          </div>
        </div>
      </section>
      <section className="panel" id="experimentos">
        <h2>Por qué elegimos este modelo</h2>
        <p>
          En las mismas 31 noches reservadas de Sleep Cassette, añadir las cinco
          probabilidades elevó el F1 macro sobre EDF completos de 0,7630 a{" "}
          <strong>0,7806</strong>. La mejora fue de 1,76 puntos, con un
          intervalo de confianza del 95 % de +0,91 a +2,67. La probabilidad
          adicional de movimiento no mostró una mejora consistente.
        </p>
        <div className="scroll">
          <table className="compare">
            <caption>
              Resultados de la corrida publicada con cuatro canales
            </caption>
            <thead>
              <tr>
                <th scope="col">Evaluación</th>
                <th scope="col">Épocas</th>
                <th scope="col">F1 macro</th>
                <th scope="col">F1 de N1</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">Prueba · EDF completo</th>
                <td>83.875</td>
                <td>
                  <strong>0,7806</strong>
                </td>
                <td>0,4841</td>
              </tr>
              <tr>
                <th scope="row">Prueba · recorte W30</th>
                <td>43.417</td>
                <td>0,7970</td>
                <td>0,5326</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          El desarrollo usó 62 participantes y cinco pliegues separados por
          persona. Las probabilidades usadas para entrenar el híbrido proceden
          de redes que desconocían al participante evaluado. El F1 macro medio
          de desarrollo en W30 fue 0,7940. W30 conserva hasta 30 minutos de
          vigilia a cada lado del sueño anotado. La herramienta procesa el EDF
          completo.
        </p>
        <p className="caption">
          Corrida MLflow <code>a64415fc2bd649b39c6fd158994b0479</code>. La
          prueba incluyó 16 participantes de Sleep Cassette.{" "}
          <a href={report}>Consultar protocolo e intervalos por paciente</a>.
        </p>
      </section>
    </>
  );
}

