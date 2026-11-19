"""Genera el borrador del artículo final (Word) con el formato del curso.

Formato: carta, márgenes de 2,5 cm, Times New Roman 10, interlineado sencillo,
texto justificado, secciones numeradas, tablas con título encima y figuras con
título y fuente debajo, referencias IEEE en orden de aparición.

Los textos marcados como [PENDIENTE] o [VERIFICAR] se resaltan en amarillo.

Uso (desde la raíz del repositorio):
    uv run --with python-docx python docs/articulo/construir_articulo.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "Articulo_final_borrador.docx"
FONT = "Times New Roman"

doc = Document()
_fig = 0
_tab = 0


# --------------------------------------------------------------------------- formato
def setup() -> None:
    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21.59), Cm(27.94)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(s, side, Cm(2.5))
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    pf = st.paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(4), 1.0
    # Numeración de página en el pie
    p = s.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)


def runs(p, text: str, size: float | None = None, italic: bool = False) -> None:
    """Añade texto con **negrita**, *cursiva* y resaltado de [PENDIENTE…]/[VERIFICAR…]."""
    for part in re.split(r"(\*\*.+?\*\*|\*.+?\*|\[(?:PENDIENTE|VERIFICAR)[^\]]*\])", text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("*"):
            r = p.add_run(part[1:-1])
            r.italic = True
        elif part.startswith("["):
            r = p.add_run(part)
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            r.bold = True
        else:
            r = p.add_run(part)
        if italic:
            r.italic = True
        r.font.name = FONT
        if size:
            r.font.size = Pt(size)


def para(text: str, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, italic=False,
         after: float | None = None):
    p = doc.add_paragraph()
    p.alignment = align
    runs(p, text, size, italic)
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullets(items: list[str]) -> None:
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(2)
        runs(p, it)


def h1(text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold, r.font.size, r.font.name = True, Pt(11), FONT


def h2(text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold, r.font.size, r.font.name = True, Pt(10), FONT


def shade(cell, fill: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def table(title: str, header: list[str], rows: list[list[str]],
          widths: list[float] | None = None, note: str | None = None,
          bold_rows: tuple[int, ...] = ()) -> int:
    global _tab
    _tab += 1
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.keep_with_next = True
    cap.paragraph_format.space_before = Pt(4)
    r = cap.add_run(f"Tabla {_tab}. ")
    r.bold = True
    r.font.size = Pt(9)
    runs(cap, title, size=9)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        runs(c.paragraphs[0], f"**{h}**", size=8.5)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shade(c, "D9D9D9")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            txt = f"**{v}**" if ri in bold_rows and v else v
            runs(cells[i].paragraphs[0], txt, size=8.5)
            cells[i].paragraphs[0].alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            )
            cells[i].paragraphs[0].paragraph_format.space_after = Pt(0)
    for row in t.rows:  # ninguna fila se divide entre páginas
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)
    for row in t.rows[:-1]:  # mantener la tabla en una sola página
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.keep_with_next = True
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    if note:
        para(note, size=8, italic=True, after=6)
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return _tab


def figure(path: Path, title: str, width_cm: float = 16.0,
           source: str = "elaboración propia.") -> int:
    global _fig
    _fig += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Figura {_fig}. ")
    r.bold = True
    r.font.size = Pt(9)
    runs(cap, f"{title} Fuente: {source}", size=9)
    return _fig


# --------------------------------------------------------------------------- contenido
setup()

t = para("**Clasificación automática de estadios del sueño a partir de señales "
         "polisomnográficas mediante aprendizaje de máquina con contexto temporal: "
         "del modelo a una aplicación web desplegada en la nube**",
         align=WD_ALIGN_PARAGRAPH.CENTER, size=14, after=8)

para("Carlos Adrián Alarcón Delgado, Andrés Enrique Bautista Pabón, John Janer Buila Ferrin",
     align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
para("Maestría en Inteligencia Artificial, Universidad de los Andes, Bogotá, Colombia",
     align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, after=0)
para("{c.alarcond, a.bautistap, j.buila}@uniandes.edu.co",
     align=WD_ALIGN_PARAGRAPH.CENTER, after=10)

para(
    "**Resumen.** La estadificación del sueño a partir de la polisomnografía (PSG) se realiza "
    "manualmente en épocas de 30 segundos, es costosa y varía entre evaluadores, sobre todo en el "
    "estadio de transición N1. Este trabajo desarrolla y despliega una solución de aprendizaje de "
    "máquina que clasifica cinco estadios (W, N1, N2, N3 y REM) con el subconjunto Sleep Cassette "
    "de Sleep-EDF Expanded (153 noches, 78 sujetos). Un pipeline reproducible extrae 145 "
    "descriptores estadísticos, espectrales y no lineales por época a partir de dos canales de EEG, "
    "EOG y EMG, y añade 290 variables de contexto temporal. Con una partición fija por sujeto (62 "
    "de desarrollo, evaluados con validación cruzada de cinco folds, y 16 reservados para la prueba "
    "final) se compararon un clasificador mayoritario, Random Forest, CatBoost, LightGBM y redes "
    "LSTM/BiLSTM, con ajuste de hiperparámetros anidado por sujeto. El modelo seleccionado, "
    "LightGBM ajustado, obtuvo en la prueba final un F1 macro de 0,788 y un kappa de Cohen de "
    "0,768 (exactitud de 0,834), prácticamente iguales a los de validación cruzada y por encima "
    "del umbral de éxito de 0,70; la línea base con un solo canal alcanzaba 0,655 de F1 macro. La "
    "fusión multicanal y el contexto temporal explican la mayor parte de la mejora, mientras que N1 "
    "sigue siendo el estadio más difícil (F1 = 0,52). El modelo se desplegó como una aplicación web "
    "pública sobre AWS y Vercel, definida con Terraform, que procesa una noche completa en unos dos "
    "segundos y presenta el hipnograma estimado."
)
para("**Palabras clave:** estadificación del sueño, electroencefalografía, gradient boosting, "
     "contexto temporal, despliegue en la nube.", after=8)

# 1 ------------------------------------------------------------------------------------------
h1("1. Introducción")
para(
    "El sueño se organiza en ciclos de estadios con características fisiológicas diferenciadas. "
    "Su análisis clínico se basa en la polisomnografía (PSG), que registra simultáneamente "
    "señales como el electroencefalograma (EEG), el electrooculograma (EOG) y el electromiograma "
    "(EMG). Un especialista divide la noche en épocas de 30 segundos y asigna a cada una un "
    "estadio según las reglas de la American Academy of Sleep Medicine (AASM) [1] o, en registros "
    "más antiguos, de Rechtschaffen y Kales [2]. El resultado, el hipnograma, es la base para diagnosticar trastornos como la "
    "apnea, el insomnio o la narcolepsia."
)
para(
    "La puntuación manual de una noche de ocho horas implica revisar cerca de 960 épocas y "
    "consume varias horas de trabajo especializado. Además, la concordancia entre evaluadores es "
    "imperfecta: en un estudio con más de 2.500 evaluadores la concordancia global fue de 82,6 % y "
    "descendió a 63 % en N1 [3]; otros estudios reportan un kappa entre expertos cercano a 0,76 [4]. "
    "Estas cifras ofrecen una referencia realista de lo que puede esperarse de un sistema automático "
    "y justifican el interés por herramientas que reduzcan el tiempo de análisis y aporten "
    "consistencia."
)
para(
    "El **objetivo general** del proyecto es desarrollar, evaluar y desplegar una solución de "
    "aprendizaje de máquina que clasifique automáticamente los estadios del sueño a partir de "
    "señales PSG y presente el hipnograma resultante en una aplicación accesible. Los objetivos "
    "específicos son: (i) construir un pipeline reproducible de preparación de datos sobre "
    "Sleep-EDF Expanded; (ii) comparar modelos tradicionales y de aprendizaje profundo bajo un "
    "protocolo de validación por sujeto; (iii) alcanzar un F1 macro y un kappa iguales o "
    "superiores a 0,70 en el conjunto de prueba, superando la línea base; y (iv) desplegar la "
    "solución de forma que procese un registro completo en menos de cinco minutos."
)
para(
    "Las **contribuciones** del trabajo son: un protocolo experimental con partición fija por "
    "sujeto, ajuste anidado y trazabilidad completa en MLflow; un análisis de ablación que "
    "cuantifica el aporte de los canales adicionales, de los nuevos descriptores y del contexto "
    "temporal; la comparación controlada entre gradient boosting y redes recurrentes con las mismas "
    "particiones; y una aplicación web desplegada con infraestructura como código. El artículo se "
    "organiza así: la sección 2 revisa el estado del arte; la sección 3 describe datos, métodos y "
    "despliegue; la sección 4 presenta los resultados; la sección 5 los discute; y la sección 6 "
    "concluye y propone trabajo futuro."
)

# 2 ------------------------------------------------------------------------------------------
h1("2. Estado del arte")
para(
    "Sleep-EDF Expanded [5]–[7] es uno de los conjuntos públicos más utilizados para evaluar "
    "estadificación automática. Su subconjunto Sleep Cassette, con 153 noches de 78 sujetos sanos "
    "de 25 a 101 años, se conoce en la literatura como Sleep-EDF-78. Los trabajos sobre este "
    "conjunto pueden agruparse en tres familias, a las que se suman tendencias más recientes."
)
para(
    "**Métodos basados en características.** Extraen descriptores temporales, espectrales y no "
    "lineales de cada época y los clasifican con modelos como bosques aleatorios o gradient "
    "boosting. Las potencias por banda estimadas con el método de Welch [8], los parámetros de "
    "Hjorth [9] y la entropía de permutación [10] son descriptores habituales. YASA [11] mostró que "
    "un clasificador LightGBM con descriptores de EEG, EOG y EMG, más versiones suavizadas en "
    "ventanas temporales, alcanza una exactitud cercana a la de los expertos en varias cohortes, "
    "con un modelo liviano e interpretable."
)
para(
    "**Aprendizaje profundo de extremo a extremo.** DeepSleepNet [12] combina redes "
    "convolucionales que aprenden representaciones del EEG crudo con una BiLSTM que modela las "
    "transiciones entre épocas. TinySleepNet [13] simplificó esa arquitectura y mejoró su "
    "eficiencia. SeqSleepNet [14] y XSleepNet [15] modelan secuencias de épocas con "
    "representaciones tiempo-frecuencia, y AttnSleep [16] incorpora mecanismos de atención. Estos "
    "modelos reportan en Sleep-EDF-78, con validación por sujeto y un solo canal Fpz-Cz, "
    "exactitudes de 78 % a 83 % y F1 macro de 72 % a 78 % [16]."
)
para(
    "**Fusión multicanal.** Trabajos recientes combinan EEG, EOG y EMG y analizan mediante "
    "ablaciones el aporte de cada canal y del contexto temporal [11], [15]. La información del EOG "
    "ayuda a reconocer REM y la del EMG a separar vigilia y REM, estadios que el EEG por sí solo "
    "confunde con frecuencia."
)
para(
    "**Tendencias recientes.** La investigación se ha desplazado hacia modelos más generales y "
    "auditables. U-Sleep [17] se entrenó con registros de más de 15.000 participantes de 16 "
    "estudios clínicos y mostró que un solo modelo puede generalizar a cohortes y montajes no "
    "vistos. SleepTransformer [18] usa solo transformers y aporta interpretabilidad mediante los "
    "pesos de atención y una medida de incertidumbre por época, útil para remitir al experto los "
    "casos dudosos. L-SeqSleepNet [19] modela ciclos de sueño completos (unos 90 minutos) y reduce "
    "la sobrepredicción de N2. Más recientemente, los modelos fundacionales preentrenados con "
    "grandes volúmenes de PSG se adaptan a la estadificación y a otras tareas, como la detección de "
    "patologías [20]. Estos enfoques mejoran la generalidad a costa de más datos, más cómputo y "
    "despliegues más exigentes."
)
para(
    "**Brechas identificadas.** Primero, las cifras publicadas no siempre son comparables: los "
    "trabajos difieren en el número de sujetos, la cantidad de vigilia conservada, los canales y "
    "el esquema de validación, y algunos dividen por época y no por sujeto, lo que sobrestima el "
    "desempeño. Segundo, pocos trabajos comparan de forma controlada modelos basados en "
    "características y redes recurrentes con exactamente las mismas particiones y objetivos. "
    "Tercero, la mayoría se detiene en la evaluación offline y no aborda el despliegue, la "
    "reproducibilidad ni el costo de inferencia, que crece con los modelos más recientes. Este proyecto se centra en esas tres brechas: "
    "validación estrictamente por sujeto, comparación controlada entre familias y una solución "
    "desplegada y trazable."
)

# 3 ------------------------------------------------------------------------------------------
h1("3. Metodología")
para(
    "La metodología sigue el ciclo de vida del aprendizaje de máquina: comprensión y preparación "
    "de datos, ingeniería de características, modelado y validación, registro del modelo y "
    "despliegue. La Figura 1 resume la arquitectura completa."
)
figure(ROOT / "docs/articulo/figura_arquitectura.png",
       "Arquitectura de la solución. A: ciclo de aprendizaje de máquina sobre los sujetos de "
       "desarrollo. B: servicio desplegado; las flechas dobles indican solicitud y respuesta, y "
       "las discontinuas el paso de un modelo aprobado a producción.")

h2("3.1. Datos")
para(
    "Se usó Sleep-EDF Expanded 1.0.0 de PhysioNet [5]–[7]. Se descargaron y verificaron con "
    "SHA-256 los 197 pares PSG-hipnograma del conjunto (8,1 GiB). El estudio emplea el "
    "subconjunto Sleep Cassette: 153 registros de 78 sujetos, con EEG Fpz-Cz y Pz-Oz y EOG "
    "horizontal muestreados a 100 Hz, y una envolvente de EMG submentoniano a 1 Hz. Los "
    "hipnogramas siguen las reglas de Rechtschaffen y Kales; los estadios 3 y 4 se unificaron en "
    "N3 y R se renombró como REM, siguiendo la convención AASM [1]."
)
para(
    "Antes de leer cualquier señal se fijó una partición por sujeto (semilla 42): 62 sujetos de "
    "desarrollo, distribuidos en cinco folds de 13, 13, 12, 12 y 12 sujetos, y 16 sujetos "
    "reservados para la evaluación final. Todas las noches de una persona pertenecen al mismo "
    "grupo, lo que evita que el modelo se evalúe sobre sujetos que ya vio en el entrenamiento."
)

h2("3.2. Preparación de los datos")
para(
    "Cada registro se segmentó en épocas de 30 segundos sincronizadas con el hipnograma. Al EEG se "
    "aplicó un filtro pasabanda Butterworth de cuarto orden entre 0,3 y 35 Hz. Se excluyeron las "
    "épocas sin puntuar (marcadas como «?»). Las épocas de movimiento (M) no se usan como objetivo "
    "en la tarea de cinco clases, pero se conservan en la secuencia para calcular el contexto "
    "temporal. Como los registros de Sleep Cassette incluyen muchas horas de vigilia antes y "
    "después de dormir, se conservaron solo 30 minutos de vigilia antes del primer estadio de "
    "sueño y después del último, criterio habitual en la literatura sobre Sleep-EDF-78 [12], [16]. "
    "La Tabla 1 resume el flujo de inclusión y exclusión, y la Tabla 2 la distribución por clase."
)
table(
    "Flujo de inclusión y exclusión de épocas de 30 segundos.",
    ["Concepto", "Desarrollo", "Prueba", "Total"],
    [
        ["Sujetos / registros", "62 / 122", "16 / 31", "78 / 153"],
        ["Épocas en los hipnogramas", "332.534", "83.890", "416.424"],
        ["Excluidas: sin puntuar (?)", "1.333", "2", "1.335"],
        ["Excluidas: recorte de vigilia (±30 min)", "179.034", "40.458", "219.492"],
        ["Movimiento (M): solo contexto, no objetivo", "115", "13", "128"],
        ["Épocas analizadas (5 clases)", "152.052", "43.417", "195.469"],
    ],
    widths=[7.0, 3.0, 3.0, 3.0],
    bold_rows=(5,),
    note="Generado con scripts/epoch_inventory.py. 165 épocas analizadas tienen algún canal "
         "marcado como plano; la marca es informativa y no excluye épocas.",
)
table(
    "Distribución por clase en los 62 sujetos de desarrollo, antes y después del recorte de vigilia.",
    ["Estadio", "Noche completa", "Analizadas", "% analizadas"],
    [
        ["W", "227.811", "48.777", "32,08 %"],
        ["N1", "16.603", "16.603", "10,92 %"],
        ["N2", "54.955", "54.955", "36,14 %"],
        ["N3", "10.656", "10.656", "7,01 %"],
        ["REM", "21.061", "21.061", "13,85 %"],
        ["Total", "331.086", "152.052", "100,00 %"],
    ],
    widths=[3.0, 3.5, 3.5, 3.5],
    bold_rows=(5,),
    note="La distribución del conjunto de prueba no se inspeccionó para no influir en las decisiones de modelado.",
)
para(
    "El recorte reduce la vigilia de 69 % a 32 % de las épocas, pero el conjunto sigue "
    "desbalanceado: N3 representa 7 % y N1 11 %. Por ello todos los modelos usan pesos de clase "
    "inversos a la frecuencia, y la métrica principal es el F1 macro, que pondera por igual los "
    "cinco estadios."
)

h2("3.3. Ingeniería de características")
para(
    "Para cada época y canal se calcularon los descriptores de la Tabla 3. En el EEG y el "
    "EOG las potencias por banda se estimaron con el método de Welch [8] en las bandas delta "
    "(0,5–4 Hz), theta (4–8 Hz), alfa (8–12 Hz), sigma (12–16 Hz) y beta (16–30 Hz). La envolvente "
    "de EMG, muestreada a 1 Hz, recibió solo descriptores de amplitud. El resultado son 145 "
    "variables por época."
)
table(
    "Grupos de características por época.",
    ["Grupo", "Descriptores"],
    [
        ["Estadísticos", "Media, desviación estándar, RMS, cuantiles, asimetría, curtosis, IQR, "
                         "desviación absoluta mediana, cruces por cero"],
        ["Espectrales", "Potencias absolutas y relativas, log-potencias, cocientes entre bandas "
                        "(p. ej. sigma/beta, delta/beta), frecuencia de borde espectral al 95 %, "
                        "entropía espectral"],
        ["No lineales", "Parámetros de Hjorth [9], entropía de permutación de orden 3 [10], "
                        "dimensión fractal de Petrosian, longitud de la señal"],
        ["Estabilidad", "Variabilidad del RMS entre seis subventanas de 5 s, indicador de señal plana"],
        ["Contexto temporal", "Por cada variable: media de las cuatro épocas previas y media "
                              "triangular centrada de 15 épocas (7,5 min), normalizadas por noche "
                              "(290 variables)"],
    ],
    widths=[3.0, 13.0],
)
para(
    "El contexto temporal se inspira en YASA [11]: un especialista no puntúa una época de forma "
    "aislada, sino observando las vecinas. Las ventanas se reinician en cada registro y en cada "
    "hueco temporal, nunca usan etiquetas y se normalizan por noche con la mediana y los "
    "percentiles 5 y 95. Como emplean épocas futuras, el sistema corresponde a un análisis "
    "retrospectivo de una noche completa y no a una clasificación en tiempo real. La entrada final "
    "de los modelos tabulares tiene 435 variables."
)

h2("3.4. Modelos")
para(
    "Se compararon cinco familias de modelos con las mismas particiones (Tabla 4). El clasificador "
    "mayoritario (Dummy) fija el desempeño mínimo. Random Forest [21] fue la línea base del avance. "
    "LightGBM [22] y CatBoost [23] son implementaciones de gradient boosting sobre árboles, "
    "eficientes con cientos de variables y robustas al desbalance mediante pesos de clase. Las "
    "redes LSTM y BiLSTM [24] modelan explícitamente la secuencia de épocas. La CNN 1D sobre EEG "
    "crudo se evaluó solo en la fase piloto (ocho sujetos), donde obtuvo un F1 macro de 0,354 "
    "frente a 0,583 de Random Forest; por su bajo desempeño y su costo se priorizaron las redes "
    "recurrentes sobre características."
)
table(
    "Configuración de los modelos evaluados.",
    ["Modelo", "Entrada", "Configuración principal"],
    [
        ["Dummy", "—", "Predice la clase más frecuente"],
        ["Random Forest", "25 variables Fpz-Cz", "64 árboles, pesos balanceados"],
        ["CatBoost", "25 a 435", "Búsqueda: 300–500 árboles, profundidad 4–6, lr 0,05–0,08, L2 3–9"],
        ["LightGBM (final)", "435", "261 iteraciones, 63 hojas, mín. 200 muestras por hoja, lr 0,05, "
                                    "L2 = 5, 63 bins, 80 % de filas y columnas, pesos inversos de clase"],
        ["LSTM / BiLSTM", "Ventanas de 15 épocas × 145 o 435",
         "Proyección a 64 unidades + ReLU, una capa recurrente (64 por dirección), dropout 0,3–0,4, "
         "AdamW (lr 3·10⁻⁴ o 10⁻³, weight decay 10⁻⁴), lote 256, máx. 50 épocas, parada temprana "
         "por F1 macro con paciencia 7, recorte de gradiente a norma 1"],
        ["CNN 1D (piloto)", "3.000 muestras EEG", "Conv1d 16 filtros (kernel 25, paso 5) → MaxPool 4 → "
                                                  "Conv1d 32 filtros (kernel 9) → AdaptiveAvgPool 6 → "
                                                  "capa lineal; Adam lr 10⁻³, lote 32, hasta 20 épocas "
                                                  "con parada temprana"],
    ],
    widths=[2.8, 3.4, 9.8],
)

h2("3.5. Validación y ajuste de hiperparámetros")
para(
    "La evaluación usa validación cruzada externa de cinco folds por sujeto. En cada iteración, "
    "cuatro folds (49 o 50 sujetos) se usan para entrenar y uno (12 o 13 sujetos) para evaluar. "
    "Dentro de cada entrenamiento externo se reserva 20 % de los sujetos para elegir la "
    "configuración y el número de iteraciones o épocas; luego se reentrena con todos los sujetos "
    "de entrenamiento y se predice el fold externo una sola vez. Para LightGBM se compararon 12 "
    "configuraciones (número de hojas, regularización, muestreo y potencia de los pesos de "
    "clase) con parada temprana por F1 macro (paciencia de 40 iteraciones); para la BiLSTM, seis. "
    "La imputación, el escalado y los pesos de clase se ajustan solo con datos de entrenamiento."
)
para(
    "Las métricas son el F1 macro (principal), la exactitud, el kappa de Cohen [25], el F1 por "
    "estadio y la matriz de confusión. Se reporta el promedio y la desviación estándar entre "
    "folds, y para comparar dos modelos se calcula un intervalo de confianza del 95 % mediante "
    "2.000 remuestreos bootstrap emparejados de sujetos completos sobre las predicciones fuera de "
    "muestra (OOF) [26]. Los 16 sujetos de prueba se evalúan una sola vez, con la configuración "
    "congelada."
)

h2("3.6. Herramientas y entorno")
para(
    "La solución se implementó en Python. MNE [27] y SciPy se usaron para leer y filtrar las "
    "señales; pandas, NumPy y scikit-learn [28] para la manipulación de datos y los modelos "
    "clásicos; LightGBM y CatBoost para boosting; PyTorch [29] para las redes; y MLflow [30] para "
    "registrar parámetros, métricas, predicciones y modelos. Las configuraciones se declaran en "
    "YAML, los datos intermedios se guardan en Parquet y NumPy con identificadores por hash de "
    "entradas, parámetros y código, y el entorno se fija con uv. El repositorio incluye pruebas "
    "unitarias y de integración (59 pruebas aprobadas), además de lint y verificación de tipos."
)

h2("3.7. Despliegue")
para(
    "La solución se desplegó en AWS (región us-west-2) y Vercel, definida completamente con "
    "Terraform (Figura 1B). El frontend, en React y Vite, se publica en Vercel. Una API sobre "
    "API Gateway y AWS Lambda (imagen de contenedor) emite formularios firmados para cargar el "
    "archivo directamente en S3 y garantiza la idempotencia de cada solicitud. Una máquina de "
    "estados de Step Functions invoca un worker Lambda que verifica el SHA-256 del archivo y del "
    "modelo, deriva las 290 variables de contexto y genera la predicción con sus cinco "
    "probabilidades. Los resultados en JSON y CSV quedan en S3 durante siete días."
)
para(
    "Se eligió una arquitectura sin servidor porque el modelo LightGBM es liviano y su "
    "inferencia en CPU tarda pocos segundos por noche, de modo que no requiere GPU ni servidores "
    "encendidos permanentemente: el costo es proporcional al uso y la plataforma escala "
    "automáticamente con las solicitudes simultáneas. El reentrenamiento se ejecuta bajo demanda en "
    "una instancia EC2 efímera, registra el experimento en SageMaker MLflow y produce un "
    "candidato que solo se promueve si reproduce las predicciones de referencia. Cada versión del "
    "flujo apunta a una versión fija del modelo, lo que permite revertir un cambio. "
    "[PENDIENTE: describir la carga directa de archivos EDF, en desarrollo.]"
)

# 4 ------------------------------------------------------------------------------------------
h1("4. Resultados")
h2("4.1. Comparación de modelos y ablaciones")
para(
    "La Tabla 5 muestra la evolución del desempeño en validación cruzada al incorporar cada "
    "mejora. Todos los modelos usan los mismos 152.052 segmentos y los mismos cinco folds."
)
table(
    "Resultados de validación cruzada (62 sujetos, cinco folds). Media ± desviación entre folds.",
    ["Modelo", "Variables", "F1 macro", "Exactitud", "Kappa"],
    [
        ["Dummy (mayoritario)", "—", "0,106 ± 0,005", "0,362", "0,000"],
        ["Random Forest · EEG Fpz-Cz", "25", "0,654 ± 0,021", "0,713", "0,612"],
        ["CatBoost · EEG Fpz-Cz", "25", "0,644 ± 0,029", "0,687", "0,588"],
        ["CatBoost · EEG ampliado", "42", "0,649 ± 0,028", "0,692", "0,595"],
        ["CatBoost · multicanal", "88", "0,727 ± 0,032", "0,760", "0,682"],
        ["CatBoost · multicanal ampliado", "145", "0,731 ± 0,033", "0,764", "0,688"],
        ["CatBoost · contexto ajustado", "435", "0,775 ± 0,019", "0,821", "0,756"],
        ["LSTM · 15 épocas pasadas", "145", "0,767 ± 0,018", "0,805", "0,739"],
        ["BiLSTM ajustada · 15 épocas centradas", "145", "0,780 ± 0,017", "0,819", "0,755"],
        ["LightGBM · contexto", "435", "0,782 ± 0,016", "0,817", "0,754"],
        ["LightGBM ajustado (seleccionado)", "435", "0,789 ± 0,018", "0,830", "0,769"],
    ],
    widths=[6.0, 2.0, 3.0, 2.5, 2.5],
    bold_rows=(10,),
)
para(
    "Los dos cambios con mayor aporte fueron la fusión multicanal (+0,083 de F1 macro al pasar de "
    "Fpz-Cz a cuatro señales) y el contexto temporal (+0,038 adicional). Ampliar los descriptores "
    "de un solo canal aportó poco (+0,005). El ajuste anidado elevó LightGBM de 0,782 a 0,789; el "
    "intervalo bootstrap de esa diferencia, [0,003; 0,011], excluye el cero. LightGBM ajustado "
    "mejoró el F1 macro en los cinco folds y superó a la mejor BiLSTM, aunque la diferencia frente "
    "a esta no es concluyente (Δ = −0,002; IC 95 % [−0,009; 0,006]). La Figura 2 compara los "
    "modelos finales de cada familia."
)
figure(ROOT / "docs/assets/refinement/model-comparison.png",
       "F1 macro medio por modelo con la desviación entre folds.", width_cm=13.5)

h2("4.2. Desempeño por estadio")
table(
    "F1 por estadio sobre las predicciones fuera de muestra (OOF) agrupadas.",
    ["Estadio", "RF · Fpz-Cz", "BiLSTM ajustada", "LightGBM ajustado"],
    [
        ["W", "0,838", "0,923", "0,924"],
        ["N1", "0,311", "0,513", "0,516"],
        ["N2", "0,789", "0,832", "0,849"],
        ["N3", "0,780", "0,791", "0,808"],
        ["REM", "0,557", "0,848", "0,854"],
    ],
    widths=[3.0, 3.5, 3.5, 3.5],
)
para(
    "Frente a la línea base, la mejora más notable se da en REM (de 0,557 a 0,854) y en N1 (de "
    "0,311 a 0,516), estadios en los que el EOG y el EMG aportan información que el EEG frontal no "
    "contiene. N1 sigue siendo el estadio más difícil: la matriz de confusión (Figura 3) muestra "
    "que se confunde principalmente con N2, W y REM, lo que coincide con la menor concordancia "
    "entre expertos reportada para este estadio [3]."
)
figure(ROOT / "docs/assets/refinement/best-new-confusion.png",
       "Matriz de confusión OOF de LightGBM ajustado, en conteos y proporciones por fila.",
       width_cm=15.5)

h2("4.3. Evaluación sobre el conjunto de prueba")
para(
    "Con la configuración congelada, el modelo se reentrenó con los 62 sujetos de desarrollo "
    "(261 iteraciones, semilla 542) y se evaluó **una sola vez** sobre los 16 sujetos reservados: "
    "31 noches y 43.417 épocas que no intervinieron en ninguna decisión de modelado. Los EDF de "
    "prueba se verificaron con SHA-256 y se procesaron con el mismo protocolo que los de "
    "desarrollo. La Tabla 7 muestra los resultados; los intervalos del 95 % se obtuvieron con "
    "2.000 remuestreos bootstrap de sujetos completos."
)
table(
    "Desempeño en el conjunto de prueba (16 sujetos) frente a la validación cruzada.",
    ["Métrica", "Validación cruzada", "Prueba", "IC 95 % (prueba)"],
    [
        ["F1 macro", "0,789", "0,788", "0,743 – 0,822"],
        ["Exactitud", "0,830", "0,834", "0,810 – 0,861"],
        ["Kappa de Cohen", "0,769", "0,768", "0,731 – 0,809"],
        ["F1 W / N1 / N2", "0,924 / 0,516 / 0,849", "0,924 / 0,516 / 0,849", "—"],
        ["F1 N3 / REM", "0,808 / 0,854", "0,793 / 0,858", "—"],
    ],
    widths=[3.6, 4.4, 4.0, 4.0],
    note="Validación cruzada: media de cinco folds (métricas globales) y predicciones OOF "
         "agrupadas (F1 por estadio). Por sujeto, el F1 macro de prueba varió entre 0,552 y 0,888 "
         "y el kappa entre 0,611 y 0,885.",
)
para(
    "El desempeño en prueba prácticamente coincide con el estimado en desarrollo, lo que indica "
    "que el protocolo de validación anidada por sujeto no sobrestimó la capacidad de "
    "generalización. Incluso los límites inferiores de los intervalos superan 0,70 en F1 macro "
    "y en kappa. La matriz de confusión (Figura 4) repite el patrón observado en desarrollo: N1 "
    "es el estadio más difícil (recall 0,57) y se confunde sobre todo con N2 (27 %) y W (10 %); "
    "21 % de las épocas N3 se clasifican como N2."
)
figure(ROOT / "docs/articulo/figura_confusion_prueba.png",
       "Matriz de confusión en el conjunto de prueba (16 sujetos, 43.417 épocas), en conteos y "
       "proporciones por fila.", width_cm=15.5)
para(
    "**Análisis cualitativo.** La Figura 5 compara el hipnograma del experto con el predicho para "
    "la noche SC4771, del sujeto de prueba cuyo kappa (0,764) es el más cercano a la mediana; no es "
    "el mejor caso, sino uno típico. Es una noche fragmentada, con 26 % de N1 y numerosos "
    "despertares breves. El modelo reproduce la arquitectura de la noche: la alternancia de ciclos, "
    "los periodos REM y la vigilia prolongada a mitad del registro, y coincide en 80,5 % de las "
    "épocas. Los desacuerdos se agrupan en las fronteras entre estadios, sobre todo N1/N2, N1/W y "
    "N1/REM, "
    "y en despertares de una o dos épocas; además, el modelo predice algunas épocas aisladas de N3 "
    "que el experto puntuó como N2. Estos errores de transición son los mismos que generan "
    "discrepancias entre evaluadores humanos [3]."
)
figure(ROOT / "docs/articulo/figura_hipnograma.png",
       "Hipnograma de referencia (arriba) y predicho (abajo) para la noche de prueba SC4771; los "
       "puntos marcan las épocas en desacuerdo.", width_cm=16.0)

h2("4.4. Desempeño del servicio desplegado")
para(
    "El 18 de noviembre de 2026 se verificó el flujo completo con la noche pública SC4001 "
    "(841 épocas). Las clases y las cinco probabilidades coincidieron con la referencia local "
    "(diferencia máxima de 5,6·10⁻¹⁶). La Tabla 8 resume los tiempos medidos."
)
table(
    "Tiempos del servicio desplegado.",
    ["Prueba", "Resultado"],
    [
        ["Una predicción con el servicio activo", "Aceptación 0,50 s; resultado en 1,83 s"],
        ["Cinco predicciones simultáneas", "p95 de aceptación 0,59 s; p95 de finalización 6,51 s"],
        ["Reentrenamiento en EC2 (261 árboles)", "318,9 s; un segundo ajuste reprodujo el mismo SHA-256"],
    ],
    widths=[6.5, 9.5],
)

# 5 ------------------------------------------------------------------------------------------
h1("5. Discusión")
para(
    "**Cumplimiento de los criterios de éxito.** La propuesta definió tres criterios: (i) F1 macro "
    "y kappa ≥ 0,70 en el conjunto de prueba; (ii) superar la línea base con las mismas "
    "particiones; y (iii) procesar un registro en menos de cinco minutos y presentar el "
    "hipnograma. Los tres se cumplen: en el conjunto de prueba el modelo alcanza un F1 macro de "
    "0,788 y un kappa de 0,768, con intervalos cuyo límite inferior también supera 0,70; supera "
    "ampliamente a la línea base de Random Forest (0,655 de F1 macro con las mismas particiones "
    "en desarrollo) y al clasificador mayoritario; y el servicio responde en menos de dos "
    "segundos por noche."
)
para(
    "**Comparación con la literatura.** El F1 macro de 0,788 y el kappa de 0,768 en prueba están en "
    "el rango de los modelos profundos publicados para Sleep-EDF-78 (Tabla 9). La comparación es "
    "orientativa: este trabajo usa cuatro señales y no solo Fpz-Cz, valida con cinco folds por "
    "sujeto y conserva 30 minutos de vigilia, condiciones que no coinciden exactamente con las de "
    "cada referencia."
)
table(
    "Comparación orientativa con trabajos sobre Sleep-EDF-78 (validación por sujeto y ±30 min de vigilia).",
    ["Trabajo", "Señales", "Exactitud", "F1 macro", "Kappa"],
    [
        ["DeepSleepNet [12], según [16]", "Fpz-Cz", "0,778", "0,718", "0,70"],
        ["AttnSleep [16]", "Fpz-Cz", "0,813", "0,751", "0,74"],
        ["SeqSleepNet [14], 3 épocas, según [16]", "Fpz-Cz", "0,826", "0,763", "0,76"],
        ["AttnSleep [16], 3 épocas", "Fpz-Cz", "0,829", "0,781", "0,77"],
        ["Este trabajo · CV (LightGBM ajustado)", "EEG ×2, EOG, EMG", "0,830", "0,789", "0,769"],
        ["Este trabajo · prueba (16 sujetos)", "EEG ×2, EOG, EMG", "0,834", "0,788", "0,768"],
    ],
    widths=[5.8, 3.3, 2.3, 2.3, 2.3],
    bold_rows=(4, 5),
    note="Cifras de [16], Tablas V y VI: validación cruzada de 20 folds por sujeto. Este trabajo usa "
         "cinco folds por sujeto y cuatro señales, por lo que la comparación es orientativa.",
)
para(
    "**Fortalezas.** El protocolo evita la fuga de información entre sujetos, separa la selección "
    "de hiperparámetros de la evaluación y deja trazabilidad completa de cada resultado. El "
    "modelo final es liviano, interpretable mediante la importancia de variables (destacan el "
    "cociente sigma/beta de Fpz-Cz, la envolvente del EMG y descriptores del EOG con contexto) y "
    "barato de servir sin GPU. La comparación controlada muestra que, con descriptores bien "
    "diseñados y contexto temporal, el gradient boosting iguala o supera a redes recurrentes "
    "entrenadas con las mismas particiones."
)
para(
    "**Limitaciones.** Primero, el recorte de vigilia usa el hipnograma de referencia para "
    "ubicar el inicio y el fin del sueño; en un registro nuevo sin anotaciones esa información no "
    "existe, por lo que el desempeño reportado corresponde a un protocolo de recorte supervisado. "
    "Segundo, N1 conserva un F1 cercano a 0,52. Tercero, la clase de movimiento (M), prevista en "
    "la propuesta, tiene solo 128 épocas y no se modeló como objetivo. Cuarto, todos los sujetos "
    "son de una sola cohorte de sujetos sanos; la generalización a pacientes o a otros equipos de "
    "adquisición no está demostrada. Quinto, la selección se basa en una sola semilla y una sola "
    "partición interna por fold."
)
para(
    "**Aplicaciones.** La solución puede servir como herramienta de apoyo a la puntuación: "
    "proponer un hipnograma inicial que el especialista revise y corrija, priorizar la revisión "
    "de épocas con baja confianza o procesar grandes volúmenes de registros en estudios de "
    "investigación. No sustituye el criterio clínico."
)

# 6 ------------------------------------------------------------------------------------------
h1("6. Conclusiones y trabajo futuro")
para(
    "Se desarrolló y desplegó una solución de estadificación automática del sueño en cinco "
    "clases que, evaluada una sola vez sobre 16 sujetos reservados de Sleep-EDF-78, alcanza un "
    "F1 macro de 0,788 y un kappa de 0,768, por encima del umbral de 0,70 definido en la "
    "propuesta, cerca de la concordancia entre expertos y prácticamente igual a lo estimado en "
    "validación cruzada (0,789 y 0,769). La fusión de EEG, EOG y EMG y el "
    "contexto temporal fueron los factores decisivos: juntos elevaron el F1 macro en 0,13 "
    "puntos respecto de la línea base. Un modelo LightGBM ajustado igualó a las redes "
    "recurrentes con un costo de inferencia mucho menor, lo que facilitó un despliegue sin "
    "servidor que responde en segundos y se reproduce con infraestructura como código."
)
para(
    "Entre las lecciones aprendidas se destacan la importancia de fijar la partición por sujeto "
    "antes de explorar los datos, de comparar modelos con exactamente los mismos objetivos y de "
    "registrar cada experimento para poder auditar y reproducir los resultados."
)
para("Como trabajo futuro se proponen:")
bullets([
    "Reemplazar el recorte de vigilia supervisado por una detección automática del periodo de "
    "sueño, para clasificar archivos EDF sin anotaciones.",
    "Mejorar N1 con modelos secuenciales de mayor alcance, como transformers sobre las "
    "características, o con un suavizado de la secuencia de predicciones (modelo oculto de Markov).",
    "Evaluar la clasificación en seis clases incluyendo movimiento, y validar en Sleep Telemetry "
    "y en cohortes clínicas.",
    "Calibrar las probabilidades y mostrar la incertidumbre por época en la interfaz.",
])

# Referencias -------------------------------------------------------------------------------
h1("Referencias")
refs = [
    "R. B. Berry et al., *The AASM Manual for the Scoring of Sleep and Associated Events: Rules, "
    "Terminology and Technical Specifications*, v. 2.6. Darien, IL, USA: American Academy of Sleep "
    "Medicine, 2020.",
    "A. Rechtschaffen y A. Kales, *A Manual of Standardized Terminology, Techniques and Scoring "
    "System for Sleep Stages of Human Subjects*. Washington, DC, USA: U.S. Government Printing "
    "Office, 1968.",
    "R. S. Rosenberg y S. Van Hout, “The American Academy of Sleep Medicine inter-scorer "
    "reliability program: sleep stage scoring,” *J. Clin. Sleep Med.*, vol. 9, n.º 1, pp. 81–87, "
    "2013, doi: 10.5664/jcsm.2350.",
    "H. Danker-Hopfe et al., “Interrater reliability for sleep scoring according to the "
    "Rechtschaffen & Kales and the new AASM standard,” *J. Sleep Res.*, vol. 18, n.º 1, pp. 74–84, "
    "2009, doi: 10.1111/j.1365-2869.2008.00700.x.",
    "B. Kemp, A. H. Zwinderman, B. Tuk, H. A. C. Kamphuisen y J. J. L. Oberyé, “Analysis of a "
    "sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG,” *IEEE "
    "Trans. Biomed. Eng.*, vol. 47, n.º 9, pp. 1185–1194, 2000, doi: 10.1109/10.867928.",
    "A. L. Goldberger et al., “PhysioBank, PhysioToolkit, and PhysioNet: components of a new "
    "research resource for complex physiologic signals,” *Circulation*, vol. 101, n.º 23, "
    "pp. e215–e220, 2000, doi: 10.1161/01.CIR.101.23.e215.",
    "B. Kemp, *Sleep-EDF Database Expanded* (v. 1.0.0), PhysioNet, 2013, doi: 10.13026/C2X676.",
    "P. Welch, “The use of fast Fourier transform for the estimation of power spectra,” *IEEE "
    "Trans. Audio Electroacoust.*, vol. 15, n.º 2, pp. 70–73, 1967, doi: 10.1109/TAU.1967.1161901.",
    "B. Hjorth, “EEG analysis based on time domain properties,” *Electroencephalogr. Clin. "
    "Neurophysiol.*, vol. 29, n.º 3, pp. 306–310, 1970, doi: 10.1016/0013-4694(70)90143-4.",
    "C. Bandt y B. Pompe, “Permutation entropy: a natural complexity measure for time series,” "
    "*Phys. Rev. Lett.*, vol. 88, n.º 17, art. 174102, 2002, doi: 10.1103/PhysRevLett.88.174102.",
    "R. Vallat y M. P. Walker, “An open-source, high-performance tool for automated sleep "
    "staging,” *eLife*, vol. 10, art. e70092, 2021, doi: 10.7554/eLife.70092.",
    "A. Supratak, H. Dong, C. Wu y Y. Guo, “DeepSleepNet: a model for automatic sleep stage "
    "scoring based on raw single-channel EEG,” *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 25, "
    "n.º 11, pp. 1998–2008, 2017, doi: 10.1109/TNSRE.2017.2721116.",
    "A. Supratak y Y. Guo, “TinySleepNet: an efficient deep learning model for sleep stage "
    "scoring based on raw single-channel EEG,” en *Proc. 42nd Annu. Int. Conf. IEEE EMBC*, 2020, "
    "pp. 641–644, doi: 10.1109/EMBC44109.2020.9176741.",
    "H. Phan, F. Andreotti, N. Cooray, O. Y. Chén y M. De Vos, “SeqSleepNet: end-to-end "
    "hierarchical recurrent neural network for sequence-to-sequence automatic sleep staging,” "
    "*IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 27, n.º 3, pp. 400–410, 2019, "
    "doi: 10.1109/TNSRE.2019.2896659.",
    "H. Phan, O. Y. Chén, M. C. Tran, P. Koch, A. Mertins y M. De Vos, “XSleepNet: multi-view "
    "sequential model for automatic sleep staging,” *IEEE Trans. Pattern Anal. Mach. Intell.*, "
    "vol. 44, n.º 9, pp. 5903–5915, 2022, doi: 10.1109/TPAMI.2021.3070057.",
    "E. Eldele et al., “An attention-based deep learning approach for sleep stage classification "
    "with single-channel EEG,” *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 29, pp. 809–818, "
    "2021, doi: 10.1109/TNSRE.2021.3076234.",
    "M. Perslev, S. Darkner, L. Kempfner, M. Nikolic, P. J. Jennum y C. Igel, “U-Sleep: resilient "
    "high-frequency sleep staging,” *npj Digit. Med.*, vol. 4, art. 72, 2021, "
    "doi: 10.1038/s41746-021-00440-5.",
    "H. Phan, K. Mikkelsen, O. Y. Chén, P. Koch, A. Mertins y M. De Vos, “SleepTransformer: "
    "automatic sleep staging with interpretability and uncertainty quantification,” *IEEE Trans. "
    "Biomed. Eng.*, vol. 69, n.º 8, pp. 2456–2467, 2022, doi: 10.1109/TBME.2022.3147187.",
    "H. Phan et al., “L-SeqSleepNet: whole-cycle long sequence modeling for automatic sleep "
    "staging,” *IEEE J. Biomed. Health Inform.*, vol. 27, n.º 10, pp. 4748–4757, 2023, "
    "doi: 10.1109/JBHI.2023.3303197.",
    "W. Huang et al., “A unified time-frequency foundation model for sleep decoding,” *Nat. "
    "Commun.*, vol. 17, art. 1198, 2026, doi: 10.1038/s41467-025-67970-4.",
    "L. Breiman, “Random forests,” *Mach. Learn.*, vol. 45, pp. 5–32, 2001, "
    "doi: 10.1023/A:1010933404324.",
    "G. Ke et al., “LightGBM: a highly efficient gradient boosting decision tree,” en *Adv. Neural "
    "Inf. Process. Syst.*, vol. 30, 2017.",
    "L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush y A. Gulin, “CatBoost: unbiased "
    "boosting with categorical features,” en *Adv. Neural Inf. Process. Syst.*, vol. 31, 2018.",
    "S. Hochreiter y J. Schmidhuber, “Long short-term memory,” *Neural Comput.*, vol. 9, n.º 8, "
    "pp. 1735–1780, 1997, doi: 10.1162/neco.1997.9.8.1735.",
    "J. Cohen, “A coefficient of agreement for nominal scales,” *Educ. Psychol. Meas.*, vol. 20, "
    "n.º 1, pp. 37–46, 1960, doi: 10.1177/001316446002000104.",
    "B. Efron y R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: "
    "Chapman & Hall, 1993.",
    "A. Gramfort et al., “MEG and EEG data analysis with MNE-Python,” *Front. Neurosci.*, "
    "vol. 7, art. 267, 2013, doi: 10.3389/fnins.2013.00267.",
    "F. Pedregosa et al., “Scikit-learn: machine learning in Python,” *J. Mach. Learn. Res.*, "
    "vol. 12, pp. 2825–2830, 2011.",
    "A. Paszke et al., “PyTorch: an imperative style, high-performance deep learning library,” "
    "en *Adv. Neural Inf. Process. Syst.*, vol. 32, 2019.",
    "M. Zaharia et al., “Accelerating the machine learning lifecycle with MLflow,” *IEEE Data "
    "Eng. Bull.*, vol. 41, n.º 4, pp. 39–45, 2018.",
]
for i, ref in enumerate(refs, 1):
    p = para(f"[{i}] {ref}", align=WD_ALIGN_PARAGRAPH.LEFT, size=9, after=2)
    p.paragraph_format.left_indent = Cm(0.7)
    p.paragraph_format.first_line_indent = Cm(-0.7)

h1("Anexo. Repositorio y aplicación")
para("Código, configuraciones, notebooks y documentación: https://github.com/aladelca/maia_proyecto. "
     "Aplicación desplegada: https://maia-sueno-dev-60306.vercel.app.", align=WD_ALIGN_PARAGRAPH.LEFT)

doc.save(OUT)
print(OUT)
