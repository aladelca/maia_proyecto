"""Genera el documento breve del entregable de despliegue (Word).

Uso (desde la raíz del repositorio):
    uv run --with python-docx python docs/articulo/construir_entregable_despliegue.py
"""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

OUT = Path(__file__).resolve().parent / "Entregable_despliegue.docx"
FONT = "Times New Roman"
REPO = "https://github.com/aladelca/maia_proyecto"
APP = "https://maia-sueno-dev-60306.vercel.app"

doc = Document()
s = doc.sections[0]
s.page_width, s.page_height = Cm(21.59), Cm(27.94)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(s, side, Cm(2.5))
st = doc.styles["Normal"]
st.font.name, st.font.size = FONT, Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st.paragraph_format.space_after = Pt(6)


def p(text="", bold=False, size=None, center=False, highlight=False):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.JUSTIFY
    r = par.add_run(text)
    r.bold, r.font.name = bold, FONT
    if size:
        r.font.size = Pt(size)
    if highlight:
        r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return par


def item(label, value):
    par = doc.add_paragraph(style="List Bullet")
    r = par.add_run(f"{label}: ")
    r.bold = True
    par.add_run(value)


p("Proyecto: Despliegue de soluciones", bold=True, size=12, center=True)
p("Clasificación automática de estadios del sueño mediante aprendizaje de máquina "
  "a partir de señales polisomnográficas", bold=True, size=14, center=True)
p("Entregable de despliegue", size=12, center=True)

p("Integrantes", bold=True)
for name, mail in [
    ("Carlos Adrián Alarcón Delgado", "c.alarcond@uniandes.edu.co"),
    ("Andrés Enrique Bautista Pabón", "a.bautistap@uniandes.edu.co"),
    ("John Janer Buila Ferrin", "j.buila@uniandes.edu.co"),
]:
    item(name, mail)

p("Enlaces", bold=True)
item("Repositorio en GitHub", REPO)
item("Aplicación desplegada", APP)
item("Guía de despliegue y operación", f"{REPO}/blob/main/docs/cloud-operations.md")
item("Ficha e instrucciones del modelo", f"{REPO}/blob/main/modelos/stack_five/README.md")

p("Acceso", bold=True)
p("La aplicación es pública y no requiere credenciales. El usuario carga un archivo EDF de "
  "polisomnografía (dos EEG, EOG y envolvente EMG, hasta 128 MiB) y obtiene el hipnograma, el "
  "tiempo en cada estadio y un CSV con la predicción y las probabilidades por época. Sin archivo "
  "propio, «Probar ejemplo público» procesa diez minutos de señal en el servidor y «Explorar modo "
  "demo» muestra un ejemplo local. Cada pestaña del navegador recibe una clave de invitado "
  "aleatoria, de modo que cada visitante solo ve sus propios resultados. La API documenta su "
  "contrato en https://9u8vgivsce.execute-api.us-west-2.amazonaws.com/docs.")
p("La infraestructura corre en una cuenta de AWS Academy: el equipo encenderá el laboratorio a la "
  "hora asignada para la evaluación.")
p("El repositorio es público y puede consultarse sin iniciar sesión en GitHub.")

p("Resumen de la solución desplegada", bold=True)
p("El modelo desplegado es un híbrido: un LightGBM que recibe 435 variables de EEG, EOG y EMG "
  "con contexto temporal y las cinco probabilidades por estadio de FullNightNet, una red "
  "convolucional que lee la noche completa. En los 16 sujetos de prueba clasificó el EDF completo, "
  "sin anotaciones, con un F1 macro de 0,781 y un kappa de 0,808; el LightGBM base obtuvo un F1 "
  "macro de 0,788 con el recorte de vigilia del protocolo. Se despliega sin servidor: el frontend "
  "React está en Vercel y la API, la orquestación y la inferencia en AWS (API Gateway, Lambda, "
  "Step Functions y S3), con toda la infraestructura definida en Terraform. Los modelos corren en "
  "CPU, se verifican por SHA-256 y se cargan una vez por contenedor.")

doc.save(OUT)
print(OUT)
