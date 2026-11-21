# MAIA · Laboratorio del sueño

## Desarrollo local

Desde `frontend/`: `npm ci`, luego `npm run dev`.
La aplicación consulta `/config.json` con el contrato existente `{ "api": "…" }`.
Si no hay configuración, se puede explorar el **modo demo**, sin configurar AWS.
El ejemplo público sigue usando el servidor: no se sustituye por datos sintéticos
si falla. El modo demo se activa explícitamente, no sube archivos, no ejecuta el
modelo y no ofrece una descarga de probabilidades ficticias.

La entrada pública es una polisomnografía EDF de hasta 128 MiB, con EEG
Fpz-Cz, EEG Pz-Oz y EOG horizontal a 100 Hz y envolvente EMG submental a 1 Hz.
La interfaz valida extensión y tamaño; el servidor valida las señales.
Se utiliza `model: stack_five`, `format: edf` para la carga y la versión
`stack-five-a64415fc2bd6`. La consulta al ejemplo público confirma la versión
antes de habilitar solicitudes; los resultados de otras versiones se rechazan.

La pestaña **Analizar una noche** conserva el formulario, el demo y el hipnograma
interactivo. **Sobre el proyecto** incorpora el contenido de `main` del PR #21:
las 435 características, las cinco probabilidades de FullNightNet y la selección
del híbrido. Las métricas y los enlaces provienen del contenido de Adrián, no del
demo. Cambiar de pestaña no reinicia el trabajo, el archivo ni la selección del
gráfico. Se admiten flechas, Inicio/Fin y enlaces `#modelo` / `#experimentos`.

Esta integración adapta el frontend público en la rama UX; no fusiona `main`
ni incorpora cambios del backend o de despliegue. Las aplicaciones Python de
inferencia local añadidas en upstream (`local` / `local-stack`) no se integran en
esta vista previa Vite. Antes de publicar se debe conciliar la rama con `main`
y verificar el flujo real de EDF contra la versión desplegada.

## Verificación

- `npm run build`: TypeScript y compilación Vite.
- `BROWSER_CHANNEL=chrome npm run test:e2e`: pruebas locales con Chrome instalado,
  respuestas interceptadas y sin acceso a AWS. Playwright inicia Vite solo.
- `npm run test:e2e`: lo mismo con Chromium de Playwright, si está instalado.
- `DEMO_ORIGIN=https://… EXAMPLE_EDF=/ruta/registro.edf npm run test:e2e`:
  ejecuta la prueba de integración EDF real. Crea trabajos y carga el
  archivo en el entorno indicado; requiere un servicio operativo.

## Auditoría y alcance

El frontend original tenía un hipnograma estático, un bloqueo completo cuando
fallaba la configuración, errores HTTP poco descriptivos y pruebas dependientes
del servicio remoto. Ahora incluye selección por clic y teclado, ampliación de
la época seleccionada, detalles de elegibilidad, huecos para épocas sin
predicción, estados de carga/error, reconsulta del trabajo y datos demo
explícitamente identificados. El gráfico usa minutos desde el inicio del
registro; las duraciones por estadio cuentan épocas válidas de 30 segundos.

Se preservan endpoints, token de invitado, cargas firmadas, SHA-256,
`x-amz-version-id`, clave de idempotencia, recuperación del trabajo por URL y
sesión y renovación del enlace CSV. No se modifican backend ni despliegue.
Las pruebas locales no acreditan la recuperación del error 500 de AWS.

La interfaz incluye enlace para saltar al contenido, etiquetas, foco visible,
control de épocas accesible por teclado, mensajes anunciados y diseño móvil.
El gráfico tiene desplazamiento horizontal en pantallas pequeñas y el control
inferior permite explorar todos los datos sin apuntar a segmentos pequeños.

### Conectar la vista previa al servicio existente

Copia `.env.example` a `.env.local` y asigna `MAIA_DEV_API` al valor `api`
del `/config.json` del laboratorio publicado. Reinicia `npm run dev`.
Vite sirve una configuración local y reenvía `/dev-api/public/*` al mismo
endpoint, conservando métodos, cuerpos y cabeceras. Esto evita el bloqueo CORS
de las consultas a la API desde localhost. No se incluye esta configuración
en el build ni se modifica el archivo generado por Terraform en producción.

Para descargar resultados desde localhost, configura `MAIA_DEV_STORAGE_HOST`
con el hostname exacto del bucket S3 de trabajos. Vite sirve `/dev-results`
exclusivamente para GET firmados de JSON y CSV de ese bucket. No reenvía cookies,
tokens de invitado ni Origin; tampoco sigue redirecciones. La configuración
publicada sigue descargando directamente desde S3.
Vite también sirve `/dev-upload` para POST multipart firmados dirigidos exclusivamente
a la raíz del mismo bucket. Conserva el formulario y la versión de S3, limita el
tamaño a 129 MiB incluido el multipart y no reenvía cookies ni tokens del invitado.
No se modifican permisos de AWS.

El seguimiento distingue preparación, subida, envío, procesamiento y lectura del
hipnograma. Variables y predicción comparten etapa porque la API solo anuncia
`processing`, `succeeded` o `failed`; no se simulan porcentajes ni subetapas. Los errores 500 del servicio se muestran sin sustituirlos por
datos demo.
