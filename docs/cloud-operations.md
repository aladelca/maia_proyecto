# Operación del laboratorio de sueño

El laboratorio sirve el híbrido FullNightNet y LightGBM `stack-five-a64415fc2bd6` (corrida MLflow a64415fc2bd649b39c6fd158994b0479). Una rama aporta 435 características y otra aporta cinco probabilidades neuronales. El clasificador final predice cinco etapas por época de 30 segundos. Sus pesos están en `modelos/stack_five/` y se copian en la imagen del worker. [Ficha del modelo](../modelos/README.md) · [Arquitectura y resultados](comparacion-fullnight-lightgbm.md)

## Probar la aplicación

En el [laboratorio de sueño](https://maia-sueno-dev-60306.vercel.app), selecciona **Probar ejemplo público** para procesar diez minutos de señal sintética. Para analizar una noche propia, selecciona un EDF y pulsa **Generar predicción**. El archivo debe contener EEG Fpz-Cz, EEG Pz-Oz y EOG horizontal a 100 Hz, y EMG submental en envolvente a 1 Hz. El tamaño máximo es 128 MiB y la duración admitida es de 30 segundos a 48 horas. El ejemplo y las cargas usan los mismos pesos incluidos en la imagen del worker. [Ejemplo EDF](../output/demo-sueno-sintetico-10min.edf) · [Ejecución local](../modelos/stack_five/README.md)

### API

El laboratorio se abre directamente. Las operaciones de la tabla están disponibles bajo `/public/v1`. Para las cargas y predicciones, el navegador genera una clave aleatoria de 32 bytes, la guarda durante la vida de la pestaña y la envía en la cabecera `X-Guest-Token` como 64 caracteres hexadecimales en minúsculas. Recargar la página conserva el acceso al resultado. Otra persona puede abrir el laboratorio y ejecutar su propia predicción. Las integraciones que ya usan Cognito pueden seguir usando `/v1` con su token de acceso.

La referencia completa está en `{api}/docs` (Swagger UI) y en la especificación OpenAPI `{api}/openapi.json`. La guía de la API explica los esquemas, los endpoints de salud y la elección de Flask. [Guía de la API](api.md)

| Solicitud | Respuesta |
| --- | --- |
| `GET /health` | Estado de la función, versión desplegada y entradas que acepta el backend. No requiere clave. |
| `GET /health/ready` | HTTP 200 si la configuración, S3 y Step Functions responden; HTTP 503 con la comprobación que falla. No requiere clave. |
| `GET /public/v1/examples/default` | Descripción del ejemplo. |
| `POST /public/v1/uploads` con `{"size": bytes,"sha256": digest}` | Formulario firmado válido diez minutos. Enviar sus campos y, al final, el archivo. Conservar la cabecera S3 `x-amz-version-id`. |
| `POST /public/v1/predictions` con cabecera UUID `Idempotency-Key` | Acepta `{"example":"default"}` o `{"upload_id":"…","version_id":"…"}`. Devuelve HTTP 202 y `status_url`. Reutilizar la clave con otro archivo devuelve HTTP 409. |
| `GET /public/v1/predictions/{job}` | Estado y enlaces temporales al JSON y al CSV. Consultar de nuevo para renovar los enlaces. |

La carga y la predicción envían el formato EDF y el identificador de modelo stack_five. El ejemplo acepta {"example":"default","model":"stack_five"}. La respuesta identifica la release, la corrida MLflow, el perfil, la ventana completa y las cinco clases. Las cargas y resultados expiran a partir de siete días por política de S3. El worker fija la versión del archivo y verifica su SHA-256.

## Infraestructura y releases históricas

Se necesitan AWS CLI, Terraform ≥1.10, Docker con buildx, Node 22 y uv. Para las credenciales temporales locales usar variables AWS estándar en `.env` con permisos 600. Cargarlas en la sesión con `set -a; source .env; set +a`. El proveedor Vercel lee `VERCEL_API_TOKEN`. Los contenedores ejecutados en AWS reciben credenciales de rol.

```bash
uv sync --locked --extra cloud --extra boosting
export AWS_REGION=us-west-2
infra/scripts/bootstrap.sh
for stack in platform network training aws vercel; do infra/scripts/init.sh "$stack"; done
terraform -chdir=infra/platform apply
terraform -chdir=infra/network apply
for name in api worker training; do infra/scripts/push-image.sh "$name"; done
```

Para la primera publicación, los artefactos de las campañas locales deben estar disponibles. Para un entorno existente, recuperar las referencias aprobadas con el procedimiento de restauración que figura abajo.

```bash
uv run --extra cloud --extra boosting --extra dl python scripts/export_serving_bundle.py --output var/cloud/bundle
uv run --extra cloud --extra boosting --extra dl python scripts/publish_cloud_inputs.py \
  --edf-run var/research/ibio-fullnight/development/681a52f21d273bf190ae \
  --stack-run var/research/ibio-fullnight/development/681a52f21d273bf190ae
```

`--stack-run` publica en el bucket de investigación las dos entradas del entrenamiento del híbrido: `fullnight_oof`, con las probabilidades fuera de pliegue de FullNightNet en las 152.052 épocas de desarrollo, y `stack_example`, con las 440 entradas del EDF de ejemplo. Antes comprueba que la red reajustada de la campaña sea la misma `six_direct.pt` que sirve el worker. Calcular el ejemplo requiere torch, por eso el comando incluye `--extra dl`.

`--edf-run` publica los pesos congelados del modelo EDF (`freeze.json` → `edf_model`) en `releases/ibio-edf/<sha256>/model.pt` y añade `objects.edf_model` a `inputs.json`. Con ese puntero, Terraform activa `EDF_ENABLED` en la API, pasa la clave, la versión y el SHA-256 al worker y publica `edfEnabled: true` en `public_config`. Para habilitar EDF en un entorno existente basta con añadir el puntero a `inputs.objects` de `delivery/config/aws.json` y aplicar `infra/aws`.

La release histórica de LightGBM desde EDF utiliza el modelo tabular ya publicado. Para habilitarlo en un entorno, establecer `edf_lgbm_enabled=true` en la configuración AWS aprobada (`delivery/config/aws.json`) y desplegar la API y el worker. Al crear una configuración desde cero, `infra/scripts/configure-app.py --edf-lgbm` incluye ese indicador. Verificar la paridad con `scripts/smoke_cloud.py --edf-model lightgbm --edf noche.edf --edf-expected hipnograma.csv` y las opciones habituales de entorno. Publicar después el valor `edfLgbmEnabled` de `public_config` en `MAIA_PUBLIC_CONFIG` y recompilar el frontend con Vercel. La opción está desactivada por defecto. Para retirarla, desactivar ambos indicadores.

Definir `TF_VAR_project_name` y `TF_VAR_team_id` con el proyecto y equipo Vercel. La primera aplicación crea el proyecto para fijar el origen canónico. En una cuenta normal Terraform crea los roles por componente. En el laboratorio, proporcionar `--existing-role-arn` para la API y `--instance-profile` y `--job-role` para entrenamiento con los roles preexistentes.

```bash
python3 infra/scripts/configure-app.py --origin "https://${TF_VAR_project_name}.vercel.app"
python3 infra/scripts/configure-training.py
terraform -chdir=infra/training apply
terraform -chdir=infra/aws apply
terraform -chdir=infra/vercel apply
```

Terraform guarda el estado cifrado, versionado y con bloqueo en S3, incluido el de bootstrap. La creación de buckets comprueba propietario y región mediante AWS CLI desde Terraform. Esto permite operar con las restricciones de lectura de Object Lock de la cuenta docente. Vercel usa su integración con GitHub para compilar los archivos de `frontend/`. La configuración pública del navegador se carga en el proyecto de Vercel. [Configuración de la entrega continua](#integración-y-entrega-continuas)

## Entrenar y promover

```bash
# Reajustar el híbrido servido (modo por defecto).
infra/scripts/run-training.sh stack
# Componente tabular: reajustar el LightGBM o repetir su calibración completa, con mayor consumo de EC2.
infra/scripts/run-training.sh refit
infra/scripts/run-training.sh tune
```

`stack` descarga por versión y SHA-256 el dataset de variables, la partición, el LightGBM aprobado, `fullnight_oof` y `stack_example`. Clona la configuración del LightGBM (261 árboles, semilla 542) y la ajusta con las 435 columnas más las cinco probabilidades de FullNightNet de los 62 participantes de desarrollo. Registra un run hijo `stack_five` con parámetros, métricas y el modelo, y deja en `s3://<research>/candidates/<UUID>/` el clasificador, su manifiesto y sus predicciones sobre el ejemplo. Con las mismas entradas, el código reprodujo en local el SHA-256 del `stack_five.joblib` publicado. FullNightNet no se reentrena en la nube: necesita GPU o MPS y la caché de señales de la campaña IBIO.

`refit` reajusta los 261 árboles del LightGBM tabular con la configuración del campeón y semilla 542. `tune` prueba las 12 configuraciones en una partición interna por sujeto en cada uno de los cinco folds, reajusta cada ganador, calcula predicciones externas y repite la selección para el ajuste final. Registra 78 ajustes, métricas y predicciones fuera de muestra (OOF). Ese modelo es la base del híbrido y la referencia histórica; la API desplegada no lo sirve.

El lanzador imprime el UUID y la instancia, espera el marcador de salida y retira el cómputo. Cada campaña tiene su propio estado Terraform. EC2 usa una subred con ruta de salida, grupo de seguridad sin acceso entrante, IMDSv2 y volumen cifrado que se borra al terminar. El apagado de emergencia se programa a los 90 minutos para `stack` y `refit`, y a las 12 horas para la calibración. CloudWatch registra el contenedor en `/maia/sleep/dev/training`. Los intentos guardan su estado de salida y log de arranque en S3. Para reanudar, usar `CAMPAIGN_ID=<UUID> infra/scripts/run-training.sh <modo>` con idénticos datos, modo e imagen. Por cada ajuste completado se conservan su versión de modelo y su ID de MLflow.

La demo usa SageMaker MLflow App. Para consultar el experimento, exportar `MLFLOW_TRACKING_URI` con el ARN del campo `tracking` de `terraform -chdir=infra/training output -json runner`. El cliente usa el complemento `sagemaker-mlflow`. Para cuentas sin el servicio, `configure-training.py --sqlite` configura MLflow por campaña con SQLite y artefactos en S3. La referencia `db` de `progress.json` permite restaurar el último snapshot por versión. [Integración de MLflow con AWS](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-track-experiments.html)

En cuentas que permiten Auto Scaling, `configure-training.py --batch` añade un entorno AWS Batch sobre EC2 con mínimo cero. Enviar el job usando los outputs `queue` y `definition` con `aws batch submit-job`. Sin argumentos, el contenedor reajusta el híbrido; para el LightGBM tabular, añadir `--container-overrides '{"command":["--mode","refit"]}'` o `tune`. El laboratorio usa EC2 directo porque denegó `CreateAutoScalingGroup`.

**Promover un híbrido.** El modelo servido vive en Git, así que la promoción es un PR:

```bash
uv run --extra cloud --extra boosting --extra dl python scripts/promote_stack_candidate.py "$JOB_ID"
git diff --stat   # modelos/stack_five/, stack_release.py y la versión que fija la web
make test
```

El script comprueba que el candidato proviene de la campaña y de las entradas aprobadas, y que usó las probabilidades de la red servida. Luego puntúa `output/demo-sueno-sintetico-10min.edf` con el candidato y exige las mismas etapas y probabilidades que calculó el entrenamiento, con una tolerancia de 10⁻⁶. Si el candidato es idéntico al modelo servido, no modifica nada. Si no lo es, reemplaza `stack_five.joblib` y `example-expected.csv`, regenera `src/sleep_staging/stack_release.py` y actualiza la versión en la web y en sus pruebas. Al mergear el PR, `deploy.yml` publica el worker y repite la paridad en AWS. Para revertir, se revierte el commit.

**Promover el LightGBM tabular.** La promoción verifica las predicciones congeladas del candidato antes de copiarlo al bucket de releases. Para decidir si reemplaza al campeón, revisar por separado las métricas externas de desarrollo. Conservar el ARN previo de Step Functions para revertir.

```bash
uv run --extra cloud --extra boosting python scripts/promote_cloud_candidate.py "$JOB_ID" --output var/cloud/approved.json
python3 infra/scripts/configure-app.py --origin "$ORIGIN" --inputs var/cloud/approved.json
terraform -chdir=infra/aws plan -out=release.tfplan
terraform -chdir=infra/aws apply release.tfplan
```

Al regenerar la configuración del laboratorio conservar `--existing-role-arn`. Cada versión de Step Functions invoca una versión fija de Lambda. Para revertir, aplicar `-var="live_version_arn=$PREVIOUS_WORKFLOW_ARN"` en `infra/aws`. Guardar el valor en la configuración del entorno mientras deba seguir activo y ejecutar la prueba de paridad.

## Restaurar y automatizar la entrega

Los ajustes aprobados se guardan en `delivery/config/{aws,vercel,training}.json` del bucket de investigación. Para respaldarlos ejecutar `uv run --extra cloud python infra/scripts/save-config.py` con los tres archivos locales `infra/<stack>/deploy.auto.tfvars.json`. El de Vercel identifica el equipo y el proyecto. Las copias anteriores pueden incluir `publish`. Para restaurar un checkout:

```bash
mkdir -p var/cloud
for stack in aws training vercel; do
  aws s3 cp "s3://$RESEARCH_BUCKET/delivery/config/$stack.json" "infra/$stack/deploy.auto.tfvars.json"
done
python3 -c 'import json; from pathlib import Path; Path("var/cloud/inputs.json").write_text(json.dumps(json.load(open("infra/aws/deploy.auto.tfvars.json"))["inputs"]))'
```

`infra/delivery` prepara la conexión a GitHub. La instalación de GitHub se autoriza una vez en la cuenta propietaria. Con una conexión disponible, `enabled=true`, `repository`, `research_bucket` y `deployment_role_arns` activan CodePipeline sobre `main`, con CodeBuild para validar, publicar imágenes por digest y desplegar los componentes de AWS. La política de release permite actualizar los servicios existentes. Los cambios de IAM y red se aplican antes desde la sesión de infraestructura. El laboratorio denegó `codestar-connections:CreateConnection`. Su entorno se desplegó con Terraform desde el checkout. Actualmente GitHub Actions automatiza la entrega del backend.

## Integración y entrega continuas

**Cada merge a `main` inicia una publicación de producción directamente en Vercel**, incluso cuando cambian archivos fuera de `frontend/`. El proyecto `maia-sueno-dev-60306` sigue la rama `main` de `aladelca/maia_proyecto`. Usa la raíz `frontend`, Vite, Node 22, `npm ci` y la salida `dist`. Las ramas de PR generan previews. El historial de producción de Vercel identifica el SHA publicado. [Despliegues Git de Vercel](https://vercel.com/docs/git)

En la configuración del proyecto de Vercel, define `MAIA_PUBLIC_CONFIG` para **Production** y **Preview** con el JSON que devuelve `terraform -chdir=infra/aws output -json public_config`. Incluye la URL pública de la API, el indicador `edfEnabled` y los identificadores públicos de Cognito. El build escribe estos campos en `dist/config.json`. Al cambiar el endpoint o la disponibilidad de un modelo en AWS, actualiza el valor y vuelve a desplegar la revisión. El build de la web usa únicamente su repositorio y esta configuración pública.

GitHub Actions valida los PR y despliega la inferencia cuando cambian archivos del backend. Los buildspecs de `ci/` siguen disponibles para una cuenta con CodeConnections.

| Workflow | Cuándo corre | Qué hace |
| --- | --- | --- |
| `ci.yml` | Cada PR y antes de cada despliegue | `make check` y `make test` con torch CPU, `terraform fmt` y `validate` de cada stack, build del frontend y de las tres imágenes, con una importación del código dentro de cada una. |
| `deploy.yml` | Push a `main` que modifica la inferencia, sus contenedores, la infraestructura AWS o sus scripts de entrega. También admite ejecución manual. | Repite CI, publica `api` y `worker` por digest, aplica `infra/aws`, ejecuta la prueba de paridad y registra las imágenes. |
| `code-review.yml` | PR del propio repositorio, con `CLAUDE_REVIEW=true` | Claude comenta hallazgos sobre fuga de datos, paridad, seguridad y Terraform. |
| `security.yml` | PR, `main` y cada lunes | CodeQL para Python, TypeScript y workflows. Revisión de dependencias nuevas en los PR. |
| `dependabot.yml` | Semanal o mensual | Propone actualizaciones de actions, npm y uv con siete días de espera desde su publicación. |

`make check` incluye `ci/check-pins.py`, que exige que las versiones de `containers/` coincidan con `uv.lock`. Así el worker y el entrenamiento usan las mismas bibliotecas numéricas que las corridas locales.

### Qué hace un despliegue

1. Lee `delivery/config/aws.json` del bucket de investigación y conserva su ETag.
2. Si cambió el backend, construye `api` y `worker`, los sube a ECR por digest y actualiza solo `images`. Un cambio de frontend o de documentación no publica versiones nuevas de Lambda.
3. Planifica `infra/aws`. `ci/check-plan.py` rechaza cualquier destrucción o reemplazo, que debe aplicarse desde una sesión de infraestructura. El resumen del job lista los recursos modificados.
4. Aplica el plan y ejecuta `scripts/smoke_cloud.py` con la noche de referencia de la release activa, descargada por VersionId y verificada por SHA-256. La prueba exige las mismas clases y probabilidades con una tolerancia de 10⁻⁶.
5. Si la aplicación o la prueba fallan, vuelve a aplicar las imágenes registradas y repite la prueba. Un cambio de Terraform se revierte con un commit de revert, que pasa de nuevo por el workflow.
6. Si la prueba pasa, escribe las imágenes nuevas en `delivery/config/aws.json` con `If-Match`. La escritura falla si alguien cambió el archivo durante el despliegue.
7. Con `RUN_E2E=true` o la opción manual `e2e`, ejecuta Playwright contra el sitio público.

La promoción de modelos y el entrenamiento siguen siendo manuales. El workflow conserva el modelo de `inputs` y nunca modifica `live_version_arn`. Si esa variable fija una versión anterior, el job lo advierte.

### Configurar GitHub

Crear el environment `dev` y limitar sus despliegues a la rama `main`. Antes del primer despliegue, respaldar la configuración vigente con `uv run --extra cloud python infra/scripts/save-config.py`, porque el workflow parte de ella.

| Tipo | Nombre | Valor |
| --- | --- | --- |
| Secreto de `dev` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` | Sesión de AWS Academy. `make ci-credentials` las copia desde `.env` con `gh`. |
| Secreto de `dev` | `DEMO_USERNAME`, `DEMO_PASSWORD` | Usuario Cognito de prueba. |
| Variable de `dev` | `AWS_REGION` | Opcional. `us-west-2` por defecto. |
| Variable de `dev` | `AWS_ROLE_ARN` | En una cuenta propia, rol con confianza OIDC para `repo:aladelca/maia_proyecto:environment:dev` y los permisos de build y release de `infra/delivery/roles.tf`. Reemplaza a los tres secretos de AWS. |
| Variable de `dev` | `RUN_E2E` | `true` ejecuta Playwright en cada despliegue. |
| Variable de repositorio | `CLAUDE_REVIEW` | `true` activa la revisión automática. |
| Secreto de repositorio | `ANTHROPIC_API_KEY` o `CLAUDE_CODE_OAUTH_TOKEN` | Credencial de la revisión. `claude setup-token` genera el token OAuth. |
| Variable de repositorio | `CLAUDE_REVIEW_MODEL` | Opcional. `claude-opus-5-5` por defecto. |

Las credenciales de AWS Academy vencen al terminar la sesión del laboratorio. Al iniciar una nueva, actualizar `.env` y ejecutar `make ci-credentials`. Si un despliegue falla en el paso de credenciales con `ExpiredToken`, se vuelve a ejecutar después de actualizar los secretos. El repositorio es público: los logs enmascaran el ID de cuenta y los secretos.

## Verificar y dar acceso

Para probar desde Chrome, exportar `DEMO_ORIGIN` con la URL pública del entorno. La prueba de compatibilidad con la API de Cognito usa `DEMO_USERNAME` y `DEMO_PASSWORD` cuando se ejecuta el smoke sin `--guest`.

```bash
uv run --extra cloud python scripts/smoke_cloud.py --guest
uv run --extra cloud python scripts/smoke_cloud.py --guest \
  --edf noche-PSG.edf --edf-expected noche-local.csv   # CSV de serve_fullnight.py --edf
BROWSER_CHANNEL=chrome npm --prefix frontend run test:e2e
```

La prueba de API consulta primero `/health/ready` y `/health`, y anota en `health.config_drift` los modelos en que `config.json` y el backend discrepan. Después verifica permisos, carga firmada, idempotencia y las cinco probabilidades contra la referencia de la noche. Playwright prueba el acceso público desde un navegador nuevo, el aislamiento de los resultados entre visitantes, el procesamiento, la recarga, la carga del EDF de ejemplo y la descarga del CSV. CloudWatch conserva siete días de logs y dispone de alarmas por errores del worker, throttling de la API y fallos del flujo. El almacenamiento, ECR, tracking e invocaciones pueden generar cargos aun cuando no haya una instancia de entrenamiento activa.

## Evidencia del despliegue de desarrollo

Las filas del 18 de noviembre corresponden al LightGBM tabular que precedió al híbrido, y la del 20 de noviembre a FullNightNet sola. El 18 de noviembre de 2026 se comprobó el flujo en `us-west-2` con la noche pública SC4001. Se ejecutaron tanto el botón de ejemplo como la carga de archivo desde Chrome. En las 841 épocas coincidieron todas las clases y las cinco probabilidades con la referencia local dentro de 5,6 × 10⁻¹⁶. Un segundo usuario autenticado recibió HTTP 404 al consultar el trabajo del primero.

| Prueba | Resultado observado |
| --- | --- |
| Un ejemplo con el servicio caliente | Aceptación en 0,50 s y resultado en 1,83 s. |
| Cinco ejemplos simultáneos | p95 de aceptación de 0,59 s y p95 de finalización de 6,51 s. Es una medición de cinco trabajos. |
| Primer refit en EC2 | 261 árboles en 318,9 s de ajuste. Run hijo `76b5b4f69a8f4ff196ce666adf83cdd1`, experimento `sleep-cassette-cloud` en SageMaker MLflow. |
| Comparación del candidato Linux con el campeón local | 841 de 841 etiquetas iguales y diferencia máxima de probabilidad de 8,7 × 10⁻¹⁹ antes de serializar a JSON. |
| Segundo refit y recuperación | El segundo ajuste tardó 277,1 s y produjo el mismo SHA-256 de modelo. Al reanudar la campaña en una instancia nueva, se conservó el run hijo `a4b951281b78499bb06ccc8153da6e52` y su checkpoint. Se verificó que las dos campañas quedaron finalizadas en MLflow y se retiraron las instancias. |
| Cambio y reversión | La API procesó el ejemplo y la carga con el candidato EC2 `lgbm-a0f29b53352d6156`. Se restauró el campeón `lgbm-49f74e944ce5457b`. |
| Ejemplo público del híbrido (23 de noviembre) | `stack-five-a64415fc2bd6`, 20 épocas, perfil con los cuatro canales y ventana completa. La primera solicitud tardó 10,3 s de punta a punta (5,6 s de inferencia) y las dos siguientes 4,9 s (4,05 s). La primera fila del CSV coincidió con `modelos/stack_five/example-expected.csv`, y otra clave de invitado recibió HTTP 404 al consultar el trabajo. |
| Carga EDF (20 de noviembre) | Noche completa SC4121E0 de 48 MiB, 2.786 épocas, modelo `681a52f21d273bf190ae/six`. Con `auto` (dos EEG, EOG y EMG) y con `eeg_pz_oz+eog`, todas las clases coincidieron con `serve_fullnight.py` en CPU local. La diferencia máxima de probabilidad fue de 3,9 × 10⁻⁷. La inferencia tardó 48,4 s en frío (carga de torch y del modelo) y 2,1 s en caliente. Un Parquet enviado como EDF y un perfil sin EEG se rechazaron con mensajes controlados. |

La comparación de la noche verifica portabilidad e implementación. Las métricas de generalización se calculan con los folds por sujeto.

El stack de plataforma admite `monthly_budget` con `limit_usd` y `recipient` para un presupuesto mensual de toda la cuenta y un aviso al 80 %. En la demo quedó desactivado porque no se definieron un monto ni un destinatario.
