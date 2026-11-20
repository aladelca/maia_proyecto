# Operación del laboratorio de sueño

El frontend procesa noches Sleep Cassette preparadas y devuelve W, N1, N2, N3 y REM por época de 30 segundos. El modelo inicial es el LightGBM ajustado del proyecto. Su F1 macro histórico de desarrollo es 0,7893. El refit usa los 62 sujetos de desarrollo y reserva los 16 sujetos del test. [Resultados científicos](refinement-results.md)

## Probar la aplicación

Abre [el laboratorio en Vercel](https://maia-sueno-dev-60306.vercel.app), inicia sesión y selecciona **Probar ejemplo público**. Puedes recargar la página mientras se procesa la noche. Al terminar verás el hipnograma, el tiempo por estadio, la versión del modelo y la descarga del CSV. Para analizar una noche propia, selecciona el Parquet y pulsa **Generar predicción**.

| Entrada | Contrato |
| --- | --- |
| Registro | Un `recording_id`, `epoch_id` únicos, `onset_s` entero sobre una cuadrícula de 30 s y `eligible` booleano. Los identificadores comienzan por un carácter alfanumérico y admiten letras, dígitos, punto, dos puntos, guion y guion bajo. |
| Señales | Las 145 columnas flotantes y finitas de `raw_columns` del manifiesto. Incluir las épocas excluidas que aportan contexto. El worker deriva las otras 290 columnas dentro de la noche. |
| Procedencia | Metadatos Parquet `input_schema=sleep-cassette-night-v1` y `preprocessing_id=sc-w30-context435-v1`. El recorte usa el protocolo retrospectivo del proyecto. |
| Límites | 32 MiB comprimidos, 3.000 épocas, 128 grupos de filas y 64 MiB de tamaño declarado sin comprimir. |

El exportador escribe la noche de ejemplo en `var/cloud/bundle/example.parquet`. Un operador puede recuperarla con `aws s3api get-object` usando `buckets.releases` y el par `key/version` de `objects["example.parquet"]` en `inputs.json`.

### API

Cognito usa Authorization Code con PKCE. Las rutas exigen un token de acceso y el scope `aws.cognito.signin.user.admin`. Las claves de entrada y salida se derivan del propietario autenticado.

| Solicitud | Respuesta |
| --- | --- |
| `GET /v1/examples/default` | Descripción del ejemplo. |
| `POST /v1/uploads` con `{"size": bytes,"sha256": digest}` | Formulario firmado válido diez minutos. Enviar sus campos y, al final, el archivo. Conservar la cabecera S3 `x-amz-version-id`. |
| `POST /v1/predictions` con cabecera UUID `Idempotency-Key` | Acepta `{"example":"default"}` o `{"upload_id":"…","version_id":"…"}`. Devuelve HTTP 202 y `status_url`. Reutilizar la clave con otro archivo devuelve HTTP 409. |
| `GET /v1/predictions/{job}` | Estado y enlaces temporales al JSON y al CSV. Consultar de nuevo para renovar los enlaces. |

Las cargas y resultados expiran a partir de siete días por política de S3. El worker fija la versión del archivo y verifica su SHA-256. Verifica también el manifiesto y el modelo antes de cargar el pipeline.

## Desplegar desde la raíz del repositorio

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
uv run --extra cloud --extra boosting python scripts/export_serving_bundle.py --output var/cloud/bundle
uv run --extra cloud python scripts/publish_cloud_inputs.py
```

Definir `TF_VAR_project_name` y `TF_VAR_team_id` con el proyecto y equipo Vercel. La primera aplicación crea el proyecto para fijar el origen canónico. En una cuenta normal Terraform crea los roles por componente. En el laboratorio, proporcionar `--existing-role-arn` para la API y `--instance-profile` y `--job-role` para entrenamiento con los roles preexistentes.

```bash
terraform -chdir=infra/vercel apply -var=publish=false
python3 infra/scripts/configure-app.py --origin "https://${TF_VAR_project_name}.vercel.app"
python3 infra/scripts/configure-training.py
terraform -chdir=infra/training apply
terraform -chdir=infra/aws apply
npm --prefix frontend ci
npm --prefix frontend run build
terraform -chdir=infra/aws output -json public_config > frontend/dist/config.json
terraform -chdir=infra/vercel apply
```

Terraform guarda el estado cifrado, versionado y con bloqueo en S3, incluido el de bootstrap. La creación de buckets comprueba propietario y región mediante AWS CLI desde Terraform. Esto permite operar con las restricciones de lectura de Object Lock de la cuenta docente. Vercel recibe los archivos compilados de `frontend/dist`.

## Entrenar y promover

```bash
infra/scripts/run-training.sh refit
# Repetir la calibración completa, con mayor consumo de EC2.
infra/scripts/run-training.sh tune
```

`refit` descarga los datos, partición y configuración por versión y SHA-256 y reajusta los 261 árboles con la configuración del campeón y semilla 542. `tune` prueba las 12 configuraciones en una partición interna por sujeto en cada uno de los cinco folds, reajusta cada ganador, calcula predicciones externas y repite la selección para el ajuste final. Registra 78 ajustes, métricas y predicciones fuera de muestra (OOF). Cada ajuste tiene su run hijo de MLflow. La campaña escribe en `s3://<research>/candidates/<UUID>/`.

El lanzador imprime el UUID y la instancia, espera el marcador de salida y retira el cómputo. Cada campaña tiene su propio estado Terraform. EC2 usa una subred con ruta de salida, grupo de seguridad sin acceso entrante, IMDSv2 y volumen cifrado que se borra al terminar. El apagado de emergencia se programa a los 90 minutos para el refit y a las 12 horas para la calibración. CloudWatch registra el contenedor en `/maia/sleep/dev/training`. Los intentos guardan su estado de salida y log de arranque en S3. Para reanudar, usar `CAMPAIGN_ID=<UUID> infra/scripts/run-training.sh refit` con idénticos datos, modo e imagen. Por cada ajuste completado se conservan su versión de modelo y su ID de MLflow.

La demo usa SageMaker MLflow App. Para consultar el experimento, exportar `MLFLOW_TRACKING_URI` con el ARN del campo `tracking` de `terraform -chdir=infra/training output -json runner`. El cliente usa el complemento `sagemaker-mlflow`. Para cuentas sin el servicio, `configure-training.py --sqlite` configura MLflow por campaña con SQLite y artefactos en S3. La referencia `db` de `progress.json` permite restaurar el último snapshot por versión. [Integración de MLflow con AWS](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-track-experiments.html)

En cuentas que permiten Auto Scaling, `configure-training.py --batch` añade un entorno AWS Batch sobre EC2 con mínimo cero. Enviar el job usando los outputs `queue` y `definition` con `aws batch submit-job`. Para recalibrar, añadir `--container-overrides '{"command":["--mode","tune"]}'`. El laboratorio usa EC2 directo porque denegó `CreateAutoScalingGroup`.

La promoción verifica las predicciones congeladas del candidato antes de copiarlo al bucket de releases. Para decidir si reemplaza al campeón, revisar por separado las métricas externas de desarrollo. Conservar el ARN previo de Step Functions para revertir.

```bash
uv run --extra cloud --extra boosting python scripts/promote_cloud_candidate.py "$JOB_ID" --output var/cloud/approved.json
python3 infra/scripts/configure-app.py --origin "$ORIGIN" --inputs var/cloud/approved.json
terraform -chdir=infra/aws plan -out=release.tfplan
terraform -chdir=infra/aws apply release.tfplan
```

Al regenerar la configuración del laboratorio conservar `--existing-role-arn`. Cada versión de Step Functions invoca una versión fija de Lambda. Para revertir, aplicar `-var="live_version_arn=$PREVIOUS_WORKFLOW_ARN"` en `infra/aws`. Guardar el valor en la configuración del entorno mientras deba seguir activo y ejecutar la prueba de paridad.

## Restaurar y automatizar la entrega

Los ajustes aprobados se guardan en `delivery/config/{aws,vercel,training}.json` del bucket de investigación. Para respaldarlos ejecutar `uv run --extra cloud python infra/scripts/save-config.py` con los tres archivos locales `infra/<stack>/deploy.auto.tfvars.json`. El de Vercel incluye únicamente `team_id`, `project_name` y `publish`. Para restaurar un checkout:

```bash
mkdir -p var/cloud
for stack in aws training vercel; do
  aws s3 cp "s3://$RESEARCH_BUCKET/delivery/config/$stack.json" "infra/$stack/deploy.auto.tfvars.json"
done
python3 -c 'import json; from pathlib import Path; Path("var/cloud/inputs.json").write_text(json.dumps(json.load(open("infra/aws/deploy.auto.tfvars.json"))["inputs"]))'
```

`infra/delivery` prepara la conexión a GitHub. La instalación de GitHub se autoriza una vez en la cuenta propietaria. Con una conexión disponible, `enabled=true`, `repository`, `research_bucket` y `deployment_role_arns` activan CodePipeline sobre `main`, con CodeBuild para validar, publicar imágenes por digest y desplegar AWS y Vercel. La política de release permite actualizar los servicios existentes. Los cambios de IAM y red se aplican antes desde la sesión de infraestructura. El valor del token de Vercel se carga en Secrets Manager fuera de Terraform. El laboratorio denegó `codestar-connections:CreateConnection`. Su entorno se desplegó con Terraform desde el checkout. Desde entonces, GitHub Actions automatiza la entrega, como describe la sección siguiente.

## Integración y entrega continuas

GitHub Actions valida cada pull request y despliega la inferencia al integrar en `main`. Reemplaza a CodePipeline en el laboratorio. Los buildspecs de `ci/` se conservan para una cuenta con CodeConnections.

| Workflow | Cuándo corre | Qué hace |
| --- | --- | --- |
| `ci.yml` | Cada PR y antes de cada despliegue | `make check` y `make test` con torch CPU, `terraform fmt` y `validate` de cada stack, build del frontend y de las tres imágenes, con una importación del código dentro de cada una. |
| `deploy.yml` | Push a `main` que modifica código servido, contenedores, `infra/aws`, `infra/vercel` o `ci/`. También se ejecuta a mano. | Repite CI, publica `api` y `worker` por digest, aplica `infra/aws`, ejecuta la prueba de paridad, registra las imágenes y publica Vercel. |
| `code-review.yml` | PR del propio repositorio, con `CLAUDE_REVIEW=true` | Claude comenta hallazgos sobre fuga de datos, paridad, seguridad y Terraform. |
| `security.yml` | PR, `main` y cada lunes | CodeQL para Python, TypeScript y workflows. Revisión de dependencias nuevas en los PR. |
| `dependabot.yml` | Semanal o mensual | Propone actualizaciones de actions, npm y uv con siete días de espera desde su publicación. |

`make check` incluye `ci/check-pins.py`, que exige que las versiones de `containers/` coincidan con `uv.lock`. Así el worker y el entrenamiento usan las mismas bibliotecas numéricas que las corridas locales.

### Qué hace un despliegue

1. Lee `delivery/config/aws.json` y `vercel.json` del bucket de investigación y conserva el ETag del primero.
2. Si cambió el backend, construye `api` y `worker`, los sube a ECR por digest y actualiza solo `images`. Un cambio de frontend o de documentación no publica versiones nuevas de Lambda.
3. Planifica `infra/aws`. `ci/check-plan.py` rechaza cualquier destrucción o reemplazo, que debe aplicarse desde una sesión de infraestructura. El resumen del job lista los recursos modificados.
4. Aplica el plan y ejecuta `scripts/smoke_cloud.py` con la noche de referencia de la release activa, descargada por VersionId y verificada por SHA-256. La prueba exige las mismas clases y probabilidades con una tolerancia de 10⁻⁶.
5. Si la aplicación o la prueba fallan, vuelve a aplicar las imágenes registradas y repite la prueba. Un cambio de Terraform se revierte con un commit de revert, que pasa de nuevo por el workflow.
6. Si la prueba pasa, escribe las imágenes nuevas en `delivery/config/aws.json` con `If-Match`. La escritura falla si alguien cambió el archivo durante el despliegue.
7. Compila el frontend con la configuración pública y lo publica en Vercel. Con `RUN_E2E=true` o la opción manual `e2e`, ejecuta además Playwright contra el sitio.

La promoción de modelos y el entrenamiento siguen siendo manuales. El workflow conserva el modelo de `inputs` y nunca modifica `live_version_arn`. Si esa variable fija una versión anterior, el job lo advierte.

### Configurar GitHub

Crear el environment `dev` y limitar sus despliegues a la rama `main`. Antes del primer despliegue, respaldar la configuración vigente con `uv run --extra cloud python infra/scripts/save-config.py`, porque el workflow parte de ella.

| Tipo | Nombre | Valor |
| --- | --- | --- |
| Secreto de `dev` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` | Sesión de AWS Academy. `make ci-credentials` las copia desde `.env` con `gh`. |
| Secreto de `dev` | `VERCEL_API_TOKEN` | Token de despliegue de Vercel. |
| Secreto de `dev` | `DEMO_USERNAME`, `DEMO_PASSWORD` | Usuario Cognito de prueba. |
| Variable de `dev` | `AWS_REGION` | Opcional. `us-west-2` por defecto. |
| Variable de `dev` | `AWS_ROLE_ARN` | En una cuenta propia, rol con confianza OIDC para `repo:aladelca/maia_proyecto:environment:dev` y los permisos de build y release de `infra/delivery/roles.tf`. Reemplaza a los tres secretos de AWS. |
| Variable de `dev` | `RUN_E2E` | `true` ejecuta Playwright en cada despliegue. |
| Variable de repositorio | `CLAUDE_REVIEW` | `true` activa la revisión automática. |
| Secreto de repositorio | `ANTHROPIC_API_KEY` o `CLAUDE_CODE_OAUTH_TOKEN` | Credencial de la revisión. `claude setup-token` genera el token OAuth. |
| Variable de repositorio | `CLAUDE_REVIEW_MODEL` | Opcional. `claude-opus-5-5` por defecto. |

Las credenciales de AWS Academy vencen al terminar la sesión del laboratorio. Al iniciar una nueva, actualizar `.env` y ejecutar `make ci-credentials`. Si un despliegue falla en el paso de credenciales con `ExpiredToken`, se vuelve a ejecutar después de actualizar los secretos. El repositorio es público: los logs enmascaran el ID de cuenta y los secretos.

## Verificar y dar acceso

Cognito admite usuarios por invitación. Un administrador puede crearlos con `admin-create-user --message-action SUPPRESS` y darles acceso de forma privada. Para las pruebas, exportar `DEMO_USERNAME`, `DEMO_PASSWORD` y `DEMO_ORIGIN`.

```bash
uv run --extra cloud python scripts/smoke_cloud.py
BROWSER_CHANNEL=chrome npm --prefix frontend run test:e2e
```

La prueba de API verifica permisos, carga firmada, idempotencia y las cinco probabilidades contra la referencia de la noche. Playwright prueba Vercel, Cognito, procesamiento, recarga, carga de Parquet y descarga del CSV. CloudWatch conserva siete días de logs y dispone de alarmas por errores del worker, throttling de la API y fallos del flujo. El almacenamiento, ECR, tracking e invocaciones pueden generar cargos aun cuando no haya una instancia de entrenamiento activa.

## Evidencia del despliegue de desarrollo

El 18 de noviembre de 2026 se comprobó el flujo en `us-west-2` con la noche pública SC4001. Se ejecutaron tanto el botón de ejemplo como la carga de archivo desde Chrome. En las 841 épocas coincidieron todas las clases y las cinco probabilidades con la referencia local dentro de 5,6 × 10⁻¹⁶. Un segundo usuario autenticado recibió HTTP 404 al consultar el trabajo del primero.

| Prueba | Resultado observado |
| --- | --- |
| Un ejemplo con el servicio caliente | Aceptación en 0,50 s y resultado en 1,83 s. |
| Cinco ejemplos simultáneos | p95 de aceptación de 0,59 s y p95 de finalización de 6,51 s. Es una medición de cinco trabajos. |
| Primer refit en EC2 | 261 árboles en 318,9 s de ajuste. Run hijo `76b5b4f69a8f4ff196ce666adf83cdd1`, experimento `sleep-cassette-cloud` en SageMaker MLflow. |
| Comparación del candidato Linux con el campeón local | 841 de 841 etiquetas iguales y diferencia máxima de probabilidad de 8,7 × 10⁻¹⁹ antes de serializar a JSON. |
| Segundo refit y recuperación | El segundo ajuste tardó 277,1 s y produjo el mismo SHA-256 de modelo. Al reanudar la campaña en una instancia nueva, se conservó el run hijo `a4b951281b78499bb06ccc8153da6e52` y su checkpoint. Se verificó que las dos campañas quedaron finalizadas en MLflow y se retiraron las instancias. |
| Cambio y reversión | La API procesó el ejemplo y la carga con el candidato EC2 `lgbm-a0f29b53352d6156`. Se restauró el campeón `lgbm-49f74e944ce5457b`. |

La comparación de la noche verifica portabilidad e implementación. Las métricas de generalización se calculan con los folds por sujeto.

El stack de plataforma admite `monthly_budget` con `limit_usd` y `recipient` para un presupuesto mensual de toda la cuenta y un aviso al 80 %. En la demo quedó desactivado porque no se definieron un monto ni un destinatario.
