# API HTTP

La API recibe una noche en EDF, lanza la inferencia del híbrido `stack_five` en Step Functions y devuelve el hipnograma por época. El código conserva rutas históricas para Parquet y otros modelos EDF, pero la imagen publicada las desactiva (ver `features` en `/health`). Es una aplicación Flask (`services/api/app.py`) que corre en Lambda detrás de API Gateway. No ejecuta el modelo: valida la solicitud, firma las cargas en S3, inicia el flujo y consulta su estado.

| Recurso | Dónde |
| --- | --- |
| Documentación interactiva (Swagger UI) | `GET {api}/docs` |
| Especificación OpenAPI 3.1 | `GET {api}/openapi.json`, versionada en [`services/api/openapi.json`](../services/api/openapi.json) |
| URL base `{api}` | Campo `api` de `config.json` del frontend o `terraform -chdir=infra/aws output -json public_config` |
| Flujo de uso de la aplicación y límites de entrada | [Operación en la nube](cloud-operations.md#api) |

## Espacios de rutas

| Prefijo | Autenticación | Uso |
| --- | --- | --- |
| `/health`, `/health/ready`, `/openapi.json`, `/docs` | Ninguna | Operación y documentación |
| `/public/v1` | Cabecera `X-Guest-Token` con 64 caracteres hexadecimales en minúsculas | Aplicación web pública. Cada clave ve solo sus trabajos. |
| `/v1` | `Authorization: Bearer <access token de Cognito>` | Integraciones con usuario. API Gateway valida el JWT y rechaza el ID token. |

`/public/v1` y `/v1` exponen las mismas cuatro operaciones con el mismo contrato: `GET examples/default`, `POST uploads`, `POST predictions` y `GET predictions/{job}`. API Gateway limita la API a 10 solicitudes/s con ráfagas de 15.

## Salud del servicio

| Endpoint | Qué comprueba | Respuesta |
| --- | --- | --- |
| `GET /health` | *Liveness*: que la función se ejecuta. No consulta AWS, así que es barato y siempre responde si Lambda corre. | `200 {"status":"ok","service":"maia-sleep-api","version":"8fc1e91474f6","features":{"parquet":false,"edf_fullnight":false,"edf_lightgbm":false,"edf_stack_five":true}}` |
| `GET /health/ready` | *Readiness*: que la API puede aceptar trabajos. Revisa la configuración, el bucket de trabajos (existe y tiene versionado, porque cada trabajo fija la versión S3 de su archivo) y el alias `live` de Step Functions. | `200 {"status":"ready","checks":{"config":"ok","storage":"ok","workflow":"ok"}}` o `503` con `"status":"unavailable"` y `error` o `skipped` en la comprobación afectada |

`version` son los 12 primeros caracteres del digest de la imagen desplegada. Sirve para confirmar que un despliegue llegó a Lambda. `features` indica qué entradas acepta el backend. La imagen publicada fija `ACTIVE_EDF_MODEL=stack_five`; con esa release híbrida activa solo `edf_stack_five` es verdadero y la API rechaza Parquet y los otros modelos EDF. Sin ella, `edf_fullnight` y `edf_lightgbm` siguen a `EDF_ENABLED` y `EDF_LGBM_ENABLED`. La web no consulta `/health`: acepta solo EDF con `stack_five` y, antes de habilitar el formulario, comprueba que `model_version` de `GET /public/v1/examples/default` sea la release esperada. Por seguridad, `/health/ready` no devuelve mensajes de AWS. El código del error (por ejemplo `AccessDeniedException`) queda en CloudWatch, en `/aws/lambda/maia-sleep-dev-api`, con la línea `Readiness check <nombre> failed`.

Cómo leer las respuestas desde fuera:

| Observación | Significado |
| --- | --- |
| `/health` 200 y `/health/ready` 200 | Servicio disponible. |
| `/health` 200 y `/health/ready` 503 | La función corre, pero falta configuración o falla S3 o Step Functions. Revisar el log. |
| `500 {"message":"Internal Server Error"}` en `/health` | API Gateway no pudo ejecutar la Lambda. En el laboratorio de AWS Academy ocurre cuando la sesión está detenida. |
| `404 {"message":"Not Found"}` en `/health` | La ruta no está desplegada en API Gateway: el entorno es anterior a este endpoint. |

```bash
api=$(jq -r .api frontend/dist/config.json)
curl -s "$api/health" | jq
curl -s -w '\nHTTP %{http_code}\n' "$api/health/ready"
```

`scripts/smoke_cloud.py` consulta ambos endpoints antes de la prueba de paridad y se detiene si la API no está lista. Por eso el despliegue de `deploy.yml` también los verifica. El informe añade `health` con la versión, `features` y `config_drift`, que lista los modelos en que `config.json` y el backend discrepan. Esa discrepancia no detiene la prueba ni afecta a la web, que ya no lee esos indicadores. Solo indica que `MAIA_PUBLIC_CONFIG` en Vercel quedó desactualizado.

## Ejemplo de uso como invitado

```bash
api=$(jq -r .api frontend/dist/config.json)
token=$(openssl rand -hex 32)
job=$(uuidgen | tr 'A-Z' 'a-z')
curl -s -X POST "$api/public/v1/predictions" \
  -H "X-Guest-Token: $token" -H "Idempotency-Key: $job" \
  -H 'content-type: application/json' -d '{"example":"default"}'
curl -s "$api/public/v1/predictions/$job" -H "X-Guest-Token: $token" | jq '.state, .files'
```

Para un archivo propio se pide `POST uploads` con `size` y `sha256`, se envía el formulario firmado a S3 y se inicia la predicción con `upload_id` y la cabecera `x-amz-version-id` de S3 como `version_id`. Con la release híbrida activa, ambas solicitudes deben incluir `"format":"edf"`. Los esquemas, los campos EDF (`format`, `model`, `profile`) y los códigos de error de cada operación están en `/docs`.

Todas las respuestas de error de la aplicación tienen la forma `{"error": "mensaje en español"}`. Las que genera API Gateway (JWT inválido, throttling o Lambda no disponible) tienen la forma `{"message": "…"}`.

## Mantener la documentación

`services/api/openapi.json` es la fuente de la documentación y se escribe a mano. Al añadir o cambiar una ruta hay que tocar tres sitios: la ruta Flask, la operación en `openapi.json` y la `route_key` en `infra/aws/main.tf`. Si falta la última, API Gateway responde 404 aunque Flask tenga la ruta. `tests/unit/test_cloud_api.py::test_flask_openapi_and_gateway_declare_the_same_routes` compara los tres conjuntos y falla si alguno no coincide. Para validar el documento:

```bash
uvx openapi-spec-validator services/api/openapi.json
```

La página `/docs` carga Swagger UI 5.17.14 desde jsDelivr con hashes de integridad (SRI). Si se actualiza la versión, hay que recalcular los hashes. El botón **Authorize** acepta la clave de invitado o el token de Cognito, y **Try it out** llama a la misma API.

## Por qué Flask y no FastAPI

FastAPI es la opción habitual para APIs nuevas en Python. Para esta API se evaluó con los criterios que importan en su despliegue real, y se mantuvo Flask.

**Qué hace la API.** Expone ocho rutas en unas 430 líneas. Cada solicitud hace una o dos llamadas a boto3 y responde en milisegundos. El trabajo pesado (descargar el modelo, leer el EDF y predecir) corre en otra Lambda que orquesta Step Functions. La autenticación JWT, CORS y el límite de solicitudes ya los resuelve API Gateway según Terraform, no el framework.

| Criterio | Ventaja de FastAPI | Situación en este proyecto | Peso |
| --- | --- | --- | --- |
| Concurrencia asíncrona (ASGI) | Un proceso atiende muchas solicitudes de E/S a la vez. | Lambda asigna un entorno por solicitud concurrente, de modo que un proceso nunca atiende dos solicitudes simultáneas. Además, boto3 es síncrono, así que FastAPI ejecutaría estos endpoints en un *threadpool*. | Nulo |
| Validación con Pydantic | Esquemas declarativos y errores 422 automáticos. | La validación es manual (UUID, SHA-256, límites de tamaño, perfiles EDF) y responde 400 o 503 con mensajes en español que el frontend muestra tal cual. Migrar cambiaría el formato de error o exigiría reescribir el manejador de 422. | Medio. Es la ventaja real. |
| OpenAPI y `/docs` automáticos | Se generan desde los tipos. | Se escribieron a mano en `openapi.json` y una prueba impide que el documento se separe del código y de API Gateway. | Medio |
| Integración con Lambda | Requiere un adaptador adicional (Mangum o Lambda Web Adapter). | Flask traduce el evento HTTP API 2.0 con su `test_client()` en unas 20 líneas, sin dependencias nuevas. | Bajo |
| Dependencias de la imagen | 12 paquetes y 16 MB (FastAPI, Starlette, Pydantic, pydantic-core compilado, AnyIO, Mangum…). | 7 paquetes y 6 MB. | Bajo |
| Arranque en frío | Importa más módulos. | Ver la medición. La diferencia solo afecta al primer arranque de cada contenedor. | Bajo |
| Coherencia interna | — | El servidor local de EDF (`src/sleep_staging/serving/local_edf.py`) también es Flask y sirve el frontend y la carga *multipart*. | Bajo |

**Medición.** Se usó la misma imagen base de la Lambda (`public.ecr.aws/lambda/python:3.12`, Python 3.12.16, x86_64) con siete repeticiones y las versiones de `uv.lock`. Cada repetición importa el framework y crea una aplicación con una ruta; en la de FastAPI se añade un modelo Pydantic y el adaptador Mangum. Se emuló x86_64 en Apple Silicon, así que los valores absolutos están inflados y lo comparable es la proporción.

| Medición | Flask 3.1.3 | FastAPI 0.156.4 + Pydantic 2.14.1 + Mangum 0.22.0 |
| --- | --- | --- |
| Importar y crear la app (mediana) | 107 ms (103–115) | 232 ms (208–246) |
| Tamaño instalado | 6 MB, 7 paquetes | 16 MB, 12 paquetes |
| Referencia: importar boto3 y crear un cliente S3 | 186 ms en caliente; 2,2–2,3 s en el primer arranque | Igual |

FastAPI añade unos 125 ms por arranque en frío en esta medición, pero boto3 domina el arranque de todas formas. El tamaño y el tiempo de arranque no deciden la elección por sí solos. Lo que decide es que la ventaja principal de FastAPI, la concurrencia asíncrona, no existe en Lambda con un cliente AWS síncrono. Sus otras ventajas, validación y documentación automáticas, se cubren aquí con validación explícita, una especificación OpenAPI escrita a mano y una prueba de consistencia. Migrar implicaría reescribir los ocho manejadores, añadir un adaptador, cambiar el formato de error que consumen el frontend y el smoke test, y volver a desplegar sin ganar capacidad ni latencia.

FastAPI aparece en `uv.lock` solo como dependencia transitiva de `mlflow-skinny`. La API no la usa.

**Cuándo reconsiderarlo.**

- La API crece hasta muchas operaciones o esquemas anidados y mantener `openapi.json` a mano empieza a costar más que migrar.
- La API pasa a un proceso de larga duración (ECS, App Runner o EC2) que atiende muchas solicitudes de E/S concurrentes.
- Se quiere notificar el progreso del trabajo con *streaming* (SSE o WebSocket) en lugar del sondeo actual cada 2 s.

Si llega ese momento, la migración tiene poco riesgo: las pruebas de `tests/unit/test_cloud_api.py` invocan `handler(event)` con eventos de API Gateway, así que validarían también una versión FastAPI detrás de Mangum sin reescribirse.
