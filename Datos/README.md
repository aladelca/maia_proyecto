# Muestra de datos Sleep-EDF

Esta carpeta incluye un registro real completo de Sleep Cassette, de Sleep-EDF Expanded 1.0.0, y su hipnograma de referencia. Permite explorar la estructura de los datos y probar la aplicación sin descargar toda la base.

| Archivo | Contenido |
| --- | --- |
| `SC4011E0-PSG.edf` | Señales fisiológicas del registro (48,74 MiB). Es el archivo que se carga en la aplicación. |
| `SC4011EH-Hypnogram.edf` | Estadios anotados por expertos, en EDF+. Se incluye como referencia; no es una entrada para la predicción. |
| `SHA256SUMS.txt` | Huellas de los dos archivos, verificadas contra las publicadas por PhysioNet. |

## Probar la aplicación

1. Descargar `SC4011E0-PSG.edf` desde esta carpeta (opción de descarga del archivo en GitHub), o usarlo desde un clon del repositorio.
2. Abrir [la aplicación](https://maia-sueno-dev-60306.vercel.app), seleccionar el EDF y pulsar **Generar predicción**. La sesión de AWS Academy debe estar activa.
3. Esperar el análisis y explorar el hipnograma y las probabilidades descargables.

El registro contiene EEG Fpz-Cz, EEG Pz-Oz y EOG horizontal a 100 Hz, y envolvente EMG submental a 1 Hz, además de canales auxiliares. La aplicación utiliza los cuatro canales requeridos por el modelo. El registro incluye vigilia; su duración no equivale al tiempo total de sueño.

Las anotaciones originales distinguen W, R, 1, 2, 3, 4, M y ?. El proyecto agrupa 3 y 4 como N3 y utiliza cinco clases finales: W, N1, N2, N3 y REM. La muestra no se ha recortado ni transformado.

## Procedencia y licencia

Datos de Bob Kemp, [Sleep-EDF Database Expanded, versión 1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/), PhysioNet, DOI: [10.13026/C2X676](https://doi.org/10.13026/C2X676). Se redistribuyen bajo [Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/); conservar esta atribución y la licencia al reutilizarlos.

Archivos originales:

- [SC4011E0-PSG.edf](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4011E0-PSG.edf)
- [SC4011EH-Hypnogram.edf](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4011EH-Hypnogram.edf)
- [Sumas SHA-256 oficiales](https://physionet.org/files/sleep-edfx/1.0.0/SHA256SUMS.txt)

Referencias:

- Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE Transactions on Biomedical Engineering. 2000;47(9):1185–1194. DOI: [10.1109/10.867928](https://doi.org/10.1109/10.867928).
- Pollard TJ et al. PhysioNet as a global platform for biomedical research. Nature Health. 2026. DOI: [10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z).

Para verificar la integridad desde esta carpeta: `shasum -a 256 -c SHA256SUMS.txt`. Los archivos se almacenan directamente en Git; esta muestra no requiere Git LFS. La base completa y los pasos de descarga se describen en el [README principal](../README.md) y en el [contrato de datos](../docs/data-contract.md).
