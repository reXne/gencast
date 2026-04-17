# GenCast Repro

Implementacion completa, entrenable y reproducible en PyTorch de un modelo de pronostico meteorologico probabilistico inspirado en GenCast.

Este repositorio no afirma ser una copia bit a bit del codigo interno de Google DeepMind. Lo que si hace es dejar una reimplementacion publica, autocontenida y editable de los componentes centrales descritos en las fuentes publicas:

- GenCast usa un modelo de difusion condicional para generar ensembles de trayectorias meteorologicas a 12 horas por paso.
- El denoiser sigue el esquema de preacondicionamiento EDM/Karras.
- La arquitectura mezcla un encoder `grid -> mesh`, un processor sparse-transformer sobre una malla icosaedrica y un decoder `mesh -> grid`.
- El entrenamiento se hace sobre residuos respecto del ultimo estado conocido, con ponderacion por latitud y por variable.

## Estado del proyecto

El repo queda listo para:

- entrenar con ERA5 o WeatherBench2 en formato `zarr`, `netcdf` o multi-file netcdf,
- calcular estadisticas de normalizacion,
- entrenar un denoiser tipo GenCast,
- generar ensembles autoregresivos,
- evaluar RMSE, bias, CRPS y spread-skill,
- hacer pruebas de humo con un dataset sintetico cuando no tienes ERA5 local.

## Instalacion

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Comandos principales

```bash
gencast-repro fit-normalizer --config configs/mini.yaml
gencast-repro train --config configs/mini.yaml
gencast-repro evaluate --config configs/mini.yaml --checkpoint artifacts/checkpoints/best.pt
gencast-repro sample --config configs/mini.yaml --checkpoint artifacts/checkpoints/best.pt --split test --index 0
```

## Configuraciones incluidas

- `configs/mini.yaml`: modo de humo con dataset sintetico y malla pequena.
- `configs/era5_1deg.yaml`: configuracion publica razonable para una replica a 1 grado.
- `configs/era5_0p25deg.yaml`: configuracion ambiciosa mas cercana al paper, pensada para hardware serio.

## Fuentes publicas usadas

- Paper GenCast: <https://arxiv.org/abs/2312.15796>
- Repo oficial GraphCast/GenCast de Google DeepMind: <https://github.com/google-deepmind/graphcast>
- Pagina de publicacion de Google DeepMind: <https://deepmind.google/research/publications/gencast-learning-skillful-ensemble-forecasting-of-medium-range-weather/>

## Aviso importante

La paridad real con GenCast de produccion depende de:

- acceso a ERA5/WeatherBench2 completo,
- normalizacion y preprocessing alineados,
- hyperparameters de gran escala,
- hardware importante para entrenamiento global a 0.25 grados.

Este repo te deja la base de investigacion y entrenamiento para replicarlo publicamente, no pesos propietarios ni garantia de igualar el skill del paper sin esos recursos.

