# Pronósticos macroeconómicos con machine learning

Material reproducible del curso **Proyecciones Macroeconómicas con Machine
Learning y Deep Learning**. Cada sesión combina fundamentos generales de
forecasting con una aplicación real: la inflación peruana (Sesiones 1 y 2) y el
crecimiento del PIB de EE.UU. con FRED-QD (Sesión 3).

Los casos sirven para estudiar el método. Sus resultados no implican que un
modelo concreto sea superior en otros países, variables, horizontes o regímenes.

## Mensaje central

Forecasting no es solamente una carrera de algoritmos. Es un contrato de
evaluación definido por:

1. el objetivo y el horizonte;
2. la información disponible en cada origen;
3. la representación del target y de los predictores;
4. el benchmark;
5. la validación y el backtest temporal;
6. la pérdida relevante para la decisión.

La arquitectura del ejercicio debe fijarse antes de interpretar un ranking.

## Contenido

### Sesión 1: fundamentos del forecasting

[`session1/s1_fundamentos.ipynb`](session1/s1_fundamentos.ipynb)

- Componentes de una serie temporal y descomposición X13 (X-13ARIMA-SEATS vía MacroPy).
- Estacionariedad y uso prudente del test ADF.
- Sesgos inductivos de OLS y Random Forest.
- No linealidad, soporte observado y extrapolación.
- Tres experimentos controlados: relación lineal, umbral y tendencia temporal.

### Sesión 2: feature engineering, validación temporal y backtesting

[`session2/s2_pipeline.ipynb`](session2/s2_pipeline.ipynb)

- Definición del origen y del target a un mes (inflación peruana).
- Calendario aproximado de publicación y ragged edge.
- Construcción de 41 features económicas y dinámicas.
- Train, validación temporal y test bloqueado.
- Tuning de Ridge, Lasso y Random Forest dentro de development.
- Backtest expansivo frente a RW y AR(3)-OLS.
- Diagnóstico por regímenes y test de Diebold-Mariano.
- Coeficientes estandarizados y contribuciones a una predicción de Ridge.

### Sesión 3: métodos de ensamble con FRED-MD

[`session3/s3_A_ensambles.ipynb`](session3/s3_A_ensambles.ipynb) y
[`session3/s3_B_fredmd.ipynb`](session3/s3_B_fredmd.ipynb)

- Del árbol inestable al bagging y Random Forest; boosting, XGBoost y LightGBM.
- Los frenos que importan: learning rate, profundidad, hojas mínimas y early
  stopping con validación temporal.
- FRED-MD (McCracken y Ng): 126 series mensuales de EE.UU. con códigos de
  transformación, en un vintage congelado y documentado.
- Pronóstico de la inflación mensual de EE.UU., replicando en espíritu a
  Medeiros, Vasconcelos, Veiga y Zilberman (2021, JBES): rezagos de todo el
  panel más factores PCA estimados dentro de cada ventana, rolling de 30 años
  y 36 años de test.
- Importancias por gain y por permutación como puente a SHAP (Sesión 4).
- Notebook extra [`session3/s3_extra_pbi_fredqd.ipynb`](session3/s3_extra_pbi_fredqd.ipynb):
  la otra cara, PBI trimestral con FRED-QD y muestra corta, donde los
  benchmarks mandan (estilo Goulet Coulombe et al. 2022, JAE).

## Estructura del repositorio

```text
macro-forecasting-ml/
├── pyproject.toml                proyecto y dependencias para uv
├── requirements.txt             instalación alternativa con pip
├── uv.lock                      entorno reproducible de uv
├── session1/
│   ├── data/
│   │   ├── monthly.csv           panel mensual consolidado (base compartida)
│   │   └── metadata.csv          etiquetas, grupos y rezagos de publicación
│   ├── s1_fundamentos.ipynb
│   └── utils.py                  carga, estilo y helpers compartidos del curso
├── session2/
│   └── s2_pipeline.ipynb         usa session1/utils.py y session1/data/
└── session3/
    ├── data/
    │   ├── fredmd.csv            vintage congelado de FRED-MD (2026-07)
    │   ├── fredqd.csv            vintage congelado de FRED-QD (2026-07)
    │   └── VINTAGE.txt           fechas, URLs y citas de las descargas
    ├── output/                   caches parquet de los backtests
    ├── s3_A_ensambles.ipynb
    ├── s3_B_fredmd.ipynb
    ├── s3_extra_pbi_fredqd.ipynb
    └── utils.py                  loaders de FRED-MD/QD; reusa session1/utils.py
```

## Requisitos

- Python 3.11.
- `uv`, recomendado, o `pip` como alternativa.
- En macOS, `xgboost` y `lightgbm` necesitan OpenMP: `brew install libomp`.
- Una distribución de LaTeX con `latexmk` y LuaLaTeX para compilar las slides.

## Instalación

### Opción recomendada: uv

```bash
uv sync
uv run jupyter lab
```

`uv sync` usa `pyproject.toml` y `uv.lock` para reconstruir el entorno.

### Alternativa: pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

En Windows, la activación equivalente es `.venv\Scripts\activate`.

## Ejecución

1. `session1/s1_fundamentos.ipynb`: ejecutar todas las celdas.
2. `session2/s2_pipeline.ipynb`: ejecutar todas las celdas.
3. `session3/s3_A_ensambles.ipynb`: ejecutar todas las celdas.
4. `session3/s3_B_fredmd.ipynb`: la primera corrida ejecuta el backtest completo
   (varios minutos) y lo guarda en `session3/output/`; las siguientes cargan el
   cache en segundos. Lo mismo aplica al notebook extra de PBI.

Las figuras se guardan vía los `utils.py` de cada sesión en dos formatos: PDF
vectorial para las slides y PNG a 300 dpi para previsualización.

## Datos

`session1/data/monthly.csv` contiene un panel mensual consolidado de inflación,
actividad, encuestas, dinero y crédito, precios y variables externas del Perú
(enero de 1996 a junio de 2026, con borde irregular al final).
`session1/data/metadata.csv` documenta frecuencia, etiqueta, familia y rezago
aproximado de publicación por serie. Fuente: exportaciones del sistema de
nowcasting del profesor a partir de BCRP, INEI y FRED.

`session3/data/fredmd.csv` y `session3/data/fredqd.csv` son copias congeladas
de los `current.csv` mensual y trimestral de FRED-MD/QD (vintage 2026-07; ver
`VINTAGE.txt`), las bases públicas de McCracken y Ng para investigación macro
con "big data". Sus filas especiales (códigos de transformación y, en QD,
flags de factores) las separan automáticamente los loaders de
`session3/utils.py`.

Ninguna sesión necesita una API ni una clave: todo corre offline.

## Diseño empírico

**Sesión 2 (inflación peruana a un mes).** El origen es la publicación del IPC
del mes `t`; el objetivo es la inflación interanual de `t+1`. El Random Walk usa
la última observación; Ridge, Lasso y Random Forest aprenden una corrección al
mismo ancla. Esquema: train hasta 2015, validación 2016-2020 en cinco folds
expansivos, test bloqueado 2021-2026 con reestimación expansiva mensual. En la
versión actual Ridge obtiene el menor RMSE descriptivo (0.447 frente a 0.497
del RW), con ventaja concentrada en 2021-2023 y un DM limítrofe (p cerca de
0.050): evidencia interesante, no un veredicto universal.

**Sesión 3 (inflación mensual de EE.UU., estilo Medeiros et al. 2021).**
Target `1200*dlog(CPIAUCSL)`; benchmarks RW, media histórica y AR(4) directo;
challengers Ridge, Lasso, RF, XGBoost y LightGBM sobre ~490 features (4
rezagos de cada serie de FRED-MD y de la inflación) más 4 factores PCA
estimados dentro de cada ventana vía Pipeline. Esquema del paper: ventana
móvil de 360 meses con reestimación mensual; tuning en development (hasta
1989); test 1990-2026 con los subperiodos del paper (1990-2000, 2001-2015) y
la extensión 2016-2026; DM predefinido RF vs RW y diagnóstico declarado sin
2020-2021. En la versión actual Random Forest gana el test completo (RMSE
relativo al RW de 0.90; DM con p = 0.084) y XGBoost domina los subperiodos del
paper (0.82 en 1990-2000 y 0.89 en 2001-2015); la extensión 2016-2026 es más
pareja y el AR(4) recupera terreno. El mensaje del paper reproduce, con sus
matices.

El notebook extra (`s3_extra_pbi_fredqd.ipynb`) aplica el mismo pipeline al
crecimiento trimestral del PBI de EE.UU. con FRED-QD (test 2015-2026, regla de
los dos extremos para COVID). Ahí los benchmarks mandan: la comparación entre
ambos ejercicios es la lección sobre el tamaño de muestra.

## Limitaciones

- Los calendarios son aproximados (Sesión 2) o inexistentes por convención
  (Sesión 3, estándar de la literatura FRED): los datos son series revisadas,
  no vintages en tiempo real.
- Los tests están correctamente bloqueados dentro del código, pero las
  notebooks se diseñaron cuando esos periodos ya eran historia observada: son
  holdouts pedagógicos, no evaluaciones prospectivas vírgenes.
- Coeficientes, contribuciones e importancias son lecturas predictivas
  condicionadas a la parametrización; no identifican efectos causales.
- Los diagnósticos por subperiodos usan muestras pequeñas y no sustituyen
  nueva evidencia fuera de muestra.

## Solución de problemas

- macOS, error al importar `xgboost` o `lightgbm`: falta OpenMP. Instalar con
  `brew install libomp`.
- Linux, `OSError: libgomp.so.1: cannot open shared object file`: falta el
  runtime OpenMP de GCC. Instalar con `sudo apt-get install libgomp1`
  (Debian/Ubuntu) o `sudo dnf install libgomp` (Fedora). En Google Colab ya
  viene instalado.

## Fuentes y licencia

Las series peruanas proceden de exportaciones basadas en BCRP, INEI y FRED;
FRED-QD es de la Federal Reserve Bank of St. Louis (McCracken y Ng). Quien
redistribuya los datos debe comprobar las condiciones de uso de cada proveedor
original.

El repositorio aún no contiene un archivo `LICENSE`. Antes de publicar o
redistribuir el código y el material docente debe añadirse una licencia
explícita.
