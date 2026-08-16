# Pronósticos macroeconómicos con machine learning

Material reproducible del curso **Proyecciones Macroeconómicas con Machine
Learning y Deep Learning**. Cada sesión combina fundamentos generales de
forecasting con una aplicación real: la inflación peruana (Sesiones 1 y 2), la
inflación de EE.UU. con FRED-MD (Sesión 3) y su curva de Phillips con redes
neuronales (Sesión 4).

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

- Del árbol inestable al bagging y Random Forest; del bagging al boosting y XGBoost.
- Los frenos que importan: learning rate, profundidad, hojas mínimas y early
  stopping con validación temporal.
- FRED-MD (McCracken y Ng): 126 series mensuales de EE.UU. con códigos de
  transformación, en un vintage congelado y documentado.
- Pronóstico de la inflación mensual de EE.UU., replicando en espíritu a
  Medeiros, Vasconcelos, Veiga y Zilberman (2021, JBES): rezagos de todo el
  panel más factores PCA estimados dentro de cada ventana, rolling de 30 años
  y 36 años de test.
- Importancias por gain y por permutación: dos lupas sobre el mismo modelo y por
  qué no coinciden cuando las features están correlacionadas.

### Sesión 4: redes neuronales, estructura económica e incertidumbre

[`session4/s4_A_redes.ipynb`](session4/s4_A_redes.ipynb),
[`session4/s4_B_npc.ipynb`](session4/s4_B_npc.ipynb) y
[`session4/s4_C_incertidumbre.ipynb`](session4/s4_C_incertidumbre.ipynb)

- La intuición de las redes neuronales: la neurona como regresión con bisagra,
  aproximación universal, descenso de gradiente y backpropagation.
- Los frenos que importan con muestras cortas: early stopping con validación
  temporal, weight decay, dropout y ensembles.
- Extrapolación comparada: redes, árboles y OLS fuera del rango observado.
- La **curva de Phillips neuronal**: una Hemisphere Neural Network (HNN) en
  PyTorch, replicando en espíritu a Goulet Coulombe (2022). Una subred por
  bloque económico (actividad real, expectativas, costos, financiero) y el
  pronóstico como suma de contribuciones: interpretabilidad por construcción.
- Incertidumbre y pronóstico de densidad: bandas por bootstrap de bloques,
  predicción conforme (split y adaptativa) y medición de cobertura empírica.
- La frontera, conceptual: DensityHNN y la varianza condicional aprendida.

## Estructura del repositorio

```text
macro-forecasting-ml/
├── pyproject.toml                proyecto y dependencias para uv
├── requirements.txt             instalación alternativa con pip
├── uv.lock                      entorno reproducible de uv
├── session1/
│   ├── data/
│   │   ├── monthly.csv           panel mensual consolidado (BCRP, INEI, FRED)
│   │   └── metadata.csv          etiquetas, grupos y rezagos de publicación
│   ├── s1_fundamentos.ipynb
│   └── utils.py                  carga, estilo y helpers del curso
├── session2/
│   ├── data/                     misma copia del panel mensual y su diccionario
│   ├── s2_pipeline.ipynb
│   └── utils.py
├── session3/
│   ├── data/
│   │   ├── fredmd.csv            vintage congelado de FRED-MD (2026-07)
│   │   └── VINTAGE.txt           fecha, URL y cita de la descarga
│   ├── output/                   caches parquet de los backtests
│   ├── s3_A_ensambles.ipynb
│   ├── s3_B_fredmd.ipynb
│   └── utils.py                  loader de FRED-MD, métricas y estilo
└── session4/
    ├── data/
    │   ├── fredqd.csv            vintage congelado de FRED-QD (2026-07)
    │   └── VINTAGE.txt           fecha, URL y cita de la descarga
    ├── s4_A_redes.ipynb
    ├── s4_B_npc.ipynb
    ├── s4_C_incertidumbre.ipynb
    └── utils.py                  loader de FRED-QD, hemisferios, métricas
```

Cada sesión es autocontenida: su notebook encuentra el `utils.py` y el `data/`
que están junto a él, así que la carpeta puede copiarse o compartirse sola.

## Requisitos

- Python 3.11.
- `uv`, recomendado, o `pip` como alternativa.
- En macOS, `xgboost` necesita OpenMP: `brew install libomp`.
- `torch` se instala en su versión CPU; no hace falta GPU para el curso.
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
   cache en segundos.
5. `session4/s4_A_redes.ipynb`, `s4_B_npc.ipynb` y `s4_C_incertidumbre.ipynb`:
   cada uno corre en un par de minutos en CPU.

Las figuras se guardan vía los `utils.py` de cada sesión en dos formatos: PDF
vectorial para las slides y PNG a 300 dpi para previsualización.

## Datos

`monthly.csv` contiene un panel mensual consolidado de inflación, actividad,
encuestas, dinero y crédito, precios y variables externas del Perú (enero de
1996 a junio de 2026, con borde irregular al final). `metadata.csv` documenta
frecuencia, etiqueta, familia y rezago aproximado de publicación por serie.
Ambos viven en `session1/data/` y `session2/data/`, con copias idénticas, para
que cada sesión corra por su cuenta. Fuente: exportaciones del sistema de
nowcasting del profesor a partir de BCRP, INEI y FRED.

`session3/data/fredmd.csv` es una copia congelada del `current.csv` mensual de
FRED-MD (vintage 2026-07; ver `VINTAGE.txt`), la base pública de McCracken y Ng
para investigación macro con "big data". Su fila especial de códigos de
transformación la separa automáticamente el loader de `session3/utils.py`.

`session4/data/fredqd.csv` es la versión trimestral de la misma base
(vintage 2026-07), que la Sesión 4 usa para construir los bloques económicos de
la curva de Phillips neuronal.

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
challengers Ridge, Lasso, RF y XGBoost sobre ~490 features (4
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


**Sesión 4 (curva de Phillips neuronal, estilo Goulet Coulombe 2022).** Target
`400*dlog(CPIAUCSL)` trimestral; features: cuatro bloques de FRED-QD (actividad
real, expectativas de corto plazo, costos y commodities, dinero y financiero),
cada uno resumido en cinco componentes principales estimados solo con el train.
La HNN asigna una subred por bloque y suma sus salidas, así que cada término es
la contribución de ese bloque en puntos porcentuales. Train hasta 1994,
validación 1995-2006 para elegir configuración, test bloqueado desde 2007. En la
versión actual la HNN obtiene RMSE 2.269 sin COVID frente a 2.425 del AR(2) y
2.399 de una red densa con la misma información: la estructura mejora la
precisión además de dar interpretabilidad. La contribución del bloque de
actividad real se mueve al revés que el desempleo (la curva de Phillips emerge
sin imponerla) y su pendiente implícita es plana en los 2010 y la más empinada
de la muestra en los 2020.

La parte de incertidumbre compara cuatro maneras de construir bandas del 90%:
gaussiana del ensemble (cobertura 45%), bootstrap por bloques (43%), conforme
split (69%) y conforme adaptativo (90%), al precio de bandas casi el doble de
anchas.

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

- macOS, error al importar `xgboost`: falta OpenMP. Instalar con
  `brew install libomp`.
- Linux, `OSError: libgomp.so.1: cannot open shared object file`: falta el
  runtime OpenMP de GCC. No se arregla con pip, porque no es un paquete de
  Python:

  ```bash
  sudo apt-get update && sudo apt-get install -y libgomp1   # Debian/Ubuntu
  sudo dnf install -y libgomp                               # Fedora
  ```

  Sin permisos de administrador: `conda install -c conda-forge libgomp`. En
  Google Colab ya viene instalado.

## Fuentes y licencia

Las series peruanas proceden de exportaciones basadas en BCRP, INEI y FRED;
FRED-MD es de la Federal Reserve Bank of St. Louis (McCracken y Ng). Quien
redistribuya los datos debe comprobar las condiciones de uso de cada proveedor
original.

El repositorio aún no contiene un archivo `LICENSE`. Antes de publicar o
redistribuir el código y el material docente debe añadirse una licencia
explícita.
