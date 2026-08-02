# Pronósticos macroeconómicos con machine learning

Material reproducible de la Sesión 1 del curso **Proyecciones Macroeconómicas
con Machine Learning y Deep Learning**. La sesión combina fundamentos generales
de forecasting con una aplicación a la inflación peruana.

El caso sirve para estudiar el método. Sus resultados no implican que un modelo
concreto sea superior en otros países, variables, horizontes o regímenes.

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

### Parte A: fundamentos

[`session1/s1_A_fundamentos.ipynb`](session1/s1_A_fundamentos.ipynb)

- Componentes de una serie temporal y descomposición STL.
- Estacionariedad y uso prudente del test ADF.
- Sesgos inductivos de OLS y Random Forest.
- No linealidad, soporte observado y extrapolación.
- Tres experimentos controlados: relación lineal, umbral y tendencia temporal.

### Parte B: pipeline temporal

[`session1/s1_B_pipeline.ipynb`](session1/s1_B_pipeline.ipynb)

- Definición del origen y del target a un mes.
- Calendario aproximado de publicación y ragged edge.
- Construcción de 41 features económicas y dinámicas.
- Train, validación temporal y test bloqueado.
- Tuning de Ridge, Lasso y Random Forest dentro de development.
- Backtest expansivo frente a RW y AR(3)-OLS.
- Diagnóstico por regímenes y test de Diebold-Mariano.
- Coeficientes estandarizados y contribuciones a una predicción de Ridge.

## Estructura del repositorio

```text
macro-forecasting-ml/
├── pyproject.toml                proyecto y dependencias para uv
├── requirements.txt             instalación alternativa con pip
├── uv.lock                      entorno reproducible de uv
└── session1/
    ├── data/
    │   ├── monthly.csv           panel mensual consolidado
    │   └── metadata.csv          etiquetas, grupos y rezagos de publicación
    ├── s1_A_fundamentos.ipynb
    ├── s1_B_pipeline.ipynb
    └── utils.py                  carga, estilo y helpers de figuras
```

Las notebooks cargan exclusivamente los archivos de `session1/data/`.

## Requisitos

- Python 3.11.
- `uv`, recomendado, o `pip` como alternativa.
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

1. Abrir `session1/s1_A_fundamentos.ipynb` y ejecutar todas las celdas.
2. Abrir `session1/s1_B_pipeline.ipynb` y ejecutar todas las celdas.
3. Comprobar que cada notebook termina sin error y que actualiza sus figuras.

Las figuras se guardan mediante `session1/utils.py` en dos formatos:

- PDF vectorial para la presentación;
- PNG a 300 dpi para previsualización o uso externo.

Los PDF vectoriales se versionan porque son insumos de las slides. Los PNG son
reproducibles y están ignorados por Git.

## Datos

`session1/data/monthly.csv` contiene un panel mensual consolidado de inflación,
actividad, encuestas, dinero y crédito, precios y variables externas. La muestra
abarca de enero de 1996 a junio de 2026, con un borde irregular al final. El
archivo trimestral abarca de 1995-T1 a 2026-T1.

`session1/data/metadata.csv` documenta la frecuencia, la etiqueta, la familia y
un rezago aproximado de publicación para cada serie. Las fechas representan el
periodo de referencia; el valor solo entra al conjunto de información después
de aplicar su rezago.

Los datos fueron exportados del sistema de nowcasting del profesor a partir de
fuentes del BCRP, INEI y FRED. No se necesita una API ni una clave para ejecutar
la sesión.

## Diseño empírico de la Parte B

El origen es la publicación del IPC del mes `t`. El objetivo es la inflación
interanual del mes `t+1`. Todos los modelos observan la inflación de `t`.

El Random Walk usa esa última observación como pronóstico. Ridge, Lasso y Random
Forest aprenden una corrección al mismo ancla:

```text
pronóstico final = inflación observada en t + corrección estimada
```

El esquema temporal es:

- train inicial: objetivos hasta diciembre de 2015;
- validación: cinco folds anuales expansivos, 2016-2020;
- test: enero de 2021 a junio de 2026;
- reestimación: expansiva, mes a mes, usando solo observaciones pasadas.

La validación selecciona los hiperparámetros. El test evalúa la configuración
congelada. Las métricas principales son RMSE, MAE, sesgo y desempeño relativo al
RW; el contraste predefinido compara Ridge con RW mediante Diebold-Mariano.

## Cómo interpretar los resultados

En la versión actual, Ridge obtiene el menor RMSE descriptivo del test completo:
0.447 frente a 0.497 del RW. La ventaja se concentra en 2021-2023. Durante
2024-2026, el RW vuelve a superar ligeramente a Ridge. El contraste DM entre
Ridge y RW produce un valor p cercano a 0.050, por lo que la evidencia es
limítrofe y no justifica una afirmación universal de superioridad.

El ranking depende de la variable, el horizonte, el periodo y el régimen. La
regularización controla complejidad y redundancia; no identifica causalidad ni
descubre por sí sola las variables económicas verdaderas.

## Limitaciones

- El calendario es **pseudo-tiempo-real**: respeta rezagos mensuales aproximados,
  pero utiliza datos históricos revisados, no vintages disponibles en cada día.
- `metadata.csv` todavía no registra proveedor, código original, fecha de
  extracción ni vintage por serie. Esas columnas son el siguiente paso para una
  trazabilidad completa.
- El test está bloqueado correctamente dentro de la implementación final, pero
  la notebook fue diseñada cuando 2021-2026 ya era historia observada. Es un
  holdout pedagógico, no una evaluación prospectiva completamente virgen.
- Los coeficientes y contribuciones son explicaciones predictivas condicionadas
  a la parametrización. No deben leerse como efectos causales.
- Los diagnósticos por subperiodos ayudan a detectar inestabilidad, pero sus
  muestras son pequeñas y no sustituyen nueva evidencia fuera de muestra.

## Fuentes y licencia

Las series proceden de exportaciones basadas en BCRP, INEI y FRED. Quien
redistribuya los datos debe comprobar las condiciones de uso de cada proveedor
original.

El repositorio aún no contiene un archivo `LICENSE`. Antes de publicar o
redistribuir el código y el material docente debe añadirse una licencia
explícita y documentarse por serie el proveedor y su código original.
