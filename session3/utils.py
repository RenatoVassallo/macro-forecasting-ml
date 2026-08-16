"""Utilidades de la Sesion 3: metodos de ensamble con FRED-MD.

Modulo autocontenido: carga de datos, estilo grafico y metricas de evaluacion.
Los loops importantes (backtest, tuning) viven DENTRO de los notebooks a
proposito: son la leccion, no un detalle de implementacion.

Datos: session3/data/fredmd.csv es una copia congelada del current.csv de
McCracken y Ng (ver VINTAGE.txt). No se necesita internet ni API key.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figures"

# paleta del curso
INK, MUTED = "#2a231d", "#6b5f52"
ACCENT, GOLD, OLIVE, BLUE = "#b0472a", "#9a7b2e", "#5a7d43", "#3a5f7d"
TINT, BORDER = "#f0e0d6", "#d3bba9"
MODEL_COLORS = [ACCENT, BLUE, OLIVE, GOLD, MUTED, "#7d3a5f", "#4d7d7d", "#8c7f6f"]


# --------------------------------------------------------------------------- #
# datos: FRED-MD (vintage congelado)
# --------------------------------------------------------------------------- #
def apply_tcode(x: pd.Series, tcode: int) -> pd.Series:
    """Aplica el codigo de transformacion 1-7 de McCracken y Ng a una serie.

    1: nivel                      5: dlog(x)
    2: dx                         6: d2log(x)
    3: d2x                        7: d(x_t / x_{t-1} - 1)
    4: log(x)
    """
    if tcode == 1:
        return x
    if tcode == 2:
        return x.diff()
    if tcode == 3:
        return x.diff().diff()
    logx = np.log(x.where(x > 0))
    if tcode == 4:
        return logx
    if tcode == 5:
        return logx.diff()
    if tcode == 6:
        return logx.diff().diff()
    if tcode == 7:
        return (x / x.shift(1) - 1.0).diff()
    raise ValueError(f"tcode desconocido: {tcode}")


def _load_fred(nombre: str, freq: str):
    """Lee el vintage congelado de FRED-MD y separa sus filas especiales.

    Tras el header viene una fila `transform` con los tcodes 1-7. No es un
    dato: se separa detectando las filas cuyo primer campo no es una fecha.

    Devuelve (raw, tcodes, panel):
      raw    DataFrame (PeriodIndex) con las series originales;
      tcodes Series con el codigo de transformacion por columna;
      panel  DataFrame con cada serie ya transformada segun su tcode.
    """
    df = pd.read_csv(DATA / nombre)
    first = df.iloc[:, 0].astype(str).str.strip()
    fechas = pd.to_datetime(first, format="%m/%d/%Y", errors="coerce")

    specials = df[fechas.isna()].copy()
    specials.index = first[fechas.isna()].str.lower()
    trans_rows = [ix for ix in specials.index if ix.startswith("transform")]
    if not trans_rows:
        raise ValueError(f"{nombre} no contiene la fila 'transform' esperada.")
    tcodes = specials.loc[trans_rows[0]].iloc[1:].astype(float).astype(int)

    data = df[fechas.notna()].copy()
    raw = data.iloc[:, 1:].astype(float)
    raw.index = pd.PeriodIndex(fechas[fechas.notna()], freq=freq)
    raw.index.name = "period"
    tcodes = tcodes.reindex(raw.columns)

    panel = pd.DataFrame(
        {c: apply_tcode(raw[c], int(tcodes[c])) for c in raw.columns},
        index=raw.index,
    )
    return raw, tcodes, panel


def load_fredmd():
    """FRED-MD mensual (vintage congelado en data/fredmd.csv)."""
    return _load_fred("fredmd.csv", "M")


def target_cpi_inflation(raw: pd.DataFrame) -> pd.Series:
    """Inflacion mensual anualizada de EE.UU.: 1200 * dlog(CPIAUCSL)."""
    y = 1200 * np.log(raw["CPIAUCSL"]).diff()
    y.name = "cpi_inflation"
    return y



# --------------------------------------------------------------------------- #
# evaluacion
# --------------------------------------------------------------------------- #
def rmse(e):
    e = np.asarray(e, dtype=float)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2)))


def dm_test(e1, e2, h: int = 1):
    """Test de Diebold y Mariano (perdida cuadratica) con correccion de Harvey.

    H0: ambos modelos tienen la misma precision. Devuelve (estadistico, p-valor).
    Estadistico negativo: el modelo 1 es mas preciso que el 2.
    """
    from scipy import stats

    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    ok = np.isfinite(e1) & np.isfinite(e2)
    d = e1[ok] ** 2 - e2[ok] ** 2
    n = d.size
    if n < 10:
        return np.nan, np.nan
    dbar = d.mean()
    # varianza de largo plazo con h-1 autocovarianzas (errores h-pasos se solapan)
    gamma = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    v = (gamma[0] + 2 * sum(gamma[1:])) / n
    if v <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(v)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)      # Harvey et al.
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


def coverage(y_true, lo, hi):
    """Cobertura empirica: fraccion de realizaciones dentro de la banda."""
    y, lo, hi = (np.asarray(v, dtype=float) for v in (y_true, lo, hi))
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])))


# --------------------------------------------------------------------------- #
# estilo y graficos
# --------------------------------------------------------------------------- #
def set_style():
    plt.rcParams.update({
        "figure.figsize": (10, 4.5), "figure.dpi": 110,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": BORDER, "axes.labelcolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.grid": True, "grid.color": BORDER, "grid.alpha": 0.4,
        "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "font.size": 10.5,
        "legend.frameon": False, "figure.autolayout": True,
    })


def save_fig(fig, name: str):
    """Guarda PDF vectorial y PNG de alta resolucion en session3/figures."""
    FIGS.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight", "dpi": 300, "facecolor": "white"}
    fig.savefig(FIGS / f"{name}.pdf", **options)
    fig.savefig(FIGS / f"{name}.png", **options)


def plot_fan(ax, x, centre, bands: dict, color=ACCENT, label=None):
    """Fan chart simple: `bands` = {cobertura: (lo, hi)} de menor a mayor."""
    alphas = np.linspace(0.45, 0.14, num=len(bands))
    for (cov_, (lo, hi)), a in zip(sorted(bands.items()), alphas):
        ax.fill_between(x, lo, hi, color=color, alpha=a, linewidth=0,
                        label=f"{int(cov_ * 100)}%")
    ax.plot(x, centre, color=color, lw=1.8, ls="dashed", label=label)
    return ax
