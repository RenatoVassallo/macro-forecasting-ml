"""Utilidades del curso: Pronostico Macroeconomico con ML y DL.

Este modulo contiene la plomeria (carga de datos, estilo grafico, tests) para
que los notebooks se concentren en el METODO. Los bucles importantes (backtest,
ventanas, conformal) estan escritos DENTRO de los notebooks a proposito: son la
leccion, no un detalle de implementacion.

Datos: panel mensual consolidado y archivos auxiliares en ./data, exportados del
sistema de nowcasting del profesor (fuentes BCRP, INEI, FRED). No se necesita
ninguna API ni clave.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figures"

GDP_COLUMNS = [
    "g_pbim_yoy", "cem", "g_elec", "exp_eco3m", "exp_sec3m", "g_circ",
    "g_credmn", "int", "g_tc", "g_tdi", "g_p_cu", "g_p_wti", "vix",
    "g_us_indpro", "ip_cum_yoy",
]
INFLATION_COLUMNS = [
    "ipc_yoy", "ipc_mom", "g_tc", "g_p_wti", "g_ipm", "int", "g_emiprim",
]

# paleta del curso
INK, MUTED = "#2a231d", "#6b5f52"
ACCENT, GOLD, OLIVE, BLUE = "#b0472a", "#9a7b2e", "#5a7d43", "#3a5f7d"
TINT, BORDER = "#f0e0d6", "#d3bba9"
MODEL_COLORS = [ACCENT, BLUE, OLIVE, GOLD, MUTED, "#7d3a5f", "#4d7d7d", "#8c7f6f"]


# --------------------------------------------------------------------------- #
# datos
# --------------------------------------------------------------------------- #
def load_monthly():
    """Panel mensual consolidado de inflación, actividad y drivers macro."""
    return pd.read_csv(DATA / "monthly.csv", index_col=0, parse_dates=True)


def load_gdp():
    """Indicadores mensuales de actividad + PBI trimestral (var. % 12m)."""
    m = load_monthly()[GDP_COLUMNS]
    q = pd.read_csv(DATA / "gdp_quarterly.csv", index_col=0, parse_dates=True)
    return m, q["g_pbiq"]


def load_inflation():
    """Inflacion (IPC var. % 12m y mensual) + determinantes."""
    return load_monthly()[INFLATION_COLUMNS]


def load_metadata():
    """Rezagos de publicacion por serie (dias despues del fin del periodo)."""
    return pd.read_csv(DATA / "metadata.csv").set_index("column")


def release_date(period_end: pd.Timestamp, delay_days: int) -> pd.Timestamp:
    """Fecha en que un dato referido a `period_end` se hace publico."""
    return pd.Timestamp(period_end) + pd.Timedelta(days=int(delay_days))


def as_of(monthly: pd.DataFrame, meta: pd.DataFrame, fecha) -> pd.DataFrame:
    """El conjunto de informacion en `fecha`: enmascara todo dato aun no publicado.

    Esta funcion ES la disciplina de pseudo tiempo real: un dato del mes m con
    rezago de publicacion d solo existe para el pronosticador desde m_fin + d.
    Olvidarla produce 'leakage' y backtests con precision imposible de replicar
    en vivo.
    """
    fecha = pd.Timestamp(fecha)
    out = monthly.copy()
    for c in out.columns:
        d = int(meta.loc[c, "delay_days"]) if c in meta.index else 15
        ends = out.index + pd.offsets.MonthEnd(0)
        out.loc[ends + pd.Timedelta(days=d) > fecha, c] = np.nan
    return out


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
    """Guarda PDF vectorial y PNG de alta resolución para las láminas."""
    FIGS.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight", "dpi": 300, "facecolor": "white"}
    fig.savefig(FIGS / f"{name}.pdf", **options)
    fig.savefig(FIGS / f"{name}.png", **options)


def plot_windows_scheme():
    """Esquema didactico: ventana expansiva vs ventana movil."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 3.6), sharex=True)
    n, n_or = 20, 6
    for ax, kind in zip(axes, ["expansiva", "móvil"]):
        for i in range(n_or):
            t0 = 0 if kind == "expansiva" else i
            t1 = n - n_or + i
            ax.barh(i, t1 - t0, left=t0, height=0.62, color=TINT,
                    edgecolor=BORDER, zorder=2)
            ax.scatter([t1 + 1], [i], color=ACCENT, s=45, zorder=3,
                       label="predicción" if i == 0 else None)
        ax.set_yticks(range(n_or))
        ax.set_yticklabels([f"origen {i + 1}" for i in range(n_or)], fontsize=8)
        ax.set_title(f"Ventana {kind}", loc="left", fontsize=10)
        ax.invert_yaxis()
        ax.grid(False)
    axes[1].set_xlabel("tiempo (la barra es la muestra de entrenamiento)")
    axes[0].legend(loc="lower right", fontsize=8)
    return fig


def plot_ragged_edge(info: pd.DataFrame, meta: pd.DataFrame, fecha,
                     n_meses: int = 14):
    """Mapa de calor de disponibilidad: el borde irregular ('ragged edge')."""
    fecha = pd.Timestamp(fecha)
    sub = info.tail(n_meses)
    order = (meta.reindex(sub.columns)["delay_days"].fillna(15)
             .sort_values().index.tolist())
    sub = sub[order]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    avail = sub.notna().T.astype(float).to_numpy()
    ax.imshow(avail, aspect="auto", cmap=plt.matplotlib.colors
              .ListedColormap(["#efe7da", OLIVE]), vmin=0, vmax=1)
    ax.set_yticks(range(len(sub.columns)))
    labels = [f"{c}  ({int(meta.loc[c, 'delay_days']) if c in meta.index else 15}d)"
              for c in sub.columns]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(len(sub.index)))
    ax.set_xticklabels([d.strftime("%b%y") for d in sub.index], fontsize=8,
                       rotation=45, ha="right")
    ax.set_title(f"Conjunto de informacion al {fecha.date()} "
                 "(verde = publicado, beige = aun no existe)", loc="left")
    ax.grid(False)
    return fig


def plot_fan(ax, x, centre, bands: dict, color=ACCENT, label=None):
    """Fan chart simple: `bands` = {cobertura: (lo, hi)} de menor a mayor."""
    alphas = np.linspace(0.45, 0.14, num=len(bands))
    for (cov_, (lo, hi)), a in zip(sorted(bands.items()), alphas):
        ax.fill_between(x, lo, hi, color=color, alpha=a, linewidth=0,
                        label=f"{int(cov_ * 100)}%")
    ax.plot(x, centre, color=color, lw=1.8, ls="dashed", label=label)
    return ax
