from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

# ====================================================================
# Caminhos
# ====================================================================
AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")

DADOS_DIR = AOL / "results" / "completo" / "sarima" / "dados"
OUTPUT_DIR = AOL / "results" / "plots"
OUTPUT_FILE = OUTPUT_DIR / "05_diagnostico_residuos_sarima.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# modelo final: SARIMA(1,1,0)(1,1,0)[24]
ORDER = (1, 1, 0)
SEASONAL_ORDER = (1, 1, 0, 24)


# ====================================================================
# Carrega a serie (mesma logica do 05_treinar_melhor_modelo.py)
# ====================================================================
def load_series(filename):
    df = pd.read_csv(DADOS_DIR / filename)
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = df.sort_values("window_start").set_index("window_start")
    return df["requests"].asfreq("h")


print("Carregando treino + validacao...")
treino = load_series("treino.csv")
validacao = load_series("validacao.csv")
serie = pd.concat([treino, validacao]).sort_index().asfreq("h")

if serie.isna().any():
    raise ValueError("A serie treino + validacao possui lacunas inesperadas.")


# ====================================================================
# Ajusta o mesmo modelo final
# ====================================================================
print("Ajustando SARIMA(1,1,0)(1,1,0)[24]...")
modelo = SARIMAX(
    serie,
    order=ORDER,
    seasonal_order=SEASONAL_ORDER,
    enforce_stationarity=False,
    enforce_invertibility=False,
    simple_differencing=False,
)
resultados = modelo.fit(disp=False, maxiter=80)


# ====================================================================
# Diagnostico padrao com quatro paineis, titulos em portugues
# ====================================================================
print("Gerando diagnostico de residuos...")
fig = resultados.plot_diagnostics(lags=48, figsize=(12, 8))

titulos = [
    "Resíduos padronizados",
    "Histograma e densidade estimada",
    "Q-Q normal",
    "Correlograma (ACF dos resíduos)",
]
for eixo, titulo in zip(fig.axes, titulos):
    eixo.set_title(titulo)

# eixo 0 = residuos no tempo | eixo 1 = histograma | eixo 2 = Q-Q | eixo 3 = correlograma
fig.axes[0].set_xlabel("Data/hora")
fig.axes[2].set_xlabel("Quantis teóricos")
fig.axes[2].set_ylabel("Quantis amostrais")
fig.axes[3].set_xlabel("Defasagem")

fig.axes[1].legend(["Histograma", "Densidade estimada", "N(0,1)"])

fig.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figura salva em: {OUTPUT_FILE}")


# ====================================================================
# Confirmacao do Ljung-Box nos lags 24 e 48
# ====================================================================
lb = acorr_ljungbox(resultados.resid, lags=[24, 48], return_df=True)
print("\nLjung-Box nos lags sazonais (resíduos):")
print(lb)
print(
    "\nConfira com results/completo/sarima/modelos/05_resumo_modelo_final.txt: "
    "Prob(Q) (L1) = 0,22 e estrutura remanescente nos lags 24 e 48."
)
