import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pathlib import Path

plt.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
CARACT = AOL / "results" / "treino" / "caracterizacao"
PLOTS_DIR = AOL / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
INPUT_FILE = CARACT / "requests_per_hour_window.csv"
OUTPUT_FILE = PLOTS_DIR / "decomposicao_semanal.png"
OUTPUT_FILE2 = PLOTS_DIR / "decomposicao.png"


# =========================
# LEITURA E PREPARO DA SÉRIE HORÁRIA
# =========================
df_hourly = pd.read_csv(INPUT_FILE)
df_hourly["window_start"] = pd.to_datetime(df_hourly["window_start"])
df_hourly = df_hourly.set_index("window_start").sort_index()
df_hourly = df_hourly.asfreq("h", fill_value=0)

serie = df_hourly["requests"]


# =========================
# DECOMPOSIÇÃO DIÁRIA (período de 24 horas)
# =========================
decomposicao_diaria = seasonal_decompose(serie, model="additive", period=24)

fig = decomposicao_diaria.plot()
fig.set_size_inches(14, 8)
plt.suptitle("Decomposição da série temporal - sazonalidade diária", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_FILE2, dpi=300, bbox_inches="tight")
plt.show()


# =========================
# DECOMPOSIÇÃO SEMANAL (período de 168 horas)
# =========================
decomposicao_semanal = seasonal_decompose(serie, model="additive", period=168)

fig = decomposicao_semanal.plot()
fig.set_size_inches(14, 8)
plt.suptitle("Decomposição da série temporal - sazonalidade semanal", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()
