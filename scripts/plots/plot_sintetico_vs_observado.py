import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ==== Caminhos ====
AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
LOG_SINTETICO = AOL / "results" / "log_sintetico" / "log_sintetico.csv"
PREVISAO_TESTE = AOL / "results" / "completo" / "sarima" / "previsoes" / "05_previsao_teste_sarima.csv"
OUTPUT_DIR = AOL / "results" / "plots"
OUTPUT_FILE = OUTPUT_DIR / "sintetico_vs_observado.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== Série observada (coluna real do teste) ====
print("Lendo", PREVISAO_TESTE)
df_obs = pd.read_csv(PREVISAO_TESTE)
df_obs["window_start"] = pd.to_datetime(df_obs["window_start"])
serie_obs = df_obs.set_index("window_start")["real"]

# ==== Série sintética (contagem de registros por hora) ====
print("Lendo", LOG_SINTETICO)
df_sint = pd.read_csv(LOG_SINTETICO, usecols=["QueryTime"])
df_sint["QueryTime"] = pd.to_datetime(df_sint["QueryTime"])
serie_sint = df_sint.set_index("QueryTime").resample("h").size()

# ==== Gráfico ====
plt.figure(figsize=(14, 5))
plt.plot(serie_obs.index, serie_obs.values, linewidth=1.2, label="Observado")
plt.plot(serie_sint.index, serie_sint.values, linewidth=1.2, label="Sintético")

plt.title("Série sintética x série observada (teste, 18-31/05/2006)", fontsize=15)
plt.xlabel("Tempo", fontsize=16)
plt.ylabel("Número de requisições por hora", fontsize=16)
plt.legend(fontsize=14)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
plt.xticks(rotation=45)
plt.tick_params(axis="both", labelsize=16)

plt.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print("Figura salva em", OUTPUT_FILE)
