import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ==== Caminhos ====
# Re-plota a previsao futura (figura 06) a partir dos CSVs ja salvos pelo script
# 4_sarima/06_previsoes.py, sem re-treinar o modelo.
AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
SARIMA = AOL / "results" / "completo" / "sarima"
DADOS = SARIMA / "dados"
PREVISOES = SARIMA / "previsoes"
OUTPUT_DIR = AOL / "results" / "plots"
OUTPUT_FILE = OUTPUT_DIR / "06_previsoes_futuras.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== Leitura dos dados ====
print("Lendo série de modelagem e previsões futuras")
serie = pd.read_csv(DADOS / "serie_modelagem_sem_dias_incompletos.csv")
serie["window_start"] = pd.to_datetime(serie["window_start"])
serie = serie.sort_values("window_start").set_index("window_start")["requests"]

futuro = pd.read_csv(PREVISOES / "06_previsoes_futuras_sarima.csv")
futuro["window_start"] = pd.to_datetime(futuro["window_start"])

# ==== Gráfico ====
plt.figure(figsize=(14, 6))

plt.plot(serie.iloc[-24 * 14:], linewidth=1.2, label="Histórico (últimos 14 dias)")
plt.plot(futuro["window_start"], futuro["forecast"], linewidth=1.2, label="Previsão futura SARIMA")
plt.fill_between(
    futuro["window_start"],
    futuro["lower_95"],
    futuro["upper_95"],
    alpha=0.2,
    label="Intervalo de 95%",
)

plt.title("Previsão futura de requisições por hora", fontsize=15)
plt.xlabel("Data/hora", fontsize=16)
plt.ylabel("Número de requisições", fontsize=16)
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
