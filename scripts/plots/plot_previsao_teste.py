import json

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ==== Caminhos ====
# Re-plota a previsao no conjunto de teste (figura 05) a partir dos CSVs ja salvos
# pelo script 4_sarima/05_treinar_melhor_modelo.py, sem re-treinar o modelo.
AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
SARIMA = AOL / "results" / "completo" / "sarima"
DADOS = SARIMA / "dados"
PREVISOES = SARIMA / "previsoes"
MODELOS = SARIMA / "modelos"
OUTPUT_DIR = AOL / "results" / "plots"
OUTPUT_FILE = OUTPUT_DIR / "05_previsao_teste_melhor_modelo.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ler_serie(arquivo):
    df = pd.read_csv(DADOS / arquivo)
    df["window_start"] = pd.to_datetime(df["window_start"])
    return df.sort_values("window_start").set_index("window_start")["requests"]


# ==== Leitura dos dados ====
print("Lendo dados de treino, validação e previsão do teste")
treino = ler_serie("treino.csv")
validacao = ler_serie("validacao.csv")
treino_validacao = pd.concat([treino, validacao]).sort_index()

previsao = pd.read_csv(PREVISOES / "05_previsao_teste_sarima.csv")
previsao["window_start"] = pd.to_datetime(previsao["window_start"])

with open(MODELOS / "melhor_modelo_validacao.json", encoding="utf-8") as f:
    nome_modelo = json.load(f)["model"]

# ==== Gráfico ====
plt.figure(figsize=(14, 6))

plt.plot(treino_validacao.iloc[-24 * 14:], linewidth=1.2, label="Treino + validação (últimas duas semanas)")
plt.plot(previsao["window_start"], previsao["real"], linewidth=1.2, label="Teste final")
plt.plot(previsao["window_start"], previsao["forecast_sarima"], linewidth=1.2, label="Previsão SARIMA")

plt.title(f"Previsão no conjunto de teste - {nome_modelo}", fontsize=15)
plt.xlabel("Data/hora", fontsize=16)
plt.ylabel("Número de requisições", fontsize=16)
plt.legend(fontsize=14)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
plt.xticks(rotation=45)
plt.tick_params(axis="both", labelsize=16)

plt.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print("Figura salva em", OUTPUT_FILE)
