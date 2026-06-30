import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==== Caminhos ====
AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
INPUT_FILE = AOL / "results" / "treino" / "caracterizacao" / "requests_per_hour.csv"
OUTPUT_DIR = AOL / "results" / "plots"
OUTPUT_FILE = OUTPUT_DIR / "requisicoes_por_hora_do_dia.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== Leitura dos dados ====
print("Lendo", INPUT_FILE)
df = pd.read_csv(INPUT_FILE)
df = df.sort_values("hour")

# ==== Gráfico ====
plt.figure(figsize=(10, 5))
plt.plot(df["hour"], df["Requests"], marker="o", linewidth=1.8, color="#1f77b4")

plt.title("Requisições por hora do dia (soma do período de treino)", fontsize=15)
plt.xlabel("Hora do dia", fontsize=16)
plt.ylabel("Número de requisições", fontsize=16)

plt.xticks(range(0, 24))
plt.xlim(-0.5, 23.5)
plt.tick_params(axis="both", labelsize=16)
plt.grid(True, axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print("Figura salva em", OUTPUT_FILE)
