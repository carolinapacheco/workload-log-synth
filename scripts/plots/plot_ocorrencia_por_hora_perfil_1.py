import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==== Caminhos ====
AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
INPUT_FILE = AOL / "results" / "treino" / "perfis_kmeans" / "distribuicao_perfis_por_hora.csv"
OUTPUT_DIR = AOL / "results" / "plots"
OUTPUT_FILE = OUTPUT_DIR / "ocorrencia_por_hora_perfil_1.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== Leitura e filtro do perfil 1 ====
print("Lendo", INPUT_FILE)
df = pd.read_csv(INPUT_FILE)
df = df[df["perfil_carga"] == 1]

serie = df.set_index("hour")["quantidade_janelas"].reindex(range(24), fill_value=0)

# ==== Gráfico ====
plt.figure(figsize=(11, 5.5))
plt.bar(serie.index, serie.values, color="#1f77b4")

for hora, valor in zip(serie.index, serie.values):
    if valor > 0:
        plt.text(hora, valor + 0.5, str(int(valor)), ha="center", va="bottom", fontsize=12)

plt.title("Ocorrência do perfil 1 por hora do dia", fontsize=15)
plt.xlabel("Hora do dia", fontsize=16)
plt.ylabel("Número de janelas", fontsize=16)

plt.xticks(range(0, 24))
plt.xlim(-0.5, 23.5)
plt.ylim(0, serie.max() * 1.12)
plt.tick_params(axis="both", labelsize=16)
plt.grid(True, axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print("Figura salva em", OUTPUT_FILE)
