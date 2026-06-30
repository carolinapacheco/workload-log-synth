import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
CARACT = AOL / "results" / "treino" / "caracterizacao"
PLOTS_DIR = AOL / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
INPUT_FILE = CARACT / "acf_requests_per_hour.csv"
OUTPUT_FILE = PLOTS_DIR / "plot_acf_requests_per_hour.png"

df = pd.read_csv(INPUT_FILE)

plt.figure(figsize=(10, 5.5))
plt.plot(df["lag"], df["acf"], marker="o", markersize=4, linewidth=1.6)

plt.axvline(24, linestyle="--", linewidth=1.5, label="Lag 24")
plt.axvline(168, linestyle=":", linewidth=1.5, label="Lag 168")

plt.title("Autocorrelação das requisições por hora", fontsize=14)
plt.xlabel("Lag", fontsize=16)
plt.ylabel("ACF", fontsize=16)
plt.tick_params(axis="both", labelsize=16)
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()