import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
CARACT = AOL / "results" / "treino" / "caracterizacao"
INPUT_FILE = CARACT / "query_popularity_rank.csv"
OUTPUT_DIR = AOL / "results" / "plots"
OUTPUT_FILE = OUTPUT_DIR / "plot_query_popularity_loglog.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

plt.figure(figsize=(8, 5.5))
plt.loglog(
    df["Rank"],
    df["Frequency"],
    marker=".",
    linestyle="none",
    alpha=0.7
)

plt.title("Popularidade das queries (rank × frequência)", fontsize=15)
plt.xlabel("Rank", fontsize=16)
plt.ylabel("Frequência", fontsize=16)
plt.tick_params(axis="both", which="both", labelsize=16)
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()