from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ====================================================================
# Caminhos
# ====================================================================
AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")

# Observado: popularidade do conjunto de TREINO (parametrizou o gerador)
RANK_OBSERVADO = AOL / "results" / "treino" / "caracterizacao" / "query_popularity_rank.csv"
LOG_SINTETICO = AOL / "results" / "log_sintetico" / "log_sintetico.csv"

OUTPUT_DIR = AOL / "results" / "plots"
OUTPUT_FILE = OUTPUT_DIR / "comparacao_popularidade_obs_sintetico.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ====================================================================
# Curva observada: rank x frequencia (ja calculada na caracterizacao)
# ====================================================================
print("Carregando popularidade observada...")
obs = pd.read_csv(RANK_OBSERVADO)


# ====================================================================
# Curva sintetica: conta a frequencia de cada Query e ordena por rank
# ====================================================================
print("Carregando log sintetico e calculando rank x frequencia...")
log = pd.read_csv(LOG_SINTETICO, usecols=["Query"])
log = log.dropna(subset=["Query"])
log["Query"] = log["Query"].astype(str).str.strip()
log = log[log["Query"] != ""]

freq_sint = log["Query"].value_counts().reset_index()
freq_sint.columns = ["Query", "Frequency"]
freq_sint["Rank"] = range(1, len(freq_sint) + 1)


# ====================================================================
# Grafico log-log sobreposto
# ====================================================================
print("Gerando grafico...")
plt.figure(figsize=(8, 5.5))
plt.loglog(obs["Rank"], obs["Frequency"], marker=".", linestyle="none", alpha=0.7, label="Observado")
plt.loglog(freq_sint["Rank"], freq_sint["Frequency"], marker=".", linestyle="none", alpha=0.7, label="Sintético")

plt.title("Popularidade das queries: observado × sintético", fontsize=15)
plt.xlabel("Rank", fontsize=16)
plt.ylabel("Frequência", fontsize=16)
plt.tick_params(axis="both", which="both", labelsize=16)
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figura salva em: {OUTPUT_FILE}")
