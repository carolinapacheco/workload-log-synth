import pandas as pd
from pathlib import Path

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
CARACT = AOL / "results" / "treino" / "caracterizacao"

INPUT_FILE = AOL / "data" / "processed" / "aol_query_log_clean_complete_train.csv"

OUTPUT_GLOBAL_INTERARRIVAL = CARACT / "interarrival_global.csv"
OUTPUT_GLOBAL_SUMMARY = CARACT / "interarrival_global_summary.txt"
OUTPUT_USER_INTERARRIVAL = CARACT / "interarrival_user.csv"
OUTPUT_USER_SUMMARY = CARACT / "interarrival_user_summary.txt"


def resumir(serie):
    return pd.Series({
        "count": serie.count(),
        "mean_seconds": serie.mean(),
        "std_seconds": serie.std(),
        "min_seconds": serie.min(),
        "p25_seconds": serie.quantile(0.25),
        "median_seconds": serie.median(),
        "p75_seconds": serie.quantile(0.75),
        "p90_seconds": serie.quantile(0.90),
        "p95_seconds": serie.quantile(0.95),
        "p99_seconds": serie.quantile(0.99),
        "max_seconds": serie.max(),
    })


# =========================
# LEITURA
# =========================
df = pd.read_csv(INPUT_FILE)
df["QueryTime"] = pd.to_datetime(df["QueryTime"], errors="coerce")


# =========================
# INTERARRIVAL GLOBAL
# =========================
global_df = df[["QueryTime"]].dropna().sort_values("QueryTime")
global_df["prev_time"] = global_df["QueryTime"].shift(1)
global_df["interarrival_seconds"] = (global_df["QueryTime"] - global_df["prev_time"]).dt.total_seconds()

global_df.to_csv(OUTPUT_GLOBAL_INTERARRIVAL, index=False, encoding="utf-8")

global_interarrival = global_df["interarrival_seconds"].dropna()
global_summary = resumir(global_interarrival)

global_summary_lines = [
    "RESUMO DE INTERARRIVAL GLOBAL",
    "=" * 40,
    f"Arquivo analisado: {INPUT_FILE.resolve()}",
    "",
    "Estatísticas (em segundos):",
    global_summary.to_string(),
    "",
    f"Arquivo detalhado: {OUTPUT_GLOBAL_INTERARRIVAL.resolve()}",
]

with open(OUTPUT_GLOBAL_SUMMARY, "w", encoding="utf-8") as arquivo:
    arquivo.write("\n".join(global_summary_lines))

print("\n".join(global_summary_lines))


# =========================
# INTERARRIVAL POR USUÁRIO
# =========================
user_df = df[["AnonID", "QueryTime"]].dropna().sort_values(["AnonID", "QueryTime"])
user_df["prev_time_same_user"] = user_df.groupby("AnonID")["QueryTime"].shift(1)
user_df["interarrival_seconds"] = (user_df["QueryTime"] - user_df["prev_time_same_user"]).dt.total_seconds()

user_interarrival_only = user_df.dropna(subset=["interarrival_seconds"])
user_interarrival_only.to_csv(OUTPUT_USER_INTERARRIVAL, index=False, encoding="utf-8")

user_summary = resumir(user_interarrival_only["interarrival_seconds"])

user_gap_summary = user_interarrival_only.groupby("AnonID")["interarrival_seconds"].mean().describe()

user_summary_lines = [
    "RESUMO DE INTERARRIVAL POR USUÁRIO",
    "=" * 40,
    f"Arquivo analisado: {INPUT_FILE.resolve()}",
    "",
    "Estatísticas gerais dos gaps entre requisições do mesmo usuário (em segundos):",
    user_summary.to_string(),
    "",
    "Estatísticas das médias de interarrival por usuário:",
    user_gap_summary.to_string(),
    "",
    f"Arquivo detalhado: {OUTPUT_USER_INTERARRIVAL.resolve()}",
]

with open(OUTPUT_USER_SUMMARY, "w", encoding="utf-8") as arquivo:
    arquivo.write("\n".join(user_summary_lines))

print()
print("\n".join(user_summary_lines))
