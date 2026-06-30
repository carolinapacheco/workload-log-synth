import pandas as pd
from pathlib import Path

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
CARACT = AOL / "results" / "treino" / "caracterizacao"

INPUT_FILE = AOL / "data" / "processed" / "aol_query_log_clean_complete_train.csv"

OUTPUT_USER_QUERY_STATS = CARACT / "user_query_recurrence_stats.csv"
OUTPUT_REPEAT_DIST = CARACT / "user_query_repeat_distribution.csv"
OUTPUT_SUMMARY = CARACT / "query_recurrence_summary.txt"


# =========================
# LEITURA E LIMPEZA
# =========================
df = pd.read_csv(INPUT_FILE)

df = df.dropna(subset=["AnonID", "Query"])
df["Query"] = df["Query"].astype(str).str.strip()
df = df[df["Query"] != ""]

total_requests = len(df)
total_users = df["AnonID"].nunique()


# =========================
# ESTATÍSTICAS POR USUÁRIO
# =========================
user_stats = df.groupby("AnonID").agg(
    total_requests=("Query", "size"),
    unique_queries=("Query", "nunique"),
).reset_index()

user_stats["repeated_requests"] = user_stats["total_requests"] - user_stats["unique_queries"]
user_stats["repeat_rate"] = user_stats["repeated_requests"] / user_stats["total_requests"]

users_with_repetition = (user_stats["repeated_requests"] > 0).sum()
users_with_repetition_share = users_with_repetition / total_users

total_repeated_requests = user_stats["repeated_requests"].sum()
repeated_requests_share = total_repeated_requests / total_requests

user_stats.to_csv(OUTPUT_USER_QUERY_STATS, index=False, encoding="utf-8")


# =========================
# DISTRIBUIÇÃO DE REPETIÇÃO POR PAR (USUÁRIO, QUERY)
# =========================
user_query_counts = df.groupby(["AnonID", "Query"]).size().reset_index(name="frequency")

repeat_dist = user_query_counts["frequency"].value_counts().sort_index().reset_index()
repeat_dist.columns = ["frequency", "number_of_user_query_pairs"]
repeat_dist["total_requests_represented"] = repeat_dist["frequency"] * repeat_dist["number_of_user_query_pairs"]

repeat_dist.to_csv(OUTPUT_REPEAT_DIST, index=False, encoding="utf-8")

repeated_pairs = (user_query_counts["frequency"] > 1).sum()
repeated_pairs_share = repeated_pairs / len(user_query_counts)


# =========================
# RESUMO
# =========================
linhas_resumo = [
    "RECORRÊNCIA DE QUERIES POR USUÁRIO NO AOL QUERY LOG",
    "=" * 65,
    f"Arquivo analisado: {INPUT_FILE.resolve()}",
    "",
    f"Total de requisições: {total_requests}",
    f"Total de usuários: {total_users}",
    "",
    f"Usuários com pelo menos uma query repetida: {users_with_repetition}",
    f"Proporção de usuários com repetição: {users_with_repetition_share:.6f} ({users_with_repetition_share * 100:.2f}%)",
    "",
    f"Total de requisições repetidas (além da primeira ocorrência da query pelo usuário): {total_repeated_requests}",
    f"Proporção de requisições repetidas: {repeated_requests_share:.6f} ({repeated_requests_share * 100:.2f}%)",
    "",
    f"Pares (usuário, query) com frequência > 1: {repeated_pairs}",
    f"Proporção de pares (usuário, query) repetidos: {repeated_pairs_share:.6f} ({repeated_pairs_share * 100:.2f}%)",
    "",
    "ESTATÍSTICAS DE TOTAL DE REQUISIÇÕES POR USUÁRIO:",
    user_stats["total_requests"].describe().to_string(),
    "",
    "ESTATÍSTICAS DE QUERIES ÚNICAS POR USUÁRIO:",
    user_stats["unique_queries"].describe().to_string(),
    "",
    "ESTATÍSTICAS DE REQUISIÇÕES REPETIDAS POR USUÁRIO:",
    user_stats["repeated_requests"].describe().to_string(),
    "",
    "ESTATÍSTICAS DE TAXA DE REPETIÇÃO POR USUÁRIO:",
    user_stats["repeat_rate"].describe().to_string(),
    "",
    f"Arquivo estatísticas por usuário: {OUTPUT_USER_QUERY_STATS.resolve()}",
    f"Arquivo distribuição de repetição: {OUTPUT_REPEAT_DIST.resolve()}",
]

resumo = "\n".join(linhas_resumo)

with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as arquivo:
    arquivo.write(resumo)

print(resumo)
