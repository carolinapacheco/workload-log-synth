import pandas as pd
from pathlib import Path

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
CARACT = AOL / "results" / "treino" / "caracterizacao"

INPUT_FILE = AOL / "data" / "processed" / "aol_query_log_clean_complete_train.csv"

OUTPUT_GENERAL_SUMMARY = CARACT / "general_summary_filtered.txt"
OUTPUT_TOP_QUERIES = CARACT / "queries_top.csv"
OUTPUT_REQUESTS_PER_DAY = CARACT / "requests_per_day_filtered.csv"
OUTPUT_REQUESTS_PER_HOUR = CARACT / "requests_per_hour.csv"
OUTPUT_REQUESTS_PER_USER = CARACT / "requests_per_user.csv"
OUTPUT_ITEMRANK_DIST = CARACT / "itemrank_distribution.csv"


# =========================
# LEITURA
# =========================
df = pd.read_csv(INPUT_FILE)
df["QueryTime"] = pd.to_datetime(df["QueryTime"], errors="coerce")


# =========================
# MÉTRICAS GERAIS
# =========================
total_requests = len(df)
distinct_users = df["AnonID"].nunique()
distinct_queries = df["Query"].nunique()
start_time = df["QueryTime"].min()
end_time = df["QueryTime"].max()

clicked_mask = df["ClickURL"].notna() & (df["ClickURL"].astype(str).str.strip() != "")
total_clicked = clicked_mask.sum()
click_rate = total_clicked / total_requests


# =========================
# TOP QUERIES
# =========================
top_queries = df["Query"].value_counts().reset_index()
top_queries.columns = ["Query", "Frequency"]
top_queries["Percentage"] = (top_queries["Frequency"] / total_requests) * 100
top_queries.to_csv(OUTPUT_TOP_QUERIES, index=False, encoding="utf-8")


# =========================
# REQUISIÇÕES POR USUÁRIO
# =========================
requests_per_user = df.groupby("AnonID").size().reset_index(name="Requests")
requests_per_user = requests_per_user.sort_values("Requests", ascending=False)
requests_per_user.to_csv(OUTPUT_REQUESTS_PER_USER, index=False, encoding="utf-8")


# =========================
# DISTRIBUIÇÃO DE ITEM RANK
# =========================
itemrank_dist = df[df["ItemRank"].notna()].groupby("ItemRank").size().reset_index(name="Frequency")
itemrank_dist = itemrank_dist.sort_values("ItemRank")
itemrank_dist.to_csv(OUTPUT_ITEMRANK_DIST, index=False, encoding="utf-8")


# =========================
# REQUISIÇÕES POR HORA
# =========================
df["hour"] = df["QueryTime"].dt.hour
requests_per_hour = df.groupby("hour").size().reset_index(name="Requests")
requests_per_hour = requests_per_hour.sort_values("hour")
requests_per_hour.to_csv(OUTPUT_REQUESTS_PER_HOUR, index=False, encoding="utf-8")


# =========================
# REQUISIÇÕES POR DIA
# =========================
df["date"] = df["QueryTime"].dt.date
requests_per_day = df.groupby("date").size().reset_index(name="Requests")
requests_per_day = requests_per_day.sort_values("date")
requests_per_day.to_csv(OUTPUT_REQUESTS_PER_DAY, index=False, encoding="utf-8")


# =========================
# RESUMO
# =========================
requests_per_user_stats = df.groupby("AnonID").size().describe()
requests_per_day_stats = requests_per_day["Requests"].describe()

linhas_resumo = [
    "CARACTERIZAÇÃO GERAL DO AOL QUERY LOG",
    "=" * 70,
    f"Arquivo analisado: {INPUT_FILE.resolve()}",
    "",
    f"Total de requisições (dataset completo): {total_requests}",
    f"Usuários distintos: {distinct_users}",
    f"Queries únicas: {distinct_queries}",
    f"Início do período: {start_time}",
    f"Fim do período: {end_time}",
    "",
    f"Requisições com clique registrado: {total_clicked}",
    f"Taxa de clique registrada: {click_rate:.4f} ({click_rate * 100:.2f}%)",
    "ESTATÍSTICAS DE REQUISIÇÕES POR USUÁRIO:",
    requests_per_user_stats.to_string(),
    "",
    "ESTATÍSTICAS DE REQUISIÇÕES POR DIA:",
    requests_per_day_stats.to_string(),
    "",
    f"Arquivo top queries: {OUTPUT_TOP_QUERIES.resolve()}",
    f"Arquivo requests por dia filtrado: {OUTPUT_REQUESTS_PER_DAY.resolve()}",
    f"Arquivo requests por hora: {OUTPUT_REQUESTS_PER_HOUR.resolve()}",
    f"Arquivo requests por usuário: {OUTPUT_REQUESTS_PER_USER.resolve()}",
    f"Arquivo distribuição item rank: {OUTPUT_ITEMRANK_DIST.resolve()}",
]

resumo = "\n".join(linhas_resumo)

with open(OUTPUT_GENERAL_SUMMARY, "w", encoding="utf-8") as arquivo:
    arquivo.write(resumo)

print(resumo)
