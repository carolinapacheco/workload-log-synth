import pandas as pd
from pathlib import Path

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
CARACT = AOL / "results" / "treino" / "caracterizacao"

INPUT_FILE = AOL / "data" / "processed" / "aol_query_log_clean_complete_train.csv"

OUTPUT_RANK = CARACT / "query_popularity_rank.csv"
OUTPUT_FREQUENCY_DIST = CARACT / "query_frequency_distribution.csv"
OUTPUT_SUMMARY = CARACT / "query_popularity_summary.txt"


# =========================
# LEITURA E LIMPEZA
# =========================
df = pd.read_csv(INPUT_FILE)

df = df.dropna(subset=["Query"])
df["Query"] = df["Query"].astype(str).str.strip()
df = df[df["Query"] != ""]

total_requests = len(df)


# =========================
# FREQUÊNCIA POR QUERY
# =========================
query_freq = df["Query"].value_counts().reset_index()
query_freq.columns = ["Query", "Frequency"]

query_freq["Rank"] = range(1, len(query_freq) + 1)
query_freq["RelativeFrequency"] = query_freq["Frequency"] / total_requests
query_freq["CumulativeFrequency"] = query_freq["Frequency"].cumsum()
query_freq["CumulativeShare"] = query_freq["CumulativeFrequency"] / total_requests

query_freq = query_freq[
    ["Rank", "Query", "Frequency", "RelativeFrequency", "CumulativeFrequency", "CumulativeShare"]
]

query_freq.to_csv(OUTPUT_RANK, index=False, encoding="utf-8")


# =========================
# DISTRIBUIÇÃO DAS FREQUÊNCIAS
# =========================
freq_dist = query_freq["Frequency"].value_counts().sort_index().reset_index()
freq_dist.columns = ["Frequency", "NumberOfQueries"]
freq_dist["TotalRequestsRepresented"] = freq_dist["Frequency"] * freq_dist["NumberOfQueries"]

freq_dist.to_csv(OUTPUT_FREQUENCY_DIST, index=False, encoding="utf-8")


# =========================
# RESUMO
# =========================
total_unique_queries = len(query_freq)

top_1 = query_freq.iloc[0]["Frequency"]
top_10_share = query_freq.head(10)["Frequency"].sum() / total_requests
top_100_share = query_freq.head(100)["Frequency"].sum() / total_requests
top_1000_share = query_freq.head(1000)["Frequency"].sum() / total_requests

singleton_queries = (query_freq["Frequency"] == 1).sum()
singleton_share_queries = singleton_queries / total_unique_queries
rare_queries_2 = (query_freq["Frequency"] <= 2).sum()
rare_queries_5 = (query_freq["Frequency"] <= 5).sum()

linhas_resumo = [
    "POPULARIDADE DAS QUERIES DO AOL QUERY LOG",
    "=" * 60,
    f"Arquivo analisado: {INPUT_FILE.resolve()}",
    "",
    f"Total de requisições: {total_requests}",
    f"Total de queries únicas: {total_unique_queries}",
    "",
    f"Frequência da query mais popular: {top_1}",
    f"Participação acumulada do top 10: {top_10_share:.6f} ({top_10_share * 100:.2f}%)",
    f"Participação acumulada do top 100: {top_100_share:.6f} ({top_100_share * 100:.2f}%)",
    f"Participação acumulada do top 1000: {top_1000_share:.6f} ({top_1000_share * 100:.2f}%)",
    "",
    f"Queries que aparecem 1 vez: {singleton_queries}",
    f"Proporção de queries singleton: {singleton_share_queries:.6f} ({singleton_share_queries * 100:.2f}%)",
    f"Queries com frequência <= 2: {rare_queries_2}",
    f"Queries com frequência <= 5: {rare_queries_5}",
    "",
    "ESTATÍSTICAS DA FREQUÊNCIA DAS QUERIES:",
    query_freq["Frequency"].describe().to_string(),
    "",
    f"Arquivo rank x frequência: {OUTPUT_RANK.resolve()}",
    f"Arquivo distribuição de frequências: {OUTPUT_FREQUENCY_DIST.resolve()}",
]

resumo = "\n".join(linhas_resumo)

with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as arquivo:
    arquivo.write(resumo)

print(resumo)
