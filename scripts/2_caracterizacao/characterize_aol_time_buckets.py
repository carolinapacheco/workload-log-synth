import pandas as pd
from pathlib import Path

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
TIME_BUCKETS = AOL / "results" / "completo" / "time_buckets"

INPUT_FILE = AOL / "data" / "processed" / "aol_query_log_clean_complete.csv"

OUTPUT_PER_MINUTE = TIME_BUCKETS / "requests_per_minute.csv"
OUTPUT_PER_5MIN = TIME_BUCKETS / "requests_per_5min.csv"
OUTPUT_PER_HOUR = TIME_BUCKETS / "requests_per_hour_window.csv"
OUTPUT_SUMMARY = TIME_BUCKETS / "time_buckets_summary.txt"


def bucketizar(df, freq):
    temp = df.copy()
    temp["window_start"] = temp["QueryTime"].dt.floor(freq)
    temp["has_click"] = temp["ClickURL"].notna() & (temp["ClickURL"].astype(str).str.strip() != "")

    grouped = temp.groupby("window_start").agg(
        requests=("QueryTime", "size"),
        unique_users=("AnonID", "nunique"),
        unique_queries=("Query", "nunique"),
        clicks=("has_click", "sum"),
    ).reset_index().sort_values("window_start")

    grouped["click_rate"] = grouped["clicks"] / grouped["requests"]
    return grouped


# =========================
# LEITURA
# =========================
df = pd.read_csv(INPUT_FILE)
df["QueryTime"] = pd.to_datetime(df["QueryTime"], errors="coerce")

df = df.dropna(subset=["QueryTime"])


# =========================
# AGREGAÇÃO EM JANELAS
# =========================
per_minute = bucketizar(df, "min")
per_5min = bucketizar(df, "5min")
per_hour = bucketizar(df, "h")

per_minute.to_csv(OUTPUT_PER_MINUTE, index=False, encoding="utf-8")
per_5min.to_csv(OUTPUT_PER_5MIN, index=False, encoding="utf-8")
per_hour.to_csv(OUTPUT_PER_HOUR, index=False, encoding="utf-8")


# =========================
# RESUMO
# =========================
linhas_resumo = [
    "CARACTERIZAÇÃO TEMPORAL EM BUCKETS DO AOL QUERY LOG",
    "=" * 60,
    f"Arquivo analisado: {INPUT_FILE.resolve()}",
    "",
    f"Total de registros usados nos buckets: {len(df)}",
    f"Início do período: {df['QueryTime'].min()}",
    f"Fim do período: {df['QueryTime'].max()}",
    "",
    "ESTATÍSTICAS DE REQUISIÇÕES POR MINUTO:",
    per_minute["requests"].describe().to_string(),
    "",
    "ESTATÍSTICAS DE REQUISIÇÕES POR 5 MINUTOS:",
    per_5min["requests"].describe().to_string(),
    "",
    "ESTATÍSTICAS DE REQUISIÇÕES POR HORA:",
    per_hour["requests"].describe().to_string(),
    "",
    "ESTATÍSTICAS DE USUÁRIOS DISTINTOS POR MINUTO:",
    per_minute["unique_users"].describe().to_string(),
    "",
    "ESTATÍSTICAS DE USUÁRIOS DISTINTOS POR 5 MINUTOS:",
    per_5min["unique_users"].describe().to_string(),
    "",
    "ESTATÍSTICAS DE USUÁRIOS DISTINTOS POR HORA:",
    per_hour["unique_users"].describe().to_string(),
    "",
    "ESTATÍSTICAS DE QUERIES ÚNICAS POR MINUTO:",
    per_minute["unique_queries"].describe().to_string(),
    "",
    "ESTATÍSTICAS DE QUERIES ÚNICAS POR 5 MINUTOS:",
    per_5min["unique_queries"].describe().to_string(),
    "",
    "ESTATÍSTICAS DE QUERIES ÚNICAS POR HORA:",
    per_hour["unique_queries"].describe().to_string(),
    "",
    "ESTATÍSTICAS DE TAXA DE CLIQUE POR MINUTO:",
    per_minute["click_rate"].describe().to_string(),
    "",
    "ESTATÍSTICAS DE TAXA DE CLIQUE POR 5 MINUTOS:",
    per_5min["click_rate"].describe().to_string(),
    "",
    "ESTATÍSTICAS DE TAXA DE CLIQUE POR HORA:",
    per_hour["click_rate"].describe().to_string(),
    "",
    f"Arquivo por minuto: {OUTPUT_PER_MINUTE.resolve()}",
    f"Arquivo por 5 minutos: {OUTPUT_PER_5MIN.resolve()}",
    f"Arquivo por hora: {OUTPUT_PER_HOUR.resolve()}",
    f"Arquivo de resumo: {OUTPUT_SUMMARY.resolve()}",
]

resumo = "\n".join(linhas_resumo)

with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as arquivo:
    arquivo.write(resumo)

print(resumo)
