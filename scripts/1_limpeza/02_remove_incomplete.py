import pandas as pd
from pathlib import Path

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")

INPUT_FILE = AOL / "data" / "processed" / "aol_query_log_clean.csv"
OUTPUT_FILE = AOL / "data" / "processed" / "aol_query_log_clean_complete_days.csv"
OUTPUT_REMOVED_FILE = AOL / "data" / "processed" / "aol_query_log_incomplete_days_removed.csv"
OUTPUT_SUMMARY_FILE = AOL / "results" / "limpeza" / "remove_incomplete_days_summary.txt"

MIN_HORAS_DIA_COMPLETO = 18


# =========================
# LEITURA
# =========================
df = pd.read_csv(INPUT_FILE)
df["QueryTime"] = pd.to_datetime(df["QueryTime"], errors="coerce")

sem_data = df[df["QueryTime"].isna()]
df = df.dropna(subset=["QueryTime"])


# =========================
# COBERTURA DE HORAS POR DIA
# =========================
df["date"] = df["QueryTime"].dt.date

cobertura_diaria = df.groupby("date").agg(
    requests=("QueryTime", "size"),
    first_time=("QueryTime", "min"),
    last_time=("QueryTime", "max"),
).reset_index()

cobertura_diaria["coverage_hours"] = (cobertura_diaria["last_time"] - cobertura_diaria["first_time"]).dt.total_seconds() / 3600
cobertura_diaria["complete_day"] = cobertura_diaria["coverage_hours"] >= MIN_HORAS_DIA_COMPLETO

dias_completos = cobertura_diaria.loc[cobertura_diaria["complete_day"], "date"].tolist()
dias_incompletos = cobertura_diaria.loc[~cobertura_diaria["complete_day"], "date"].tolist()


# =========================
# SEPARAÇÃO DOS REGISTROS
# =========================
df_completos = df[df["date"].isin(dias_completos)]
df_removidos = df[df["date"].isin(dias_incompletos)]

df_completos = df_completos.drop(columns=["date"])
df_removidos = df_removidos.drop(columns=["date"])

df_completos.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
df_removidos.to_csv(OUTPUT_REMOVED_FILE, index=False, encoding="utf-8")


# =========================
# RESUMO
# =========================
linhas_resumo = [
    "REMOÇÃO DE DIAS INCOMPLETOS DO AOL QUERY LOG",
    "=" * 50,
    f"Arquivo de entrada: {INPUT_FILE.resolve()}",
    f"Critério de dia completo: cobertura >= {MIN_HORAS_DIA_COMPLETO} horas",
    "",
    f"Total de registros lidos: {len(df) + len(sem_data)}",
    f"Linhas com timestamp inválido removidas: {len(sem_data)}",
    f"Total de registros válidos para análise: {len(df)}",
    "",
    f"Quantidade de dias completos: {len(dias_completos)}",
    f"Quantidade de dias incompletos: {len(dias_incompletos)}",
    f"Dias incompletos identificados: {dias_incompletos}",
    "",
    f"Registros mantidos no arquivo final: {len(df_completos)}",
    f"Registros removidos por pertencerem a dias incompletos: {len(df_removidos)}",
    "",
    f"Arquivo final sem dias incompletos: {OUTPUT_FILE.resolve()}",
    f"Arquivo contendo registros removidos: {OUTPUT_REMOVED_FILE.resolve()}",
]

resumo = "\n".join(linhas_resumo)

with open(OUTPUT_SUMMARY_FILE, "w", encoding="utf-8") as arquivo:
    arquivo.write(resumo)

print(resumo)
