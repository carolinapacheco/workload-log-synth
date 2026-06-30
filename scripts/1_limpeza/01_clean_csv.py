import pandas as pd
from pathlib import Path


AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")

INPUT_FILE = AOL / "data" / "raw" / "aol_query_log.csv"
OUTPUT_CLEAN_FILE = AOL / "data" / "processed" / "aol_query_log_clean.csv"
OUTPUT_INVALID_FILE = AOL / "data" / "processed" / "aol_query_log_invalid.csv"
OUTPUT_DUPLICATES_FILE = AOL / "data" / "processed" / "aol_query_log_duplicates.csv"
OUTPUT_SUMMARY_FILE = AOL / "results" / "limpeza" / "aol_query_log_cleaning_summary.txt"

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
COLUNAS_OBRIGATORIAS = ["AnonID", "Query", "QueryTime", "ItemRank", "ClickURL"]


# =========================
# LEITURA
# =========================
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_FILE}")

# Le tudo como texto para ter controle total da limpeza
df = pd.read_csv(INPUT_FILE, dtype=str)

df.columns = [coluna.strip() for coluna in df.columns]
faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
if faltando:
    raise ValueError(f"Colunas obrigatórias ausentes no CSV: {faltando}")

total_original = len(df)

for coluna in df.columns:
    df[coluna] = df[coluna].astype(str).str.strip()

textos_nulos = ["", "nan", "NaN", "None", "NULL", "null", "<NA>"]
df = df.replace(textos_nulos, pd.NA)


# =========================
# VALIDAÇÃO DAS LINHAS
# =========================
anon_id = pd.to_numeric(df["AnonID"], errors="coerce")
query_time = pd.to_datetime(df["QueryTime"], format=DATETIME_FORMAT, errors="coerce")
query_vazia = df["Query"].isna() | (df["Query"].str.strip() == "")

linha_invalida = anon_id.isna() | query_time.isna() | query_vazia

invalidas = df[linha_invalida].copy()
validas = df[~linha_invalida].copy()


# =========================
# PADRONIZAÇÃO DOS TIPOS (apenas nas linhas válidas)
# =========================
validas["AnonID"] = pd.to_numeric(validas["AnonID"], errors="coerce").astype("Int64")
validas["QueryTime"] = pd.to_datetime(validas["QueryTime"], format=DATETIME_FORMAT, errors="coerce")
validas["ItemRank"] = pd.to_numeric(validas["ItemRank"], errors="coerce").astype("Int64")
validas["ClickURL"] = validas["ClickURL"].astype("string")
validas["Query"] = validas["Query"].astype("string").str.strip()


# =========================
# REMOÇÃO DE DUPLICATAS
# =========================
eh_duplicata_exata = validas.duplicated(
    subset=["AnonID", "Query", "QueryTime", "ItemRank", "ClickURL"],
    keep="first",
)
duplicatas_exatas = validas[eh_duplicata_exata].copy()
sem_duplicatas_exatas = validas[~eh_duplicata_exata].copy()

eh_mesmo_usuario_horario = sem_duplicatas_exatas.duplicated(
    subset=["AnonID", "QueryTime"],
    keep="first",
)
duplicatas_usuario_horario = sem_duplicatas_exatas[eh_mesmo_usuario_horario].copy()
limpas = sem_duplicatas_exatas[~eh_mesmo_usuario_horario].copy()

duplicatas = pd.concat([duplicatas_exatas, duplicatas_usuario_horario], ignore_index=True)


# =========================
# ORDENAÇÃO
# =========================
limpas = limpas.sort_values(by=["AnonID", "QueryTime"]).reset_index(drop=True)


# =========================
# SALVANDO OS ARQUIVOS
# =========================
limpas.to_csv(OUTPUT_CLEAN_FILE, index=False, encoding="utf-8")
invalidas.to_csv(OUTPUT_INVALID_FILE, index=False, encoding="utf-8")
duplicatas.to_csv(OUTPUT_DUPLICATES_FILE, index=False, encoding="utf-8")


# =========================
# RESUMO
# =========================
linhas_resumo = [
    "RESUMO DA LIMPEZA DO AOL CSV",
    "=" * 40,
    f"Arquivo de entrada: {INPUT_FILE.resolve()}",
    f"Total de linhas originais: {total_original}",
    f"Linhas inválidas removidas: {len(invalidas)}",
    f"Duplicatas exatas removidas: {len(duplicatas_exatas)}",
    f"Linhas removidas por mesmo AnonID + QueryTime: {len(duplicatas_usuario_horario)}",
    f"Total de duplicatas removidas: {len(duplicatas)}",
    f"Total final de linhas limpas: {len(limpas)}",
    "",
    "VALORES NULOS NO DATASET LIMPO:",
    limpas.isna().sum().to_string(),
    "",
    "TIPOS DAS COLUNAS NO DATASET LIMPO:",
    limpas.dtypes.to_string(),
    "",
    f"Usuários distintos no dataset limpo: {limpas['AnonID'].nunique()}",
    f"Início do período: {limpas['QueryTime'].min()}",
    f"Fim do período: {limpas['QueryTime'].max()}",
    "",
    f"Arquivo limpo: {OUTPUT_CLEAN_FILE.resolve()}",
    f"Arquivo de inválidas: {OUTPUT_INVALID_FILE.resolve()}",
    f"Arquivo de duplicatas: {OUTPUT_DUPLICATES_FILE.resolve()}",
]

resumo = "\n".join(linhas_resumo)

with open(OUTPUT_SUMMARY_FILE, "w", encoding="utf-8") as arquivo:
    arquivo.write(resumo)

print(resumo)
