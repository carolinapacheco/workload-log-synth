import re
import pandas as pd
from pathlib import Path

INPUT_FILE = "user-ct-test-collection-02.txt"
OUTPUT_FILE = "aol_query_log.csv"
ERROR_FILE = "aol_query_log_erros.txt"

pattern = re.compile(
    r"^(\d+)\s+(.*?)\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})(?:\s+(\d+)\s+(\S+))?$"
)


print("Convertendo o AOL Query Log de txt para csv...")


# =========================
# LEITURA E PARSE DAS LINHAS
# =========================
registros = []
erros = []

with open(INPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
    for line_number, raw_line in enumerate(f, start=1):
        line = raw_line.strip()

        if not line:
            continue

        if line.lower().startswith("anonid"):
            continue

        match = pattern.match(line)
        if match:
            anon_id, query, query_time, item_rank, click_url = match.groups()

            registros.append({
                "AnonID": int(anon_id),
                "Query": query,
                "QueryTime": query_time,
                "ItemRank": int(item_rank) if item_rank else pd.NA,
                "ClickURL": click_url if click_url else pd.NA
            })
        else:
            erros.append(f"Linha {line_number}: {line}")


# =========================
# MONTAGEM DO DATAFRAME
# =========================
df = pd.DataFrame(registros)
df["QueryTime"] = pd.to_datetime(df["QueryTime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
df = df.sort_values(by=["AnonID", "QueryTime"]).reset_index(drop=True)


# =========================
# SALVANDO CSV E ERROS
# =========================
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

with open(ERROR_FILE, "w", encoding="utf-8") as ferr:
    for erro in erros:
        ferr.write(erro + "\n")

print("Conversão concluída.")
print(f"Linhas convertidas: {len(df)}")
print(f"Linhas com erro: {len(erros)}")
print(f"CSV salvo em: {Path(OUTPUT_FILE).resolve()}")
print(f"Arquivo de erros: {Path(ERROR_FILE).resolve()}")


# =========================
# CHECAGEM INICIAL DO DATASET
# =========================
print("\n=== Checagem inicial do dataset ===")
print("\nTipos das colunas:")
print(df.dtypes)

print("\nValores nulos por coluna:")
print(df.isna().sum())

print("\nQuantidade de usuários distintos:")
print(df["AnonID"].nunique())

print("\nPeríodo coberto:")
print(f"Início: {df['QueryTime'].min()}")
print(f"Fim:    {df['QueryTime'].max()}")

print("\nTop 10 queries mais frequentes:")
print(df["Query"].value_counts().head(10))
