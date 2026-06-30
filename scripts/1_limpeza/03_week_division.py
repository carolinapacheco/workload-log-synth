import pandas as pd
from pathlib import Path

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
PROCESSED_DIR = AOL / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "aol_query_log_clean_complete.csv"
OUTPUT_FILE = PROCESSED_DIR / "aol_query_log_clean_complete_train.csv"

# recorte de treino: março e abril de 2006
DATA_INICIO = pd.Timestamp("2006-03-01 00:00:00")
DATA_FIM = pd.Timestamp("2006-04-30 23:00:00")


# =========================
# LEITURA
# =========================
df = pd.read_csv(INPUT_FILE)
df["QueryTime"] = pd.to_datetime(df["QueryTime"])
df = df.sort_values("QueryTime")

print("Período original:")
print(f"Início: {df['QueryTime'].min()}")
print(f"Fim: {df['QueryTime'].max()}")
print(f"Total de registros: {len(df)}")


# =========================
# RECORTE DO PERÍODO DE TREINO
# =========================
df_treino = df[(df["QueryTime"] >= DATA_INICIO) & (df["QueryTime"] < DATA_FIM)]

print("\nPeríodo após filtro:")
print(f"Início: {df_treino['QueryTime'].min()}")
print(f"Fim: {df_treino['QueryTime'].max()}")
print(f"Total de registros: {len(df_treino)}")
print(f"Quantidade de dias no recorte: {df_treino['QueryTime'].dt.date.nunique()}")

df_treino.to_csv(OUTPUT_FILE, index=False)
print(f"\nArquivo salvo em: {OUTPUT_FILE}")
