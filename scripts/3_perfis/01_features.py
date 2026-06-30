import pandas as pd
from pathlib import Path

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
OUTPUT_DIR = AOL / "results" / "completo" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = AOL / "data" / "processed" / "aol_query_log_clean_complete.csv"
OUTPUT_FILE = OUTPUT_DIR / "features_janelas_horarias.csv"


# =========================
# LEITURA
# =========================
df = pd.read_csv(INPUT_FILE)
df["QueryTime"] = pd.to_datetime(df["QueryTime"])
df = df.sort_values("QueryTime")

print("Período dos dados:")
print(f"Início: {df['QueryTime'].min()}")
print(f"Fim: {df['QueryTime'].max()}")
print(f"Total de registros: {len(df)}")


# =========================
# JANELA TEMPORAL E ATRIBUTOS POR REQUISIÇÃO
# =========================
df["window_start"] = df["QueryTime"].dt.floor("h")
df["has_click"] = df["ClickURL"].notna() & (df["ClickURL"].astype(str).str.strip() != "")

df["interarrival_seconds"] = df["QueryTime"].diff().dt.total_seconds()
df.loc[df["interarrival_seconds"] < 0, "interarrival_seconds"] = None


# =========================
# AGREGAÇÃO POR JANELA HORÁRIA
# =========================
print("\nAgregando dados por janela horária...")

features = df.groupby("window_start").agg(
    requests=("QueryTime", "count"),
    unique_users=("AnonID", "nunique"),
    unique_queries=("Query", "nunique"),
    clicks=("has_click", "sum"),
    mean_interarrival=("interarrival_seconds", "mean"),
    median_interarrival=("interarrival_seconds", "median"),
    std_interarrival=("interarrival_seconds", "std"),
).reset_index()

features["click_rate"] = features["clicks"] / features["requests"]

features["hour"] = features["window_start"].dt.hour
features["day_of_week"] = features["window_start"].dt.dayofweek
features["is_weekend"] = features["day_of_week"].isin([5, 6]).astype(int)

features["mean_interarrival"] = features["mean_interarrival"].fillna(0)
features["median_interarrival"] = features["median_interarrival"].fillna(0)
features["std_interarrival"] = features["std_interarrival"].fillna(0)

features.to_csv(OUTPUT_FILE, index=False)
print(f"\nFeatures salvas em: {OUTPUT_FILE}")
