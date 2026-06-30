from pathlib import Path
import pandas as pd

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")

INPUT_FILE = AOL / "data" / "processed" / "aol_query_log_clean.csv"
OUTPUT_FILE = AOL / "results" / "limpeza" / "daily_coverage_check.csv"

MIN_HORAS_DIA_COMPLETO = 18


df = pd.read_csv(INPUT_FILE)
df["QueryTime"] = pd.to_datetime(df["QueryTime"])

daily_coverage = df.groupby(df["QueryTime"].dt.date).agg(
    requests=("QueryTime", "size"),
    first_time=("QueryTime", "min"),
    last_time=("QueryTime", "max"),
).reset_index().rename(columns={"QueryTime": "date"})

daily_coverage["coverage_hours"] = (daily_coverage["last_time"] - daily_coverage["first_time"]).dt.total_seconds() / 3600
daily_coverage["complete_day"] = daily_coverage["coverage_hours"] >= MIN_HORAS_DIA_COMPLETO

print(daily_coverage.sort_values("coverage_hours").head(10))

daily_coverage.to_csv(OUTPUT_FILE, index=False)
