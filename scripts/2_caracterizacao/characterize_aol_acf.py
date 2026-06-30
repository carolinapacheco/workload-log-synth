import pandas as pd
from pathlib import Path

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
CARACT = AOL / "results" / "treino" / "caracterizacao"

INPUT_MINUTE = CARACT / "requests_per_minute.csv"
INPUT_5MIN = CARACT / "requests_per_5min.csv"
INPUT_HOUR = CARACT / "requests_per_hour_window.csv"

OUTPUT_ACF_MINUTE = CARACT / "acf_requests_per_minute.csv"
OUTPUT_ACF_5MIN = CARACT / "acf_requests_per_5min.csv"
OUTPUT_ACF_HOUR = CARACT / "acf_requests_per_hour.csv"

OUTPUT_SUMMARY = CARACT / "acf_summary.txt"

SERIES = [
    (INPUT_MINUTE, OUTPUT_ACF_MINUTE, 120, "Requisições por minuto"),
    (INPUT_5MIN, OUTPUT_ACF_5MIN, 120, "Requisições por 5 minutos"),
    (INPUT_HOUR, OUTPUT_ACF_HOUR, 168, "Requisições por hora"),
]


# =========================
# CÁLCULO DA AUTOCORRELAÇÃO
# =========================
linhas_resumo = [
    "AUTOCORRELAÇÃO DAS SÉRIES TEMPORAIS DO AOL QUERY LOG",
    "=" * 70,
    "",
]

for input_file, output_file, max_lag, label in SERIES:
    df = pd.read_csv(input_file)
    df["window_start"] = pd.to_datetime(df["window_start"], errors="coerce")

    valores = df["requests"].astype(float).reset_index(drop=True)
    linhas_acf = []
    for lag in range(1, max_lag + 1):
        linhas_acf.append({"lag": lag, "acf": valores.autocorr(lag=lag)})

    acf_df = pd.DataFrame(linhas_acf)
    acf_df.to_csv(output_file, index=False, encoding="utf-8")

    top_positive = acf_df.dropna().sort_values("acf", ascending=False).head(10)

    linhas_resumo += [
        f"SÉRIE: {label}",
        "-" * 50,
        f"Arquivo analisado: {input_file.resolve()}",
        f"Quantidade de observações: {len(df)}",
        f"Máximo lag analisado: {max_lag}",
        f"Média da série: {df['requests'].mean():.6f}",
        f"Desvio padrão da série: {df['requests'].std():.6f}",
        f"Arquivo ACF gerado: {output_file.resolve()}",
        "",
        "Top 10 maiores autocorrelações:",
        top_positive.to_string(index=False),
        "",
    ]


# =========================
# RESUMO
# =========================
resumo = "\n".join(linhas_resumo)

with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as arquivo:
    arquivo.write(resumo)

print(resumo)
