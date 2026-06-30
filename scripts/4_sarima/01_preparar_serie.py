import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")

INPUT_FILE = AOL / "results" / "completo" / "time_buckets" / "requests_per_hour_window.csv"
DADOS_DIR = AOL / "results" / "completo" / "sarima" / "dados"
GRAFICOS_DIR = AOL / "results" / "plots"
DADOS_DIR.mkdir(parents=True, exist_ok=True)
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

# dia identificado como incompleto, removido da série usada na modelagem
DIAS_INCOMPLETOS = ["2006-05-17"]


# garante frequência horária; lacunas pontuais são preenchidas por interpolação
# temporal (evita usar zero, que criaria quedas artificiais de carga)
def garantir_frequencia_horaria(series, name):
    series = series.asfreq("h")
    missing = series[series.isna()]

    if not missing.empty:
        print(f"\nAtenção: o conjunto '{name}' possui {len(missing)} hora(s) ausente(s).")
        print("Primeiras ausências encontradas:")
        print(missing.head())

        series = series.interpolate(method="time")
        series = series.ffill().bfill()

        print(f"Lacunas do conjunto '{name}' preenchidas por interpolação temporal.")

    return series


def salvar_serie(series, filename):
    output = series.reset_index()
    output.columns = ["window_start", "requests"]
    output.to_csv(DADOS_DIR / filename, index=False)
    print(f"Arquivo salvo: {DADOS_DIR / filename}")


print("Preparando série temporal para SARIMA...")


# =========================
# LEITURA DA SÉRIE
# =========================
df = pd.read_csv(INPUT_FILE)
df["window_start"] = pd.to_datetime(df["window_start"])
df = df.sort_values("window_start").set_index("window_start")
series_original = df["requests"].astype(float)


# =========================
# REMOÇÃO DO DIA INCOMPLETO
# =========================
series_modelagem = series_original.copy()
for day in DIAS_INCOMPLETOS:
    day = pd.to_datetime(day).date()
    series_modelagem = series_modelagem[series_modelagem.index.date != day]


# =========================
# DIVISÃO TREINO / VALIDAÇÃO / TESTE
# =========================
# o teste começa em 18/05 porque o dia 17/05 foi removido por estar incompleto
train = series_modelagem.loc["2006-03-01 00:00:00":"2006-04-30 23:00:00"]
validation = series_modelagem.loc["2006-05-01 00:00:00":"2006-05-16 23:00:00"]
test = series_modelagem.loc["2006-05-18 00:00:00":"2006-05-31 23:00:00"]

train = garantir_frequencia_horaria(train, "treino")
validation = garantir_frequencia_horaria(validation, "validacao")
test = garantir_frequencia_horaria(test, "teste_final")


# =========================
# SALVANDO OS CONJUNTOS
# =========================
salvar_serie(series_original, "serie_original.csv")
salvar_serie(series_modelagem, "serie_modelagem_sem_dias_incompletos.csv")
salvar_serie(train, "treino.csv")
salvar_serie(validation, "validacao.csv")
salvar_serie(test, "teste_final.csv")


# =========================
# RESUMO DOS CONJUNTOS
# =========================
rows = []
for nome, serie in [
    ("serie_original", series_original),
    ("serie_modelagem_sem_dias_incompletos", series_modelagem),
    ("treino", train),
    ("validacao", validation),
    ("teste_final", test),
]:
    rows.append({
        "conjunto": nome,
        "inicio": serie.index.min(),
        "fim": serie.index.max(),
        "observacoes": len(serie),
        "requests_total": serie.sum(),
        "requests_media_por_hora": serie.mean(),
        "requests_mediana_por_hora": serie.median(),
        "requests_min": serie.min(),
        "requests_max": serie.max(),
    })

summary = pd.DataFrame(rows)
summary.to_csv(DADOS_DIR / "resumo_conjuntos.csv", index=False)
print(f"Resumo salvo: {DADOS_DIR / 'resumo_conjuntos.csv'}")


# =========================
# GRÁFICO DA DIVISÃO
# =========================
plt.figure(figsize=(14, 6))
plt.plot(train, label="Treino")
plt.plot(validation, label="Validação")
plt.plot(test, label="Teste final")
plt.title("Divisão da série temporal de requisições por hora")
plt.xlabel("Data/hora")
plt.ylabel("Número de requisições")
plt.legend()
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "01_divisao_treino_validacao_teste.png", dpi=300, bbox_inches="tight")
plt.close()


print("\nSérie preparada com sucesso.")
print(f"Série original: {series_original.index.min()} até {series_original.index.max()} | {len(series_original)} observações")
print(f"Série de modelagem: {series_modelagem.index.min()} até {series_modelagem.index.max()} | {len(series_modelagem)} observações")
print(f"Treino: {train.index.min()} até {train.index.max()} | {len(train)} observações")
print(f"Validação: {validation.index.min()} até {validation.index.max()} | {len(validation)} observações")
print(f"Teste final: {test.index.min()} até {test.index.max()} | {len(test)} observações")
