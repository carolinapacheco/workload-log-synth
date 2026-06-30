import pandas as pd
import warnings
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, kpss

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")

DADOS_DIR = AOL / "results" / "completo" / "sarima" / "dados"
DIAGNOSTICO_DIR = AOL / "results" / "completo" / "sarima" / "diagnostico"
DIAGNOSTICO_DIR.mkdir(parents=True, exist_ok=True)


def teste_adf(series, name, filename):
    result = adfuller(series.dropna(), autolag="AIC")
    adf_statistic = result[0]
    p_value = result[1]
    used_lags = result[2]
    n_obs = result[3]
    critical_values = result[4]
    icbest = result[5]

    output_path = DIAGNOSTICO_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Teste ADF - {name}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Hipóteses do teste:\n")
        f.write("H0: a série possui raiz unitária, ou seja, não é estacionária.\n")
        f.write("H1: a série não possui raiz unitária, ou seja, é estacionária.\n\n")
        f.write(f"ADF Statistic: {adf_statistic}\n")
        f.write(f"p-value: {p_value}\n")
        f.write(f"Used lags: {used_lags}\n")
        f.write(f"Number of observations: {n_obs}\n")
        f.write(f"IC best: {icbest}\n\n")
        f.write("Critical values:\n")
        for key, value in critical_values.items():
            f.write(f"{key}: {value}\n")
        f.write("\nInterpretação:\n")
        if p_value < 0.05:
            f.write(
                "Como o p-value é menor que 0,05, rejeita-se H0. "
                "Assim, a série pode ser considerada estacionária ao nível de 5%.\n"
            )
        else:
            f.write(
                "Como o p-value é maior ou igual a 0,05, não se rejeita H0. "
                "Assim, a série não pode ser considerada estacionária ao nível de 5%.\n"
            )

    print(f"\nTeste ADF - {name}")
    print(f"ADF Statistic: {adf_statistic}")
    print(f"p-value: {p_value}")
    print("Resultado:", "Estacionária" if p_value < 0.05 else "Não estacionária")
    print(f"Arquivo salvo em: {output_path}")

    return {
        "serie": name,
        "adf_statistic": adf_statistic,
        "p_value": p_value,
        "used_lags": used_lags,
        "n_obs": n_obs,
        "icbest": icbest,
        "estacionaria_5_pct": p_value < 0.05,
    }


def teste_kpss(series, name, filename):
    # o KPSS emite InterpolationWarning quando a estatística cai fora da tabela de p-valores
    # nesse caso o p-valor é truncado em 0,01 ou 0,10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        statistic, p_value, used_lags, critical_values = kpss(series.dropna(), regression="c", nlags="auto")

    output_path = DIAGNOSTICO_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Teste KPSS - {name}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Hipóteses do teste:\n")
        f.write("H0: a série é estacionária.\n")
        f.write("H1: a série não é estacionária, ou seja, possui raiz unitária.\n\n")
        f.write(f"KPSS Statistic: {statistic}\n")
        f.write(f"p-value: {p_value}\n")
        f.write(f"Used lags: {used_lags}\n\n")
        f.write("Critical values:\n")
        for key, value in critical_values.items():
            f.write(f"{key}: {value}\n")
        f.write("\nInterpretação:\n")
        if p_value < 0.05:
            f.write(
                "Como o p-value é menor que 0,05, rejeita-se H0. "
                "Assim, a série não pode ser considerada estacionária ao nível de 5%.\n"
            )
        else:
            f.write(
                "Como o p-value é maior ou igual a 0,05, não se rejeita H0. "
                "Assim, a série pode ser considerada estacionária ao nível de 5%.\n"
            )

    print(f"\nTeste KPSS - {name}")
    print(f"KPSS Statistic: {statistic}")
    print(f"p-value: {p_value}")
    print("Resultado:", "Estacionária" if p_value >= 0.05 else "Não estacionária")
    print(f"Arquivo salvo em: {output_path}")

    return {
        "serie": name,
        "kpss_statistic": statistic,
        "kpss_p_value": p_value,
        "kpss_used_lags": used_lags,
        "kpss_estacionaria_5_pct": p_value >= 0.05,
    }


print("Iniciando diagnóstico de estacionariedade...")


# =========================
# LEITURA DO TREINO
# =========================
df = pd.read_csv(DADOS_DIR / "treino.csv")
df["window_start"] = pd.to_datetime(df["window_start"])
df = df.sort_values("window_start").set_index("window_start")
train = df["requests"].astype(float).asfreq("h")

print("\nConjunto de treino carregado.")
print(f"Início: {train.index.min()}")
print(f"Fim: {train.index.max()}")
print(f"Observações: {len(train)}")


# =========================
# TESTES NAS TRANSFORMAÇÕES DA SÉRIE
# =========================
transformacoes = [
    (train, "série de treino original", "adf_treino_original.txt"),
    (train.diff().dropna(), "série de treino com diferenciação comum", "adf_treino_diferenciacao_comum.txt"),
    (train.diff(24).dropna(), "série de treino com diferenciação sazonal de 24 horas", "adf_treino_diferenciacao_sazonal_24h.txt"),
    (train.diff(24).diff().dropna(), "série de treino com diferenciação comum e sazonal", "adf_treino_diferenciacao_comum_e_sazonal.txt"),
]

adf_results = []
kpss_results = []
for serie, nome, arquivo in transformacoes:
    adf_results.append(teste_adf(serie, nome, arquivo))
    kpss_results.append(teste_kpss(serie, nome, arquivo.replace("adf_", "kpss_")))

adf_summary = pd.DataFrame(adf_results)
kpss_summary = pd.DataFrame(kpss_results)
adf_summary.to_csv(DIAGNOSTICO_DIR / "resumo_adf.csv", index=False)
kpss_summary.to_csv(DIAGNOSTICO_DIR / "resumo_kpss.csv", index=False)

print("\nResumo dos testes ADF:")
print(adf_summary)
print("\nResumo dos testes KPSS:")
print(kpss_summary)


# =========================
# VISÃO COMBINADA (concordância entre ADF e KPSS)
# =========================
combinado = adf_summary.merge(kpss_summary, on="serie")

conclusoes = []
for _, row in combinado.iterrows():
    adf_estac = row["estacionaria_5_pct"]
    kpss_estac = row["kpss_estacionaria_5_pct"]
    if adf_estac and kpss_estac:
        conclusoes.append("estacionária (ADF e KPSS concordam)")
    elif (not adf_estac) and (not kpss_estac):
        conclusoes.append("não-estacionária (ADF e KPSS concordam)")
    else:
        conclusoes.append("inconclusivo (ADF e KPSS discordam)")
combinado["conclusao"] = conclusoes

combinado.to_csv(DIAGNOSTICO_DIR / "resumo_adf_kpss.csv", index=False)

print("\nResumo combinado ADF + KPSS:")
print(combinado[["serie", "estacionaria_5_pct", "kpss_estacionaria_5_pct", "conclusao"]])
print("Diagnóstico de estacionariedade finalizado.")
