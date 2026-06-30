import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")

INPUT_FILE = AOL / "results" / "treino" / "sarima_perfil_1" / "serie_perfil_1_treino.csv"

OUTPUT_DIR = AOL / "results" / "treino" / "sarima_perfil_1" / "diagnostico"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = AOL / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_ADF = OUTPUT_DIR / "adf_perfil_1.txt"


def executar_adf(serie, nome):
    serie = serie.dropna()

    resultado = adfuller(serie)

    texto = f"""
Teste ADF - {nome}

Estatística ADF: {resultado[0]}
p-valor: {resultado[1]}
Lags usados: {resultado[2]}
Número de observações: {resultado[3]}

Valores críticos:
1%: {resultado[4]["1%"]}
5%: {resultado[4]["5%"]}
10%: {resultado[4]["10%"]}

Interpretação:
"""

    if resultado[1] <= 0.05:
        texto += "A série pode ser considerada estacionária ao nível de 5%.\n"
    else:
        texto += "A série não pode ser considerada estacionária ao nível de 5%.\n"

    return texto


def salvar_grafico_serie(serie, titulo, arquivo_saida):
    plt.figure(figsize=(14, 5))
    plt.plot(serie.index, serie.values)
    plt.title(titulo)
    plt.xlabel("Tempo")
    plt.ylabel("Requests perfil 1")
    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=300, bbox_inches="tight")
    plt.close()


def salvar_acf_pacf(serie, prefixo):
    serie = serie.dropna()

    plt.figure(figsize=(12, 5))
    plot_acf(serie, lags=72)
    plt.title(f"ACF - {prefixo}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"acf_{prefixo}.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 5))
    plot_pacf(serie, lags=72, method="ywm")
    plt.title(f"PACF - {prefixo}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"pacf_{prefixo}.png", dpi=300, bbox_inches="tight")
    plt.close()


print("Executando diagnóstico da série do perfil 1...")


# =========================
# LEITURA DO TREINO
# =========================
df = pd.read_csv(INPUT_FILE)
df["window_start"] = pd.to_datetime(df["window_start"])
df = df.set_index("window_start").asfreq("h")

serie = df["requests_perfil_1"].fillna(0)


# =========================
# GRÁFICOS E ACF/PACF
# =========================
salvar_grafico_serie(
    serie,
    "Série temporal do perfil 1 - treino",
    PLOTS_DIR / "serie_perfil_1_treino.png"
)

salvar_acf_pacf(serie, "serie_original")

serie_diff = serie.diff().dropna()

salvar_grafico_serie(
    serie_diff,
    "Série do perfil 1 com diferenciação comum",
    PLOTS_DIR / "serie_perfil_1_diff_comum.png"
)

salvar_acf_pacf(serie_diff, "diff_comum")

serie_diff_sazonal = serie.diff(24).dropna()

salvar_grafico_serie(
    serie_diff_sazonal,
    "Série do perfil 1 com diferenciação sazonal 24h",
    PLOTS_DIR / "serie_perfil_1_diff_sazonal_24h.png"
)

salvar_acf_pacf(serie_diff_sazonal, "diff_sazonal_24h")


# =========================
# TESTES ADF
# =========================
texto = ""
texto += executar_adf(serie, "série original do perfil 1")
texto += "\n" + "=" * 80 + "\n"
texto += executar_adf(serie_diff, "série com diferenciação comum")
texto += "\n" + "=" * 80 + "\n"
texto += executar_adf(serie_diff_sazonal, "série com diferenciação sazonal 24h")

OUTPUT_ADF.write_text(texto, encoding="utf-8")

print(texto)
print(f"Diagnóstico salvo em: {OUTPUT_DIR}")
