import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")

DADOS_DIR = AOL / "results" / "completo" / "sarima" / "dados"
GRAFICOS_DIR = AOL / "results" / "plots"
DIAGNOSTICO_DIR = AOL / "results" / "completo" / "sarima" / "diagnostico"
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTICO_DIR.mkdir(parents=True, exist_ok=True)


print("Iniciando geração de ACF e PACF...")


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
# ACF E PACF DE CADA VERSÃO DA SÉRIE
# =========================
versoes = [
    (train, "treino_original", "série de treino original", None),
    (train.diff().dropna(), "treino_diferenciacao_comum", "série de treino com diferenciação comum", "serie_treino_diferenciacao_comum.csv"),
    (train.diff(24).dropna(), "treino_diferenciacao_sazonal_24h", "série de treino com diferenciação sazonal de 24 horas", "serie_treino_diferenciacao_sazonal_24h.csv"),
    (train.diff(24).diff().dropna(), "treino_diferenciacao_comum_e_sazonal", "série de treino com diferenciação comum e sazonal", "serie_treino_diferenciacao_comum_e_sazonal.csv"),
]

for serie, suffix, title, filename in versoes:
    if filename is not None:
        output = serie.reset_index()
        output.columns = ["window_start", "requests"]
        output.to_csv(DIAGNOSTICO_DIR / filename, index=False)
        print(f"Série transformada salva em: {DIAGNOSTICO_DIR / filename}")

    lags = min(120, max(1, len(serie.dropna()) // 2 - 1))

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_acf(serie.dropna(), lags=lags, alpha=0.05, ax=ax)
    ax.set_title(f"ACF - {title}")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelação")
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / f"03_acf_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_pacf(serie.dropna(), lags=lags, alpha=0.05, method="ywm", ax=ax)
    ax.set_title(f"PACF - {title}")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelação parcial")
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / f"03_pacf_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"ACF/PACF salvas para: {title}")


print("\nGeração de ACF e PACF finalizada.")
print(f"Gráficos salvos em: {GRAFICOS_DIR}")
