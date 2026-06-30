import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")

INPUT_FILE = AOL / "results" / "treino" / "sarima_perfil_1" / "serie_perfil_1_treino.csv"

OUTPUT_DIR = AOL / "results" / "treino" / "sarima_perfil_1" / "modelo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = AOL / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_METRICAS = OUTPUT_DIR / "metricas_sarima_perfil_1.csv"
OUTPUT_PREVISAO = OUTPUT_DIR / "previsao_validacao_sarima_perfil_1.csv"
OUTPUT_RESUMO = OUTPUT_DIR / "resumo_modelo_sarima_perfil_1.txt"
OUTPUT_GRAFICO = PLOTS_DIR / "grafico_previsao_validacao_sarima_perfil_1.png"


def carregar_serie(caminho):
    df = pd.read_csv(caminho)
    df["window_start"] = pd.to_datetime(df["window_start"])

    df = df.set_index("window_start").sort_index()

    df = df.asfreq("h")
    serie = df["requests_perfil_1"].fillna(0)

    return serie


def calcular_metricas(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mascara = y_true != 0
    if mascara.sum() > 0:
        mape = np.mean(np.abs((y_true[mascara] - y_pred[mascara]) / y_true[mascara])) * 100
    else:
        mape = np.nan

    return mae, rmse, mape


def treinar_e_prever(serie_treino, passos, order, seasonal_order):
    modelo = SARIMAX(
        serie_treino,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    resultado = modelo.fit(disp=False)
    previsao = resultado.forecast(steps=passos)

    previsao = previsao.clip(lower=0)

    return resultado, previsao


print("Treinando SARIMA para o perfil 1...")


# =========================
# LEITURA E DIVISÃO INTERNA
# =========================
serie = carregar_serie(INPUT_FILE)

print(f"Série carregada: {INPUT_FILE}")
print(f"Início: {serie.index.min()}")
print(f"Fim: {serie.index.max()}")
print(f"Observações: {len(serie)}")

if len(serie) < 100:
    raise ValueError("A série tem poucas observações para treinar SARIMA com segurança.")

ponto_corte = int(len(serie) * 0.8)

treino = serie.iloc[:ponto_corte]
validacao = serie.iloc[ponto_corte:]

print("\nDivisão interna:")
print(f"Treino interno: {treino.index.min()} até {treino.index.max()} | {len(treino)} observações")
print(f"Validação interna: {validacao.index.min()} até {validacao.index.max()} | {len(validacao)} observações")


# =========================
# AJUSTE E AVALIAÇÃO DOS CANDIDATOS
# =========================
parametros = [
    ((1, 0, 1), (1, 0, 1, 24)),
    ((1, 0, 1), (1, 1, 1, 24)),
    ((2, 0, 1), (1, 0, 1, 24)),
    ((1, 0, 2), (1, 0, 1, 24)),
    ((2, 0, 2), (1, 0, 1, 24)),

    ((1, 0, 1), (0, 1, 1, 24)),
    ((1, 0, 1), (1, 1, 0, 24)),
    ((2, 0, 1), (0, 1, 1, 24)),
    ((1, 0, 2), (1, 1, 0, 24)),

    ((1, 1, 1), (1, 1, 1, 24)),
]

resultados_metricas = []

melhor_modelo = None
melhor_previsao = None
melhor_rmse = float("inf")
melhor_config = None

for order, seasonal_order in parametros:
    print(f"\nTestando SARIMA{order}x{seasonal_order}...")

    try:
        resultado, previsao = treinar_e_prever(
            treino,
            passos=len(validacao),
            order=order,
            seasonal_order=seasonal_order
        )

        mae, rmse, mape = calcular_metricas(validacao, previsao)

        resultados_metricas.append({
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            "aic": resultado.aic,
            "bic": resultado.bic,
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        })

        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")

        if rmse < melhor_rmse:
            melhor_rmse = rmse
            melhor_modelo = resultado
            melhor_previsao = previsao
            melhor_config = (order, seasonal_order)

    except Exception as e:
        print(f"Erro ao treinar SARIMA{order}x{seasonal_order}: {e}")

        resultados_metricas.append({
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            "aic": np.nan,
            "bic": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "erro": str(e)
        })

metricas_df = pd.DataFrame(resultados_metricas)
metricas_df = metricas_df.sort_values("rmse", na_position="last")
metricas_df.to_csv(OUTPUT_METRICAS, index=False)


# =========================
# SALVANDO O MELHOR MODELO
# =========================
if melhor_modelo is None:
    print("Nenhum modelo foi treinado com sucesso.")
else:
    previsao_df = pd.DataFrame({
        "window_start": validacao.index,
        "real": validacao.values,
        "previsto": melhor_previsao.values
    })

    previsao_df.to_csv(OUTPUT_PREVISAO, index=False)

    plt.figure(figsize=(14, 5))
    plt.plot(previsao_df["window_start"], previsao_df["real"], label="Real")
    plt.plot(previsao_df["window_start"], previsao_df["previsto"], label="Previsto")
    plt.title("Previsão SARIMA para o perfil 1 - validação interna")
    plt.xlabel("Tempo")
    plt.ylabel("Requests perfil 1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_GRAFICO, dpi=300, bbox_inches="tight")
    plt.close()

    melhor_order, melhor_seasonal_order = melhor_config

    resumo = f"""
Resumo do modelo SARIMA para o perfil 1

Série utilizada:
{INPUT_FILE}

Divisão utilizada:
Treino interno: {treino.index.min()} até {treino.index.max()} | {len(treino)} observações
Validação interna: {validacao.index.min()} até {validacao.index.max()} | {len(validacao)} observações

Melhor configuração:
SARIMA{melhor_order}x{melhor_seasonal_order}

Melhor RMSE:
{melhor_rmse}

Arquivos gerados:
Métricas: {OUTPUT_METRICAS}
Previsão: {OUTPUT_PREVISAO}
Gráfico: {OUTPUT_GRAFICO}

Observação metodológica:
Como a análise do perfil 1 foi realizada sobre o conjunto de treino, a avaliação do SARIMA foi feita por divisão interna da própria série temporal do perfil 1. As janelas pertencentes ao perfil 1 mantêm seus valores observados de requisições, enquanto as demais janelas recebem valor zero, representando ausência daquele comportamento de carga.
"""

    OUTPUT_RESUMO.write_text(resumo, encoding="utf-8")

    print(resumo)
    print("Treinamento finalizado.")
