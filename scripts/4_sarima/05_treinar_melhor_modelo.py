import ast
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

DADOS_DIR = AOL / "results" / "completo" / "sarima" / "dados"
MODELOS_DIR = AOL / "results" / "completo" / "sarima" / "modelos"
PREVISOES_DIR = AOL / "results" / "completo" / "sarima" / "previsoes"
GRAFICOS_DIR = AOL / "results" / "plots"
for directory in [MODELOS_DIR, PREVISOES_DIR, GRAFICOS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def load_series(filename, require_continuous=True):
    input_file = DADOS_DIR / filename

    if not input_file.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_file}")

    df = pd.read_csv(input_file)
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = df.sort_values("window_start").set_index("window_start")

    series = df["requests"]

    if require_continuous:
        series = series.asfreq("h")

        if series.isna().any():
            missing = series[series.isna()]
            raise ValueError(
                f"A série {filename} possui {len(missing)} valores ausentes após asfreq('h'). "
                f"Primeiras ausências: {list(missing.index[:5])}. "
                "Verifique se há lacunas dentro desse subconjunto."
            )

    return series


def parse_best_model():
    input_file = MODELOS_DIR / "melhor_modelo_validacao.txt"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {input_file}. "
            "Execute primeiro o script 04_ajustar_modelos_otimizado.py."
        )

    content = input_file.read_text(encoding="utf-8")

    order = None
    seasonal_order = None
    model_name = None

    for line in content.splitlines():
        if line.startswith("model:"):
            model_name = line.split("model:", 1)[1].strip()
        elif line.startswith("order:"):
            order = ast.literal_eval(line.split("order:", 1)[1].strip())
        elif line.startswith("seasonal_order:"):
            seasonal_order = ast.literal_eval(line.split("seasonal_order:", 1)[1].strip())

    if order is None or seasonal_order is None:
        raise ValueError(
            "Não foi possível ler order e seasonal_order do arquivo melhor_modelo_validacao.txt."
        )

    return model_name, order, seasonal_order


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    if not mask.any():
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0

    if not mask.any():
        return np.nan

    return np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100


def evaluate(real, forecast):
    real = pd.Series(real).astype(float)
    forecast = pd.Series(forecast).astype(float)

    data = pd.concat([real, forecast], axis=1).dropna()
    data.columns = ["real", "forecast"]

    if data.empty:
        return {
            "mae": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "smape": np.nan,
            "n_observacoes": 0
        }

    mae = mean_absolute_error(data["real"], data["forecast"])
    mse = mean_squared_error(data["real"], data["forecast"])
    rmse = np.sqrt(mse)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape(data["real"], data["forecast"]),
        "smape": smape(data["real"], data["forecast"]),
        "n_observacoes": len(data)
    }


def fit_sarima(series, order, seasonal_order):
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        # Não usar simple_differencing=True aqui, para manter a previsão na escala original.
        simple_differencing=False
    )

    return model.fit(disp=False, maxiter=80)


# O teste começa em 18/05, mas o treino+validação termina em 16/05. O dia 17/05 foi
# removido por estar incompleto. Portanto, não podemos fazer forecast(steps=len(test))
# e atribuir diretamente ao índice do teste, pois a primeira previsão interna
# corresponderia a 17/05 00h, não a 18/05 00h. Esta função prevê também o intervalo
# removido e depois seleciona apenas as datas do conjunto de teste.
def forecast_test_with_gap(results, train_validation, test):
    forecast_start = train_validation.index.max() + pd.Timedelta(hours=1)
    forecast_end = test.index.max()

    full_forecast_index = pd.date_range(
        start=forecast_start,
        end=forecast_end,
        freq="h"
    )

    forecast = results.forecast(steps=len(full_forecast_index))
    forecast.index = full_forecast_index

    return forecast.loc[test.index], forecast


# Cria baselines usando apenas valores observados disponíveis. Como 17/05 foi removido,
# o baseline t-24 não existirá para 18/05. Isso é esperado. As métricas dos baselines
# são calculadas apenas nas linhas em que a referência histórica existe.
def build_baselines(train_validation, test):
    observed = pd.concat([train_validation, test]).sort_index()

    baselines = pd.DataFrame(index=test.index)
    baselines["real"] = test

    baselines["baseline_t_1"] = observed.shift(1).reindex(test.index)
    baselines["baseline_t_24"] = observed.shift(24).reindex(test.index)
    baselines["baseline_t_168"] = observed.shift(168).reindex(test.index)

    return baselines


def plot_test_forecast(train_validation, test, forecast, model_name):
    plt.figure(figsize=(14, 6))

    plt.plot(train_validation.iloc[-24 * 14:], label="Treino + validação - últimas duas semanas")
    plt.plot(test, label="Teste final")
    plt.plot(forecast, label="Previsão SARIMA")

    plt.title(f"Previsão no conjunto de teste - {model_name}")
    plt.xlabel("Data/hora")
    plt.ylabel("Número de requisições")
    plt.legend()
    plt.tight_layout()

    output_path = GRAFICOS_DIR / "05_previsao_teste_melhor_modelo.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráfico salvo em: {output_path}")


print("Treinando melhor modelo com treino + validação e avaliando no teste...")


# =========================
# LEITURA DOS CONJUNTOS
# =========================
train = load_series("treino.csv", require_continuous=True)
validation = load_series("validacao.csv", require_continuous=True)
test = load_series("teste_final.csv", require_continuous=True)

train_validation = pd.concat([train, validation]).sort_index().asfreq("h")

if train_validation.isna().any():
    raise ValueError(
        "A série treino + validação possui lacunas. "
        "Como esses períodos são contínuos, isso não era esperado."
    )


# =========================
# LEITURA DO MELHOR MODELO
# =========================
model_name, order, seasonal_order = parse_best_model()

print(f"\nMelhor modelo lido: {model_name}")
print(f"order: {order}")
print(f"seasonal_order: {seasonal_order}")


# =========================
# TREINO DO MODELO FINAL
# =========================
print("\nTreinando modelo final em treino + validação...")
results = fit_sarima(train_validation, order, seasonal_order)


# =========================
# PREVISÃO NO TESTE (respeitando a lacuna do dia 17/05)
# =========================
print("\nGerando previsão para o teste, respeitando a lacuna do dia 17/05...")
test_forecast, full_forecast_with_gap = forecast_test_with_gap(
    results=results,
    train_validation=train_validation,
    test=test
)


# =========================
# MÉTRICAS E BASELINES
# =========================
metrics_sarima = evaluate(test, test_forecast)
metrics_rows = [{
    "modelo": "SARIMA",
    **metrics_sarima
}]

baselines = build_baselines(train_validation, test)

for baseline_name in ["baseline_t_1", "baseline_t_24", "baseline_t_168"]:
    metrics_rows.append({
        "modelo": baseline_name,
        **evaluate(baselines["real"], baselines[baseline_name])
    })

metrics_df = pd.DataFrame(metrics_rows)


# =========================
# SALVANDO AS SAÍDAS
# =========================
metrics_path = MODELOS_DIR / "05_metricas_teste_com_baselines.csv"
metrics_df.to_csv(metrics_path, index=False)

forecast_output = pd.DataFrame({
    "window_start": test.index,
    "real": test.values,
    "forecast_sarima": test_forecast.values,
    "error": test.values - test_forecast.values
})

forecast_path = PREVISOES_DIR / "05_previsao_teste_sarima.csv"
forecast_output.to_csv(forecast_path, index=False)

full_forecast_output = pd.DataFrame({
    "window_start": full_forecast_with_gap.index,
    "forecast_sarima": full_forecast_with_gap.values
})

full_forecast_path = PREVISOES_DIR / "05_previsao_teste_incluindo_lacuna_17_05.csv"
full_forecast_output.to_csv(full_forecast_path, index=False)

baselines_path = PREVISOES_DIR / "05_baselines_teste.csv"
baselines.reset_index().rename(columns={"index": "window_start"}).to_csv(
    baselines_path,
    index=False
)

summary_path = MODELOS_DIR / "05_resumo_modelo_final.txt"
summary_path.write_text(str(results.summary()), encoding="utf-8")


# =========================
# GRÁFICO E RESUMO
# =========================
plot_test_forecast(train_validation, test, test_forecast, model_name)

print("\nMétricas no teste:")
print(metrics_df)

print("\nArquivos gerados:")
print(f"- {metrics_path}")
print(f"- {forecast_path}")
print(f"- {full_forecast_path}")
print(f"- {baselines_path}")
print(f"- {summary_path}")

print("\nAvaliação no teste finalizada.")
