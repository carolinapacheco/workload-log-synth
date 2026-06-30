import gc
import ast
import json
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

ARQUIVO_RESULTADOS = MODELOS_DIR / "04_modelos_candidatos_validacao.csv"
ARQUIVO_MELHOR_MODELO = MODELOS_DIR / "melhor_modelo_validacao.txt"
ARQUIVO_MELHOR_MODELO_JSON = MODELOS_DIR / "melhor_modelo_validacao.json"
ARQUIVO_RESUMO_MELHOR = MODELOS_DIR / "04_resumo_melhor_modelo_validacao.txt"

MAXITER = 50
SAZONALIDADE = 24


def load_series(filename):
    df = pd.read_csv(DADOS_DIR / filename)
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = df.sort_values("window_start").set_index("window_start")

    series = df["requests"].asfreq("h")

    if series.isna().any():
        raise ValueError(
            f"A série {filename} possui valores ausentes. "
            "Verifique o script 01_preparar_serie.py."
        )

    return series.astype(float)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    if not mask.any():
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0

    if not mask.any():
        return np.nan

    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def evaluate_forecast(real, forecast):
    mae = mean_absolute_error(real, forecast)
    mse = mean_squared_error(real, forecast)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(real, forecast)
    smape = symmetric_mean_absolute_percentage_error(real, forecast)

    return mae, mse, rmse, mape, smape


def fit_sarima(train, order, seasonal_order):
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        concentrate_scale=True,
        simple_differencing=False
    )

    results = model.fit(
        disp=False,
        maxiter=MAXITER,
        method="lbfgs"
    )

    return results


def get_candidate_models():
    d_values = [0, 1]

    non_seasonal_orders_base = [
        (1, None, 0),
        (0, None, 1),
        (1, None, 1),
        (2, None, 1),
    ]

    seasonal_orders = [
        (1, 1, 0, SAZONALIDADE),
        (0, 1, 1, SAZONALIDADE),
        (1, 1, 1, SAZONALIDADE),
    ]

    candidates = []

    for d in d_values:
        for p, _, q in non_seasonal_orders_base:
            order = (p, d, q)

            for seasonal_order in seasonal_orders:
                name = f"SARIMA{order}{seasonal_order}"
                candidates.append({
                    "model": name,
                    "order": order,
                    "seasonal_order": seasonal_order
                })

    return candidates


def read_done_models():
    if not ARQUIVO_RESULTADOS.exists():
        return set()

    try:
        df = pd.read_csv(ARQUIVO_RESULTADOS)
        if "model" not in df.columns:
            return set()
        return set(df["model"].dropna().astype(str))
    except Exception:
        return set()


def append_result(row):
    row_df = pd.DataFrame([row])

    write_header = not ARQUIVO_RESULTADOS.exists()
    row_df.to_csv(
        ARQUIVO_RESULTADOS,
        mode="a",
        header=write_header,
        index=False
    )


def safe_model_name(name):
    return (
        name.replace("SARIMA", "sarima")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
        .replace(" ", "")
    )


def save_forecast(validation, forecast, model_name):
    forecast_output = pd.DataFrame({
        "window_start": validation.index,
        "real": validation.values,
        "forecast": forecast.values,
        "error": validation.values - forecast.values,
        "model": model_name
    })

    output_path = PREVISOES_DIR / f"04_forecast_validacao_{safe_model_name(model_name)}.csv"
    forecast_output.to_csv(output_path, index=False)


def _to_native(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_best_model_info(best_row):
    with open(ARQUIVO_MELHOR_MODELO, "w", encoding="utf-8") as f:
        f.write("Melhor modelo SARIMA selecionado na validação\n")
        f.write("=" * 70 + "\n\n")
        for column, value in best_row.items():
            f.write(f"{column}: {value}\n")

    best_dict = {k: _to_native(v) for k, v in best_row.to_dict().items()}
    best_dict["order"] = list(ast.literal_eval(best_dict["order"]))
    best_dict["seasonal_order"] = list(ast.literal_eval(best_dict["seasonal_order"]))

    with open(ARQUIVO_MELHOR_MODELO_JSON, "w", encoding="utf-8") as f:
        json.dump(best_dict, f, indent=4, ensure_ascii=False)

    print(f"Informações do melhor modelo salvas em: {ARQUIVO_MELHOR_MODELO}")
    print(f"Arquivo JSON do melhor modelo salvo em: {ARQUIVO_MELHOR_MODELO_JSON}")


def plot_best_validation_forecast(train, validation, forecast, model_name):
    plt.figure(figsize=(14, 6))

    plt.plot(train.iloc[-24 * 14:], label="Treino - últimas duas semanas")
    plt.plot(validation, label="Validação")
    plt.plot(forecast, label="Previsão SARIMA")

    plt.title(f"Previsão no conjunto de validação - {model_name}")
    plt.xlabel("Data/hora")
    plt.ylabel("Número de requisições")
    plt.legend()
    plt.tight_layout()

    output_path = GRAFICOS_DIR / "04_previsao_validacao_melhor_modelo.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráfico do melhor modelo salvo em: {output_path}")


print("Iniciando ajuste otimizado dos modelos SARIMA candidatos...")


# =========================
# LEITURA DOS CONJUNTOS
# =========================
train = load_series("treino.csv")
validation = load_series("validacao.csv")

print("\nConjuntos carregados:")
print(f"Treino: {train.index.min()} até {train.index.max()} | {len(train)} observações")
print(f"Validação: {validation.index.min()} até {validation.index.max()} | {len(validation)} observações")


# =========================
# AJUSTE E AVALIAÇÃO NA VALIDAÇÃO
# =========================
candidates = get_candidate_models()
done_models = read_done_models()

print(f"\nModelos candidatos: {len(candidates)}")
print(f"Modelos já registrados no CSV: {len(done_models)}")

for i, candidate in enumerate(candidates, start=1):
    model_name = candidate["model"]
    order = candidate["order"]
    seasonal_order = candidate["seasonal_order"]

    if model_name in done_models:
        print(f"\n[{i}/{len(candidates)}] Pulando modelo já executado: {model_name}")
        continue

    print(f"\n[{i}/{len(candidates)}] Ajustando {model_name}...")

    try:
        results = fit_sarima(train, order, seasonal_order)

        forecast = results.forecast(steps=len(validation))
        forecast.index = validation.index

        mae, mse, rmse, mape, smape = evaluate_forecast(validation, forecast)

        min_forecast = float(np.min(forecast))
        max_forecast = float(np.max(forecast))
        negative_forecasts = int((forecast < 0).sum())

        row = {
            "model": model_name,
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            "aic": results.aic,
            "bic": results.bic,
            "mae_validation": mae,
            "mse_validation": mse,
            "rmse_validation": rmse,
            "mape_validation": mape,
            "smape_validation": smape,
            "min_forecast": min_forecast,
            "max_forecast": max_forecast,
            "negative_forecasts": negative_forecasts,
            "converged": results.mle_retvals.get("converged", None),
            "iterations": results.mle_retvals.get("iterations", None),
            "status": "ok"
        }

        append_result(row)
        save_forecast(validation, forecast, model_name)

        print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}% | sMAPE: {smape:.2f}%")
        print(f"Previsões: min={min_forecast:.2f}, max={max_forecast:.2f}, negativas={negative_forecasts}")
        print(f"AIC: {results.aic:.4f} | BIC: {results.bic:.4f}")

        try:
            results.remove_data()
        except Exception:
            pass

        del results, forecast

    except Exception as error:
        print(f"Erro ao ajustar {model_name}: {error}")

        row = {
            "model": model_name,
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            "aic": np.nan,
            "bic": np.nan,
            "mae_validation": np.nan,
            "mse_validation": np.nan,
            "rmse_validation": np.nan,
            "mape_validation": np.nan,
            "smape_validation": np.nan,
            "min_forecast": np.nan,
            "max_forecast": np.nan,
            "negative_forecasts": np.nan,
            "converged": None,
            "iterations": None,
            "status": f"erro: {error}"
        }

        append_result(row)

    gc.collect()


# =========================
# SELEÇÃO DO MELHOR MODELO
# =========================
evaluation_df = pd.read_csv(ARQUIVO_RESULTADOS)
valid_models = evaluation_df[evaluation_df["status"] == "ok"].copy()

if valid_models.empty:
    raise RuntimeError("Nenhum modelo candidato foi ajustado com sucesso.")

# penaliza modelos que geram muitas previsões negativas para uma variável de contagem
# não exclui automaticamente, mas coloca mais pra baixo no ranking
valid_models["penalizacao_negativas"] = valid_models["negative_forecasts"].fillna(0) * 1000

best_row = valid_models.sort_values(
    by=["rmse_validation", "mae_validation", "aic"],
    ascending=True
).iloc[0]

print("\nMelhor modelo selecionado na validação:")
print(best_row)

save_best_model_info(best_row)


# =========================
# GRÁFICO E RESUMO DO MELHOR MODELO
# =========================
best_order = tuple(int(x.strip()) for x in best_row["order"].strip("()").split(","))
best_seasonal_order = tuple(int(x.strip()) for x in best_row["seasonal_order"].strip("()").split(","))

print("\nReajustando o melhor modelo para gerar gráfico e resumo...")
best_results = fit_sarima(train, best_order, best_seasonal_order)
best_forecast = best_results.forecast(steps=len(validation))
best_forecast.index = validation.index

plot_best_validation_forecast(
    train=train,
    validation=validation,
    forecast=best_forecast,
    model_name=best_row["model"]
)

with open(ARQUIVO_RESUMO_MELHOR, "w", encoding="utf-8") as f:
    f.write(str(best_results.summary()))

print(f"Resumo estatístico salvo em: {ARQUIVO_RESUMO_MELHOR}")
print("\nAjuste dos modelos candidatos finalizado.")
