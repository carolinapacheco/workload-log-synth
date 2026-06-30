import ast
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

HORIZONTE_HORAS = 24 * 7


# carrega a série de modelagem que tem o dia 17/05 removido de propósito
# não força asfreq('h'), pois a lacuna é esperada e não deve virar erro
def load_series_modelagem(filename):
    input_file = DADOS_DIR / filename
    df = pd.read_csv(input_file)
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = df.sort_values("window_start").set_index("window_start")
    return df["requests"].astype(float)


def load_best_model_params():
    json_path = MODELOS_DIR / "melhor_modelo_validacao.json"
    csv_path = MODELOS_DIR / "04_modelos_candidatos_validacao.csv"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            best = json.load(f)
        return best["model"], tuple(best["order"]), tuple(best["seasonal_order"])

    if not csv_path.exists():
        raise FileNotFoundError(
            "Não encontrei os parâmetros do melhor modelo. Execute o script 04 primeiro."
        )

    df = pd.read_csv(csv_path)
    valid = df[df["status"] == "ok"].copy()
    best = valid.sort_values(by=["rmse_validation", "mae_validation", "aic"]).iloc[0]
    return best["model"], ast.literal_eval(best["order"]), ast.literal_eval(best["seasonal_order"])


def fit_final_model(series_modelagem):
    model_path = MODELOS_DIR / "06_modelo_final_historico_completo.pkl"

    if model_path.exists():
        print(f"Carregando modelo final salvo em: {model_path}")
        return SARIMAXResults.load(model_path)

    print("Ajustando modelo final no histórico completo (treino + validação + teste)...")
    _, order, seasonal_order = load_best_model_params()

    model = SARIMAX(
        series_modelagem,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)
    results.save(model_path)
    return results


def plot_future(series_modelagem, forecast_df):
    plt.figure(figsize=(14, 6))

    plt.plot(series_modelagem.iloc[-24 * 14:], label="Histórico - últimos 14 dias")
    plt.plot(forecast_df["window_start"], forecast_df["forecast"], label="Previsão futura SARIMA")
    plt.fill_between(
        forecast_df["window_start"],
        forecast_df["lower_95"],
        forecast_df["upper_95"],
        alpha=0.2,
        label="Intervalo de 95%",
    )

    plt.title("Previsão futura de requisições por hora")
    plt.xlabel("Data/hora")
    plt.ylabel("Número de requisições")
    plt.legend()
    plt.tight_layout()

    output_path = GRAFICOS_DIR / "06_previsoes_futuras.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráfico de previsões futuras salvo em: {output_path}")


print("Gerando previsões futuras com o melhor modelo SARIMA...")


# =========================
# LEITURA DA SÉRIE E DO MODELO
# =========================
series_modelagem = load_series_modelagem("serie_modelagem_sem_dias_incompletos.csv")
results = fit_final_model(series_modelagem)
model_name, order, seasonal_order = load_best_model_params()

print(f"\nModelo: {model_name}")
print(f"order: {order} | seasonal_order: {seasonal_order}")
print(f"Fim da série observada: {series_modelagem.index.max()}")


# =========================
# PREVISÃO FUTURA
# =========================
forecast_result = results.get_forecast(steps=HORIZONTE_HORAS)

forecast = forecast_result.predicted_mean.clip(lower=0)
conf_int = forecast_result.conf_int(alpha=0.05)
conf_int.columns = ["lower_95", "upper_95"]
conf_int = conf_int.clip(lower=0)

last_timestamp = series_modelagem.index.max()
future_index = pd.date_range(
    start=last_timestamp + pd.Timedelta(hours=1),
    periods=HORIZONTE_HORAS,
    freq="h",
)

forecast_df = pd.DataFrame({
    "window_start": future_index,
    "forecast": forecast.values,
    "lower_95": conf_int["lower_95"].values,
    "upper_95": conf_int["upper_95"].values,
    "model": model_name,
    "order": str(order),
    "seasonal_order": str(seasonal_order),
})


# =========================
# SALVANDO E GRÁFICO
# =========================
output_path = PREVISOES_DIR / "06_previsoes_futuras_sarima.csv"
forecast_df.to_csv(output_path, index=False)

plot_future(series_modelagem, forecast_df)

print(f"\nPrevisão futura: {future_index.min()} até {future_index.max()} "
      f"({HORIZONTE_HORAS} horas)")
print(f"Volume previsto - mín: {forecast.min():.0f} | "
      f"máx: {forecast.max():.0f} | média/h: {forecast.mean():.0f}")
print(f"Previsões futuras salvas em: {output_path}")
print("Geração de previsões futuras finalizada.")
