import argparse
import numpy as np
import pandas as pd
from pathlib import Path

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")

PERFIS_DIR = AOL / "results" / "treino" / "perfis_kmeans"
TREINO_DIR = AOL / "results" / "treino" / "caracterizacao"

# Entrada: previsao do sarima para o teste. Aceita qualquer csv no formato "window_start,forecast_sarima"
FORECAST_PADRAO = AOL / "results" / "completo" / "sarima" / "previsoes" / "05_previsao_teste_sarima.csv"

SAIDA_DIR = AOL / "results" / "log_sintetico"
SAIDA_DIR.mkdir(parents=True, exist_ok=True)

PERFIS_ANOMALOS = [4]
CATALOGO_QUERIES = 50_000
SEED = 42


def carregar_inputs():
    janelas = pd.read_csv(PERFIS_DIR / "janelas_com_perfis_kmeans.csv")
    perfis = pd.read_csv(PERFIS_DIR / "resumo_perfis_kmeans.csv")
    req_user = pd.read_csv(TREINO_DIR / "requests_per_user.csv")
    q_freq = pd.read_csv(TREINO_DIR / "query_frequency_distribution.csv")
    itemrank = pd.read_csv(TREINO_DIR / "itemrank_distribution.csv")
    return janelas, perfis, req_user, q_freq, itemrank


# Perfil mais comum de cada hora (moda), ignorando o perfil anômalo. Empate: pega o primeiro.
def perfil_dominante_por_hora(janelas):
    base = janelas[~janelas["perfil_carga"].isin(PERFIS_ANOMALOS)]
    dominante = base.groupby("hour")["perfil_carga"].agg(lambda s: s.mode().iloc[0])
    return dominante.to_dict()


def razoes_por_perfil(perfis):
    razoes = {}
    for _, r in perfis.iterrows():
        p = int(r["perfil_carga"])
        razoes[p] = {
            "users_por_req": r["unique_users_media"] / r["requests_media"],
            "click_rate": r["click_rate_media"],
        }
    return razoes


def montar_catalogo_queries(q_freq, rng):
    freqs = q_freq["Frequency"].to_numpy().astype(float)
    n_por_faixa = q_freq["NumberOfQueries"].to_numpy().astype(float)
    pesos = rng.choice(freqs, size=CATALOGO_QUERIES, p=n_por_faixa / n_por_faixa.sum())
    catalogo = np.arange(CATALOGO_QUERIES)
    probs = pesos / pesos.sum()
    return catalogo, probs


def preparar_itemrank(itemrank):
    valores = itemrank["ItemRank"].to_numpy()
    freq = itemrank["Frequency"].to_numpy().astype(float)
    return valores, freq / freq.sum()


def gerar_hora(ts, n_req, perfil, razoes, n_usuarios_total,
               catalogo, probs_query, ranks, probs_rank, rng):
    if n_req <= 0:
        return None

    r = razoes[perfil]

    u = max(1, int(round(n_req * r["users_por_req"])))
    u = min(u, n_req, n_usuarios_total)
    usuarios_hora = rng.choice(n_usuarios_total, size=u, replace=False) + 1
    anon = rng.choice(usuarios_hora, size=n_req)

    queries = rng.choice(catalogo, size=n_req, p=probs_query)

    segundos = np.sort(rng.integers(0, 3600, size=n_req))
    instantes = ts + pd.to_timedelta(segundos, unit="s")

    clicou = rng.random(n_req) < r["click_rate"]
    rank = rng.choice(ranks, size=n_req, p=probs_rank)

    return pd.DataFrame({
        "AnonID": anon,
        "Query": [f"query_{q}" for q in queries],
        "QueryTime": instantes.strftime("%Y-%m-%d %H:%M:%S"),
        "ItemRank": np.where(clicou, rank, np.nan),
        "ClickURL": np.where(clicou,
                             "http://synthetic.example/" + pd.Series(queries).astype(str),
                             ""),
        "perfil_carga": perfil,
    })


def carregar_forecast(caminho):
    fc = pd.read_csv(caminho)
    col_ts = next((c for c in ["window_start", "timestamp", "data", "date"]
                   if c in fc.columns), None)
    col_val = next((c for c in ["forecast_sarima", "previsto", "forecast",
                                "predicted_mean", "yhat", "requests"]
                    if c in fc.columns), None)
    if col_ts is None or col_val is None:
        raise ValueError(f"Nao identifiquei tempo/volume em: {list(fc.columns)}")

    fc = fc[[col_ts, col_val]].rename(columns={col_ts: "window_start", col_val: "requests"})
    fc["window_start"] = pd.to_datetime(fc["window_start"])
    fc["requests"] = fc["requests"].round().clip(lower=0).astype(int)
    return fc


ap = argparse.ArgumentParser(description="Gera log sintetico (SARIMA + perfis k-means).")
ap.add_argument("--forecast", default=str(FORECAST_PADRAO))
ap.add_argument("--saida", default=str(SAIDA_DIR / "log_sintetico.csv"))
args = ap.parse_args()

rng = np.random.default_rng(SEED)


# =========================
# CARREGAMENTO DOS INSUMOS
# =========================
janelas, perfis, req_user, q_freq, itemrank = carregar_inputs()

dominante = perfil_dominante_por_hora(janelas)
razoes = razoes_por_perfil(perfis)
n_usuarios_total = len(req_user)
catalogo, probs_query = montar_catalogo_queries(q_freq, rng)
ranks, probs_rank = preparar_itemrank(itemrank)

print(f"Carregando previsao do SARIMA: {args.forecast}")
fc = carregar_forecast(Path(args.forecast))
print(f"Horas a gerar: {len(fc)} | requisicoes previstas: {fc['requests'].sum():,}")


# =========================
# GERAÇÃO DO LOG
# =========================
partes = []
for _, linha in fc.iterrows():
    ts = linha["window_start"]
    n = int(linha["requests"])
    perfil = dominante.get(ts.hour, perfis["perfil_carga"].iloc[0])
    bloco = gerar_hora(ts, n, perfil, razoes, n_usuarios_total,
                       catalogo, probs_query, ranks, probs_rank, rng)
    if bloco is not None:
        partes.append(bloco)

log = pd.concat(partes, ignore_index=True)
log.to_csv(args.saida, index=False)
print(f"\nLog sintetico salvo em: {args.saida} ({len(log):,} registros)")


# =========================
# VALIDAÇÃO (reconstrução das features por hora)
# =========================
log["window_start"] = pd.to_datetime(log["QueryTime"]).dt.floor("h")
val = (log.groupby("window_start")
          .agg(requests=("AnonID", "size"),
               unique_users=("AnonID", "nunique"),
               unique_queries=("Query", "nunique"),
               clicks=("ItemRank", lambda s: s.notna().sum()))
          .reset_index())
val["click_rate"] = val["clicks"] / val["requests"]
val = val.merge(fc.rename(columns={"requests": "requests_sarima"}),
                on="window_start", how="left")

saida_val = Path(args.saida).with_name("validacao_log_sintetico.csv")
val.to_csv(saida_val, index=False)

print(f"\nResumo da validacao salvo em: {saida_val}")
