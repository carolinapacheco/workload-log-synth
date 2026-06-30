from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


# =========================
# CAMINHOS
# =========================
AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")

LOG_SINTETICO = AOL / "results" / "log_sintetico" / "log_sintetico.csv"

TESTE_FINAL = AOL / "results" / "completo" / "sarima" / "dados" / "teste_final.csv"

# Distribuicoes do TREINO: foram elas que parametrizaram o gerador
TREINO_DIR = AOL / "results" / "treino" / "caracterizacao"
ITEMRANK_DIST = TREINO_DIR / "itemrank_distribution.csv"
HORA_WINDOW = TREINO_DIR / "requests_per_hour_window.csv"
QUERY_FREQ_DIST = TREINO_DIR / "query_frequency_distribution.csv"

SAIDA_DIR = AOL / "results" / "log_sintetico"
SAIDA_DIR.mkdir(parents=True, exist_ok=True)
SAIDA_CSV = SAIDA_DIR / "comparacao_log_sintetico.csv"
SAIDA_TXT = SAIDA_DIR / "comparacao_log_sintetico.txt"


# =========================
# ESTATISTICA: QUI-QUADRADO DE ADERENCIA + V DE CRAMER
# =========================

def efeito_cramer(v):
    if v < 0.1:
        return "desprezivel"
    if v < 0.3:
        return "pequeno"
    if v < 0.5:
        return "medio"
    return "grande"


def comparar(nome, contagem_original, contagem_sintetico):
    categorias = list(contagem_original.keys())
    linha_orig = np.array([contagem_original[c] for c in categorias], dtype=float)
    linha_sint = np.array([contagem_sintetico[c] for c in categorias], dtype=float)

    tabela = np.vstack([linha_orig, linha_sint])
    chi2, p, gl, _ = chi2_contingency(tabela)
    n = tabela.sum()
    v = np.sqrt(chi2 / n)

    return {
        "comparacao": nome,
        "k": len(categorias),
        "gl": gl,
        "N": int(n),
        "chi2": chi2,
        "p": p,
        "cramer_v": v,
        "efeito": efeito_cramer(v),
    }


# =========================
# FAIXAS (MESMAS CATEGORIAS PARA ORIGINAL E SINTETICO)
# =========================
def faixa_itemrank(r):
    if r == 1:
        return "1"
    if r <= 3:
        return "2-3"
    if r <= 5:
        return "4-5"
    if r <= 10:
        return "6-10"
    return "11+"


FAIXAS_ITEMRANK = ["1", "2-3", "4-5", "6-10", "11+"]


def faixa_popularidade(f):
    if f == 1:
        return "1"
    if f == 2:
        return "2"
    if f == 3:
        return "3"
    if f <= 5:
        return "4-5"
    if f <= 10:
        return "6-10"
    if f <= 50:
        return "11-50"
    if f <= 100:
        return "51-100"
    return "101+"


FAIXAS_POPULARIDADE = ["1", "2", "3", "4-5", "6-10", "11-50", "51-100", "101+"]


def somar_por_faixa(df, coluna_valor, coluna_faixa, ordem):
    soma = df.groupby(coluna_faixa)[coluna_valor].sum()
    return {f: float(soma.get(f, 0.0)) for f in ordem}


def contar_por_faixa(serie_faixas, ordem):
    cont = serie_faixas.value_counts()
    return {f: float(cont.get(f, 0.0)) for f in ordem}


# =========================
# CARREGA O LOG SINTETICO
# =========================
print("Carregando log sintetico...")
log = pd.read_csv(LOG_SINTETICO)
log["QueryTime"] = pd.to_datetime(log["QueryTime"])
print(f"Registros sinteticos: {len(log):,}")

resultados = []


# =========================
# PERFIL DIARIO: REQUISICOES POR HORA DO DIA (K = 24)
# =========================
print("Comparando perfil diario (hora do dia)...")
teste = pd.read_csv(TESTE_FINAL)
teste["window_start"] = pd.to_datetime(teste["window_start"])

orig_hora = teste.groupby(teste["window_start"].dt.hour)["requests"].sum()
sint_hora = log.groupby(log["QueryTime"].dt.hour).size()

horas = list(range(24))
resultados.append(comparar(
    "Perfil diario (req. x hora do dia)",
    {h: float(orig_hora.get(h, 0.0)) for h in horas},
    {h: float(sint_hora.get(h, 0.0)) for h in horas},
))


# =========================
# DIA DA SEMANA: REQUISICOES POR DIA (K = 7)
# =========================
print("Comparando dia da semana...")
orig_dia = teste.groupby(teste["window_start"].dt.dayofweek)["requests"].sum()
sint_dia = log.groupby(log["QueryTime"].dt.dayofweek).size()

dias = list(range(7))
resultados.append(comparar(
    "Dia da semana (req. x dia)",
    {d: float(orig_dia.get(d, 0.0)) for d in dias},
    {d: float(sint_dia.get(d, 0.0)) for d in dias},
))


# =========================
# POSICAO DO CLIQUE (ITEMRANK), K = 5 FAIXAS
# =========================
print("Comparando posicao do clique (ItemRank)...")
itemrank = pd.read_csv(ITEMRANK_DIST)
# so contam as posicoes de clique de fato (ItemRank >= 1)
itemrank = itemrank[itemrank["ItemRank"] >= 1].copy()
itemrank["faixa"] = itemrank["ItemRank"].apply(faixa_itemrank)
orig_rank = somar_por_faixa(itemrank, "Frequency", "faixa", FAIXAS_ITEMRANK)

cliques = log[log["ItemRank"].notna()].copy()
cliques["faixa"] = cliques["ItemRank"].apply(faixa_itemrank)
sint_rank = contar_por_faixa(cliques["faixa"], FAIXAS_ITEMRANK)

resultados.append(comparar("Posicao do clique (ItemRank)", orig_rank, sint_rank))


# =========================
# PRESENCA DE CLIQUE: COM X SEM (K = 2)
# =========================
print("Comparando presenca de clique...")
hora_window = pd.read_csv(HORA_WINDOW)
orig_total = hora_window["requests"].sum()
orig_com = hora_window["clicks"].sum()
orig_clique = {"com": float(orig_com), "sem": float(orig_total - orig_com)}

sint_com = log["ItemRank"].notna().sum()
sint_clique = {"com": float(sint_com), "sem": float(len(log) - sint_com)}

resultados.append(comparar("Presenca de clique (com x sem)", orig_clique, sint_clique))


# =========================
# POPULARIDADE DAS CONSULTAS (K = 8 FAIXAS)
# =========================
print("Comparando popularidade das consultas...")
qfreq = pd.read_csv(QUERY_FREQ_DIST)
qfreq["faixa"] = qfreq["Frequency"].apply(faixa_popularidade)
orig_pop = somar_por_faixa(qfreq, "TotalRequestsRepresented", "faixa", FAIXAS_POPULARIDADE)

freq_sint = log["Query"].value_counts()
freq_df = pd.DataFrame({"Frequency": freq_sint.values})
freq_df["faixa"] = freq_df["Frequency"].apply(faixa_popularidade)
freq_df["requests"] = freq_df["Frequency"]
sint_pop = somar_por_faixa(freq_df, "requests", "faixa", FAIXAS_POPULARIDADE)

resultados.append(comparar("Popularidade das consultas", orig_pop, sint_pop))


# =========================
# SAIDA
# =========================
res_df = pd.DataFrame(resultados)
res_df.to_csv(SAIDA_CSV, index=False)
print(f"\nResultados salvos em: {SAIDA_CSV}")

linhas = []
linhas.append("Comparacao log sintetico x log original")
linhas.append("Qui-quadrado de aderencia + V de Cramer (efeito)")
linhas.append("=" * 70)
linhas.append("")
for r in resultados:
    linhas.append(f"{r['comparacao']}")
    linhas.append(f"  k = {r['k']} | gl = {r['gl']} | N = {r['N']:,}")
    linhas.append(f"  chi2 = {r['chi2']:.4g} | p = {r['p']:.4g}")
    linhas.append(f"  V de Cramer = {r['cramer_v']:.4f} ({r['efeito']})")
    linhas.append("")

texto = "\n".join(linhas)
SAIDA_TXT.write_text(texto, encoding="utf-8")
print(f"Resumo salvo em: {SAIDA_TXT}\n")
print(texto)
