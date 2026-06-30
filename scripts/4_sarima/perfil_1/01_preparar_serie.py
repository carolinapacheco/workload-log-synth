import pandas as pd
from pathlib import Path

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")

INPUT_FILE = AOL / "results" / "treino" / "perfis_kmeans" / "janelas_com_perfis_kmeans.csv"

OUTPUT_DIR = AOL / "results" / "treino" / "sarima_perfil_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SERIE_COMPLETA = OUTPUT_DIR / "serie_perfil_1_com_zeros.csv"
OUTPUT_TREINO = OUTPUT_DIR / "serie_perfil_1_treino.csv"
OUTPUT_VALIDACAO = OUTPUT_DIR / "serie_perfil_1_validacao.csv"
OUTPUT_TESTE = OUTPUT_DIR / "serie_perfil_1_teste.csv"
OUTPUT_RESUMO = OUTPUT_DIR / "resumo_serie_perfil_1.txt"


def encontrar_coluna(df, opcoes):
    for col in opcoes:
        if col in df.columns:
            return col
    raise ValueError(f"Nenhuma das colunas esperadas foi encontrada: {opcoes}")


print("Preparando série temporal do perfil 1...")


# =========================
# LEITURA E PREPARO
# =========================
df = pd.read_csv(INPUT_FILE)

coluna_tempo = encontrar_coluna(df, ["window_start", "datetime", "QueryTime", "hora"])
coluna_requests = encontrar_coluna(df, ["requests", "Requests"])
coluna_perfil = encontrar_coluna(df, ["perfil_carga", "cluster", "profile", "kmeans_cluster"])

df[coluna_tempo] = pd.to_datetime(df[coluna_tempo])
df = df.sort_values(coluna_tempo)

print(f"Arquivo carregado: {INPUT_FILE}")
print(f"Coluna de tempo: {coluna_tempo}")
print(f"Coluna de requests: {coluna_requests}")
print(f"Coluna de perfil: {coluna_perfil}")

df = df[[coluna_tempo, coluna_requests, coluna_perfil]].copy()
df[coluna_tempo] = df[coluna_tempo].dt.floor("h")

df = (
    df.groupby(coluna_tempo, as_index=False)
    .agg({
        coluna_requests: "sum",
        coluna_perfil: "first"
    })
)

inicio = df[coluna_tempo].min()
fim = df[coluna_tempo].max()
indice_horario = pd.date_range(start=inicio, end=fim, freq="h")

df = df.set_index(coluna_tempo).reindex(indice_horario)
df.index.name = "window_start"

df[coluna_requests] = df[coluna_requests].fillna(0)
df[coluna_perfil] = df[coluna_perfil].fillna(-1)


# =========================
# SÉRIE DO PERFIL 1
# =========================
df["requests_perfil_1"] = df.apply(
    lambda row: row[coluna_requests] if int(row[coluna_perfil]) == 1 else 0,
    axis=1
)

df_saida = df.reset_index()[["window_start", "requests_perfil_1"]]

df_saida = df_saida[
    df_saida["window_start"].dt.date != pd.to_datetime("2006-05-17").date()
].copy()


# =========================
# DIVISÃO TREINO / VALIDAÇÃO / TESTE
# =========================
treino = df_saida[
    (df_saida["window_start"] >= "2006-03-01") &
    (df_saida["window_start"] <= "2006-04-30 23:00:00")
].copy()

validacao = df_saida[
    (df_saida["window_start"] >= "2006-05-01") &
    (df_saida["window_start"] <= "2006-05-16 23:00:00")
].copy()

teste = df_saida[
    (df_saida["window_start"] >= "2006-05-18") &
    (df_saida["window_start"] <= "2006-05-31 23:00:00")
].copy()

df_saida.to_csv(OUTPUT_SERIE_COMPLETA, index=False)
treino.to_csv(OUTPUT_TREINO, index=False)
validacao.to_csv(OUTPUT_VALIDACAO, index=False)
teste.to_csv(OUTPUT_TESTE, index=False)


# =========================
# RESUMO
# =========================
total_horas = len(df_saida)
horas_perfil_1 = (df_saida["requests_perfil_1"] > 0).sum()
horas_zero = (df_saida["requests_perfil_1"] == 0).sum()

resumo = f"""
Resumo da série temporal do perfil 1

Arquivo de entrada:
{INPUT_FILE}

Arquivo de saída:
{OUTPUT_SERIE_COMPLETA}

Período:
{df_saida["window_start"].min()} até {df_saida["window_start"].max()}

Total de observações horárias:
{total_horas}

Horas com ocorrência do perfil 1:
{horas_perfil_1}

Horas com valor zero:
{horas_zero}

Percentual de horas com perfil 1:
{horas_perfil_1 / total_horas * 100:.2f}%

Partições:
Treino: {treino["window_start"].min()} até {treino["window_start"].max()} | {len(treino)} observações
Validação: {validacao["window_start"].min()} até {validacao["window_start"].max()} | {len(validacao)} observações
Teste: {teste["window_start"].min()} até {teste["window_start"].max()} | {len(teste)} observações
"""

OUTPUT_RESUMO.write_text(resumo, encoding="utf-8")

print(resumo)
print("Série do perfil 1 preparada com sucesso.")
