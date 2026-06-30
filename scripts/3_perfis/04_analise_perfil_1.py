import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
PERFIS_KMEANS_DIR = AOL / "results" / "treino" / "perfis_kmeans"
INPUT_FILE = PERFIS_KMEANS_DIR / "janelas_com_perfis_kmeans.csv"
OUTPUT_DIR = PERFIS_KMEANS_DIR / "graficos_perfil_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = AOL / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PERFIL_ANALISADO = 1
BINS_HISTOGRAMA = 20

FEATURES_NUMERICAS = [
    "requests",
    "unique_users",
    "unique_queries",
    "clicks",
    "click_rate",
    "mean_interarrival",
    "median_interarrival",
    "std_interarrival",
]

NOMES_FEATURES = {
    "requests": "Requisições por hora",
    "unique_users": "Usuários distintos por hora",
    "unique_queries": "Consultas distintas por hora",
    "clicks": "Cliques por hora",
    "click_rate": "Taxa de cliques",
    "mean_interarrival": "Intervalo médio entre requisições",
    "median_interarrival": "Mediana do intervalo entre requisições",
    "std_interarrival": "Desvio padrão do intervalo entre requisições",
}


def salvar(nome_arquivo):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / nome_arquivo, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# LEITURA E FILTRO DO PERFIL
# =========================
df = pd.read_csv(INPUT_FILE)
df_perfil = df[df["perfil_carga"] == PERFIL_ANALISADO]

print(f"Total de janelas do Perfil {PERFIL_ANALISADO}: {len(df_perfil)}")


# =========================
# RESUMO ESTATÍSTICO
# =========================
estatisticas = df_perfil[FEATURES_NUMERICAS].describe().T
estatisticas["mediana"] = df_perfil[FEATURES_NUMERICAS].median()
estatisticas["variancia"] = df_perfil[FEATURES_NUMERICAS].var()
estatisticas["coeficiente_variacao"] = df_perfil[FEATURES_NUMERICAS].std() / df_perfil[FEATURES_NUMERICAS].mean()
estatisticas.to_csv(OUTPUT_DIR / f"resumo_estatistico_perfil_{PERFIL_ANALISADO}.csv", index=True)


# =========================
# HISTOGRAMAS E BOXPLOTS POR FEATURE
# =========================
print("Gerando histogramas e boxplots...")
for coluna in FEATURES_NUMERICAS:
    dados = df_perfil[coluna].dropna()
    contagens, limites = np.histogram(dados, bins=BINS_HISTOGRAMA)

    plt.figure(figsize=(12, 6))
    plt.bar(limites[:-1], contagens, width=np.diff(limites), align="edge", edgecolor="black", color="#1f77b4")

    if dados.max() >= 100:
        rotulos = [f"{v:.0f}" for v in limites]
    else:
        rotulos = [f"{v:.2f}" for v in limites]
    plt.xticks(limites, rotulos, rotation=45, ha="right")

    plt.title(f"Distribuição de {NOMES_FEATURES.get(coluna, coluna)} - Perfil {PERFIL_ANALISADO}")
    plt.xlabel(NOMES_FEATURES.get(coluna, coluna))
    plt.ylabel("Quantidade de janelas")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    salvar(f"histograma_{coluna}_perfil_{PERFIL_ANALISADO}.png")

    plt.figure(figsize=(8, 6))
    plt.boxplot(df_perfil[coluna].dropna(), vert=True)
    plt.title(f"Boxplot de {NOMES_FEATURES.get(coluna, coluna)} - Perfil {PERFIL_ANALISADO}")
    plt.ylabel(NOMES_FEATURES.get(coluna, coluna))
    plt.xticks([1], [f"Perfil {PERFIL_ANALISADO}"])
    salvar(f"boxplot_{coluna}_perfil_{PERFIL_ANALISADO}.png")


# =========================
# OCORRÊNCIA POR HORA DO DIA
# =========================
print("Gerando gráficos temporais...")
contagem_por_hora = df_perfil["hour"].value_counts().reindex(range(24), fill_value=0)

plt.figure(figsize=(12, 6))
plt.bar(contagem_por_hora.index, contagem_por_hora.values, color="#1f77b4")
for hora, valor in zip(contagem_por_hora.index, contagem_por_hora.values):
    if valor > 0:
        plt.text(hora, valor + 0.5, str(int(valor)), ha="center", va="bottom", fontsize=12)
plt.title(f"Ocorrência do Perfil {PERFIL_ANALISADO} por hora do dia")
plt.xlabel("Hora do dia")
plt.ylabel("Quantidade de janelas")
plt.xticks(range(0, 24))
plt.ylim(0, contagem_por_hora.max() * 1.12)
plt.grid(True, axis="y", linestyle="--", alpha=0.4)
salvar(f"ocorrencia_por_hora_perfil_{PERFIL_ANALISADO}.png")


# =========================
# OCORRÊNCIA POR DIA DA SEMANA
# =========================
dias_map = {0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta", 4: "Sexta", 5: "Sábado", 6: "Domingo"}
contagem_por_dia = df_perfil["day_of_week"].value_counts().reindex(range(7), fill_value=0)
labels = [dias_map[dia] for dia in contagem_por_dia.index]

plt.figure(figsize=(10, 6))
plt.bar(labels, contagem_por_dia.values, color="#1f77b4")
for x, valor in enumerate(contagem_por_dia.values):
    if valor > 0:
        plt.text(x, valor + 0.8, str(int(valor)), ha="center", va="bottom", fontsize=12)
plt.title(f"Ocorrência do Perfil {PERFIL_ANALISADO} por dia da semana")
plt.xlabel("Dia da semana")
plt.ylabel("Quantidade de janelas")
plt.xticks(rotation=30)
plt.ylim(0, contagem_por_dia.max() * 1.12)
plt.grid(True, axis="y", linestyle="--", alpha=0.4)
salvar(f"ocorrencia_por_dia_semana_perfil_{PERFIL_ANALISADO}.png")


# =========================
# MÉDIA DE CADA FEATURE POR HORA
# =========================
medias_por_hora = df_perfil.groupby("hour")[FEATURES_NUMERICAS].mean().sort_index()
for coluna in FEATURES_NUMERICAS:
    plt.figure(figsize=(10, 6))
    plt.plot(medias_por_hora.index, medias_por_hora[coluna], marker="o")
    plt.title(f"Média de {NOMES_FEATURES.get(coluna, coluna)} por hora - Perfil {PERFIL_ANALISADO}")
    plt.xlabel("Hora do dia")
    plt.ylabel(NOMES_FEATURES.get(coluna, coluna))
    plt.xticks(range(0, 24))
    plt.grid(True)
    salvar(f"media_por_hora_{coluna}_perfil_{PERFIL_ANALISADO}.png")


# =========================
# VARIAÇÃO RELATIVA DAS FEATURES (coeficiente de variação)
# =========================
print("Gerando gráfico de variação relativa...")
medias = df_perfil[FEATURES_NUMERICAS].mean()
desvios = df_perfil[FEATURES_NUMERICAS].std()

resumo = pd.DataFrame({
    "feature": FEATURES_NUMERICAS,
    "media": medias.values,
    "desvio": desvios.values,
})
resumo["coeficiente_variacao"] = resumo["desvio"] / resumo["media"]

plt.figure(figsize=(12, 7))
plt.bar([NOMES_FEATURES.get(f, f) for f in resumo["feature"]], resumo["coeficiente_variacao"], color="#1f77b4")
for x, valor in enumerate(resumo["coeficiente_variacao"]):
    plt.text(x, valor + 0.005, f"{valor:.2f}", ha="center", va="bottom", fontsize=12)
plt.title(f"Variação relativa das features - Perfil {PERFIL_ANALISADO}")
plt.xlabel("Feature")
plt.ylabel("Coeficiente de variação")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, resumo["coeficiente_variacao"].max() * 1.15)
plt.grid(True, axis="y", linestyle="--", alpha=0.4)
salvar(f"variacao_relativa_features_perfil_{PERFIL_ANALISADO}.png")

print(f"\nProcesso concluído! Gráficos salvos em: {OUTPUT_DIR}")
