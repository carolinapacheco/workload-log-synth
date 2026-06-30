import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
OUTPUT_DIR = AOL / "results" / "treino" / "anomalias_dbscan"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = AOL / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = AOL / "results" / "treino" / "perfis_kmeans" / "janelas_com_perfis_kmeans.csv"

# data já identificada como possível anomalia pelo kmeans
DATA_ANALISE = "2006-04-13"

EPS_PERCENTIL = 95

cluster_columns = [
    "requests",
    "unique_users",
    "unique_queries",
    "clicks",
    "click_rate",
    "mean_interarrival",
    "median_interarrival",
    "std_interarrival",
    "hour",
    "day_of_week",
    "is_weekend",
]


# =========================
# LEITURA DAS JANELAS
# =========================
features = pd.read_csv(INPUT_FILE)
features["window_start"] = pd.to_datetime(features["window_start"])
features["date"] = features["window_start"].dt.date

print(f"Total de janelas: {len(features)}")
print(f"Período: {features['window_start'].min()} até {features['window_start'].max()}")

X = features[cluster_columns]


# =========================
# NORMALIZAÇÃO
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# heurística comum: min_samples = 2 * número de variáveis
min_samples = 2 * len(cluster_columns)
print(f"min_samples utilizado: {min_samples}")


# =========================
# DISTÂNCIA ATÉ O K-ÉSIMO VIZINHO (para escolher eps)
# =========================
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)

k_distances = distances[:, -1]
k_distances_sorted = sorted(k_distances)

eps = pd.Series(k_distances).quantile(EPS_PERCENTIL / 100)
print(f"eps utilizado no DBSCAN (percentil {EPS_PERCENTIL}): {eps:.4f}")


# =========================
# GRÁFICO DE DISTÂNCIA AO K-ÉSIMO VIZINHO
# =========================
plt.figure(figsize=(10, 6))
plt.plot(k_distances_sorted)
plt.axhline(y=eps, linestyle="--", label=f"eps = {eps:.4f}")
plt.xlabel("Janelas ordenadas")
plt.ylabel(f"Distância até o {min_samples}º vizinho")
plt.title("Gráfico de distância para escolha de eps no DBSCAN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "grafico_k_distances_dbscan.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================
# TESTE DE DIFERENTES VALORES DE EPS
# =========================
print("\nTestando diferentes valores de eps...")

metricas_eps = []
for q in [0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99]:
    eps_teste = pd.Series(k_distances).quantile(q)
    labels_teste = DBSCAN(eps=eps_teste, min_samples=min_samples).fit_predict(X_scaled)

    qtd_ruido = (labels_teste == -1).sum()
    pct_ruido = qtd_ruido / len(labels_teste) * 100
    qtd_clusters = len(set(labels_teste) - {-1})

    metricas_eps.append({
        "eps": eps_teste,
        "min_samples": min_samples,
        "qtd_clusters": qtd_clusters,
        "qtd_ruido": qtd_ruido,
        "pct_ruido": pct_ruido,
    })
    print(f"eps={eps_teste:.4f} | clusters={qtd_clusters} | ruído={qtd_ruido} ({pct_ruido:.2f}%)")

metricas_eps_df = pd.DataFrame(metricas_eps)
metricas_eps_df.to_csv(OUTPUT_DIR / "metricas_eps_dbscan.csv", index=False)


# =========================
# DBSCAN FINAL
# =========================
print("\nAplicando DBSCAN final...")

features["cluster_dbscan"] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
features["is_ruido_dbscan"] = features["cluster_dbscan"] == -1

qtd_ruido = features["is_ruido_dbscan"].sum()
pct_ruido = qtd_ruido / len(features) * 100
clusters_finais = set(features["cluster_dbscan"]) - {-1}

print(f"Clusters encontrados (sem ruído): {len(clusters_finais)}")
print(f"Janelas marcadas como ruído: {qtd_ruido} ({pct_ruido:.2f}%)")


# =========================
# RESUMO DOS CLUSTERS DO DBSCAN
# =========================
resumo_dbscan = features.groupby("cluster_dbscan").agg(
    qtd_janelas=("window_start", "count"),
    requests_media=("requests", "mean"),
    requests_mediana=("requests", "median"),
    requests_min=("requests", "min"),
    requests_max=("requests", "max"),
    unique_users_media=("unique_users", "mean"),
    unique_queries_media=("unique_queries", "mean"),
    clicks_media=("clicks", "mean"),
    click_rate_media=("click_rate", "mean"),
    mean_interarrival_media=("mean_interarrival", "mean"),
    median_interarrival_media=("median_interarrival", "mean"),
    std_interarrival_media=("std_interarrival", "mean"),
).reset_index().sort_values("cluster_dbscan")

resumo_dbscan.to_csv(OUTPUT_DIR / "resumo_clusters_dbscan.csv", index=False)


# =========================
# COMPARAÇÃO ENTRE K-MEANS E DBSCAN
# =========================
comparacao = pd.crosstab(
    features["perfil_carga"],
    features["cluster_dbscan"],
    rownames=["perfil_carga_kmeans"],
    colnames=["cluster_dbscan"],
)
comparacao.to_csv(OUTPUT_DIR / "comparacao_kmeans_dbscan.csv")


# =========================
# ANÁLISE DA DATA ANÔMALA
# =========================
data_analise = pd.to_datetime(DATA_ANALISE).date()
janelas_data = features[features["date"] == data_analise]

print(f"\nAnálise da data {DATA_ANALISE}: {len(janelas_data)} janelas")
if len(janelas_data) > 0:
    qtd_ruido_data = janelas_data["is_ruido_dbscan"].sum()
    pct_ruido_data = qtd_ruido_data / len(janelas_data) * 100
    print(f"Janelas dessa data marcadas como ruído: {qtd_ruido_data} ({pct_ruido_data:.2f}%)")
    janelas_data.to_csv(OUTPUT_DIR / f"janelas_{DATA_ANALISE}_dbscan.csv", index=False)


# =========================
# DIAS COM MAIS RUÍDO
# =========================
anomalias_por_dia = features.groupby("date").agg(
    qtd_janelas=("window_start", "count"),
    qtd_ruido_dbscan=("is_ruido_dbscan", "sum"),
    requests_media=("requests", "mean"),
    click_rate_media=("click_rate", "mean"),
).reset_index()

anomalias_por_dia["pct_ruido_dbscan"] = anomalias_por_dia["qtd_ruido_dbscan"] / anomalias_por_dia["qtd_janelas"] * 100
anomalias_por_dia = anomalias_por_dia.sort_values(["qtd_ruido_dbscan", "pct_ruido_dbscan"], ascending=False)

anomalias_por_dia.to_csv(OUTPUT_DIR / "resumo_anomalias_por_dia.csv", index=False)


# =========================
# SALVA JANELAS COM RESULTADO DO DBSCAN
# =========================
features.to_csv(OUTPUT_DIR / "janelas_com_dbscan.csv", index=False)
features[features["is_ruido_dbscan"]].to_csv(OUTPUT_DIR / "janelas_ruido_dbscan.csv", index=False)


# =========================
# GRÁFICO: QUANTIDADE DE RUÍDOS POR DIA
# =========================
plt.figure(figsize=(12, 6))
plt.bar(anomalias_por_dia["date"].astype(str), anomalias_por_dia["qtd_ruido_dbscan"])
plt.xlabel("Data")
plt.ylabel("Quantidade de janelas marcadas como ruído")
plt.title("Quantidade de janelas anômalas por dia segundo DBSCAN")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "grafico_ruido_por_dia_dbscan.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================
# GRÁFICO: DISTRIBUIÇÃO DOS CLUSTERS DBSCAN POR HORA
# =========================
distribuicao_hora = features.groupby(["hour", "cluster_dbscan"]).size().reset_index(name="quantidade_janelas")
distribuicao_hora.to_csv(OUTPUT_DIR / "distribuicao_clusters_dbscan_por_hora.csv", index=False)

plt.figure(figsize=(12, 6))
for cluster in sorted(features["cluster_dbscan"].unique()):
    dados_cluster = distribuicao_hora[distribuicao_hora["cluster_dbscan"] == cluster]
    label = "Ruído (-1)" if cluster == -1 else f"Cluster {cluster}"
    plt.plot(dados_cluster["hour"], dados_cluster["quantidade_janelas"], marker="o", label=label)

plt.xlabel("Hora do dia")
plt.ylabel("Quantidade de janelas")
plt.title("Distribuição dos clusters DBSCAN por hora do dia")
plt.xticks(range(0, 24))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "grafico_distribuicao_clusters_dbscan_por_hora.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nProcesso concluído com sucesso.")
