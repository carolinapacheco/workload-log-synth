import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

AOL = next(p for p in Path(__file__).resolve().parents if p.name == "AOL Query Log")
OUTPUT_DIR = AOL / "results" / "treino" / "perfis_kmeans"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = AOL / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = AOL / "results" / "treino" / "features" / "features_janelas_horarias.csv"

K_ESCOLHIDO = 5

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
# LEITURA DAS FEATURES
# =========================
features = pd.read_csv(INPUT_FILE)
features["window_start"] = pd.to_datetime(features["window_start"])

print(f"Total de janelas: {len(features)}")
print(f"Período: {features['window_start'].min()} até {features['window_start'].max()}")

X = features[cluster_columns]


# =========================
# NORMALIZAÇÃO
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =========================
# TESTE DE DIFERENTES VALORES DE K
# =========================
print("\nTestando diferentes valores de k...")

metricas = []
for k in range(2, 11):
    modelo = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = modelo.fit_predict(X_scaled)

    inertia = modelo.inertia_
    silhouette = silhouette_score(X_scaled, labels)
    metricas.append({"k": k, "inertia": inertia, "silhouette": silhouette})

    print(f"k={k} | inertia={inertia:.2f} | silhouette={silhouette:.4f}")

metricas_df = pd.DataFrame(metricas)
metricas_df.to_csv(OUTPUT_DIR / "metricas_kmeans.csv", index=False)


# =========================
# GRÁFICO DO METO DO DO COTOVELO
# =========================
plt.figure(figsize=(10, 6))
plt.plot(metricas_df["k"], metricas_df["inertia"], marker="o")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inertia")
plt.title("Método do cotovelo para escolha de k")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "grafico_cotovelo_kmeans.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================
# GRÁFICO DO SILHOUETTE SCORE
# =========================
plt.figure(figsize=(10, 6))
plt.plot(metricas_df["k"], metricas_df["silhouette"], marker="o")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette score para diferentes valores de k")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "grafico_silhouette_kmeans.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================
# K-MEANS FINAL
# =========================
print(f"\nAplicando K-Means final com k={K_ESCOLHIDO}...")

modelo_final = KMeans(n_clusters=K_ESCOLHIDO, random_state=42, n_init=20)
features["perfil_carga"] = modelo_final.fit_predict(X_scaled)

joblib.dump(modelo_final, OUTPUT_DIR / "modelo_kmeans_treino.pkl")
joblib.dump(scaler, OUTPUT_DIR / "scaler_kmeans_treino.pkl")


# =========================
# RESUMO DOS PERFIS
# =========================
resumo_perfis = features.groupby("perfil_carga").agg(
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
).reset_index()

resumo_perfis = resumo_perfis.sort_values("requests_media")

print("\nResumo dos perfis encontrados:")
print(resumo_perfis)

features.to_csv(OUTPUT_DIR / "janelas_com_perfis_kmeans.csv", index=False)
resumo_perfis.to_csv(OUTPUT_DIR / "resumo_perfis_kmeans.csv", index=False)


# =========================
# GRÁFICO: VOLUME MÉDIO POR PERFIL
# =========================
plt.figure(figsize=(10, 6))
plt.bar(resumo_perfis["perfil_carga"].astype(str), resumo_perfis["requests_media"])
plt.xlabel("Perfil de carga")
plt.ylabel("Média de requisições por hora")
plt.title("Volume médio de requisições por perfil de carga")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "grafico_requests_media_por_perfil.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================
# GRÁFICO: USUÁRIOS MÉDIOS POR PERFIL
# =========================
plt.figure(figsize=(10, 6))
plt.bar(resumo_perfis["perfil_carga"].astype(str), resumo_perfis["unique_users_media"])
plt.xlabel("Perfil de carga")
plt.ylabel("Média de usuários distintos por hora")
plt.title("Usuários distintos médios por perfil de carga")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "grafico_usuarios_media_por_perfil.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================
# DISTRIBUIÇÃO DOS PERFIS POR HORA DO DIA
# =========================
distribuicao_hora = features.groupby(["hour", "perfil_carga"]).size().reset_index(name="quantidade_janelas")
distribuicao_hora.to_csv(OUTPUT_DIR / "distribuicao_perfis_por_hora.csv", index=False)

plt.figure(figsize=(12, 6))
for perfil in sorted(features["perfil_carga"].unique()):
    dados_perfil = distribuicao_hora[distribuicao_hora["perfil_carga"] == perfil]
    plt.plot(dados_perfil["hour"], dados_perfil["quantidade_janelas"], marker="o", label=f"Perfil {perfil}")

plt.xlabel("Hora do dia")
plt.ylabel("Quantidade de janelas")
plt.title("Distribuição dos perfis de carga por hora do dia")
plt.xticks(range(0, 24))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "grafico_distribuicao_perfis_por_hora.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nProcesso concluído com sucesso.")
