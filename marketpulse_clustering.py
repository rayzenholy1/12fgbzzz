import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

np.random.seed(42)

centers = [(25, 20), (25, 80), (55, 50), (85, 20), (85, 80)]
parts = []
for c in centers:
    income = np.random.normal(c[0], 6, 60)
    spending = np.clip(np.random.normal(c[1], 8, 60), 1, 100)
    parts.append(pd.DataFrame({"annual_income": income, "spending_score": spending}))
df = pd.concat(parts, ignore_index=True)
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

print("Перші рядки даних:")
print(df.head())
print("\nОпис даних:")
print(df.describe())

X = StandardScaler().fit_transform(df)

ks = range(1, 11)
inertias = []
silhouettes = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    if k > 1:
        silhouettes.append(silhouette_score(X, labels))
    else:
        silhouettes.append(np.nan)

print("\nМетод ліктя та силует:")
for k, i, s in zip(ks, inertias, silhouettes):
    print(f"k = {k}: inertia = {i:.2f}, silhouette = {s:.4f}")

best_k = int(np.nanargmax(silhouettes)) + 1
print("\nОптимальна кількість кластерів:", best_k)

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
axs[0].plot(list(ks), inertias, marker="o", color="blue")
axs[0].axvline(best_k, color="red", linestyle="--", label=f"k = {best_k}")
axs[0].set_xlabel("Кількість кластерів k")
axs[0].set_ylabel("Inertia (WCSS)")
axs[0].set_title("Метод ліктя")
axs[0].legend()
axs[0].grid(True)
axs[1].plot(list(ks)[1:], silhouettes[1:], marker="s", color="green")
axs[1].axvline(best_k, color="red", linestyle="--", label=f"k = {best_k}")
axs[1].set_xlabel("Кількість кластерів k")
axs[1].set_ylabel("Silhouette score")
axs[1].set_title("Коефіцієнт силуету")
axs[1].legend()
axs[1].grid(True)
plt.tight_layout()
plt.show()

kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
km_labels = kmeans.fit_predict(X)

agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
agg_labels = agg.fit_predict(X)

comparison = pd.DataFrame({
    "Метод": ["K-Means", "Agglomerative (ward)"],
    "Silhouette": [round(silhouette_score(X, km_labels), 4), round(silhouette_score(X, agg_labels), 4)],
    "Davies-Bouldin": [round(davies_bouldin_score(X, km_labels), 4), round(davies_bouldin_score(X, agg_labels), 4)]
})
print("\nПорівняння методів:")
print(comparison.to_string(index=False))
print("Adjusted Rand Index між методами:", round(adjusted_rand_score(km_labels, agg_labels), 4))

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
axs[0].scatter(df["annual_income"], df["spending_score"], c=km_labels, cmap="viridis", edgecolor="black")
axs[0].set_xlabel("Річний дохід, тис. $")
axs[0].set_ylabel("Індекс витрат")
axs[0].set_title("K-Means")
axs[0].grid(True)
axs[1].scatter(df["annual_income"], df["spending_score"], c=agg_labels, cmap="viridis", edgecolor="black")
axs[1].set_xlabel("Річний дохід, тис. $")
axs[1].set_ylabel("Індекс витрат")
axs[1].set_title("Agglomerative Clustering")
axs[1].grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(11, 5))
dendrogram(linkage(X, method="ward"), truncate_mode="lastp", p=20)
plt.xlabel("Кластери (кількість об'єктів)")
plt.ylabel("Відстань")
plt.title("Дендрограма ієрархічної кластеризації")
plt.show()

df["segment"] = km_labels
profile = df.groupby("segment").agg(
    clients=("segment", "count"),
    avg_income=("annual_income", "mean"),
    avg_spending=("spending_score", "mean")
).round(1)
print("\nПрофіль сегментів (K-Means):")
print(profile)
