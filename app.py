import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# ==========================================
# 1. BACA DATA
# ==========================================
print("Membaca data...")
df = pd.read_csv('data_transaksi.csv')
display(df.head())

# ==========================================
# 2. DATA CLEANING
# ==========================================
df_clean = df.dropna().drop_duplicates().copy()
print(f"Jumlah data setelah cleaning: {len(df_clean)} baris")

# ==========================================
# 3. LABEL ENCODING (X4 dan X5)
# ==========================================
le = LabelEncoder()
df_clean['Jenis_Transaksi'] = le.fit_transform(df_clean['Jenis_Transaksi'])
df_clean['Waktu_Dominan']   = le.fit_transform(df_clean['Waktu_Dominan'])

# ==========================================
# 4. NORMALISASI MIN-MAX
# ==========================================
features = ['Frekuensi_Mingguan', 'Avg_Nilai_Transaksi', 'Recency',
            'Jenis_Transaksi', 'Waktu_Dominan', 'Lama_Penggunaan']

scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(df_clean[features])

# Simpan sebagai dataframe biar mudah dipakai nanti
df_scaled = pd.DataFrame(X_scaled, columns=features, index=df_clean.index)

# ==========================================
# 5. CARI K OPTIMAL
# ==========================================
print("\nMenghitung nilai untuk tiap k...")

wcss, sil_scores, dbi_scores = [], [], []
rentang_k = range(2, 8)

for k in rentang_k:
    km     = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
    dbi_scores.append(davies_bouldin_score(X_scaled, labels))

k_optimal = list(rentang_k)[np.argmax(sil_scores)]
print(f"K optimal terpilih: {k_optimal}")

# ==========================================
# 6. PLOT ELBOW, SILHOUETTE, DBI
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(rentang_k, wcss, marker='o', linestyle='--', color='blue')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('k')
axes[0].set_ylabel('WCSS')
axes[0].grid(True)

axes[1].plot(rentang_k, sil_scores, marker='o', linestyle='--', color='green')
axes[1].axvline(x=k_optimal, color='red', linestyle=':', label=f'k={k_optimal}')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Score')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(rentang_k, dbi_scores, marker='o', linestyle='--', color='orange')
axes[2].axvline(x=k_optimal, color='red', linestyle=':', label=f'k={k_optimal}')
axes[2].set_title('Davies-Bouldin Index')
axes[2].set_xlabel('k')
axes[2].set_ylabel('DBI')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# 7. K-MEANS FINAL
# ==========================================
print(f"\nMenjalankan K-Means dengan k={k_optimal}...")
model = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42, n_init=10)

df_scaled['Cluster']  = model.fit_predict(X_scaled)
df_clean['Cluster']   = df_scaled['Cluster']  # untuk scatter plot pakai nilai asli

# ==========================================
# 8. EVALUASI AKHIR
# ==========================================
print(f"Silhouette Score    : {silhouette_score(X_scaled, df_scaled['Cluster']):.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X_scaled, df_scaled['Cluster']):.4f}")

# ==========================================
# 9. PCA — Reduksi ke 2D untuk visualisasi
# ==========================================
pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_scaled['PCA1'] = X_pca[:, 0]
df_scaled['PCA2'] = X_pca[:, 1]

print(f"Variansi dijelaskan PCA: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ==========================================
# 10. SCATTER PLOT — Seaborn (statis)
# ==========================================
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_scaled, x='PCA1', y='PCA2',
                hue='Cluster', palette='viridis', s=100)
plt.title(f'Scatter Plot Klaster via PCA (k={k_optimal})')
plt.tight_layout()
plt.show()

# ==========================================
# 11. SCATTER PLOT — Plotly (interaktif)
# ==========================================
fig_plotly = px.scatter(
    df_scaled,
    x='PCA1', y='PCA2',
    color=df_scaled['Cluster'].astype(str),
    title=f'Sebaran Klaster Pengguna E-Wallet (k={k_optimal})',
    labels={'color': 'Cluster'},
    hover_data=features
)
fig_plotly.show()

# ==========================================
# 12. PROFILING TIAP KLASTER (nilai ternormalisasi 0-1)
# ==========================================
print("\nRata-rata tiap klaster (nilai ternormalisasi 0–1):")
profiling = df_scaled.groupby('Cluster')[features].mean().round(4)
display(profiling)
