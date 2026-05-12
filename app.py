import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Segmentasi E-Wallet", layout="wide")
st.title("Segmentasi Pengguna E-Wallet dengan K-Means")

# ==========================================
# 1. UPLOAD DATA
# ==========================================
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Upload file CSV", type="csv")

if uploaded_file is None:
    st.info("Silakan upload file data_transaksi.csv untuk memulai.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("Preview data:")
st.dataframe(df.head())

# ==========================================
# 2. DATA CLEANING
# ==========================================
st.header("2. Data Preprocessing")

df_clean = df.dropna().drop_duplicates().copy()
st.write(f"Jumlah data setelah cleaning: **{len(df_clean)} baris**")

# Label Encoding X4 dan X5
le = LabelEncoder()
df_clean['Jenis_Transaksi'] = le.fit_transform(df_clean['Jenis_Transaksi'])
df_clean['Waktu_Dominan']   = le.fit_transform(df_clean['Waktu_Dominan'])

# Normalisasi Min-Max
features = ['Frekuensi_Mingguan', 'Avg_Nilai_Transaksi', 'Recency',
            'Jenis_Transaksi', 'Waktu_Dominan', 'Lama_Penggunaan']

scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(df_clean[features])
st.success("Preprocessing selesai: Label Encoding + MinMax Scaling")

# ==========================================
# 3. CARI K OPTIMAL
# ==========================================
st.header("3. Penentuan K Optimal")

wcss, sil_scores, dbi_scores = [], [], []
rentang_k = range(2, 8)

for k in rentang_k:
    km     = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
    dbi_scores.append(davies_bouldin_score(X_scaled, labels))

k_optimal = list(rentang_k)[np.argmax(sil_scores)]
st.write(f"K optimal terpilih otomatis: **k = {k_optimal}**")

# Plot Elbow, Silhouette, DBI
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
st.pyplot(fig)

# ==========================================
# 4. K-MEANS FINAL
# ==========================================
st.header("4. Hasil Clustering")

model = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42, n_init=10)
df_clean['Cluster'] = model.fit_predict(X_scaled)

col1, col2 = st.columns(2)
col1.metric("Silhouette Score",     f"{silhouette_score(X_scaled, df_clean['Cluster']):.4f}")
col2.metric("Davies-Bouldin Index", f"{davies_bouldin_score(X_scaled, df_clean['Cluster']):.4f}")

# ==========================================
# 5. PCA + SCATTER PLOT INTERAKTIF
# ==========================================
st.header("5. Visualisasi Klaster (PCA)")

pca  = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_clean['PCA1'] = X_pca[:, 0]
df_clean['PCA2'] = X_pca[:, 1]

st.write(f"Variansi dijelaskan PCA: **{pca.explained_variance_ratio_.sum()*100:.1f}%**")

fig_plotly = px.scatter(
    df_clean,
    x='PCA1', y='PCA2',
    color=df_clean['Cluster'].astype(str),
    title=f'Sebaran Klaster (k={k_optimal})',
    labels={'color': 'Cluster'},
    hover_data=features
)
st.plotly_chart(fig_plotly, use_container_width=True)

# ==========================================
# 6. PROFILING KLASTER
# ==========================================
st.header("6. Profiling Tiap Klaster")
profiling = df_clean.groupby('Cluster')[features].mean().round(2)
st.dataframe(profiling)