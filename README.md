# 📊 Segmentasi Pengguna E-Wallet dengan K-Means Clustering

Proyek ini melakukan segmentasi pengguna e-wallet menggunakan algoritma K-Means Clustering berdasarkan 6 variabel perilaku transaksi (X1–X6).

---

## 📁 Struktur Folder

```
project-ewallet/
├── app.py                  # Aplikasi Streamlit utama
├── data_transaksi.csv      # Dataset transaksi pengguna
├── requirements.txt        # Daftar library yang dibutuhkan
└── README.md               # File ini
```

---

## 🖥️ Cara Menjalankan di Localhost (VS Code)

### 1. Pastikan Python sudah terinstall
Cek di terminal:
```bash
python --version
```
Minimal Python 3.8.

### 2. Install semua library yang dibutuhkan
```bash
pip install -r requirements.txt
```

### 3. Jalankan aplikasi Streamlit
```bash
streamlit run app.py
```

### 4. Buka di browser
Setelah perintah di atas dijalankan, buka browser dan akses:
```
http://localhost:8501
```

### 5. Upload dataset
- Klik tombol **"Browse files"** di halaman web
- Pilih file `data_transaksi.csv`
- Aplikasi akan otomatis memproses dan menampilkan hasil

---

## 🚀 Cara Deploy ke Streamlit Cloud (Online)

### 1. Pastikan semua file sudah ada di GitHub
Repo kamu harus berisi:
- `app.py`
- `data_transaksi.csv`
- `requirements.txt`

### 2. Buka Streamlit Cloud
Kunjungi: [https://share.streamlit.io](https://share.streamlit.io)

### 3. Login dengan akun GitHub

### 4. Buat aplikasi baru
- Klik **"New app"**
- Isi form:
  - **Repository**: `BagasPermana0/project-ewallet`
  - **Branch**: `main`
  - **Main file path**: `app.py`
- Klik **"Deploy"**

### 5. Tunggu proses deploy (1–2 menit)
Setelah selesai, kamu akan mendapat link publik seperti:
```
https://project-ewallet.streamlit.app
```

---

## 📦 Library yang Digunakan

| Library | Fungsi |
|---|---|
| `pandas` | Membaca dan mengolah data |
| `numpy` | Komputasi numerik |
| `scikit-learn` | K-Means, PCA, Evaluasi klaster |
| `matplotlib` | Visualisasi grafik statis |
| `seaborn` | Visualisasi scatter plot |
| `plotly` | Visualisasi interaktif |
| `streamlit` | Antarmuka web |

---

## 📌 Variabel yang Digunakan

| Kode | Nama Variabel | Keterangan |
|---|---|---|
| X1 | Frekuensi_Mingguan | Frekuensi transaksi per minggu |
| X2 | Avg_Nilai_Transaksi | Rata-rata nilai transaksi (Rp) |
| X3 | Recency | Hari sejak transaksi terakhir |
| X4 | Jenis_Transaksi | Jenis transaksi (kategori) |
| X5 | Waktu_Dominan | Waktu paling sering bertransaksi |
| X6 | Lama_Penggunaan | Lama menggunakan e-wallet (bulan) |

---

## 👥 Tim Pengembang

Proyek ini dikembangkan sebagai bagian dari penelitian segmentasi pengguna e-wallet.
