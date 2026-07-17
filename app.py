import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="IEF-AIS Dashboard", layout="wide")

st.title("🚀 INTEGRATED EDUCATIONAL FINANCIAL AI SYSTEM (IEF-AIS)")
st.subheader("Sistem Skrining & Kalkulasi Unit Cost Berbasis TRIPLE-ML — Bidang MPI")
st.markdown("---")

# Data Internal Baseline (RAPBS MPI)
raw_data = {
    'nama_kegiatan': [
        'Gaji Guru & Staf Yayasan', 'Pengadaan Kitab Kuning & Buku Agama', 
        'Konsumsi Rapat Rutin Pengurus MPI', 'Operasional Listrik dan Internet Lab', 
        'Cetak Brosur PPDB & Spanduk', 'Wisuda Tahfidz & Khotmil Quran', 
        'Pelatihan Guru Metode Baghdadi/Utsmani', 'Pemeliharaan AC & Sarana Kelas', 
        'Perjalanan Dinas/Transport Yayasan', 'Pembelian Kertas & ATK Ujian'
    ],
    'anggaran_diajukan': [450000000, 25000000, 18000000, 36000000, 15000000, 45000000, 12000000, 16000000, 22000000, 30000000],
    'frekuensi_tahunan': [12, 1, 24, 12, 1, 1, 2, 4, 12, 2],
    'target_siswa_terdampak': [350, 350, 0, 350, 0, 150, 350, 350, 0, 350],
    'relevansi_mutu': [5, 5, 1, 4, 2, 5, 5, 3, 1, 4],
    'persen_biaya_kantor': [0.05, 0.00, 0.45, 0.10, 0.35, 0.08, 0.00, 0.12, 0.50, 0.15],
    'status_efisiensi_historis': [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
}
data = pd.DataFrame(raw_data)

# ==========================================
# [PILAR 1] CLUSTERING: Pemetaan Karakteristik
# ==========================================
X_cluster = data[['anggaran_diajukan', 'relevansi_mutu']]
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
data['klaster_anggaran'] = kmeans.fit_predict(X_cluster)

# ==========================================
# [PILAR 2] KLASIFIKASI: Skrining Efisien vs Boros
# ==========================================
X_classify = data[['persen_biaya_kantor', 'klaster_anggaran']]
y_label = data['status_efisiensi_historis']
classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
classifier.fit(X_classify, y_label)
data['rekomendasi_ai'] = classifier.predict(X_classify)
data['status_rekomendasi'] = data['rekomendasi_ai'].map({1: '✅ APPROVE', 0: '❌ REJECT (Red Flag)'})

# Perhitungan Kondisi Saat Ini
data_bersih = data[data['rekomendasi_ai'] == 1]
total_awal = data['anggaran_diajukan'].sum()
total_bersih = data_bersih['anggaran_diajukan'].sum()
hemat = total_awal - total_bersih
total_siswa_sekarang = 350
unit_cost = total_bersih / total_siswa_sekarang

# ==========================================
# [PILAR 3] REGRESI: Meramal Nominal Anggaran Masa Depan
# ==========================================
# Membuat model prediksi kebutuhan anggaran berdasarkan jumlah siswa
X_reg = np.array([[100], [200], [350], [500]]) # Data historis kapasitas siswa
y_reg = np.array([[180000000], [350000000], [614000000], [870000000]]) # Data anggaran bersih berkorelasi
model_regresi = LinearRegression()
model_regresi.fit(X_reg, y_reg)

# TAMPILAN INTERAKTIF UTAMA
col1, col2, col3 = st.columns(3)
col1.metric("Total Pengajuan Awal", f"Rp {total_awal:,}")
col2.metric("Anggaran Lolos Skrining AI", f"Rp {total_bersih:,}")
col3.metric("Dana Diselamatkan (Hemat)", f"Rp {hemat:,}", delta_color="inverse")

st.markdown("---")
st.success(f"🎯 BIAYA RIIL PER SISWA (UNIT COST BERSIH SAT INI): Rp {int(unit_cost):,} / Siswa per Tahun")
st.markdown("---")

# TAMPILAN PILAR REGRESI (SIMULASI MASA DEPAN)
st.sidebar.header("🔮 PILAR REGRESI (AI Forecasting)")
siswa_prediksi = st.sidebar.slider("Simulasi Jumlah Siswa Tahun Depan:", min_value=100, max_value=600, value=350, step=10)
anggaran_ramalan = model_regresi.predict(np.array([[siswa_prediksi]]))[0][0]

st.write("### 🔮 Peramalan Anggaran Masa Depan (Linear Regression)")
st.info(f"Jika jumlah siswa tahun depan dikunci di angka **{siswa_prediksi} Siswa**, maka prediksi kebutuhan nominal total anggaran bersih sekolah adalah **Rp {int(anggaran_ramalan):,}**")
st.markdown("---")

# Tabel Detail Laporan Kepala Sekolah
st.write("📋 **Daftar Rekomendasi Detail Kategori (Clustering & Klasifikasi AI):**")
st.dataframe(data[['nama_kegiatan', 'anggaran_diajukan', 'status_rekomendasi']], use_container_width=True)
