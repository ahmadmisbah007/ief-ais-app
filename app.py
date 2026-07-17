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

# ==========================================
# 1. DATABASE BASELINE (RAPBS MPI)
# ==========================================
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
# 2. RUNNING ENGINE ML (CLUSTERING & KLASIFIKASI)
# ==========================================
# [PILAR 1] CLUSTERING: Pemetaan Karakteristik
X_cluster = data[['anggaran_diajukan', 'relevansi_mutu']]
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
data['klaster_anggaran'] = kmeans.fit_predict(X_cluster)

# [PILAR 2] KLASIFIKASI: Skrining Efisien vs Boros
X_classify = data[['persen_biaya_kantor', 'klaster_anggaran']]
y_label = data['status_efisiensi_historis']
classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
classifier.fit(X_classify, y_label)
data['rekomendasi_ai'] = classifier.predict(X_classify)
data['status_rekomendasi'] = data['rekomendasi_ai'].map({1: '✅ APPROVE', 0: '❌ REJECT (Red Flag)'})

# ==========================================
# 3. INTERAKSI SLIDER (INPUT DINAMIS)
# ==========================================
st.sidebar.header("🔮 PENGATURAN SIMULASI (Dinamis)")
siswa_simulasi = st.sidebar.slider("Proyeksi Jumlah Siswa:", min_value=100, max_value=600, value=350, step=10)

# ==========================================
# 4. [PILAR 3] REGRESI: Peramalan Anggaran & Unit Cost Dinamis
# ==========================================
# Membuat model prediksi kebutuhan total anggaran bersih berdasarkan jumlah siswa
X_reg = np.array([[100], [200], [350], [500]]) 
y_reg = np.array([[180000000], [350000000], [614000000], [870000000]]) 
model_regresi = LinearRegression()
model_regresi.fit(X_reg, y_reg)

# 1. Ramalan Anggaran Total Bersih berdasarkan slider siswa
anggaran_ramalan_total = model_regresi.predict(np.array([[siswa_simulasi]]))[0][0]

# 2. Hitung Rasio Perubahan Skala Anggaran (Dibandingkan baseline awal 350 siswa)
# Anggaran bersih riil saat ini (baseline 350 siswa) adalah Rp 614.000.000
rasio_skala = anggaran_ramalan_total / 614000000.0

# 3. Hitung UNIT COST DINAMIS secara real-time
unit_cost_dinamis = anggaran_ramalan_total / siswa_simulasi

# ==========================================
# 5. INTEGRASI TABEL SECARA DINAMIS
# ==========================================
# Mengalikan anggaran usulan dengan rasio skala siswa agar tabel ikut berubah dinamis
data_dinamis = data.copy()
data_dinamis['anggaran_diajukan_simulasi'] = (data_dinamis['anggaran_diajukan'] * rasio_skala).astype(int)

# Hitung ulang ringkasan dinamis untuk card metrik
total_awal_dinamis = int(data['anggaran_diajukan'].sum() * rasio_skala)
total_bersih_dinamis = int(anggaran_ramalan_total)
hemat_dinamis = total_awal_dinamis - total_bersih_dinamis

# ==========================================
# 6. TAMPILAN DASHBOARD UTAMA (LIVE UPDATE)
# ==========================================
st.write(f"### 📊 Status Simulasi Kapasitas: **{siswa_simulasi} Siswa**")

# Tampilan Visual Ringkasan Angka Utama (Dinamis Mengikuti Slider)
col1, col2, col3 = st.columns(3)
col1.metric("Proyeksi Pengajuan Awal", f"Rp {total_awal_dinamis:,}")
col2.metric("Proyeksi Anggaran Bersih AI", f"Rp {total_bersih_dinamis:,}")
col3.metric("Potensi Dana Diselamatkan", f"Rp {hemat_dinamis:,}")

st.markdown("---")
# UNIT COST DINAMIS: Nilai ini sekarang berubah setiap slider digeser!
st.success(f"🎯 ESTIMASI BIAYA NYATA PER SISWA (UNIT COST DINAMIS): Rp {int(unit_cost_dinamis):,} / Siswa per Tahun")
st.markdown("---")

# TAMPILAN ANALISIS REGRESI
st.write("### 🔮 Analisis Peramalan Tren Anggaran (Linear Regression)")
st.info(f"Berdasarkan tren historis, penambahan kapasitas menjadi **{siswa_simulasi} Siswa** diproyeksikan membutuhkan total anggaran operasional bersih sebesar **Rp {int(anggaran_ramalan_total):,}**.")
st.markdown("---")

# Tabel Detail Laporan Kepala Sekolah yang Nominal Anggarannya sudah Ter-skala
st.write("📋 **Rincian Usulan Anggaran Ter-Skala (Clustering & Klasifikasi):**")
st.dataframe(
    data_dinamis[['nama_kegiatan', 'anggaran_diajukan_simulasi', 'status_rekomendasi']]
    .rename(columns={'anggaran_diajukan_simulasi': f'Proyeksi Anggaran ({siswa_simulasi} Siswa)'}), 
    use_container_width=True
)
