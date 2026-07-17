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
# 1. DATABASE BASELINE (STANDARISASI MINIMUM 100 SISWA)
# ==========================================
raw_data = {
    'nama_kegiatan': [
        'Gaji Guru & Staf Yayasan', 'Pengadaan Kitab Kuning & Buku Agama', 
        'Konsumsi Rapat Rutin Pengurus MPI', 'Operasional Listrik dan Internet Lab', 
        'Cetak Brosur PPDB & Spanduk', 'Wisuda Tahfidz & Khotmil Quran', 
        'Pelatihan Guru Metode Baghdadi/Utsmani', 'Pemeliharaan AC & Sarana Kelas', 
        'Perjalanan Dinas/Transport Yayasan', 'Pembelian Kertas & ATK Ujian'
    ],
    # Anggaran operasional dasar minimum untuk 100 siswa
    'anggaran_diajukan': [130000000, 7500000, 5000000, 11000000, 4500000, 13000000, 3500000, 5000000, 6500000, 9000000],
    'frekuensi_tahunan': [12, 1, 24, 12, 1, 1, 2, 4, 12, 2],
    'target_siswa_terdampak': [100, 100, 0, 100, 0, 40, 100, 100, 0, 100],
    'relevansi_mutu': [5, 5, 1, 4, 2, 5, 5, 3, 1, 4],
    'persen_biaya_kantor': [0.05, 0.00, 0.45, 0.10, 0.35, 0.08, 0.00, 0.12, 0.50, 0.15],
    'status_efisiensi_historis': [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
}
data = pd.DataFrame(raw_data)

# ==========================================
# 2. RUNNING ENGINE ML (CLUSTERING & KLASIFIKASI)
# ==========================================
X_cluster = data[['anggaran_diajukan', 'relevansi_mutu']]
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
data['klaster_anggaran'] = kmeans.fit_predict(X_cluster)

X_classify = data[['persen_biaya_kantor', 'klaster_anggaran']]
y_label = data['status_efisiensi_historis']
classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
classifier.fit(X_classify, y_label)
data['rekomendasi_ai'] = classifier.predict(X_classify)
data['status_rekomendasi'] = data['rekomendasi_ai'].map({1: '✅ APPROVE', 0: '❌ REJECT (Red Flag)'})

# Perhitungan Kondisi Baseline Riil (100 Siswa)
data_bersih = data[data['rekomendasi_ai'] == 1]
total_awal_100 = data['anggaran_diajukan'].sum()
total_bersih_100 = data_bersih['anggaran_diajukan'].sum() # Rp 184.000.000

# ==========================================
# 3. INTERAKSI SLIDER (INTERVAL 100 - 1000 SISWA)
# ==========================================
st.sidebar.header("🔮 PENGATURAN SIMULASI (Dinamis)")
# Rentang diubah menjadi 100 - 1000 dengan kelipatan 50
siswa_simulasi = st.sidebar.slider("Proyeksi Jumlah Siswa:", min_value=100, max_value=1000, value=100, step=50)

# ==========================================
# 4. [PILAR 3] REGRESI: Tren Kurva Dinamis & Efisiensi Skala
# ==========================================
# Menggunakan pola latih non-linear melengkung tipis (efisiensi skala)
X_reg = np.array([[100], [300], [500], [800], [1000]]) 
y_reg = np.array([[184000000], [480000000], [780000000], [1150000000], [1400000000]]) 
model_regresi = LinearRegression()
model_regresi.fit(X_reg, y_reg)

# 1. Ramalan anggaran total bersih berdasarkan slider
anggaran_ramalan_total = model_regresi.predict(np.array([[siswa_simulasi]]))[0][0]

# 2. Rasio skala terhadap baseline bersih 100 siswa
rasio_skala = anggaran_ramalan_total / 184000000.0

# 3. Hitung UNIT COST dinamis
unit_cost_dinamis = anggaran_ramalan_total / siswa_simulasi

# ==========================================
# 5. INTEGRASI TABEL SECARA DINAMIS
# ==========================================
data_dinamis = data.copy()
data_dinamis['anggaran_diajukan_simulasi'] = (data_dinamis['anggaran_diajukan'] * rasio_skala).astype(int)

total_awal_dinamis = int(data['anggaran_diajukan'].sum() * rasio_skala)
total_bersih_dinamis = int(anggaran_ramalan_total)
hemat_dinamis = total_awal_dinamis - total_bersih_dinamis

# ==========================================
# 6. TAMPILAN DASHBOARD UTAMA (LIVE UPDATE)
# ==========================================
st.write(f"### 📊 Status Simulasi Kapasitas: **{siswa_simulasi} Siswa**")

col1, col2, col3 = st.columns(3)
col1.metric("Proyeksi Pengajuan Awal", f"Rp {total_awal_dinamis:,}")
col2.metric("Proyeksi Anggaran Bersih AI", f"Rp {total_bersih_dinamis:,}")
col3.metric("Potensi Dana Diselamatkan", f"Rp {hemat_dinamis:,}")

st.markdown("---")
# Tampilan Unit Cost yang akan bergerak turun seiring bertambahnya siswa (Efisiensi Skala!)
st.success(f"🎯 ESTIMASI BIAYA NYATA PER SISWA (UNIT COST DINAMIS): Rp {int(unit_cost_dinamis):,} / Siswa per Tahun")
st.markdown("---")

st.write("### 🔮 Analisis Peramalan Tren Anggaran (Linear Regression)")
st.info(f"Berdasarkan kurva efisiensi skala ekonomi, simulasi untuk kapasitas **{siswa_simulasi} Siswa** membutuhkan estimasi total anggaran operasional bersih sebesar **Rp {int(anggaran_ramalan_total):,}**.")
st.markdown("---")

st.write("📋 **Rincian Usulan Anggaran Ter-Skala (Clustering & Klasifikasi):**")
st.dataframe(
    data_dinamis[['nama_kegiatan', 'anggaran_diajukan_simulasi', 'status_rekomendasi']]
    .rename(columns={'anggaran_diajukan_simulasi': f'Proyeksi Anggaran ({siswa_simulasi} Siswa)'}), 
    use_container_width=True
)
