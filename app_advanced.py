import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="IEF-AIS Advanced", layout="wide")

st.title("🚀 INTEGRATED EDUCATIONAL FINANCIAL AI SYSTEM (IEF-AIS) - ADVANCED")
st.subheader("Sistem Skrining & Kalkulasi Unit Cost Berbasis TRIPLE-ML — Bidang MPI")
st.markdown("---")

st.info("💡 **Petunjuk Penggunaan:** Silakan isi atau ubah nama kegiatan dan angka pada tabel di bawah sesuai kebutuhan lembaga Anda (Tersedia hingga 20 baris). Sistem AI akan langsung mengalkulasi ulang secara otomatis.")

# ==========================================
# 1. MEMBUAT DATABASE KOSONGAN / EDITABLE (20 ROWS)
# ==========================================
# Membuat 20 baris awal. Baris 1-3 diberi contoh isi agar AI punya data awal untuk belajar, sisanya kosongan.
init_rows = 20
nama_awal = ["Gaji Guru & Staf (Contoh)", "Buku & Kitab (Contoh)", "Listrik & Internet (Contoh)"] + ["(Silakan isi kegiatan di sini)"] * (init_rows - 3)
anggaran_awal = [130000000, 7500000, 11000000] + [0] * (init_rows - 3)
mutu_awal = [5, 5, 4] + [1] * (init_rows - 3)
kantor_awal = [0.05, 0.00, 0.10] + [0.00] * (init_rows - 3)

default_data = pd.DataFrame({
    'nama_kegiatan': nama_awal,
    'anggaran_diajukan': anggaran_awal,
    'relevansi_mutu': mutu_awal,
    'persen_biaya_kantor': kantor_awal
})

# Menampilkan tabel interaktif ala Excel di layar utama
st.write("### 📝 Input & Edit Rencana Anggaran Sekolah (Format Excel-Mini)")
edited_df = st.data_editor(
    default_data,
    column_config={
        "nama_kegiatan": st.column_config.TextColumn("Nama Kegiatan / Pos Anggaran", width="medium"),
        "anggaran_diajukan": st.column_config.NumberColumn("Anggaran Diajukan (Rp)", min_value=0, format="Rp %d"),
        "relevansi_mutu": st.column_config.NumberColumn("Relevansi Mutu Siswa (Skala 1-5)", min_value=1, max_value=5, help="1=Sangat Rendah, 5=Sangat Tinggi"),
        "persen_biaya_kantor": st.column_config.NumberColumn("Persen Biaya Kantor (Desimal)", min_value=0.0, max_value=1.0, format="%.2f", help="Contoh: 0.10 untuk 10% potongan kantor")
    },
    disabled=[],
    num_rows="fixed", # Mengunci di 20 baris agar rapi
    use_container_width=True
)

# Menyaring hanya baris yang diisi anggaran oleh user untuk diproses AI
data_aktif = edited_df[edited_df['anggaran_diajukan'] > 0].copy()

# Pengaman jika user belum mengisi anggaran sama sekali
if len(data_aktif) == 0:
    st.warning("⚠️ Silakan isi nilai 'Anggaran Diajukan' pada tabel di atas untuk mengaktifkan kalkulasi Engine AI.")
    st.stop()

# ==========================================
# 2. RUNNING ENGINE ML (CLUSTERING & KLASIFIKASI ADAPTIF)
# ==========================================
# Menentukan jumlah klaster secara fleksibel berdasarkan baris yang diisi
n_samples = len(data_aktif)
n_clusters = min(3, n_samples)

X_cluster = data_aktif[['anggaran_diajukan', 'relevansi_mutu']]
kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
data_aktif['klaster_anggaran'] = kmeans.fit_predict(X_cluster)

# Aturan filter otomatis untuk pelabelan training data
data_aktif['status_efisiensi_historis'] = np.where(
    (data_aktif['persen_biaya_kantor'] <= 0.20) & (data_aktif['relevansi_mutu'] >= 3), 1, 0
)

X_classify = data_aktif[['persen_biaya_kantor', 'klaster_anggaran']]
y_label = data_aktif['status_efisiensi_historis']

if len(y_label.unique()) > 1:
    classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
    classifier.fit(X_classify, y_label)
    data_aktif['rekomendasi_ai'] = classifier.predict(X_classify)
else:
    data_aktif['rekomendasi_ai'] = y_label

data_aktif['status_rekomendasi'] = data_aktif['rekomendasi_ai'].map({1: '✅ APPROVE', 0: '❌ REJECT (Red Flag)'})

# ==========================================
# 3. INTERAKSI SLIDER (INTERVAL 100 - 1000 SISWA)
# ==========================================
st.sidebar.header("🔮 PENGATURAN SIMULASI (Dinamis)")
siswa_simulasi = st.sidebar.slider("Proyeksi Jumlah Siswa:", min_value=100, max_value=1000, value=100, step=50)

# ==========================================
# 4. [PILAR 3] REGRESI: Tren Kurva Dinamis & Efisiensi Skala
# ==========================================
data_bersih = data_aktif[data_aktif['rekomendasi_ai'] == 1]
baseline_bersih = data_bersih['anggaran_diajukan'].sum() if len(data_bersih) > 0 else data_aktif['anggaran_diajukan'].sum()

# Melatih kurva regresi adaptif dari total anggaran yang diinput user
X_reg = np.array([[100], [300], [500], [800], [1000]])
y_reg = np.array([
    [baseline_bersih], 
    [baseline_bersih * 2.6], 
    [baseline_bersih * 4.2], 
    [baseline_bersih * 6.2], 
    [baseline_bersih * 7.6]
])
model_regresi = LinearRegression()
model_regresi.fit(X_reg, y_reg)

anggaran_ramalan_total = model_regresi.predict(np.array([[siswa_simulasi]]))[0][0]
rasio_skala = anggaran_ramalan_total / baseline_bersih if baseline_bersih > 0 else 1.0
unit_cost_dinamis = anggaran_ramalan_total / siswa_simulasi

# ==========================================
# 5. INTEGRASI HASIL SECARA DINAMIS
# ==========================================
data_dinamis = data_aktif.copy()
data_dinamis['anggaran_diajukan_simulasi'] = (data_dinamis['anggaran_diajukan'] * rasio_skala).astype(int)

total_awal_dinamis = int(data_aktif['anggaran_diajukan'].sum() * rasio_skala)
total_bersih_dinamis = int(anggaran_ramalan_total)
hemat_dinamis = total_awal_dinamis - total_bersih_dinamis

# ==========================================
# 6. TAMPILAN DASHBOARD UTAMA (LIVE UPDATE)
# ==========================================
st.markdown("---")
st.write(f"### 📊 Status Simulasi Kapasitas: **{siswa_simulasi} Siswa**")

col1, col2, col3 = st.columns(3)
col1.metric("Proyeksi Pengajuan Awal", f"Rp {total_awal_dinamis:,}")
col2.metric("Proyeksi Anggaran Bersih AI", f"Rp {total_bersih_dinamis:,}")
col3.metric("Potensi Dana Diselamatkan", f"Rp {hemat_dinamis:,}")

st.markdown("---")
st.success(f"🎯 ESTIMASI BIAYA NYATA PER SISWA (UNIT COST DINAMIS): Rp {int(unit_cost_dinamis):,} / Siswa per Tahun")
st.markdown("---")

st.write("📋 **Rincian Rekomendasi Hasil Skrining AI terhadap Data Input Anda:**")
st.dataframe(
    data_dinamis[['nama_kegiatan', 'anggaran_diajukan_simulasi', 'status_rekomendasi']]
    .rename(columns={'anggaran_diajukan_simulasi': f'Proyeksi Anggaran ({siswa_simulasi} Siswa)'}), 
    use_container_width=True
)
