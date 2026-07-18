import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="IEF-AIS Advanced Pro", layout="wide")

st.title("🚀 INTEGRATED EDUCATIONAL FINANCIAL AI SYSTEM (IEF-AIS) - ADVANCED")
st.subheader("Sistem Skrining & Kalkulasi Unit Cost Berbasis TRIPLE-ML — Bidang MPI")
st.markdown("---")

st.info("💡 **Petunjuk Penggunaan:** Silakan isi atau ubah nama kegiatan dan angka pada tabel di bawah sesuai kebutuhan lembaga Anda (Tersedia hingga 20 baris). Sistem AI akan langsung mengalkulasi ulang secara otomatis.")

# ==========================================
# 1. MEMBUAT DATABASE KOSONGAN / EDITABLE (20 ROWS)
# ==========================================
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

st.write("### 📝 Input & Edit Rencana Anggaran Sekolah (Format Excel-Mini)")
edited_df = st.data_editor(
    default_data,
    column_config={
        "nama_kegiatan": st.column_config.TextColumn("Nama Kegiatan / Pos Anggaran", width="medium"),
        "anggaran_diajukan": st.column_config.NumberColumn("Anggaran Diajukan (Rp)", min_value=0, format="Rp %d"),
        "relevansi_mutu": st.column_config.NumberColumn("Relevansi Mutu Siswa (Skala 1-5)", min_value=1, max_value=5),
        "persen_biaya_kantor": st.column_config.NumberColumn("Persen Biaya Kantor (Desimal)", min_value=0.0, max_value=1.0, format="%.2f")
    },
    disabled=[],
    num_rows="fixed",
    use_container_width=True
)

data_aktif = edited_df[edited_df['anggaran_diajukan'] > 0].copy()

if len(data_aktif) == 0:
    st.warning("⚠️ Silakan isi nilai 'Anggaran Diajukan' pada tabel di atas untuk mengaktifkan kalkulasi Engine AI.")
    st.stop()

# ==========================================
# 2. RUNNING ENGINE ML (CLUSTERING & KLASIFIKASI ADAPTIF)
# ==========================================
n_samples = len(data_aktif)
n_clusters = min(3, n_samples)

X_cluster = data_aktif[['anggaran_diajukan', 'relevansi_mutu']]
kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
data_aktif['klaster_anggaran'] = kmeans.fit_predict(X_cluster)

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
# 3. INTERAKSI SIDEBAR (DENGAN INPUT KESEJAHTERAAN GURU - FASE 3)
# ==========================================
st.sidebar.header("🔮 PENGATURAN SIMULASI (Dinamis)")
siswa_simulasi = st.sidebar.slider("Proyeksi Jumlah Siswa:", min_value=100, max_value=1000, value=100, step=50)

st.sidebar.markdown("---")
st.sidebar.header("💰 MODUL KESEJAHTERAAN GURU")
gaji_layak = st.sidebar.number_input("Target Honor Layak Guru / Bulan (Rp):", min_value=1000000, value=2500000, step=250000)
rasio_guru_siswa = st.sidebar.slider("Rasio Guru : Siswa (1 Guru banding X Siswa):", min_value=10, max_value=20, value=12)

# Hitung perkiraan jumlah guru berdasarkan kapasitas siswa simulasi
jumlah_guru_simulasi = max(1, int(siswa_simulasi / rasio_guru_siswa))
# Hitung total kebutuhan anggaran gaji setahun berdasarkan target kelayakan honor baru
total_gaji_layak_tahunan = jumlah_guru_simulasi * gaji_layak * 12

# ==========================================
# 4. [PILAR 3] REGRESI: Tren Kurva Dinamis & Efisiensi Skala
# ==========================================
data_bersih = data_aktif[data_aktif['rekomendasi_ai'] == 1]
baseline_bersih = data_bersih['anggaran_diajukan'].sum() if len(data_bersih) > 0 else data_aktif['anggaran_diajukan'].sum()

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

# ==========================================
# 5. INTEGRASI HASIL & KESEJAHTERAAN GURU
# ==========================================
data_dinamis = data_aktif.copy()
data_dinamis['anggaran_diajukan_simulasi'] = (data_dinamis['anggaran_diajukan'] * rasio_skala).astype(int)

# Ganti komponen gaji di tabel dinamis dengan angka simulasi kesejahteraan yang diinput di sidebar
is_gaji = data_dinamis['nama_kegiatan'].str.contains('Gaji|Guru|Staf|Honor', case=False, na=False)
if is_gaji.any():
    data_dinamis.loc[is_gaji, 'anggaran_diajukan_simulasi'] = total_gaji_layak_tahunan

# Hitung ulang total bersih akhir setelah disuntik anggaran kesejahteraan guru yang baru
total_awal_dinamis = int(data_aktif['anggaran_diajukan'].sum() * rasio_skala)
total_bersih_dinamis = int(data_dinamis[data_dinamis['rekomendasi_ai'] == 1]['anggaran_diajukan_simulasi'].sum())
hemat_dinamis = max(0, total_awal_dinamis - total_bersih_dinamis)

# Hitung Unit Cost dan SPP Bulanan Aktual
unit_cost_dinamis = total_bersih_dinamis / siswa_simulasi
spp_bulanan_ideal = unit_cost_dinamis / 12

# ==========================================
# 6. TAMPILAN DASHBOARD UTAMA & KARTU METRIK
# ==========================================
st.markdown("---")
st.write(f"### 📊 Status Simulasi Kapasitas: **{siswa_simulasi} Siswa**")

col1, col2, col3 = st.columns(3)
col1.metric("Proyeksi Pengajuan Awal", f"Rp {total_awal_dinamis:,}")
col2.metric("Proyeksi Anggaran Bersih AI", f"Rp {total_bersih_dinamis:,}")
col3.metric("Potensi Dana Diselamatkan", f"Rp {hemat_dinamis:,}")

st.markdown("---")
col_spp1, col_spp2, col_spp3 = st.columns(3)
with col_spp1:
    st.success(f"🎯 UNIT COST DINAMIS:\n\n**Rp {int(unit_cost_dinamis):,}** / Anak / Tahun")
with col_spp2:
    st.info(f"🕌 REKOMENDASI SPP BULANAN (BEP):\n\n**Rp {int(spp_bulanan_ideal):,}** / Anak / Bulan")
with col_spp3:
    st.warning(f"👥 KEBUTUHAN SDM GURU:\n\n**{jumlah_guru_simulasi} Orang Guru** (Rasio 1:{rasio_guru_siswa})")

st.markdown("---")

# ==========================================
# 7. MODUL VISUALISASI GRAFIK MEWAH
# ==========================================
st.write("### 📈 Visualisasi Analisis Keuangan Berbasis AI")
graph_col1, graph_col2 = st.columns(2)

with graph_col1:
    st.write("📋 **Komposisi Total Anggaran Berdasarkan Sensor AI (Rp)**")
    chart_data = data_dinamis.groupby('status_rekomendasi')['anggaran_diajukan_simulasi'].sum().reset_index()
    chart_data = chart_data.set_index('status_rekomendasi')
    st.bar_chart(chart_data, use_container_width=True)

with graph_col2:
    st.write("📉 **Kurva Tren Penurunan Unit Cost (Efisiensi Skala Ekonomi)**")
    rentang_siswa = list(range(100, 1050, 50))
    list_unit_cost = []
    for s in rentang_siswa:
        pred_total = model_regresi.predict(np.array([[s]]))[0][0]
        # Penyesuaian simulasi gaji di dalam kurva
        g_sim = max(1, int(s / rasio_guru_siswa)) * gaji_layak * 12
        total_gabungan = pred_total + g_sim
        list_unit_cost.append(int(total_gabungan / s))
    
    df_line = pd.DataFrame({
        'Jumlah Siswa': rentang_siswa,
        'Unit Cost (Rp)': list_unit_cost
    }).set_index('Jumlah Siswa')
    st.line_chart(df_line, use_container_width=True)

st.markdown("---")
st.write("📋 **Rincian Rekomendasi Hasil Skrining AI terhadap Data Input Anda:**")
st.dataframe(
    data_dinamis[['nama_kegiatan', 'anggaran_diajukan_simulasi', 'status_rekomendasi']]
    .rename(columns={'anggaran_diajukan_simulasi': f'Proyeksi Anggaran ({siswa_simulasi} Siswa)'}), 
    use_container_width=True
)
