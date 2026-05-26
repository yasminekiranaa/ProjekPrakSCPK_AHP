# ─────────────────────────── Library ───────────────────────────
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────── Konstanta AHP ───────────────────────────
RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

SAATY_VALUES = [1/9, 1/7, 1/5, 1/3,
                1, 3, 5, 7, 9]
SAATY_LABELS = ["1/9", "1/7", "1/5", "1/3",
                "1", "3", "5", "7", "9"]


# ─────────────────────────── Fungsi AHP ───────────────────────────────
def normalisasi_matriks(M: np.ndarray) -> np.ndarray:
    """Normalisasi kolom matriks perbandingan berpasangan."""
    return M / M.sum(axis=0)


def hitung_bobot(M: np.ndarray) -> np.ndarray:
    """Priority vector: rata-rata baris matriks ternormalisasi."""
    return normalisasi_matriks(M).mean(axis=1)


def hitung_cr(M: np.ndarray, bobot: np.ndarray):
    """Hitung lambda_max, CI, dan CR."""
    n = len(bobot)
    lambda_max = float((M @ bobot / bobot).mean())
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RI_TABLE.get(n, 1.49)
    CR = CI / ri if ri > 0 else 0.0
    return lambda_max, CI, CR


def skor_alternatif(df: pd.DataFrame, kriteria_cols: list,
                    bobot: np.ndarray, tipe_kriteria: list) -> pd.Series:
    """
    Normalisasi data alternatif per kolom, lalu bobot-kan.
    Benefit  → nilai tinggi = baik  (normalisasi sum biasa)
    Cost     → nilai rendah = baik  (invers sebelum normalisasi)
    """
    data = df[kriteria_cols].copy().astype(float)
    norm = pd.DataFrame(index=data.index, columns=kriteria_cols, dtype=float)

    for i, col in enumerate(kriteria_cols):
        vals = data[col]
        if tipe_kriteria[i] == "Cost":
            # hindari pembagian nol
            inv = 1.0 / vals.replace(0, np.nan).fillna(vals.min() * 0.01)
            norm[col] = inv / inv.sum()
        else:
            col_sum = vals.sum()
            norm[col] = vals / col_sum if col_sum != 0 else 0.0

    skor = (norm.values * bobot).sum(axis=1)
    return pd.Series(skor, index=df.index), norm


def saaty_label_to_val(label: str) -> float:
    idx = SAATY_LABELS.index(label)
    return SAATY_VALUES[idx]


# ─────────────────────────── Config Halaman ───────────────────────────
st.set_page_config(
    page_title="SPK AHP | SCPK 2025/2026",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>

/* ===== BACKGROUND APP ===== */
.stApp {

    background:
    radial-gradient(circle at top left, #2563eb 0%, transparent 30%),
    radial-gradient(circle at bottom right, #7c3aed 0%, transparent 30%),
    linear-gradient(135deg, #0f172a, #111827);

    background-attachment: fixed;
}

/* ===== CONTENT UTAMA ===== */
.main .block-container {

    background: rgba(255,255,255,0.04);

    backdrop-filter: blur(18px);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 24px;

    padding: 2rem;

    margin-top: 1rem;

    box-shadow:
        0 8px 32px rgba(0,0,0,0.25);
}


/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {

    background: rgba(15, 23, 42, 0.45);

    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);

    border-right: 1px solid rgba(255,255,255,0.08);

    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

/* ===== SEMUA TEKS ===== */
[data-testid="stSidebar"] * {
    color: white;
}

/* ===== RADIO GROUP ===== */
.stRadio > div {
    gap: 12px;
}

/* ===== MENU ===== */
.stRadio label {

    background: rgba(255,255,255,0.07);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 18px;

    padding: 14px;

    transition: all 0.25s ease;

    backdrop-filter: blur(12px);

    box-shadow:
        0 4px 12px rgba(0,0,0,0.15);
}

/* ===== HOVER ===== */
.stRadio label:hover {

    background: rgba(255,255,255,0.14);

    border: 1px solid rgba(255,255,255,0.18);

    transform: translateX(5px);

    box-shadow:
        0 8px 20px rgba(37,99,235,0.25);
}

/* ===== CARD BAWAH ===== */
.glass-card {

    background: rgba(255,255,255,0.08);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 20px;

    padding: 16px;

    backdrop-filter: blur(12px);

    box-shadow:
        0 4px 15px rgba(0,0,0,0.2);
}

/* ===== DIVIDER ===== */
hr {
    border-color: rgba(255,255,255,0.08);
}

/* ===== TITLE ===== */
h1, h2, h3 {
    color: white;
}

/* ===== HILANGKAN HEADER HITAM STREAMLIT ===== */
header {
    background: transparent !important;
}

/* ===== TOOLBAR KANAN ===== */
[data-testid="stToolbar"] {
    right: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Sidebar Nav ──────────────────────────────
with st.sidebar:

    st.image("analytics.png", width=95)

    st.markdown("""
    <h1 style='margin-bottom:0; color:white;'>
    SPK Investasi
    </h1>
    """, unsafe_allow_html=True)

    st.caption("Metode AHP")

    st.divider()

    halaman = st.radio(
        "",
        [
            "📂 Dataset",
            "📊 Hitung SPK",
            "👥 Profil Kelompok"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("""
    <div class="glass-card">
    <b>Proyek Akhir SCPK 2025/2026</b>
    <br>
    Sistem Pendukung Keputusan
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  HALAMAN 1 — DATA
# ══════════════════════════════════════════════════════════════════════
if halaman == "📂 Dataset":
    st.title("📁 Dataset")
    st.markdown("Upload dataset CSV untuk digunakan pada proses perhitungan AHP.")

    uploaded_files = st.file_uploader(
        "📤 Upload File CSV",
        type=["csv"],
        accept_multiple_files=True
    )

    if uploaded_files:

        list_df = []

        for file in uploaded_files:
            df_temp = pd.read_csv(file)
            df_temp["Sumber_File"] = file.name
            list_df.append(df_temp)

        df = pd.concat(list_df, ignore_index=True)

        st.session_state["df"] = df

        st.session_state["dataset_nama"] = ", ".join(
            [file.name for file in uploaded_files]
        )

        st.success(
            f"✅ Dataset berhasil diupload: "
            f"**{len(df):,} baris × {len(df.columns)} kolom**"
        )

        # tampilkan dataset
    if "df" in st.session_state :

        df = st.session_state["df"]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Jumlah Baris", f"{len(df):,}")
        col2.metric("Jumlah Kolom", len(df.columns))
        col3.metric(
            "Kolom Numerik",
            len(df.select_dtypes(include=np.number).columns)
        )
        col4.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        st.divider()

        st.subheader("📋 Tabel Dataset Mentah")
        st.dataframe(df, use_container_width=True, height=400)

        st.subheader("📊 Statistik Deskriptif")
        st.dataframe(
            df.describe().T.style.format("{:.3f}"),
            use_container_width=True
        )

        #     # validasi kolom
        # required_cols = [
        #     "Harga_Saham",
        #     "Return_Tahunan",
        #     "Risiko"
        # ]

        # missing_cols = [
        #     col for col in required_cols
        #     if col not in df.columns
        # ]

        # if missing_cols:
        #     st.warning(
        #         f"Kolom tidak ditemukan: {', '.join(missing_cols)}"
        #     )

# ══════════════════════════════════════════════════════════════════════
#  HALAMAN 2 — HITUNG SPK
# ══════════════════════════════════════════════════════════════════════
elif halaman == "📊 Hitung SPK":
    st.title("🧮 Sistem Pendukung Keputusan — Metode AHP")

    if "df" not in st.session_state:
        st.warning("⚠️ Dataset belum dimuat. Silakan ke halaman **📁 Data** terlebih dahulu.")
        st.stop()

    df = st.session_state["df"]
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # ── Konfigurasi ──────────────────────────────────────────────────
    st.subheader("⚙️ Konfigurasi")

    col_a, col_b = st.columns(2)
    with col_a:
        alt_col = st.selectbox(
            "Kolom Nama Alternatif",
            all_cols,
            help="Kolom yang berisi nama/identitas tiap alternatif (misal: nama produk, ID karyawan)"
        )
    with col_b:
        kriteria_cols = st.multiselect(
            "Kolom Kriteria (pilih min. 2, maks. 10)",
            num_cols,
            default=num_cols[:5] if len(num_cols) >= 5 else num_cols,
            help="Hanya kolom numerik yang bisa dijadikan kriteria"
        )

    if len(kriteria_cols) < 2:
        st.error("❗ Pilih minimal 2 kriteria untuk melanjutkan.")
        st.stop()
    if len(kriteria_cols) > 10:
        st.error("❗ Maksimal 10 kriteria (keterbatasan matriks AHP).")
        st.stop()

    n = len(kriteria_cols)

    # Tipe kriteria (widget interaktif wajib ≥ 3)
    st.subheader("🏷️ Tipe Kriteria")
    st.caption("**Benefit** = nilai besar lebih baik | **Cost** = nilai kecil lebih baik")
    tipe_cols = st.columns(min(n, 5))
    tipe_kriteria = []
    for i, kol in enumerate(kriteria_cols):
        with tipe_cols[i % 5]:
            t = st.selectbox(kol, ["Benefit", "Cost"], key=f"tipe_{kol}")
            tipe_kriteria.append(t)

    # # Jumlah alternatif teratas
    # n_alt = st.number_input(
    #     "Jumlah Alternatif yang Dievaluasi",
    #     min_value=5, max_value=min(500, len(df)),
    #     value=min(20, len(df)), step=5,
    #     help="Ambil N baris teratas dari dataset setelah drop missing values"
    # )

    st.divider()

    # ── Matriks Perbandingan Berpasangan ─────────────────────────────
    st.subheader("📊 Input Matriks Perbandingan Berpasangan Kriteria")
    st.caption(
        "Skala Saaty: **1** = Sama penting | **3** = Sedikit lebih penting | "
        "**5** = Lebih penting | **7** = Sangat penting | **9** = Mutlak lebih penting. "
        "Nilai pecahan (misal **1/3**) = kebalikannya."
    )

    # Inisialisasi session state untuk pairwise
    pw_key = "pairwise_vals"
    pair_count = n * (n - 1) // 2
    if pw_key not in st.session_state:
        st.session_state[pw_key] = {}

    pw_input = {}
    pair_idx = 0
    for i in range(n):
        cols_row = st.columns(n - i - 1) if n - i - 1 > 0 else []
        for j_idx, j in enumerate(range(i + 1, n)):
            key = f"pw_{i}_{j}"
            default_label = st.session_state[pw_key].get(key, "1")
            with (cols_row[j_idx] if cols_row else st.container()):
                label = f"**{kriteria_cols[i]}** vs **{kriteria_cols[j]}**"
                chosen = st.select_slider(
                    label,
                    options=SAATY_LABELS,
                    value=default_label,
                    key=key
                )
                pw_input[(i, j)] = saaty_label_to_val(chosen)
                st.session_state[pw_key][key] = chosen
            pair_idx += 1

    # Susun matriks M dari input
    M = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = pw_input[(i, j)]
            M[i, j] = v
            M[j, i] = 1.0 / v

    st.divider()

    # ── Tombol Eksekusi ───────────────────────────────────────────────
    hitung = st.button("🚀 Hitung SPK", type="primary", use_container_width=True)

    if hitung:
        # ── Step 1: Matriks Perbandingan ──────────────────────────────
        st.subheader("1️⃣  Matriks Perbandingan Berpasangan")
        df_M = pd.DataFrame(M, index=kriteria_cols, columns=kriteria_cols)
        st.dataframe(df_M.style.format("{:.4f}"), use_container_width=True)

        # ── Step 2: Normalisasi ────────────────────────────────────────
        norm_M = normalisasi_matriks(M)
        st.subheader("2️⃣  Matriks Normalisasi")
        df_norm = pd.DataFrame(norm_M, index=kriteria_cols, columns=kriteria_cols)
        st.dataframe(df_norm.style.format("{:.4f}"), use_container_width=True)

        # ── Step 3: Bobot (Priority Vector) ───────────────────────────
        bobot = hitung_bobot(M)
        st.subheader("3️⃣  Bobot Kriteria (Priority Vector)")
        df_bobot = pd.DataFrame({
            "Kriteria": kriteria_cols,
            "Bobot (w)": bobot,
            "Bobot (%)": bobot * 100
        })
        st.dataframe(
            df_bobot.style.format({"Bobot (w)": "{:.4f}", "Bobot (%)": "{:.2f}%"}),
            use_container_width=True, hide_index=True
        )

        # ── Step 4: Uji Konsistensi (CR) ──────────────────────────────
        lambda_max, CI, CR = hitung_cr(M, bobot)
        st.subheader("4️⃣  Uji Konsistensi")
        c1, c2, c3 = st.columns(3)
        c1.metric("λ_max", f"{lambda_max:.4f}")
        c2.metric("CI (Consistency Index)", f"{CI:.4f}")
        c3.metric("CR (Consistency Ratio)", f"{CR:.4f}")

        if CR <= 0.1:
            st.success(f"✅  CR = {CR:.4f} ≤ 0.10 → Matriks perbandingan **KONSISTEN**")
        else:
            st.error(
                f"❌  CR = {CR:.4f} > 0.10 → Matriks **TIDAK KONSISTEN**. "
                "Sesuaikan nilai perbandingan berpasangan agar lebih konsisten."
            )

        # ── Step 5: Penilaian Alternatif ──────────────────────────────
        st.subheader("5️⃣  Normalisasi Data Alternatif")
        df_eval = df.dropna(subset=kriteria_cols).copy()

        if len(df_eval) == 0:
            st.error("Tidak ada data setelah filter missing values.")
            st.stop()

        skor, df_norm_alt = skor_alternatif(df_eval, kriteria_cols, bobot, tipe_kriteria)
        df_norm_alt.index = df_eval[alt_col].values
        st.dataframe(df_norm_alt.style.format("{:.4f}"), use_container_width=True)

        # ── Step 6: Tabel Perangkingan ────────────────────────────────
        st.subheader("6️⃣  Tabel Hasil Perangkingan")
        df_result = df_eval[[alt_col] + kriteria_cols].copy()

        # reset index agar unik
        df_result = df_result.reset_index(drop=True)

        # hapus kolom duplikat kalau ada
        df_result = df_result.loc[:, ~df_result.columns.duplicated()]

        # tambah nilai preferensi
        df_result["Nilai Preferensi"] = skor.values

        # urutkan ranking
        df_result = (
            df_result
            .sort_values("Nilai Preferensi", ascending=False)
            .reset_index(drop=True)
        )

        # buat ranking baru
        df_result.insert(
            0,
            "Peringkat",
            range(1, len(df_result) + 1)
        )

        # Highlight top 3
        def highlight_top3(row):
            if row["Peringkat"] == 1:
                return ["background-color: gold"] * len(row)
            elif row["Peringkat"] == 2:
                return ["background-color: silver"] * len(row)
            elif row["Peringkat"] == 3:
                return ["background-color: #cd7f32; color:white"] * len(row)
            return [""] * len(row)

        styled_df = (
            df_result.style
            .apply(highlight_top3, axis=1)
            .format({"Nilai Preferensi": "{:.4f}"})
        )

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )

        # ── Grafik Hasil ──────────────────────────────────────────────
        st.subheader("📈 Grafik Hasil Perangkingan")

        top_n_chart = min(20, len(df_result))

        # ambil data teratas & hilangkan duplikat alternatif
        df_chart = (
            df_result.head(top_n_chart)
            .drop_duplicates(subset=[alt_col])
            .copy()
        )

        # ranking ulang setelah duplikat dihapus
        df_chart["Peringkat"] = range(1, len(df_chart) + 1)

        colors = []

        for rank in df_chart["Peringkat"]:

            if rank == 1:
                colors.append("#FFD700")   # emas

            elif rank == 2:
                colors.append("#C0C0C0")   # perak

            elif rank == 3:
                colors.append("#CD7F32")   # perunggu

            else:
                colors.append("#4A90D9")   # biru


        fig, ax = plt.subplots(
            figsize=(10, max(5, len(df_chart) * 0.5))
        )

        bars = ax.barh(
            df_chart[alt_col].astype(str),
            df_chart["Nilai Preferensi"],
            color=colors,
            edgecolor="white",
            linewidth=0.5
        )

        ax.invert_yaxis()

        ax.set_xlabel("Nilai Preferensi", fontsize=11)

        ax.set_title(
            f"Top {len(df_chart)} Alternatif Terbaik — Metode AHP",
            fontsize=13,
            fontweight="bold"
        )

        # label angka diperbaiki agar tidak bertumpuk
        ax.bar_label(
            bars,
            fmt="%.4f",
            padding=6,
            fontsize=9
        )

        ax.set_xlim(
            0,
            df_chart["Nilai Preferensi"].max() * 1.15
        )

        legend_items = [
            mpatches.Patch(color="#FFD700", label="🥇 Peringkat 1"),
            mpatches.Patch(color="#C0C0C0", label="🥈 Peringkat 2"),
            mpatches.Patch(color="#CD7F32", label="🥉 Peringkat 3"),
            mpatches.Patch(color="#4A90D9", label="Lainnya"),
        ]

        ax.legend(
            handles=legend_items,
            loc="lower right",
            fontsize=9
        )

        ax.grid(axis="x", alpha=0.3, linestyle="--")

        plt.tight_layout()

        st.pyplot(fig)

        # Kesimpulan
        st.success(
            f"🏆 **Alternatif Terbaik: {df_result.iloc[0][alt_col]}** "
            f"(Nilai Preferensi = {df_result.iloc[0]['Nilai Preferensi']:.4f})"
        )
        st.balloons()


# ══════════════════════════════════════════════════════════════════════
#  HALAMAN 3 — PROFIL KELOMPOK
# ══════════════════════════════════════════════════════════════════════
elif halaman == "👥 Profil Kelompok":
    st.title("👥 Profil Kelompok")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Anggota Kelompok")
        with st.container(border=True):
            st.markdown("""
**Anggota 1**
- **Nama:** Aisyah Yasmine Kirana
- **NIM:** 123240064

**Anggota 2**
- **Nama:** Selgy Pradista Risqi
- **NIM:** 123240074
""")

    with col2:
        st.subheader("📋 Informasi Proyek")
        with st.container(border=True):
            st.markdown("""
- **Mata Kuliah:** Sistem Pendukung Keputusan (SCPK)
- **Tahun Akademik:** 2025/2026
- **Metode SPK:** AHP (Analytical Hierarchy Process)
- **Judul Proyek:** SPK Penentuan Instrumen Investasi bagi Investor Pemula
- **Dataset:** Kaggle (CRYPTO DATASET & DATASET SAHAM INDONESIA - INDONESIA STOCK DATASET & CRYPTO DATASET.csv)
- **Kelas:** IF - C
""")

    st.divider()
    st.subheader("📖 Tentang Metode AHP")
    with st.expander("Klik untuk membaca penjelasan AHP", expanded=True):
        st.markdown("""
**Analytical Hierarchy Process (AHP)** dikembangkan oleh **Thomas L. Saaty (1970)**.  
Metode ini membantu pengambilan keputusan dengan cara memecah masalah kompleks menjadi hierarki yang terstruktur.

#### Tahapan Metode AHP:

| Tahap | Keterangan |
|-------|-----------|
| 1. Dekomposisi | Masalah dipecah menjadi tujuan, kriteria, dan alternatif |
| 2. Perbandingan Berpasangan | Setiap kriteria dibandingkan secara berpasangan menggunakan skala Saaty 1–9 |
| 3. Normalisasi Matriks | Matriks perbandingan dinormalisasi per kolom |
| 4. Priority Vector | Rata-rata baris matriks ternormalisasi = bobot tiap kriteria |
| 5. Uji Konsistensi | CR ≤ 0.10 → perbandingan dinyatakan konsisten |
| 6. Penilaian Alternatif | Data alternatif dinormalisasi, lalu dikalikan bobot |
| 7. Perangkingan | Alternatif diurutkan dari nilai preferensi tertinggi |

#### Skala Saaty:
| Nilai | Keterangan |
|-------|-----------|
| 1 | Sama penting |
| 3 | Sedikit lebih penting |
| 5 | Lebih penting |
| 7 | Sangat lebih penting |
| 9 | Mutlak lebih penting |
| 2, 4, 6, 8 | Nilai antara |
| 1/n | Kebalikan dari nilai n |
""")

    st.divider()
    st.subheader("🔗 Referensi")
    st.markdown("""
- Saaty, T. L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill.
- Sumber Dataset: [Kaggle]
- Streamlit: https://streamlit.io
""")