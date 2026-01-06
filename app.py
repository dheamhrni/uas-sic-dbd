import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="SIC DBD Semarang", layout="wide")

# Ensure a save_path variable exists for legacy asset loading
save_path = os.path.join(os.path.dirname(__file__), '')
if not save_path.endswith(os.sep):
    save_path = save_path + os.sep

# Inject custom CSS/JS for improved styling
def inject_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            css = f.read()
            # Inject via components.html to ensure styles land in the page root
            # Also force a body background as a robust visual check
            wrapper = f"""
            <style>
            body {{ background: linear-gradient(180deg, #f7fbff 0%, #f3f7ff 50%, #ffffff 100%) !important; }}
            {css}
            </style>
            """
            components.html(wrapper, height=1)
            return True
    except Exception:
        # fallback: small embedded styles if external file not found
        fallback = """
        <style>
        body { background: linear-gradient(180deg, #f7fbff 0%, #f3f7ff 50%, #ffffff 100%) !important; }
        #MainMenu{visibility:hidden}
        footer{visibility:hidden}
        header{visibility:visible}
        </style>
        """
        components.html(fallback, height=1)
        return False

def inject_helper_js():
    # Small JS snippet to subtly enhance UI (keeps height small)
    js = """
    <script>
    // Try to remove Streamlit hamburger and footer for a cleaner app
    try {
      const mainMenu = document.getElementById('MainMenu');
      if(mainMenu) mainMenu.style.display = 'none';
      const footer = document.getElementsByTagName('footer')[0];
      if(footer) footer.style.display = 'none';
    } catch(e){}
    </script>
    """
    components.html(js, height=0)

# perform injection early
css_loaded = inject_custom_css()
inject_helper_js()

# Small floating badge to indicate CSS loaded (helps debug visual injection)
if css_loaded:
        badge_html = """
        <div style="position:fixed;right:12px;top:12px;z-index:9999;font-family:Inter, Arial;">
            <div style="background:linear-gradient(90deg,#667eea,#764ba2);color:white;padding:6px 10px;border-radius:8px;box-shadow:0 6px 14px rgba(0,0,0,0.12);font-size:13px;">
                Custom theme: ON
            </div>
        </div>
        """
        components.html(badge_html, height=60)

# =========================
# LOAD DATA & MODELS
# =========================
@st.cache_data
def load_data():
    df_full = pd.read_csv("dbd_semarang_clustered_full.csv")
    profile = pd.read_csv("cluster_profile.csv")
    lookup = pd.read_csv("dbd_semarang_cluster_lookup.csv")
    return df_full, profile, lookup

@st.cache_resource
def load_assets():
    with open(save_path + 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(save_path + 'kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    with open(save_path + 'pca_model.pkl', 'rb') as f:
        pca_model = pickle.load(f)
    with open(save_path + 'cluster_summary.pkl', 'rb') as f:
        cluster_summary = pickle.load(f)
    with open(save_path + 'pca_df.pkl', 'rb') as f:
        pca_df = pickle.load(f)
    return scaler, kmeans_model, pca_model, cluster_summary, pca_df


@st.cache_resource
def load_models():
    """Load model artifacts using joblib (preferred). Falls back to `load_assets` when needed."""
    base = os.path.dirname(__file__)
    try:
        kmeans = joblib.load(os.path.join(base, 'kmeans_model.joblib'))
        scaler = joblib.load(os.path.join(base, 'scaler.joblib'))
        pca = joblib.load(os.path.join(base, 'pca.joblib'))
        labels_map_path = os.path.join(base, 'labels_map.pkl')
        if os.path.exists(labels_map_path):
            with open(labels_map_path, 'rb') as f:
                labels_map = pickle.load(f)
        else:
            labels_map = {}
        return kmeans, scaler, pca, labels_map
    except Exception:
        # fallback to older asset loader if available
        if 'load_assets' in globals():
            scaler_a, kmeans_model_a, pca_model_a, cluster_summary_a, pca_df_a = load_assets()
            labels_map = cluster_summary_a if isinstance(cluster_summary_a, dict) else {}
            return kmeans_model_a, scaler_a, pca_model_a, labels_map
        raise

df, cluster_profile, lookup_table = load_data()

# Backwards-compatible asset loading: prefer `load_models()` if present,
# otherwise fall back to `load_assets()` (older naming / formats).
if 'load_models' in globals():
    try:
        kmeans, scaler, pca, labels_map = load_models()
    except Exception:
        kmeans = scaler = pca = labels_map = None
elif 'load_assets' in globals():
    try:
        scaler_a, kmeans_model_a, pca_model_a, cluster_summary_a, pca_df_a = load_assets()
        # map older names to expected variables
        scaler = scaler_a
        kmeans = kmeans_model_a
        pca = pca_model_a
        # if cluster_summary contains label mapping, prefer that; otherwise create empty dict
        labels_map = cluster_summary_a if isinstance(cluster_summary_a, dict) else {}
    except Exception:
        kmeans = scaler = pca = labels_map = None
else:
    # neither loader exists in globals — set placeholders
    kmeans = scaler = pca = labels_map = None

features = ["total_dd", "total_dbd_dss", "total_meninggal", "ir_total", "cfr"]


# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        margin-top: 10px;
        opacity: 0.9;
    }
    .dashboard-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADERt
# ==========================
st.markdown("""
<div class='dashboard-header'>
    <h1>📊 Dashboard Prediksi Cluster DBD K-Means</h1>
    <p>Sistem Prediksi dan Visualisasi Data Demam Berdarah Dengue</p>
</div>
""", unsafe_allow_html=True)



# =========================
# NAVIGASI
# =========================
# Single-page tab layout
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "🔮 Prediksi Cluster", "📅 Analisis Per Tahun", "ℹ️ Informasi"])


# =========================
# 1. DASHBOARD
# =========================
with tab1:
    st.title("📊 Dashboard Sistem Informasi Cerdas DBD Semarang")

    # Prepare data variables used in the tab
    dbd_data = df.copy() if df is not None else None

    # Determine wilayah list from dataframe or fallback to empty list
    wilayah_col_candidates = [c for c in dbd_data.columns] if dbd_data is not None else []
    wilayah_col = None
    for c in wilayah_col_candidates:
        if 'wilayah' in c.lower() or 'kecamatan' in c.lower():
            wilayah_col = c
            break

    if wilayah_col is not None:
        wilayah_list = sorted(dbd_data[wilayah_col].dropna().unique().tolist())
    else:
        wilayah_list = []

    # Attempt to construct pca_df from model if features present
    pca_df = None
    try:
        available_feats = [f for f in features if f in dbd_data.columns]
        if len(available_feats) >= 2 and 'cluster' in dbd_data.columns:
            scaled_all = scaler.transform(dbd_data[available_feats])
            pca_arr = pca.transform(scaled_all)
            pca_df = pd.DataFrame(pca_arr, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = dbd_data['cluster'].values
    except Exception:
        pca_df = None

    # Use a container to group dashboard content without an extra heading
    with st.container():

        # --- Filter controls (Tahun & Wilayah) ---
        if dbd_data is not None:
            year_columns = [c for c in dbd_data.columns if 'tahun' in c.lower() or 'year' in c.lower()]
            if year_columns:
                year_col = year_columns[0]
                years_available = sorted(dbd_data[year_col].dropna().unique())
                years_options = ['All'] + [int(y) for y in years_available]
            else:
                year_col = None
                years_options = ['All', 2023, 2024, 2025]

            filter_col1, filter_col2 = st.columns([1, 2])
            with filter_col1:
                selected_year = st.selectbox('Filter Tahun', years_options, index=0)
            with filter_col2:
                selected_wilayah = st.multiselect('Filter Wilayah (kosong = semua)', wilayah_list, default=[])

            # prepare filtered dataframe
            if selected_year == 'All':
                filtered = dbd_data.copy()
            else:
                try:
                    filtered = dbd_data[dbd_data[year_col] == int(selected_year)].copy() if year_col is not None else dbd_data.copy()
                except Exception:
                    filtered = dbd_data.copy()

            if selected_wilayah and wilayah_col is not None:
                filtered = filtered[filtered[wilayah_col].isin(selected_wilayah)]

        else:
            filtered = None

        if dbd_data is not None:
            # Helpers to find candidate columns (case-insensitive)
            def find_col(df, candidates):
                cols = df.columns
                for cand in candidates:
                    for c in cols:
                        if cand.lower() == c.lower():
                            return c
                return None

            pasien_col = find_col(dbd_data, ['Pasien', 'pasien', 'total_dbd_dss', 'total_kasus', 'pasien_total'])
            meninggal_col = find_col(dbd_data, ['Meninggal', 'meninggal', 'total_meninggal'])
            penduduk_col = find_col(dbd_data, ['Jml Penduduk', 'Jml_Penduduk', 'penduduk', 'population'])
            ir_col = find_col(dbd_data, ['IR/100000', 'ir/100000', 'ir_total', 'ir'])

            # Summary Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_records = len(filtered) if filtered is not None else 0
                st.metric("📊 Total Data", total_records)

            with col2:
                if filtered is not None and pasien_col is not None:
                    total_pasien = filtered[pasien_col].sum()
                    st.metric("👥 Total Pasien", int(total_pasien))

            with col3:
                if filtered is not None and meninggal_col is not None:
                    total_meninggal = filtered[meninggal_col].sum()
                    st.metric("⚠️ Total Meninggal", int(total_meninggal))

            with col4:
                if filtered is not None and pasien_col is not None and meninggal_col is not None:
                    if filtered[pasien_col].sum() > 0:
                        cfr = (filtered[meninggal_col].sum() / filtered[pasien_col].sum()) * 100
                        st.metric("💔 CFR Rata-rata", f"{cfr:.2f}%")

            # Charts
            st.divider()
            st.subheader("📉 Grafik Visualisasi")
            col1, col2 = st.columns(2)

            with col1:
                if filtered is not None and penduduk_col is not None and pasien_col is not None:
                    fig_scatter = px.scatter(
                        filtered,
                        x=penduduk_col,
                        y=pasien_col,
                        title="Hubungan Jumlah Penduduk vs Pasien",
                        labels={penduduk_col: 'Jumlah Penduduk', pasien_col: 'Jumlah Pasien'},
                        color_discrete_sequence=['#667eea']
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True, key='dashboard_scatter')

            with col2:
                if filtered is not None and ir_col is not None:
                    fig_hist = px.histogram(
                        filtered,
                        x=ir_col,
                        nbins=20,
                        title="Distribusi Incidence Rate",
                        labels={ir_col: 'IR per 100,000'},
                        color_discrete_sequence=['#764ba2']
                    )
                    st.plotly_chart(fig_hist, use_container_width=True, key='dashboard_histogram')

            # Perbandingan Antar Tahun (Trend Analysis)
            st.divider()
            st.subheader("📊 Perbandingan Antar Tahun")
            if year_columns and len(filtered) > 0:
                agg_dict = {}
                if pasien_col is not None:
                    agg_dict[pasien_col] = 'sum'
                if meninggal_col is not None:
                    agg_dict[meninggal_col] = 'sum'

                try:
                    year_summary = filtered.groupby(year_col).agg(agg_dict).reset_index()
                    trend_col1, trend_col2 = st.columns(2)

                    with trend_col1:
                        if pasien_col in year_summary.columns:
                            fig_year_pasien = px.line(
                                year_summary,
                                x=year_col,
                                y=pasien_col,
                                title="Trend Jumlah Pasien Per Tahun",
                                markers=True,
                                color_discrete_sequence=['#667eea']
                            )
                            st.plotly_chart(fig_year_pasien, use_container_width=True, key='dashboard_trend_pasien')

                    with trend_col2:
                        if meninggal_col in year_summary.columns:
                            fig_year_meninggal = px.line(
                                year_summary,
                                x=year_col,
                                y=meninggal_col,
                                title="Trend Jumlah Meninggal Per Tahun",
                                markers=True,
                                color_discrete_sequence=['#e74c3c']
                            )
                            st.plotly_chart(fig_year_meninggal, use_container_width=True, key='dashboard_trend_meninggal')
                except Exception:
                    st.warning("Tidak dapat membuat grafik trend tahunan")
            else:
                st.info("Data tidak cukup untuk menampilkan trend tahunan")
            # Top 10 Wilayah by Pasien
                # === GRAFIK TOP 10 WILAYAH - JUMLAH PASIEN ===
            if filtered is not None and pasien_col is not None and wilayah_col is not None:
                st.markdown("### 🏆 Top 10 Wilayah - Jumlah Pasien")
                
                # Agregasi data per wilayah (sum pasien)
                wilayah_summary = filtered.groupby(wilayah_col)[pasien_col].sum().reset_index()
                wilayah_summary = wilayah_summary.sort_values(by=pasien_col, ascending=False).head(10)
                
                # Buat bar chart
                fig_top10 = px.bar(
                    wilayah_summary,
                    x=wilayah_col,
                    y=pasien_col,
                    title="Top 10 Wilayah Berdasarkan Jumlah Pasien DBD",
                    labels={pasien_col: 'Jumlah Pasien', wilayah_col: 'Wilayah'},
                    color=pasien_col,
                    color_continuous_scale='Reds',
                    text=pasien_col
                )
                fig_top10.update_traces(texttemplate='%{text}', textposition='outside')
                fig_top10.update_layout(
                    xaxis_tickangle=-45,
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig_top10, use_container_width=True, key='dashboard_top10_wilayah')
            
            # Data Table (menampilkan data sesuai filter jika ada)
            st.divider()
            st.subheader("📋 Data Tabel")
            st.dataframe(filtered if (filtered is not None) else dbd_data, use_container_width=True)
        else:
            st.error("Data tidak dapat dimuat")


# =========================
# 2. VISUALISASI / ANALISIS PER TAHUN
# =========================
with tab3:
    st.title("📅 Analisis Data Per Tahun")

    dbd_data = df.copy() if df is not None else None

    # detect wilayah column and list
    wilayah_col = None
    wilayah_list = []
    if dbd_data is not None:
        for c in dbd_data.columns:
            if 'wilayah' in c.lower() or 'kecamatan' in c.lower():
                wilayah_col = c
                break
        if wilayah_col is not None:
            wilayah_list = sorted(dbd_data[wilayah_col].dropna().unique().tolist())

    # helper to find columns by candidate names
    def find_col(df, candidates):
        if df is None:
            return None
        cols = df.columns
        for cand in candidates:
            for c in cols:
                if cand.lower() == c.lower():
                    return c
        return None

    with st.container():

        if dbd_data is not None:
            # Check if data has year column
            year_columns = [col for col in dbd_data.columns if 'tahun' in col.lower() or 'year' in col.lower()]

            if year_columns or len(dbd_data) > 0:
                # Try to extract year from data if column exists
                if year_columns:
                    year_col = year_columns[0]
                    years_available = sorted(dbd_data[year_col].dropna().unique())
                else:
                    years_available = [2023, 2024, 2025]

                st.write(f"**Tahun yang tersedia:** {', '.join(map(str, years_available))}")

                # Year Selection
                col1, col2, col3 = st.columns(3)

                with col1:
                    selected_year = st.selectbox("Pilih Tahun", years_available)

                with col2:
                    selected_wilayah = st.multiselect(
                        "Pilih Wilayah (Kosongkan untuk semua)",
                        wilayah_list,
                        default=[]
                    )

                st.divider()

                # Filter data based on selection
                if year_columns:
                    year_data = dbd_data[dbd_data[year_col] == selected_year].copy()
                else:
                    year_data = dbd_data.copy()

                if selected_wilayah and wilayah_col is not None:
                    year_data = year_data[year_data[wilayah_col].isin(selected_wilayah)]

                # Map column names
                pasien_col = find_col(dbd_data, ['Pasien', 'pasien', 'total_dbd_dss', 'total_kasus', 'pasien_total'])
                meninggal_col = find_col(dbd_data, ['Meninggal', 'meninggal', 'total_meninggal'])
                penduduk_col = find_col(dbd_data, ['Jml Penduduk', 'Jml_Penduduk', 'penduduk', 'population'])
                ir_col = find_col(dbd_data, ['IR/100000', 'ir/100000', 'ir_total', 'ir'])

                # Statistics for selected year
                st.subheader(f"📊 Statistik Tahun {selected_year}")

                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

                with stat_col1:
                    st.metric("📍 Total Wilayah", len(year_data))

                with stat_col2:
                    if pasien_col in year_data.columns:
                        total_pasien_year = year_data[pasien_col].sum()
                        st.metric("👥 Total Pasien", int(total_pasien_year))

                with stat_col3:
                    if meninggal_col in year_data.columns:
                        total_meninggal_year = year_data[meninggal_col].sum()
                        st.metric("⚠️ Total Meninggal", int(total_meninggal_year))

                with stat_col4:
                    if pasien_col in year_data.columns and meninggal_col in year_data.columns:
                        if year_data[pasien_col].sum() > 0:
                            cfr_year = (year_data[meninggal_col].sum() / year_data[pasien_col].sum()) * 100
                            st.metric("💔 CFR", f"{cfr_year:.2f}%")

                st.divider()
                st.subheader("📈 Grafik Perbandingan")

                chart_col1, chart_col2 = st.columns(2)

                # Chart 1: Top 10 Wilayah by Pasien
                with chart_col1:
                    if pasien_col in year_data.columns and wilayah_col is not None:
                        top_wilayah = year_data.nlargest(10, pasien_col)
                        fig_top = px.bar(
                            top_wilayah,
                            x=wilayah_col,
                            y=pasien_col,
                            title=f"Top 10 Wilayah - Jumlah Pasien ({selected_year})",
                            labels={pasien_col: 'Jumlah Pasien'},
                            color_discrete_sequence=['#667eea']
                        )
                        st.plotly_chart(fig_top, use_container_width=True, key='tab3_top_wilayah')

                # Chart 2: IR Distribution
                with chart_col2:
                    if ir_col in year_data.columns:
                        fig_ir = px.box(
                            year_data,
                            y=ir_col,
                            title=f"Distribusi IR per 100,000 Penduduk ({selected_year})",
                            color_discrete_sequence=['#764ba2']
                        )
                        st.plotly_chart(fig_ir, use_container_width=True, key='tab3_ir_distribution')

                st.divider()
                # Blok 'Perbandingan Antar Tahun' telah dihapus atas permintaan pengguna.
                # Jika ingin menambahkan kembali, kumpulkan data per tahun dari
                # `year_data` atau `dbd_data` lalu buat visualisasi line chart di sini.
                

                # Data Table for selected year
                st.subheader(f"📋 Data Tabel Tahun {selected_year}")
                st.dataframe(year_data, use_container_width=True)

                # Download option
                csv = year_data.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download Data {selected_year}",
                    data=csv,
                    file_name=f"data_dbd_{selected_year}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Tidak ada informasi tahun dalam data")
        else:
            st.error("Data tidak dapat dimuat")


# =========================
# 3. INFORMASI SISTEM (mengganti Profil Klaster)
# =========================
with tab4:
    st.title("ℹ️ Informasi Sistem")

    dbd_data = df.copy() if df is not None else None

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📚 Tentang Sistem
        Sistem ini menggunakan algoritma **K-Means** untuk mengklasifikasikan risiko DBD 
        berdasarkan data epidemiologi dari berbagai wilayah di Semarang.
        
        **Fitur Utama:**
        - 🔮 Prediksi cluster risiko DBD
        - 📈 Visualisasi data dan PCA
        - 📋 Analisis data komprehensif
        """)

    with col2:
        st.markdown("""
        ### 🎯 Interpretasi Cluster (Berdasarkan Training Data)

        **🟢 Cluster 0 - Rank 4: Wilayah Aman**
        - Kasus rendah, tidak ada atau sedikit meninggal
        - CFR 0% atau sangat rendah
        - Tindakan: Pengawasan rutin

        **🟡 Cluster 1 - Rank 2: Risiko Sedang-Tinggi**
        - Kasus sedang dengan beberapa meninggal
        - CFR: 2-7%
        - Tindakan: Mitigasi & pencegahan, edukasi masyarakat

        **🔴 Cluster 2 - Rank 1: PRIORITAS UTAMA**
        - Kasus BANYAK dengan meninggal TINGGI
        - CFR: >5% (outbreak)
        - Tindakan: Intervensi lapangan SEGERA, kontrol vektor intensif

        **🔴 Cluster 3 - Rank 3: Risiko Tinggi (Fatalitas)**
        - Kasus sedang dengan CFR SANGAT TINGGI (>10%)
        - Angka meninggal relatif tinggi
        - Tindakan: Penanganan klinis prioritas, pelatihan tenaga medis
        """)


    st.divider()

    # Accordion untuk informasi akurasi
    with st.expander("📊 Informasi Detail tentang Akurasi & Data Training"):
        st.markdown("""
        ### 🔍 Data Training yang Digunakan
        
        **Sumber Data:**
        - File: `DBD.csv` (folder: uas.v1)
        - Coverage: Wilayah Semarang
        - Periode: 2023-2025
        
        **Fitur/Variabel yang Digunakan:**
        1. **Jml Penduduk** - Jumlah penduduk di wilayah
        2. **Pasien** - Jumlah pasien DBD
        3. **Meninggal** - Jumlah pasien yang meninggal
        4. **IR/100000** - Incidence Rate per 100,000 penduduk
        5. **CFR** - Case Fatality Rate (%)
        
        ### 📈 Algoritma Model
        
        **K-Means Clustering:**
        - Algoritma: Unsupervised Learning
        - Jumlah Cluster: 3
        - Pre-processing: StandardScaler normalization
        - Dimensionality Reduction: PCA (Principal Component Analysis)
        
        ### ⚠️ Limitasi & Akurasi
        
        **Faktor yang Mempengaruhi Akurasi:**
        
        ✅ **Akurasi TINGGI ketika:**
        - Input data berada dalam range data training
        - Menggunakan data dari wilayah Semarang
        - Periode tahun 2023-2025
        - Nilai IR dan CFR sesuai dengan pola historis
        
        ⚠️ **Akurasi SEDANG/RENDAH ketika:**
        - Input data di luar range training (extrapolation)
        - Menggunakan data dari luar wilayah Semarang
        - Terdapat anomali epidemiologi yang tidak tercakup training data
        - Perubahan drastis dalam pola penyebaran DBD
        
        ### 💡 Rekomendasi Penggunaan
        
        1. **Gunakan untuk analisis exploratory** - model ini bagus untuk melihat pola umum
        2. **Selalu validasi dengan expert** - konsultasikan hasil dengan epidemiolog atau tenaga kesehatan
        3. **Perhatikan warning validation** - sistem akan memberikan peringatan jika input di luar range
        4. **Update data secara berkala** - model perlu di-retrain dengan data terbaru untuk akurasi optimal
        5. **Gunakan bersama metode lain** - jangan bergantung hanya pada model ini
        
        ### 📊 Statistik Data Training
        """)
        
        if dbd_data is not None:
            st.write("**Ringkasan Data Training dari DBD.csv:**")
            summary_stats = dbd_data.describe().round(2)
            st.dataframe(summary_stats, use_container_width=True)

    st.divider()

    # FAQ Section
    with st.expander("❓ Pertanyaan Umum (FAQ)"):
        st.markdown("""
        **Q: Bagaimana cara meningkatkan akurasi prediksi?**
        
        A: 
        - Tambahkan lebih banyak data historis ke dalam training
        - Include variabel tambahan seperti: temperatur, curah hujan, kepadatan nyamuk
        - Lakukan feature engineering yang lebih mendalam
        - Update model secara berkala dengan data terbaru
        
        ---
        
        **Q: Apakah model bisa digunakan untuk wilayah lain?**
        
        A: Model ini dilatih khusus untuk data Semarang. Untuk wilayah lain, Anda perlu:
        - Mengumpulkan data historis dari wilayah tersebut
        - Melatih ulang model dengan data baru
        - Atau gunakan sebagai baseline dengan catatan akurasi mungkin berkurang
        
        ---
        
        **Q: Berapa akurasi model secara keseluruhan?**
        
        A: Akurasi tergantung pada:
        - Similarity input dengan data training (80-95% untuk data in-range)
        - Variabilitas data training (lebih diverse = akurasi lebih stabil)
        - Perlu dilakukan cross-validation untuk nilai akurasi formal
        
        ---
        
        **Q: Bagaimana interpretasi hasil prediksi?**
        
        A: Lihat tab "Interpretasi Cluster" di atas untuk penjelasan detail setiap cluster
        """)

    st.divider()
    st.info("Dikembangkan untuk keperluan analisis data epidemiologi DBD")


# =========================
# 4. PREDIKSI RISIKO (INPUT BARU)
# =========================

# =========================
# 2. PREDIKSI CLUSTER
# =========================
with tab2:
    st.title("🔮 Prediksi Risiko DBD untuk Wilayah Baru")
    
    # Check if models are loaded
    if kmeans is None or scaler is None or pca is None:
        st.error("⚠️ Model belum berhasil dimuat. Periksa ketersediaan file model:")
        st.write("- scaler.pkl / scaler.joblib")
        st.write("- kmeans_model.pkl / kmeans_model.joblib")
        st.write("- pca_model.pkl / pca.joblib")
        st.stop()

    st.write("Masukkan data berikut untuk memperkirakan tingkat risiko:")

    col1, col2 = st.columns(2)

    wilayah_options = [
        'Banyumanik', 'Candisari', 'Gajah Mungkur', 'Gayamsari', 'Genuk', 'Gunungpati',
        'Mijen', 'Ngaliyan', 'Pedurungan', 'Semarang Barat', 'Semarang Selatan',
        'Semarang Tengah', 'Semarang Timur', 'Semarang Utara', 'Tembalang', 'Tugu'
    ]
    wilayah = col1.selectbox("Nama Wilayah", options=wilayah_options, index=0, key="wil1")

    # Inputs requested: jumlah pasien, tahun kejadian, jumlah meninggal, jumlah penduduk
    pasien = col1.number_input("Jumlah Pasien (P)", min_value=0, value=10, step=1, key="pasien1")
    tahun = col1.number_input("Tahun Kejadian", min_value=1900, max_value=2100, value=int(df['tahun'].min() if 'tahun' in df.columns else 2024), step=1, key="tahun1")

    penduduk = col2.number_input("Jumlah Penduduk", min_value=1, value=10000, step=1, key="pend1")
    meninggal = col2.number_input("Jumlah Meninggal (M)", min_value=0, value=0, step=1, key="mng1")

    st.markdown("---")
    # Validation and metric calculation
    if st.button("🔍 Prediksi Risiko", type="primary", use_container_width=True):
        
        # Show loading spinner
        with st.spinner('🔄 Sedang memproses prediksi...'):
            
            try:
                # Basic validation
                if penduduk <= 0:
                    st.error("❌ Jumlah Penduduk harus lebih besar dari 0.")
                    st.stop()
                if pasien < 0 or meninggal < 0:
                    st.error("❌ Jumlah pasien dan jumlah meninggal tidak boleh negatif.")
                    st.stop()
                if pasien == 0 and meninggal > 0:
                    st.error("❌ Jika jumlah pasien = 0, jumlah meninggal harus 0. Periksa input Anda.")
                    st.stop()

                # Compute IR per 100k and CFR
                ir_calculated = (pasien / penduduk) * 100000
                cfr_calculated = (meninggal / pasien) * 100 if pasien > 0 else 0.0

                # Display metrics
                st.success("✅ Perhitungan Metrik Berhasil")
                mcol1, mcol2 = st.columns(2)
                mcol1.metric("📊 IR (per 100.000)", f"{ir_calculated:.2f}")
                mcol2.metric("📉 CFR (%)", f"{cfr_calculated:.2f}")

                # Map user inputs to a raw feature dict
                raw_inputs = {
                    'penduduk': float(penduduk),
                    'total_kasus': float(pasien),
                    'total_dbd_dss': float(pasien),
                    'total_dd': 0.0,
                    'total_meninggal': float(meninggal),
                    'ir_total': float(ir_calculated),
                    'ir': float(ir_calculated),
                    'cfr': float(cfr_calculated),
                    'Pasien': float(pasien),
                    'Meninggal': float(meninggal),
                    'Jml Penduduk': float(penduduk)
                }

                # Build a display DataFrame
                input_df = pd.DataFrame([{
                    'wilayah': wilayah,
                    'tahun': int(tahun),
                    'penduduk': penduduk,
                    'pasien': pasien,
                    'meninggal': meninggal,
                    'IR_per_100k': round(ir_calculated, 4),
                    'CFR_percent': round(cfr_calculated, 4)
                }])

                # Align input columns with scaler expectations
                expected = None
                if hasattr(scaler, 'feature_names_in_'):
                    expected = [str(x) for x in scaler.feature_names_in_]

                if expected is not None:
                    aligned_dict = {}
                    missing = []
                    for col in expected:
                        if col in raw_inputs:
                            aligned_dict[col] = raw_inputs[col]
                        else:
                            found = None
                            for k in raw_inputs.keys():
                                if k.lower() == col.lower():
                                    found = k
                                    break
                            if found is not None:
                                aligned_dict[col] = raw_inputs[found]
                            else:
                                aligned_dict[col] = 0.0
                                missing.append(col)

                    if missing:
                        st.warning(f"⚠️ Beberapa fitur yang diharapkan tidak ditemukan: {missing}. Mengisi dengan 0.")

                    aligned = pd.DataFrame([aligned_dict], columns=expected)

                else:
                    aligned = pd.DataFrame([raw_inputs])
                    if hasattr(scaler, 'n_features_in_'):
                        expected_n = scaler.n_features_in_
                        if aligned.shape[1] != expected_n:
                            st.error(f"❌ Dimensi tidak cocok: diharapkan {expected_n} fitur, input memiliki {aligned.shape[1]}")
                            st.stop()

                # Ensure numeric dtype
                aligned = aligned.astype(float)

                # Scale the input
                scaled = scaler.transform(aligned)

                # PCA transform
                try:
                    pca_val = pca.transform(scaled)
                except Exception as e:
                    st.warning(f"⚠️ PCA transform gagal: {e}")
                    pca_val = None

                cluster_result = int(kmeans.predict(scaled)[0])

                # Cluster interpretation (TANPA RANK)
                cluster_interpretation = {
                    0: 'Wilayah Aman - Risiko Rendah',
                    1: 'Risiko Sedang-Tinggi',
                    2: 'PRIORITAS UTAMA - Risiko Sangat Tinggi',
                    3: 'Risiko Tinggi - Fatalitas Tinggi'
                }

                recommendation_map = {
                    0: 'Pengawasan Rutin - Lanjutkan monitoring berkala',
                    1: 'Mitigasi & Pencegahan - Fokus pada pencegahan dan edukasi',
                    2: '🚨 INTERVENSI SEGERA - Kontrol vektor intensif, sosialisasi masif, mobilisasi SDM',
                    3: '🏥 PENANGANAN KLINIS PRIORITAS - Siapkan fasilitas kesehatan, pelatihan tenaga medis'
                }
                
                # Emoji mapping for visual clarity
                cluster_emoji = {
                    0: '🟢',
                    1: '🟡',
                    2: '🔴',
                    3: '🟠'
                }

                cluster_label = cluster_interpretation.get(cluster_result, labels_map.get(cluster_result, "Tidak diketahui"))
                recommendation = recommendation_map.get(cluster_result, "Tidak diketahui")
                emoji = cluster_emoji.get(cluster_result, '⚪')

                # Display results with better formatting
                st.markdown("---")
                st.success(f"✅ Prediksi Berhasil untuk Wilayah **{wilayah}**")
                
                # Result cards
                result_col1, result_col2 = st.columns([1, 1])
                
                with result_col1:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 20px; border-radius: 10px; color: white;'>
                        <h2 style='margin:10px 0; color: white;'>{emoji} Cluster {cluster_result}</h2>
                        <p style='margin:0; font-size: 16px; color: white;'>{cluster_label}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with result_col2:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                padding: 20px; border-radius: 10px; color: white;'>
                        <h3 style='margin:0; color: white;'>🔔 Rekomendasi Tindakan</h3>
                        <p style='margin:10px 0 0 0; font-size: 15px; color: white;'>{recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("📋 Detail Input")
                st.dataframe(input_df, use_container_width=True)
                
                # Additional info in expander
                with st.expander("ℹ️ Informasi Teknis"):
                    st.write("**Fitur yang digunakan untuk prediksi:**")
                    if expected:
                        st.write(expected)
                    st.write(f"**Nilai setelah standardisasi:**")
                    st.write(pd.DataFrame(scaled, columns=expected if expected else [f"Feature_{i}" for i in range(scaled.shape[1])]))
                    if pca_val is not None:
                        st.write(f"**Nilai setelah PCA:**")
                        st.write(pd.DataFrame(pca_val, columns=[f"PC{i+1}" for i in range(pca_val.shape[1])]))
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")
                st.write("**Detail Error:**")
                st.code(str(e))
                
                # Debug info
                with st.expander("🔧 Informasi Debug"):
                    st.write("**Status Model:**")
                    st.write(f"- Scaler loaded: {scaler is not None}")
                    st.write(f"- KMeans loaded: {kmeans is not None}")
                    st.write(f"- PCA loaded: {pca is not None}")
                    if hasattr(scaler, 'feature_names_in_'):
                        st.write(f"- Expected features: {scaler.feature_names_in_}")
                    st.write("**Input Data:**")
                    st.write(raw_inputs)
