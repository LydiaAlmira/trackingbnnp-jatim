# =========================
# CONFIGURASI UMUM
# =========================
st.set_page_config( page_title="Dashboard Prediksi Kasus Narkoba", layout="wide", initial_sidebar_state="expanded" )

# =========================
# IMPORT LIBRARY
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA

# Prophet optional
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

sns.set_style('whitegrid')

# =========================
# SIDEBAR NAVIGASI
# =========================
menu = st.sidebar.radio(
    "📋 MENU NAVIGASI:",
    [
        "HOME 🏠",
        "INPUT DATA 📁",
        "DATA PREPROCESSING 🧹",
        "VISUALISASI DATA 📊",
        "FORECASTING 📈",
        "RINGKASAN HASIL 📋"
    ]
)

# =========================
# JUDUL UTAMA
# =========================
st.markdown(
    """
    <h1 style='text-align: center; color: white; background-color:#1f77b4;
    padding: 15px; border-radius: 10px;'>
    Dashboard Prediksi Kasus Narkoba<br>
    Forecasting & Visualisasi Data 📊
    </h1>
    """,
    unsafe_allow_html=True
)

# =========================
# HELPER FUNCTION
# =========================
def find_column(df, keywords):
    for kw in keywords:
        for c in df.columns:
            if kw.lower() in str(c).lower():
                return c
    return None


def looks_like_gender_series(s):
    s_nonnull = s.dropna().astype(str).str.strip().str.lower()

    if s_nonnull.empty:
        return False

    short_vals = sum(1 for x in s_nonnull if len(x) <= 2)

    if short_vals / len(s_nonnull) > 0.6:
        return True

    gender_words = {'l', 'p', 'lk', 'pr', 'male', 'female'}
    cnt_gender_like = sum(
        1 for x in s_nonnull
        if any(g in x for g in gender_words)
    )

    if cnt_gender_like / len(s_nonnull) > 0.5:
        return True

    return False


def forecast_prophet(series, periods=30):
    dfp = series.reset_index()
    dfp.columns = ['ds', 'y']

    model = Prophet()
    model.fit(dfp)

    future = model.make_future_dataframe(periods=periods, freq='D')
    fc = model.predict(future)

    return fc.set_index('ds')['yhat'].iloc[-periods:]


def forecast_arima(series, periods=30):
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.asfreq('D').fillna(method='ffill')

    try:
        model = ARIMA(s, order=(1,1,1))
        fit = model.fit()

        pred = fit.get_forecast(steps=periods)

        idx = pd.date_range(
            s.index.max() + timedelta(days=1),
            periods=periods,
            freq='D'
        )

        return pd.Series(pred.predicted_mean.values, index=idx)

    except Exception:
        last = float(s.dropna().iloc[-1])

        idx = pd.date_range(
            s.index.max() + timedelta(days=1),
            periods=periods,
            freq='D'
        )

        return pd.Series([last]*periods, index=idx)

# =========================
# HOME
# =========================
if menu == "HOME 🏠":

    st.info(
        "Sistem ini digunakan untuk analisis data kasus narkoba, "
        "visualisasi statistik, serta forecasting jumlah kasus "
        "30 hari ke depan."
    )

    st.markdown("### Panduan Penggunaan Sistem 🔎")

    st.markdown(
        """
        - **HOME** 🏠 : Penjelasan sistem.
        - **INPUT DATA** 📁 : Upload file Excel REG TAT.
        - **DATA PREPROCESSING** 🧹 : Pembersihan dan validasi data.
        - **VISUALISASI DATA** 📊 : Grafik bar, pie, heatmap, tren bulanan.
        - **FORECASTING** 📈 : Prediksi jumlah kasus dan statistik lainnya.
        - **RINGKASAN HASIL** 📋 : Kesimpulan hasil forecasting.
        """
    )

# =========================
# INPUT DATA
# =========================
elif menu == "INPUT DATA 📁":

    st.header("📁 Upload Data REG TAT")

    uploaded = st.file_uploader(
        "Upload file Excel (.xls/.xlsx)",
        type=['xls', 'xlsx']
    )

    if uploaded is not None:

        try:
            df = pd.read_excel(io.BytesIO(uploaded.read()))

            st.subheader("🔍 Preview Dataset")
            st.dataframe(df.head())

            # Simpan session
            st.session_state.df = df.copy()

            st.success("✅ File berhasil diupload")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

    else:
        st.info("Silakan upload file terlebih dahulu")

# =========================
# DATA PREPROCESSING
# =========================
elif menu == "DATA PREPROCESSING 🧹":

    st.header("🧹 Data Preprocessing")

    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu")
        st.stop()

    df = st.session_state.df.copy()

    # Deteksi kolom
    col_date = find_column(df, ['tanggal','tgl','date'])
    col_age = find_column(df, ['umur','usia'])
    col_gender = find_column(df, ['jenis kelamin','jk','gender'])
    col_region = find_column(df, ['asal','wilayah'])
    col_drug = find_column(df, ['narkoba','narkotika','jenis'])

    st.subheader("🔍 Hasil Deteksi Kolom")

    st.json({
        'Tanggal': col_date,
        'Umur': col_age,
        'Jenis Kelamin': col_gender,
        'Wilayah': col_region,
        'Jenis Narkoba': col_drug
    })

    # Konversi tanggal
    df[col_date] = pd.to_datetime(df[col_date], errors='coerce')

    # Hapus NA tanggal
    df = df.dropna(subset=[col_date])

    # Simpan
    st.session_state.df_processed = df
    st.session_state.col_date = col_date
    st.session_state.col_age = col_age
    st.session_state.col_gender = col_gender
    st.session_state.col_region = col_region
    st.session_state.col_drug = col_drug

    st.subheader("📊 Data Setelah Preprocessing")
    st.dataframe(df.head())

    st.success("✅ Preprocessing selesai")

# =========================
# VISUALISASI DATA
# =========================
elif menu == "VISUALISASI DATA 📊":

    st.header("📊 Visualisasi Data Kasus")

    if 'df_processed' not in st.session_state:
        st.warning("Silakan lakukan preprocessing terlebih dahulu")
        st.stop()

    df = st.session_state.df_processed

    col_date = st.session_state.col_date
    col_gender = st.session_state.col_gender
    col_region = st.session_state.col_region
    col_drug = st.session_state.col_drug

    # =========================
    # TOP WILAYAH
    # =========================
    if col_region:

        st.subheader("📌 Top Wilayah")

        top_reg = df[col_region].value_counts().nlargest(20)

        fig, ax = plt.subplots(figsize=(8,5))
        ax.barh(top_reg.index, top_reg.values)
        st.pyplot(fig)

    # =========================
    # PIE CHART GENDER
    # =========================
    if col_gender:

        st.subheader("👥 Proporsi Jenis Kelamin")

        jk = df[col_gender].value_counts()

        fig2, ax2 = plt.subplots(figsize=(5,5))
        ax2.pie(jk.values, labels=jk.index, autopct='%1.1f%%')
        ax2.axis('equal')
        st.pyplot(fig2)

    # =========================
    # TOP NARKOBA
    # =========================
    if col_drug:

        st.subheader("💊 Top Jenis Narkoba")

        top_drug = df[col_drug].value_counts().nlargest(20)

        fig3, ax3 = plt.subplots(figsize=(8,5))
        ax3.barh(top_drug.index, top_drug.values)
        st.pyplot(fig3)

    # =========================
    # HEATMAP
    # =========================
    if col_region and col_drug:

        st.subheader("🔥 Heatmap Wilayah x Jenis Narkoba")

        heat = df.groupby([col_region, col_drug]).size().unstack(fill_value=0)

        fig4, ax4 = plt.subplots(figsize=(12,6))

        sns.heatmap(
            heat,
            annot=True,
            fmt='d',
            cmap='YlOrRd',
            ax=ax4
        )

        st.pyplot(fig4)

# =========================
# FORECASTING
# =========================
elif menu == "FORECASTING 📈":

    st.header("📈 Forecasting Kasus Narkoba")

    if 'df_processed' not in st.session_state:
        st.warning("Silakan lakukan preprocessing terlebih dahulu")
        st.stop()

    df = st.session_state.df_processed
    col_date = st.session_state.col_date

    periods = st.number_input(
        'Periode prediksi (hari)',
        min_value=7,
        max_value=90,
        value=30
    )

    # Aggregasi harian
    daily = df.groupby(col_date).size().to_frame('jumlah_kasus')

    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq('D').fillna(method='ffill')

    # Pilih metode
    model_choice = st.selectbox(
        "Pilih metode forecasting",
        ["ARIMA", "Prophet"]
    )

    # Forecast
    if model_choice == "Prophet" and PROPHET_AVAILABLE:
        forecast = forecast_prophet(daily['jumlah_kasus'], periods)
    else:
        forecast = forecast_arima(daily['jumlah_kasus'], periods)

    # Tabel hasil
    forecast_df = pd.DataFrame({
        'Tanggal': forecast.index,
        'Forecast': forecast.values
    })

    st.subheader("📄 Hasil Forecast")
    st.dataframe(forecast_df)

    # Visualisasi
    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        daily.index,
        daily['jumlah_kasus'],
        label='Aktual'
    )

    ax.plot(
        forecast.index,
        forecast.values,
        '--',
        label='Forecast'
    )

    ax.legend()

    st.pyplot(fig)

    # Simpan
    st.session_state.forecast_df = forecast_df

# =========================
# RINGKASAN HASIL
# =========================
elif menu == "RINGKASAN HASIL 📋":

    st.header("📋 Ringkasan Hasil")

    if 'forecast_df' not in st.session_state:
        st.warning("Silakan lakukan forecasting terlebih dahulu")
        st.stop()

    forecast_df = st.session_state.forecast_df

    mean_case = forecast_df['Forecast'].mean()

    trend = (
        'Naik'
        if forecast_df['Forecast'].iloc[-1] > forecast_df['Forecast'].iloc[0]
        else 'Turun'
    )

    st.json({
        'Rata-rata Forecast': float(mean_case),
        'Tren Prediksi': trend,
        'Jumlah Hari Forecast': len(forecast_df)
    })

    st.success("✅ Analisis selesai")
```

Perubahan utama dari Kode 2:

1. Semua proses dipisah menjadi menu seperti Kode 1.
2. Struktur `if menu == ...` dibuat konsisten.
3. Session state dipakai seperti Kode 1.
4. Alur dibuat:

   * upload
   * preprocessing
   * visualisasi
   * forecasting
   * ringkasan
5. Tampilan dashboard jadi lebih rapi dan modular.
6. Forecasting tetap memakai isi asli Kode 2 (ARIMA/Prophet).
7. Heatmap, pie chart, dan bar chart tetap dipertahankan.

Jadi intinya:

* Struktur = mengikuti Kode 1
* Isi analisis = tetap dari Kode 2
* Dataset dan variabel = tetap khusus kasus narkoba
