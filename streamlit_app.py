import streamlit as st

# =========================
# CONFIGURASI UMUM
# =========================
st.set_page_config(
    page_title="Dashboard Prediksi Kasus Narkoba",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# =========================
# OPTIONAL PROPHET
# =========================
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

sns.set_style("whitegrid")

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
# HEADER
# =========================
st.markdown(
    """
    <h1 style='text-align:center;
    color:white;
    background-color:#1f77b4;
    padding:15px;
    border-radius:10px;'>

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


def forecast_prophet(series, periods=30):

    dfp = series.reset_index()

    dfp.columns = ["ds", "y"]

    model = Prophet()

    model.fit(dfp)

    future = model.make_future_dataframe(
        periods=periods,
        freq='D'
    )

    fc = model.predict(future)

    return fc.set_index("ds")["yhat"].iloc[-periods:]


def forecast_arima(series, periods=30):

    s = series.copy()

    s.index = pd.to_datetime(s.index)

    s = s.asfreq("D")

    s = s.ffill().fillna(0)

    try:

        model = ARIMA(
            s,
            order=(1, 1, 1)
        )

        fit = model.fit()

        pred = fit.get_forecast(
            steps=periods
        )

        idx = pd.date_range(
            start=s.index.max() + timedelta(days=1),
            periods=periods,
            freq="D"
        )

        return pd.Series(
            pred.predicted_mean.values,
            index=idx
        )

    except Exception:

        last = float(s.dropna().iloc[-1])

        idx = pd.date_range(
            start=s.index.max() + timedelta(days=1),
            periods=periods,
            freq="D"
        )

        return pd.Series(
            [last] * periods,
            index=idx
        )

# =========================
# HOME
# =========================
if menu == "HOME 🏠":

    st.info(
        "Sistem ini digunakan untuk analisis data kasus narkoba, "
        "visualisasi statistik, forecasting jumlah kasus, "
        "dan prediksi kesatuan dengan klien terbanyak."
    )

    st.markdown("### Panduan Penggunaan Sistem 🔎")

    st.markdown(
        """
        - **HOME** 🏠 : Penjelasan sistem
        - **INPUT DATA** 📁 : Upload file Excel REG TAT
        - **DATA PREPROCESSING** 🧹 : Cleaning & validasi data
        - **VISUALISASI DATA** 📊 : Grafik & heatmap
        - **FORECASTING** 📈 : Prediksi kasus dan kesatuan
        - **RINGKASAN HASIL** 📋 : Kesimpulan hasil
        """
    )

# =========================
# INPUT DATA
# =========================
elif menu == "INPUT DATA 📁":

    st.header("📁 Upload Data REG TAT")

    uploaded = st.file_uploader(
        "Upload file Excel (.xls/.xlsx)",
        type=["xls", "xlsx"]
    )

    if uploaded is not None:

        try:

            df = pd.read_excel(
                io.BytesIO(uploaded.read())
            )

            st.subheader("🔍 Preview Dataset")

            st.dataframe(df.head())

            st.write("### Nama Kolom Dataset")

            st.write(df.columns.tolist())

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

    # =========================
    # DETEKSI KOLOM
    # =========================
    col_date = find_column(
        df,
        [
            'tanggal',
            'tgl',
            'date'
        ]
    )

    col_age = find_column(
        df,
        [
            'umur',
            'usia'
        ]
    )

    col_gender = find_column(
        df,
        [
            'jenis kelamin',
            'jk',
            'gender'
        ]
    )

    col_region = find_column(
        df,
        [
            'asal',
            'wilayah',
            'kota'
        ]
    )

    col_drug = find_column(
        df,
        [
            'narkoba',
            'narkotika',
            'jenis'
        ]
    )

    # =========================
    # KOLOM KESATUAN
    # =========================
    col_kesatuan = find_column(
        df,
        [
            'asal pengajuan',
            'kesatuan',
            'satker',
            'bnnk',
            'instansi'
        ]
    )

    st.subheader("🔍 Hasil Deteksi Kolom")

    st.json({
        "Tanggal": col_date,
        "Umur": col_age,
        "Jenis Kelamin": col_gender,
        "Wilayah": col_region,
        "Jenis Narkoba": col_drug,
        "Kesatuan": col_kesatuan
    })

    # =========================
    # VALIDASI TANGGAL
    # =========================
    if col_date is None:

        st.error("Kolom tanggal tidak ditemukan")

        st.stop()

    df[col_date] = pd.to_datetime(
        df[col_date],
        errors='coerce'
    )

    df = df.dropna(subset=[col_date])

    df = df.sort_values(by=col_date)

    # =========================
    # SIMPAN SESSION
    # =========================
    st.session_state.df_processed = df

    st.session_state.col_date = col_date
    st.session_state.col_age = col_age
    st.session_state.col_gender = col_gender
    st.session_state.col_region = col_region
    st.session_state.col_drug = col_drug
    st.session_state.col_kesatuan = col_kesatuan

    st.subheader("📊 Data Setelah Preprocessing")

    st.dataframe(df.head())

    st.success("✅ Preprocessing selesai")

# =========================
# VISUALISASI DATA
# =========================
elif menu == "VISUALISASI DATA 📊":

    st.header("📊 Visualisasi Data Kasus")

    if 'df_processed' not in st.session_state:

        st.warning("Silakan preprocessing terlebih dahulu")

        st.stop()

    df = st.session_state.df_processed

    col_gender = st.session_state.col_gender
    col_region = st.session_state.col_region
    col_drug = st.session_state.col_drug
    col_date = st.session_state.col_date
    col_kesatuan = st.session_state.col_kesatuan

    # =========================
    # TOP WILAYAH
    # =========================
    if col_region:

        st.subheader("📌 Top Wilayah")

        top_reg = (
            df[col_region]
            .value_counts()
            .nlargest(10)
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.barh(
            top_reg.index,
            top_reg.values
        )

        st.pyplot(fig)

    # =========================
    # TOP KESATUAN
    # =========================
    if col_kesatuan:

        st.subheader("🏢 Top Kesatuan Pengaju")

        top_satker = (
            df[col_kesatuan]
            .value_counts()
            .nlargest(10)
        )

        figk, axk = plt.subplots(figsize=(8, 5))

        axk.barh(
            top_satker.index,
            top_satker.values
        )

        st.pyplot(figk)

    # =========================
    # PIE GENDER
    # =========================
    if col_gender:

        st.subheader("👥 Proporsi Jenis Kelamin")

        jk = df[col_gender].value_counts()

        fig2, ax2 = plt.subplots(figsize=(5, 5))

        ax2.pie(
            jk.values,
            labels=jk.index,
            autopct='%1.1f%%'
        )

        ax2.axis('equal')

        st.pyplot(fig2)

    # =========================
    # TOP NARKOBA
    # =========================
    if col_drug:

        st.subheader("💊 Top Jenis Narkoba")

        top_drug = (
            df[col_drug]
            .value_counts()
            .nlargest(10)
        )

        fig3, ax3 = plt.subplots(figsize=(8, 5))

        ax3.barh(
            top_drug.index,
            top_drug.values
        )

        st.pyplot(fig3)

    # =========================
    # TREN BULANAN
    # =========================
    st.subheader("📈 Tren Bulanan")

    monthly = (
        df.groupby(
            df[col_date].dt.to_period("M")
        )
        .size()
    )

    monthly.index = monthly.index.to_timestamp()

    fig4, ax4 = plt.subplots(figsize=(10, 4))

    ax4.plot(
        monthly.index,
        monthly.values,
        marker='o'
    )

    st.pyplot(fig4)

# =========================
# FORECASTING
# =========================
elif menu == "FORECASTING 📈":

    st.header("📈 Forecasting Kasus Narkoba")

    if 'df_processed' not in st.session_state:

        st.warning("Silakan preprocessing terlebih dahulu")

        st.stop()

    df = st.session_state.df_processed

    col_date = st.session_state.col_date

    periods = st.number_input(
        "Periode Prediksi (hari)",
        min_value=7,
        max_value=90,
        value=30
    )

    # =========================
    # AGREGASI HARIAN
    # =========================
    daily = (
        df.groupby(col_date)
        .size()
        .reset_index(name="jumlah_kasus")
    )

    daily[col_date] = pd.to_datetime(
        daily[col_date]
    )

    daily = daily.set_index(col_date)

    daily = daily.sort_index()

    daily = daily.asfreq("D")

    daily["jumlah_kasus"] = (
        daily["jumlah_kasus"]
        .ffill()
        .fillna(0)
    )

    # =========================
    # PILIH METODE
    # =========================
    model_choice = st.selectbox(
        "Pilih Metode Forecasting",
        ["ARIMA", "Prophet"]
    )

    # =========================
    # FORECAST
    # =========================
    if model_choice == "Prophet":

        if PROPHET_AVAILABLE:

            forecast = forecast_prophet(
                daily["jumlah_kasus"],
                periods
            )

        else:

            st.warning(
                "Prophet tidak tersedia. Menggunakan ARIMA."
            )

            forecast = forecast_arima(
                daily["jumlah_kasus"],
                periods
            )

    else:

        forecast = forecast_arima(
            daily["jumlah_kasus"],
            periods
        )

    # =========================
    # HASIL FORECAST
    # =========================
    forecast_df = pd.DataFrame({
        "Tanggal": forecast.index,
        "Forecast": forecast.values
    })

    st.subheader("📄 Hasil Forecast Jumlah Kasus")

    st.dataframe(forecast_df)

    # =========================
    # VISUALISASI FORECAST
    # =========================
    fig6, ax6 = plt.subplots(figsize=(12, 5))

    ax6.plot(
        daily.index,
        daily["jumlah_kasus"],
        label="Aktual"
    )

    ax6.plot(
        forecast.index,
        forecast.values,
        '--',
        label="Forecast"
    )

    ax6.legend()

    st.pyplot(fig6)

    # =========================
    # FORECAST KESATUAN
    # =========================
    st.subheader("🏢 Prediksi Kesatuan dengan Klien Terbanyak")

    col_kesatuan = st.session_state.col_kesatuan

    if col_kesatuan:

        satker_count = (
            df[col_kesatuan]
            .value_counts()
            .reset_index()
        )

        satker_count.columns = [
            'Kesatuan',
            'Jumlah_Kasus'
        ]

        satker_count['Forecast_30_Hari'] = (
            satker_count['Jumlah_Kasus'] * 1.1
        ).round().astype(int)

        top_satker = satker_count.head(10)

        st.dataframe(top_satker)

        fig7, ax7 = plt.subplots(figsize=(10,5))

        ax7.barh(
            top_satker['Kesatuan'],
            top_satker['Forecast_30_Hari']
        )

        ax7.set_xlabel("Prediksi Klien")

        ax7.set_ylabel("Kesatuan")

        st.pyplot(fig7)

    else:

        st.warning(
            "Kolom kesatuan/satker tidak ditemukan."
        )

    st.session_state.forecast_df = forecast_df

# =========================
# RINGKASAN HASIL
# =========================
elif menu == "RINGKASAN HASIL 📋":

    st.header("📋 Ringkasan Hasil Forecasting")

    if 'forecast_df' not in st.session_state:

        st.warning("Silakan lakukan forecasting terlebih dahulu")

        st.stop()

    forecast_df = st.session_state.forecast_df

    # =========================
    # STATISTIK FORECAST
    # =========================
    rata_forecast = round(
        forecast_df["Forecast"].mean(),
        2
    )

    maksimum = round(
        forecast_df["Forecast"].max(),
        2
    )

    minimum = round(
        forecast_df["Forecast"].min(),
        2
    )

    awal = forecast_df["Forecast"].iloc[0]

    akhir = forecast_df["Forecast"].iloc[-1]

    trend = (
        "📈 Meningkat"
        if akhir > awal
        else "📉 Menurun"
    )

    # =========================
    # CARD METRIK
    # =========================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rata-rata Forecast",
            f"{rata_forecast:.2f}"
        )

    with col2:
        st.metric(
            "Forecast Maksimum",
            f"{maksimum:.2f}"
        )

    with col3:
        st.metric(
            "Forecast Minimum",
            f"{minimum:.2f}"
        )

    with col4:
        st.metric(
            "Trend Prediksi",
            trend
        )

    st.divider()

    # =========================
    # INTERPRETASI
    # =========================
    st.subheader("🧠 Interpretasi Hasil")

    if akhir > awal:

        st.success(
            f"""
            Berdasarkan hasil forecasting,
            jumlah kasus diprediksi mengalami peningkatan
            dalam periode prediksi berikutnya.

            Rata-rata prediksi kasus berada di angka
            {rata_forecast:.2f} kasus per hari.
            """
        )

    else:

        st.info(
            f"""
            Berdasarkan hasil forecasting,
            jumlah kasus diprediksi cenderung menurun
            dalam periode prediksi berikutnya.

            Rata-rata prediksi kasus berada di angka
            {rata_forecast:.2f} kasus per hari.
            """
        )

    # =========================
    # TOP KESATUAN
    # =========================
    if 'df_processed' in st.session_state:

        df = st.session_state.df_processed

        col_kesatuan = st.session_state.col_kesatuan

        if col_kesatuan:

            st.subheader("🏢 Kesatuan dengan Pengajuan Tertinggi")

            top_satker = (
                df[col_kesatuan]
                .value_counts()
                .head(5)
                .reset_index()
            )

            top_satker.columns = [
                "Kesatuan",
                "Jumlah Klien"
            ]

            st.dataframe(
                top_satker,
                use_container_width=True
            )

            fig8, ax8 = plt.subplots(figsize=(9,4))

            ax8.barh(
                top_satker["Kesatuan"],
                top_satker["Jumlah Klien"]
            )

            ax8.set_xlabel("Jumlah Klien")

            ax8.set_ylabel("Kesatuan")

            st.pyplot(fig8)

    st.divider()

    # =========================
    # KESIMPULAN
    # =========================
    st.subheader("📝 Kesimpulan")

    st.write(
        f"""
        Sistem forecasting menunjukkan bahwa tren kasus
        narkoba selama {len(forecast_df)} hari ke depan
        diperkirakan {trend.lower()}.

        Analisis juga menunjukkan adanya beberapa
        kesatuan yang secara konsisten menjadi
        pengaju klien terbanyak sehingga dapat menjadi
        fokus monitoring dan evaluasi lebih lanjut.
        """
    )

    st.success("✅ Analisis forecasting selesai")
