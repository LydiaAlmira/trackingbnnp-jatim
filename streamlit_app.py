import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA

# Try to import Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

sns.set_style("whitegrid")

st.set_page_config(layout="wide", page_title="Dashboard Prediksi Kasus Narkoba")
st.title("Dashboard Prediksi Kasus Narkoba — 30 Hari ke Depan")

# =============================
# SIDEBAR
# =============================
st.sidebar.header("Upload & Pengaturan")
uploaded = st.sidebar.file_uploader("Unggah file Excel", type=["xls", "xlsx"])
periods = st.sidebar.number_input("Periode prediksi (hari)", 7, 90, 30)
use_prophet = st.sidebar.checkbox("Gunakan Prophet (jika tersedia)", True)

# =============================
# HELPER
# =============================
def find_column(df, keywords):
    for k in keywords:
        for c in df.columns:
            if k.lower() in str(c).lower():
                return c
    return None

def forecast_arima(series, periods):
    s = series.asfreq("D").fillna(method="ffill")
    try:
        model = ARIMA(s, order=(1, 1, 1))
        fit = model.fit()
        pred = fit.forecast(periods)
        idx = pd.date_range(s.index.max() + timedelta(days=1), periods=periods)
        return pd.Series(pred.values, index=idx)
    except Exception:
        last = s.dropna().iloc[-1] if not s.dropna().empty else 0
        idx = pd.date_range(s.index.max() + timedelta(days=1), periods=periods)
        return pd.Series([last] * periods, index=idx)

def forecast_prophet(series, periods):
    dfp = series.reset_index()
    dfp.columns = ["ds", "y"]
    model = Prophet()
    model.fit(dfp)
    future = model.make_future_dataframe(periods=periods)
    fc = model.predict(future)
    return fc.set_index("ds")["yhat"].iloc[-periods:]

# =============================
# MAIN PROCESS
# =============================
if uploaded is not None:
    try:
        df = pd.read_excel(uploaded)

        col_date = find_column(df, ["tanggal", "tgl", "date"])
        col_age = find_column(df, ["umur", "usia"])
        col_gender = find_column(df, ["jk", "kelamin", "gender"])

        df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
        df = df.dropna(subset=[col_date])
        df[col_date] = df[col_date].dt.floor("D")

        daily = df.groupby(col_date).size().to_frame("jumlah_kasus")

        if col_age:
            daily["rata2_umur"] = df.groupby(col_date)[col_age].mean()
        else:
            daily["rata2_umur"] = np.nan

        if col_gender:
            df["_l"] = df[col_gender].astype(str).str.lower().str.contains("l")
            g = df.groupby(col_date).agg(
                total=(col_gender, "count"),
                laki=("_l", "sum")
            )
            daily["prop_laki"] = g["laki"] / g["total"]
            daily["prop_perempuan"] = 1 - daily["prop_laki"]
        else:
            daily["prop_laki"] = np.nan
            daily["prop_perempuan"] = np.nan

        daily = daily.asfreq("D").fillna(method="ffill").fillna(0)

        st.subheader("Data Harian")
        st.dataframe(daily.head())

        # =============================
        # FORECAST
        # =============================
        st.subheader("Forecast 30 Hari")

        use_prophet_final = use_prophet and PROPHET_AVAILABLE

        total_fc = forecast_prophet(daily["jumlah_kasus"], periods) if use_prophet_final else forecast_arima(daily["jumlah_kasus"], periods)
        age_fc = forecast_arima(daily["rata2_umur"], periods) if not daily["rata2_umur"].isna().all() else None
        male_fc = forecast_arima(daily["prop_laki"], periods) if not daily["prop_laki"].isna().all() else None
        female_fc = forecast_arima(daily["prop_perempuan"], periods) if not daily["prop_perempuan"].isna().all() else None

        # =============================
        # RINGKASAN
        # =============================
        st.subheader("Ringkasan Prediksi")
        st.json({
            "mean_jumlah_kasus": float(total_fc.mean()),
            "tren_jumlah_kasus": "naik" if total_fc.iloc[-1] > total_fc.iloc[0] else "turun",
            "mean_umur": float(age_fc.mean()) if age_fc is not None else None,
            "mean_prop_laki": float(male_fc.mean()) if male_fc is not None else None,
            "mean_prop_perempuan": float(female_fc.mean()) if female_fc is not None else None
        })

        # =============================
        # TABEL HASIL
        # =============================
        forecast_df = pd.DataFrame({
            "Tanggal": total_fc.index,
            "Prediksi_Jumlah_Kasus": total_fc.values
        })

        if age_fc is not None:
            forecast_df["Prediksi_Rata2_Umur"] = age_fc.values
        if male_fc is not None:
            forecast_df["Proporsi_Laki_Laki"] = male_fc.values
        if female_fc is not None:
            forecast_df["Proporsi_Perempuan"] = female_fc.values

        st.session_state["forecast_df"] = forecast_df

        st.subheader("Tabel Hasil Prediksi")
        st.dataframe(forecast_df, use_container_width=True)

    except Exception as e:
        st.error("Terjadi kesalahan saat memproses data")
        st.exception(e)

else:
    st.info("Silakan unggah file Excel untuk memulai analisis.")

# =============================
# DOWNLOAD
# =============================
if "forecast_df" in st.session_state:
    def convert_df_to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Forecast")
        return output.getvalue()

    st.download_button(
        "📥 Download Hasil Prediksi (Excel)",
        convert_df_to_excel(st.session_state["forecast_df"]),
        "hasil_prediksi_kasus_narkoba.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
