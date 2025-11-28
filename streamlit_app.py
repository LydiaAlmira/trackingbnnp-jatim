import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA

# Try to import Prophet; app will still work without it (ARIMA fallback)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

sns.set_style('whitegrid')

st.set_page_config(layout='wide', page_title='Dashboard Prediksi Kasus Narkoba')

st.title('Dashboard Prediksi Kasus Narkoba — 30 Hari ke Depan')
st.markdown(
    'Aplikasi ini membaca file Excel **REG TAT** dan menghasilkan analisis: grafik (bar, pie, heatmap),\n'
    'serta prediksi 30 hari untuk jumlah kasus, rata-rata umur, dan proporsi jenis kelamin.\n\n'
    '**Jika Prophet tidak terpasang, aplikasi akan menggunakan ARIMA sebagai fallback.**'
)

# -----------------------------
# Sidebar: Upload & Pengaturan
# -----------------------------
st.sidebar.header('Upload & Pengaturan')
uploaded = st.sidebar.file_uploader('Unggah file Excel (xls/xlsx)', type=['xls','xlsx'])

periods = st.sidebar.number_input('Periode prediksi (hari)', min_value=7, max_value=90, value=30, step=1)
use_prophet = st.sidebar.checkbox('Gunakan Prophet jika tersedia (direkomendasikan)', value=True)

if use_prophet and not PROPHET_AVAILABLE:
    st.sidebar.warning('Prophet tidak terpasang di environment ini. Aplikasi akan pakai ARIMA.')

st.sidebar.markdown('---')
st.sidebar.markdown('Petunjuk:\n- Pastikan file memiliki kolom tanggal seperti `TANGGAL`, `TGL PENANGKAPAN`, atau `TANGGAL BERKAS DITERIMA`.\n- Kolom wilayah: `ASAL PENGAJUAN`.\n- Kolom umur: `UMUR` atau `USIA`.\n- Kolom jenis kelamin dan narkoba bila tersedia.')

# -----------------------------
# Helper functions
# -----------------------------

def find_column(df, keywords):
    for kw in keywords:
        for c in df.columns:
            if kw.lower() in str(c).lower():
                return c
    return None


def forecast_prophet(series, periods=30):
    dfp = series.reset_index()
    dfp.columns = ['ds','y']
    dfp['ds'] = pd.to_datetime(dfp['ds'])
    model = Prophet()
    model.fit(dfp)
    future = model.make_future_dataframe(periods=periods, freq='D')
    fc = model.predict(future)
    res = fc.set_index('ds')['yhat'].iloc[-periods:]
    return res


def forecast_arima(series, periods=30):
    s = series.asfreq('D').fillna(method='ffill')
    try:
        model = ARIMA(s, order=(1,1,1))
        fit = model.fit()
        pred = fit.get_forecast(steps=periods)
        idx = pd.date_range(s.index.max() + timedelta(days=1), periods=periods, freq='D')
        return pd.Series(pred.predicted_mean.values, index=idx)
    except Exception:
        last = float(s.dropna().iloc[-1]) if len(s.dropna())>0 else 0.0
        idx = pd.date_range(pd.to_datetime(series.index.max()) + timedelta(days=1), periods=periods, freq='D')
        return pd.Series([last]*periods, index=idx)

# -----------------------------
# Main processing
# -----------------------------
if uploaded is not None:
    try:
        with st.spinner('Membaca file...'):
            df = pd.read_excel(io.BytesIO(uploaded.read()))

        st.subheader('Preview data (5 baris pertama)')
        st.write(df.head())

        # Deteksi kolom
        col_date = find_column(df, ['tanggal','tgl','date'])
        col_age = find_column(df, ['umur','usia','age'])
        col_gender = find_column(df, ['jenis kelamin','jk','gender','sex'])
        col_region = find_column(df, ['asal','wilayah','kota','kabupaten','daerah'])
        col_drug = find_column(df, ['narkoba','jenis barang','jenis narkoba','barang','nama barang','jenis'])

        st.markdown('**Kolom terdeteksi:**')
        st.json({
            'Tanggal': col_date,
            'Umur': col_age,
            'Jenis Kelamin': col_gender,
            'Wilayah/Asal': col_region,
            'Jenis Narkoba': col_drug
        })

        if col_date is None:
            st.error('Kolom tanggal tidak ditemukan. Silakan pilih:')
            col_date = st.selectbox('Kolom tanggal', df.columns)

        # === Perbaikan konversi tanggal ===
        raw_dates = df[col_date].copy()

        df[col_date] = pd.to_datetime(raw_dates, errors='coerce')

        nat_ratio = df[col_date].isna().mean()
        need_excel_serial = False

        if nat_ratio > 0.2:
            need_excel_serial = True

        if need_excel_serial:
            ser = pd.to_numeric(raw_dates, errors='coerce')
            df[col_date] = pd.to_datetime(ser, unit='d', origin='1899-12-30', errors='coerce')

        if df[col_date].isna().mean() > 0.2:
            def try_convert(x):
                try:
                    return pd.to_datetime(x, errors='coerce')
                except:
                    pass
                try:
                    return pd.to_datetime(float(x), unit='d', origin='1899-12-30')
                except:
                    return pd.NaT
            df[col_date] = raw_dates.apply(try_convert)

        df = df.dropna(subset=[col_date]).copy()
        df[col_date] = pd.to_datetime(df[col_date]).dt.floor('D')

        # Filter tahun 2025 s.d November
        if (df[col_date].dt.year == 2025).any():
            df = df[df[col_date].dt.year == 2025]
            df = df[df[col_date].dt.month <= 11]

        # Normalisasi
        for col in [col_gender, col_region, col_drug]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Aggregasi harian
        daily = df.groupby(col_date).agg(jumlah_kasus=(col_date,'size'))

        if col_age:
            daily['rata2_umur'] = df.groupby(col_date)[col_age].mean()
        else:
            daily['rata2_umur'] = np.nan

        if col_gender:
            df['_laki'] = df[col_gender].str.lower().str.contains('l') | df[col_gender].str.lower().str.contains('male')
            df['_perempuan'] = df[col_gender].str.lower().str.contains('p') | df[col_gender].str.lower().str.contains('female')
            g = df.groupby(col_date).agg(
                total=(col_date,'size'),
                laki=('_laki','sum'),
                perempuan=('_perempuan','sum')
            )
            daily['prop_laki'] = g['laki']/g['total']
            daily['prop_perempuan'] = g['perempuan']/g['total']
        else:
            daily['prop_laki'] = np.nan
            daily['prop_perempuan'] = np.nan

        daily = daily.asfreq('D').fillna(method='ffill').fillna(0)

        st.subheader('Ringkasan Harian (sample)')
        st.dataframe(daily.head(20))

        # -----------------------------
        # GRAFIK
        # -----------------------------
        st.subheader('Grafik Analisis')
        col1, col2 = st.columns(2)

        with col1:
            if col_region:
                st.markdown('**Top wilayah**')
                top_reg = df[col_region].value_counts().nlargest(20)
                fig, ax = plt.subplots(figsize=(6,5))
                ax.barh(top_reg.index, top_reg.values)
                st.pyplot(fig)

            if col_gender:
                st.markdown('**Proporsi Jenis Kelamin**')
                jk = df[col_gender].value_counts()
                fig2, ax2 = plt.subplots(figsize=(5,5))
                ax2.pie(jk.values, labels=jk.index, autopct='%1.1f%%')
                ax2.axis('equal')
                st.pyplot(fig2)

        with col2:
            if col_drug:
                st.markdown('**Top Jenis Narkoba**')
                top_drug = df[col_drug].value_counts().nlargest(20)
                fig3, ax3 = plt.subplots(figsize=(6,5))
                ax3.barh(top_drug.index, top_drug.values)
                st.pyplot(fig3)

            st.markdown('**Tren Bulanan**')
            monthly = df.groupby(df[col_date].dt.to_period('M')).size()
            monthly.index = monthly.index.to_timestamp()
            fig4, ax4 = plt.subplots(figsize=(6,3))
            ax4.plot(monthly.index, monthly.values, marker='o')
            st.pyplot(fig4)

        st.markdown('**Heatmap: Wilayah x Jenis Narkoba**')
        if col_region and col_drug:
            heat = df.groupby([col_region, col_drug]).size().unstack(fill_value=0)
            top_w = df[col_region].value_counts().nlargest(20).index
            top_d = df[col_drug].value_counts().nlargest(10).index
            heat = heat.loc[top_w, top_d]
            fig5, ax5 = plt.subplots(figsize=(12,6))
            sns.heatmap(heat, annot=True, fmt='d', cmap='YlOrRd', ax=ax5)
            st.pyplot(fig5)

        # -----------------------------
        # FORECAST
        # -----------------------------
        st.subheader(f'Forecast ({periods} hari)')
        model_choice = st.selectbox('Pilih metode', ['Prophet (jika tersedia)', 'ARIMA'])
        use_prophet_final = model_choice.startswith('Prophet') and PROPHET_AVAILABLE and use_prophet

        # jumlah kasus
        if use_prophet_final:
            try:
                total_fc = forecast_prophet(daily['jumlah_kasus'], periods)
                method_used = 'Prophet'
            except:
                total_fc = forecast_arima(daily['jumlah_kasus'], periods)
                method_used = 'ARIMA'
        else:
            total_fc = forecast_arima(daily['jumlah_kasus'], periods)
            method_used = 'ARIMA'

        # umur
        age_fc = None
        if not daily['rata2_umur'].isna().all():
            try:
                age_fc = forecast_prophet(daily['rata2_umur'], periods) if use_prophet_final else forecast_arima(daily['rata2_umur'], periods)
            except:
                age_fc = forecast_arima(daily['rata2_umur'], periods)

        # laki
        male_fc = None
        if not daily['prop_laki'].isna().all():
            try:
                male_fc = forecast_prophet(daily['prop_laki'], periods) if use_prophet_final else forecast_arima(daily['prop_laki'], periods)
            except:
                male_fc = forecast_arima(daily['prop_laki'], periods)

        st.write('Metode:', method_used)

        # plot forecast
        fig6, ax6 = plt.subplots(figsize=(10,4))
        ax6.plot(daily.index, daily['jumlah_kasus'], label='historical')
        ax6.plot(total_fc.index, total_fc.values, '--', label='forecast')
        ax6.legend()
        st.pyplot(fig6)

        if age_fc is not None:
            fig7, ax7 = plt.subplots(figsize=(10,4))
            ax7.plot(daily.index, daily['rata2_umur'], label='historical')
            ax7.plot(age_fc.index, age_fc.values, '--', label='forecast')
            ax7.legend()
            st.pyplot(fig7)

        if male_fc is not None:
            fig8, ax8 = plt.subplots(figsize=(10,4))
            ax8.plot(daily.index, daily['prop_laki'], label='historical')
            ax8.plot(male_fc.index, male_fc.values, '--', label='forecast')
            ax8.legend()
            st.pyplot(fig8)

        # -----------------------------
        # RINGKASAN PREDIKSI
        # -----------------------------
        st.subheader('Ringkasan Prediksi')

        summary = {
            'total_mean_next': float(total_fc.mean()),
            'total_trend': 'naik' if total_fc.iloc[-1] > total_fc.iloc[0] else 'turun',
            'avg_age_next_mean': float(age_fc.mean()) if age_fc is not None else None,
            'male_prop_next_mean': float(male_fc.mean()) if male_fc is not None else None
        }

        st.json(summary)

        # -----------------------------
        # WILAYAH RISIKO TERTINGGI
        # -----------------------------
        st.subheader('Wilayah Risiko Tertinggi')

        if col_region:
            regions = df[col_region].dropna().unique()
            region_means = {}

            for r in regions:
                per_day = df[df[col_region] == r].groupby(col_date).size()
                per_day = per_day.reindex(daily.index).fillna(0)
                region_means[r] = per_day.mean()

            region_df = pd.DataFrame(
                region_means.items(),
                columns=['Wilayah', 'Rata-rata Kasus Harian']
            ).sort_values('Rata-rata Kasus Harian', ascending=False)

            st.dataframe(region_df.head(10))

        else:
            st.info("Kolom wilayah tidak tersedia pada dataset.")

    except Exception as e:
        st.error(f"Terjadi kesalahan dalam memproses data: {e}")
