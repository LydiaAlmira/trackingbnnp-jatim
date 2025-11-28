import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import tempfile
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
# Main: process uploaded file
# -----------------------------
if uploaded is not None:
    try:
        with st.spinner('Membaca file...'):
            in_memory = uploaded.read()
            df = pd.read_excel(io.BytesIO(in_memory))

        st.subheader('Preview data (5 baris pertama)')
        st.write(df.head())

        # Deteksi kolom penting
        col_date = find_column(df, ['tanggal','tgl','date'])
        col_age = find_column(df, ['umur','usia','age'])
        col_gender = find_column(df, ['jenis kelamin','jk','gender','sex'])
        col_region = find_column(df, ['asal','wilayah','kota','kabupaten','daerah'])
        col_drug = find_column(df, ['narkoba','jenis barang','jenis narkoba','barang','nama barang','jenis'])

        st.markdown('**Kolom terdeteksi:**')
        cols_detected = {
            'Tanggal': col_date,
            'Umur': col_age,
            'Jenis Kelamin': col_gender,
            'Wilayah / Asal': col_region,
            'Jenis Narkoba': col_drug,
        }
        st.json(cols_detected)

        # Jika kolom tanggal tidak terdeteksi -> minta user pilih
        if col_date is None:
            st.error('Kolom tanggal tidak ditemukan otomatis. Silakan pilih kolom tanggal yang benar dari daftar:')
            chosen = st.selectbox('Pilih kolom tanggal', options=list(df.columns))
            col_date = chosen

        # === Perbaikan konversi tanggal ===
        # Simpan nilai mentah dulu
        raw_dates = df[col_date].copy()

        # 1) coba konversi biasa
        df[col_date] = pd.to_datetime(raw_dates, errors='coerce')

        # 2) Jika terlalu banyak NaT atau tahun tidak masuk akal (misal min year < 1900 atau max year > 2030)
        need_excel_serial = False
        nat_ratio = df[col_date].isna().mean()
        try:
            min_year = int(df[col_date].dropna().dt.year.min()) if df[col_date].notna().any() else None
            max_year = int(df[col_date].dropna().dt.year.max()) if df[col_date].notna().any() else None
        except Exception:
            min_year = None
            max_year = None

        if (nat_ratio > 0.2) or (min_year is not None and (min_year < 1900 or max_year > 2100)):
            need_excel_serial = True

        # 3) Jika perlu, coba konversi sebagai serial Excel (unit='d', origin Excel)
        if need_excel_serial:
            try:
                # ubah ke numeric dulu (beberapa cell bisa bertipe string)
                ser = pd.to_numeric(raw_dates, errors='coerce')
                df[col_date] = pd.to_datetime(ser, unit='d', origin='1899-12-30', errors='coerce')
            except Exception:
                pass

        # 4) Jika masih banyak NaT, coba lagi per baris: bila isi adalah angka tanpa titik, konversi sebagai serial
        if df[col_date].isna().mean() > 0.2:
            def try_convert_cell(x):
                # jika sudah datetime, kembalikan
                try:
                    if pd.notna(pd.to_datetime(x, errors='coerce')):
                        return pd.to_datetime(x, errors='coerce')
                except Exception:
                    pass
                # coba numeric -> excel serial
                try:
                    n = float(x)
                    return pd.to_datetime(n, unit='d', origin='1899-12-30', errors='coerce')
                except Exception:
                    return pd.NaT
            df[col_date] = raw_dates.apply(try_convert_cell)

        # drop baris tanpa tanggal valid
        df = df.dropna(subset=[col_date]).copy()
        df[col_date] = pd.to_datetime(df[col_date], errors='coerce').dt.floor('D')

        # Jika ada data tahun 2025, filter ke 2025 Januari–November sesuai permintaan
        if (df[col_date].dt.year == 2025).any():
            df = df[df[col_date].dt.year == 2025]
            # jika memang mau hanya sampai November:
            df = df[df[col_date].dt.month <= 11]

        # Normalisasi string kolom
        for c in [col_gender, col_region, col_drug]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

        # Aggregasi harian
        daily = df.groupby(col_date).agg(jumlah_kasus=(col_date,'size'))
        if col_age in df.columns:
            daily['rata2_umur'] = df.groupby(col_date)[col_age].mean()
        else:
            daily['rata2_umur'] = np.nan

        if col_gender in df.columns:
            df['_is_laki'] = df[col_gender].astype(str).str.lower().str.contains('l') | df[col_gender].astype(str).str.lower().str.contains('male') | df[col_gender].astype(str).str.lower().str.contains('laki')
            df['_is_perempuan'] = df[col_gender].astype(str).str.lower().str.contains('p') | df[col_gender].astype(str).str.lower().str.contains('female') | df[col_gender].astype(str).str.lower().str.contains('perempuan')
            g = df.groupby(col_date).agg(total=(col_date,'size'), laki=('_is_laki','sum'), perempuan=('_is_perempuan','sum'))
            g['prop_laki'] = g['laki']/g['total']
            g['prop_perempuan'] = g['perempuan']/g['total']
            daily = daily.merge(g[['prop_laki','prop_perempuan']], left_index=True, right_index=True, how='left')
        else:
            daily['prop_laki'] = np.nan
            daily['prop_perempuan'] = np.nan

        daily = daily.asfreq('D').fillna(method='ffill').fillna(0)

        st.subheader('Ringkasan Harian (sample)')
        st.dataframe(daily.head(20))

        # ----- Grafik lengkap -----
        st.subheader('Grafik Analisis')
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('**Top wilayah (bar)**')
            if col_region:
                top_reg = df[col_region].value_counts().nlargest(20)
                fig, ax = plt.subplots(figsize=(6,5))
                sns.barplot(x=top_reg.values, y=top_reg.index, ax=ax)
                ax.set_xlabel('Jumlah kasus')
                ax.set_ylabel('Wilayah')
                st.pyplot(fig)
            else:
                st.info('Kolom wilayah tidak ditemukan.')

            st.markdown('**Proporsi Jenis Kelamin (pie)**')
            if col_gender:
                jk_counts = df[col_gender].value_counts()
                fig2, ax2 = plt.subplots(figsize=(5,5))
                ax2.pie(jk_counts.values, labels=jk_counts.index, autopct='%1.1f%%')
                ax2.axis('equal')
                st.pyplot(fig2)
            else:
                st.info('Kolom jenis kelamin tidak ditemukan.')

        with col2:
            st.markdown('**Top jenis narkoba (bar)**')
            if col_drug:
                top_drug = df[col_drug].value_counts().nlargest(20)
                fig3, ax3 = plt.subplots(figsize=(6,5))
                sns.barplot(x=top_drug.values, y=top_drug.index, ax=ax3)
                ax3.set_xlabel('Jumlah kasus')
                ax3.set_ylabel('Jenis narkoba')
                st.pyplot(fig3)
            else:
                st.info('Kolom jenis narkoba tidak ditemukan.')

            st.markdown('**Tren bulanan (line)**')
            monthly = df.groupby(df[col_date].dt.to_period('M')).size()
            monthly.index = monthly.index.to_timestamp()
            fig4, ax4 = plt.subplots(figsize=(6,3))
            ax4.plot(monthly.index, monthly.values, marker='o')
            ax4.set_title('Tren kasus per bulan')
            ax4.set_xlabel('Bulan')
            ax4.set_ylabel('Jumlah kasus')
            st.pyplot(fig4)

        # Heatmap wilayah x narkoba
        st.markdown('**Heatmap: Wilayah vs Jenis Narkoba**')
        if col_region and col_drug:
            heat = df.groupby([col_region, col_drug]).size().unstack(fill_value=0)
            top_w = df[col_region].value_counts().nlargest(20).index
            top_d = df[col_drug].value_counts().nlargest(10).index
            heat_small = heat.loc[heat.index.isin(top_w), heat.columns.isin(top_d)]
            fig5, ax5 = plt.subplots(figsize=(12,6))
            sns.heatmap(heat_small, annot=True, fmt='d', cmap='YlOrRd', ax=ax5)
            ax5.set_xlabel('Jenis Narkoba')
            ax5.set_ylabel('Wilayah')
            st.pyplot(fig5)
        else:
            st.info('Butuh kolom wilayah dan jenis narkoba untuk heatmap.')

        # ----- Forecast -----
        st.subheader('Forecast (30 hari default)')
        model_choice = st.selectbox('Pilih metode forecasting', options=['Prophet (jika tersedia)','ARIMA (fallback)'])
        use_prophet_final = (model_choice.startswith('Prophet') and PROPHET_AVAILABLE and use_prophet)

        # forecast jumlah kasus
        with st.spinner('Menjalankan forecasting...'):
            if use_prophet_final:
                try:
                    total_fc = forecast_prophet(daily['jumlah_kasus'], periods=periods)
                    method_used = 'Prophet'
                except Exception as e:
                    st.warning('Prophet gagal, fallback ke ARIMA. Error: {}'.format(e))
                    total_fc = forecast_arima(daily['jumlah_kasus'], periods=periods)
                    method_used = 'ARIMA'
            else:
                total_fc = forecast_arima(daily['jumlah_kasus'], periods=periods)
                method_used = 'ARIMA'

            # umur
            if 'rata2_umur' in daily.columns and not daily['rata2_umur'].isnull().all():
                if use_prophet_final:
                    try:
                        age_fc = forecast_prophet(daily['rata2_umur'], periods=periods)
                    except:
                        age_fc = forecast_arima(daily['rata2_umur'], periods=periods)
                else:
                    age_fc = forecast_arima(daily['rata2_umur'], periods=periods)
            else:
                age_fc = None

            # prop laki
            if 'prop_laki' in daily.columns and not daily['prop_laki'].isnull().all():
                if use_prophet_final:
                    try:
                        male_fc = forecast_prophet(daily['prop_laki'], periods=periods)
                    except:
                        male_fc = forecast_arima(daily['prop_laki'], periods=periods)
                else:
                    male_fc = forecast_arima(daily['prop_laki'], periods=periods)
            else:
                male_fc = None

        st.write('Metode dipakai:', method_used)

        # tampilkan grafik hasil forecast
        fig6, ax6 = plt.subplots(figsize=(10,4))
        ax6.plot(daily.index, daily['jumlah_kasus'], label='historical')
        ax6.plot(total_fc.index, total_fc.values, '--', label='forecast')
        ax6.set_title('Jumlah Kasus: Historical + Forecast ({})'.format(periods))
        ax6.legend()
        st.pyplot(fig6)

        if age_fc is not None:
            fig7, ax7 = plt.subplots(figsize=(10,4))
            ax7.plot(daily.index, daily['rata2_umur'], label='historical')
            ax7.plot(age_fc.index, age_fc.values, '--', label='forecast')
            ax7.set_title('Rata2 Umur: Historical + Forecast')
            ax7.legend()
            st.pyplot(fig7)

        if male_fc is not None:
            fig8, ax8 = plt.subplots(figsize=(10,4))
            ax8.plot(daily.index, daily['prop_laki'], label='historical')
            ax8.plot(male_fc.index, male_fc.values, '--', label='forecast')
            ax8.set_title('Proporsi Laki: Historical + Forecast')
            ax8.legend()
            st.pyplot(fig8)

        # ----- Ringkasan prediksi -----
        st.subheader('Ringkasan Prediksi')
        summary = {}
        summary['total_mean_next'] = float(total_fc.mean())
        summary['total_trend'] = 'naik' if total_fc.iloc[-1] > total_fc.iloc[0] else ('turun' if total_fc.iloc[-1] < total_fc.iloc[0] else 'stabil')
        if age_fc is not None:
            summary['avg_age_next_mean'] = float(age_fc.mean())
            summary['avg_age_trend'] = 'naik' if age_fc.iloc[-1] > age_fc.iloc[0] else ('turun' if age_fc.iloc[-1] < age_fc.iloc[0] else 'stabil')
        else:
            summary['avg_age_next_mean'] = None
            summary['avg_age_trend'] = None
        if male_fc is not None:
            summary['male_prop_next_mean'] = float(male_fc.mean())
            summary['male_prop_trend'] = 'naik' if male_fc.iloc[-1] > male_fc.iloc[0] else ('turun' if male_fc.iloc[-1] < male_fc.iloc[0] else 'stabil')
        else:
            summary['male_prop_next_mean'] = None
            summary['male_prop_trend'] = None

        # highest risk region
        if col_region:
            region_means = {}
            regions = df[col_region].dropna().unique()
            for r in regions:
                s = df[df[col_region]==r].groupby(col_date).size()
                s.index = pd.to_datetime(s.index)
                s = s.asfreq('D').fillna(0)
                p = forecast_arima(s, periods=periods)
                region_means[r] = float(p.mean())
            highest = max(region_means.items(), key=lambda x: x[1])
            summary['highest_risk_region'] = highest[0]
            summary['highest_risk_region_mean_daily'] = highest[1]
        else:
            summary['highest_risk_region'] = None
            summary['highest_risk_region_mean_daily'] = None

        # dominant drug naive prediction
        if col_drug:
            drug_counts = df[col_drug].value_counts().to_dict()
            total_hist = df.shape[0]
            total_next30_mean = summary['total_mean_next']
            drug_pred = {d: (cnt/total_hist)*(total_next30_mean*periods) for d,cnt in drug_counts.items()}
            dom = max(drug_pred.items(), key=lambda x: x[1])
            summary['dominant_drug_next'] = dom[0]
            summary['dominant_drug_next_total_pred'] = dom[1]
        else:
            summary['dominant_drug_next'] = None
            summary['dominant_drug_next_total_pred'] = None

        st.json(summary)

        # ----- Export hasil prediksi -----
        st.subheader('Simpan Hasil Prediksi')
        export_df = pd.DataFrame({'date': total_fc.index, 'total_cases_pred': total_fc.values})
        if age_fc is not None:
            export_df['avg_age_pred'] = age_fc.values
        if male_fc is not None:
            export_df['male_prop_pred'] = male_fc.values
        if st.button('Simpan CSV hasil prediksi'):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            export_df.to_csv(tmp.name, index=False)
            with open(tmp.name, 'rb') as f:
                st.download_button('Download CSV', data=f, file_name='forecast_results.csv', mime='text/csv')

    except Exception as e:
        st.error(f'Terjadi kesalahan saat memproses file: {e}')
else:
    st.info('Unggah file Excel di sidebar untuk memulai analisis.')
