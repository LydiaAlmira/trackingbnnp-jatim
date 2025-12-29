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

def looks_like_gender_series(s):
    """
    Return True if series s looks like gender codes (L/P, Male/Female, etc).
    Heuristics: many values are single letter 'L'/'P' or gender-like words.
    """
    s_nonnull = s.dropna().astype(str).str.strip().str.lower()
    if s_nonnull.empty:
        return False
    # sample unique small values
    uniq = s_nonnull.unique()
    # if majority are single-letter l/p or words 'l','p','male','female','lk','pr'
    short_vals = sum(1 for x in s_nonnull if len(x) <= 2)
    if short_vals / len(s_nonnull) > 0.6:
        return True
    # if contains male/female words majority
    gender_words = {'l', 'p', 'lk', 'pr', 'male', 'female', 'man', 'woman'}
    cnt_gender_like = sum(1 for x in s_nonnull if any(g in x for g in gender_words))
    if cnt_gender_like / len(s_nonnull) > 0.5:
        return True
    return False

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
    # ensure series has datetime index and daily freq
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.asfreq('D').fillna(method='ffill')
    try:
        model = ARIMA(s, order=(1,1,1))
        fit = model.fit()
        pred = fit.get_forecast(steps=periods)
        idx = pd.date_range(s.index.max() + timedelta(days=1), periods=periods, freq='D')
        return pd.Series(pred.predicted_mean.values, index=idx)
    except Exception:
        # fallback: repeat last observed value
        non_na = s.dropna()
        last = float(non_na.iloc[-1]) if len(non_na) > 0 else 0.0
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

        # -----------------------------
        # Deteksi kolom (lebih aman)
        # -----------------------------
        # initial heuristics
        col_date = find_column(df, ['tanggal','tgl','date'])
        col_age = find_column(df, ['umur','usia','age'])
        col_gender = find_column(df, ['jenis kelamin','jk','gender','sex'])
        # prefer any explicit narkotika column first
        col_drug = find_column(df, ['jenis narkotika','jenis narkoba','narkotika','narkoba','jenis barang','nama barang'])
        # fallback to 'jenis' if not found (but we'll validate)
        if col_drug is None:
            col_drug = find_column(df, ['jenis', 'barang', 'jenis barang'])

        col_region = find_column(df, ['asal','wilayah','kota','kabupaten','daerah'])
        col_weight = find_column(df, ['berat','weight','kg','gram','jumlah'])  # otomatis deteksi kolom berat jika ada

        # Validate that col_drug is not actually gender column
        if col_drug is not None and col_drug in df.columns:
            sample = df[col_drug].dropna().astype(str).str.strip()
            if looks_like_gender_series(sample):
                # detected col_drug is likely gender column -> ignore it
                st.warning(f"Deteksi otomatis menemukan kolom '{col_drug}' untuk Jenis Narkotika, tetapi isinya terlihat seperti kode jenis kelamin (L/P).")
                col_drug = None

        # If still no drug column found, try scanning for 'nark' anywhere in columns
        if col_drug is None:
            for c in df.columns:
                if 'nark' in str(c).lower() or 'narkot' in str(c).lower():
                    col_drug = c
                    break

        # If still ambiguous, ask user to select correct column for each essential field.
        st.markdown('**Kolom terdeteksi (hasil awal, sesuaikan jika salah):**')
        col_info = {
            'Tanggal (deteksi)': col_date,
            'Umur (deteksi)': col_age,
            'Jenis Kelamin (deteksi)': col_gender,
            'Wilayah/Asal (deteksi)': col_region,
            'Jenis Narkotika (deteksi)': col_drug,
            'Kolom Berat (deteksi)': col_weight
        }
        st.json(col_info)

        # Force user to pick correct columns if detection ambiguous
        # DATE
        if col_date is None or col_date not in df.columns:
            st.error('Kolom tanggal tidak terdeteksi otomatis. Silakan pilih kolom tanggal:')
            col_date = st.selectbox('Kolom tanggal', df.columns, index=0)
        else:
            # still allow manual override
            if st.checkbox('Ganti kolom tanggal (manual)', value=False):
                col_date = st.selectbox('Kolom tanggal (manual)', df.columns, index=list(df.columns).index(col_date))

        # DRUG
        if col_drug is None or col_drug not in df.columns:
            st.error('Kolom Jenis Narkotika tidak terdeteksi dengan yakin. Silakan pilih kolom Jenis Narkotika:')
            col_drug = st.selectbox('Kolom Jenis Narkotika', options=list(df.columns))
        else:
            # validate content quickly and allow override
            sample = df[col_drug].dropna().astype(str).str.strip().str.lower()
            if looks_like_gender_series(sample):
                st.error(f"Kolom terpilih ({col_drug}) terlihat seperti kolom jenis kelamin. Silakan pilih kolom Jenis Narkotika secara manual.")
                col_drug = st.selectbox('Kolom Jenis Narkotika', options=list(df.columns))
            else:
                if st.checkbox(f'Ganti kolom Jenis Narkotika (default: {col_drug})', value=False):
                    col_drug = st.selectbox('Kolom Jenis Narkotika (manual)', options=list(df.columns), index=list(df.columns).index(col_drug))

        # GENDER
        if col_gender is None or col_gender not in df.columns:
            # try to find likely gender column by heuristics (has many L/P)
            guessed = None
            for c in df.columns:
                try:
                    if looks_like_gender_series(df[c].dropna().astype(str)):
                        guessed = c
                        break
                except Exception:
                    continue
            if guessed is not None:
                col_gender = guessed
            else:
                # ask user (optional)
                col_gender = st.selectbox('Pilih kolom jenis kelamin (jika ada). Pilih "-" jika tidak ada.', options=['-'] + list(df.columns))
                if col_gender == '-':
                    col_gender = None
        else:
            if st.checkbox(f'Ganti kolom Jenis Kelamin (default: {col_gender})', value=False):
                col_gender = st.selectbox('Kolom Jenis Kelamin (manual)', options=['-'] + list(df.columns))
                if col_gender == '-':
                    col_gender = None

        # REGION (allow manual override)
        if col_region is None or col_region not in df.columns:
            if st.checkbox('Pilih kolom wilayah/manual', value=True):
                col_region = st.selectbox('Kolom wilayah/asl', options=['-'] + list(df.columns))
                if col_region == '-':
                    col_region = None
        else:
            if st.checkbox(f'Ganti kolom wilayah (default: {col_region})', value=False):
                col_region = st.selectbox('Kolom wilayah (manual)', options=['-'] + list(df.columns))
                if col_region == '-':
                    col_region = None

        # AGE override option
        if col_age is None or col_age not in df.columns:
            if st.checkbox('Pilih kolom umur/manual (opsional)', value=False):
                col_age = st.selectbox('Kolom umur', options=['-'] + list(df.columns))
                if col_age == '-':
                    col_age = None
        else:
            if st.checkbox(f'Ganti kolom umur (default: {col_age})', value=False):
                col_age = st.selectbox('Kolom umur (manual)', options=['-'] + list(df.columns))
                if col_age == '-':
                    col_age = None

        # Kolom berat manual
        if col_weight is None or col_weight not in df.columns:
            if st.checkbox('Pilih kolom berat/manual (opsional)', value=False):
                col_weight = st.selectbox('Kolom berat', options=['-'] + list(df.columns))
                if col_weight == '-':
                    col_weight = None
        else:
            if st.checkbox(f'Ganti kolom berat (default: {col_weight})', value=False):
                col_weight = st.selectbox('Kolom berat (manual)', options=['-'] + list(df.columns))
                if col_weight == '-':
                    col_weight = None

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

        # Filter tahun 2025 s.d November jika ada data 2025
        try:
            if (df[col_date].dt.year == 2025).any():
                df = df[df[col_date].dt.year == 2025]
                df = df[df[col_date].dt.month <= 11]
        except Exception:
            pass

        # Normalisasi string kolom kategori
        for c in [col_gender, col_region, col_drug]:
            if c and c in df.columns:
                df[c] = df[c].astype(str).str.strip()

        # Jika ada kolom berat, pastikan numeric
        if col_weight and col_weight in df.columns:
            # remove commas, extract numeric part
            df[col_weight] = pd.to_numeric(
                df[col_weight].astype(str).str.replace(',', '').str.extract(r'([0-9\.]+)')[0],
                errors='coerce'
            )

        # Aggregasi harian
        # Use safer groupby column reference: group by date column name
        daily = df.groupby(col_date).agg(jumlah_kasus=(col_date,'size')).rename_axis(index=col_date)

        if col_age and col_age in df.columns:
            daily['rata2_umur'] = df.groupby(col_date)[col_age].mean()
        else:
            daily['rata2_umur'] = np.nan

        if col_gender and col_gender in df.columns:
            df['_laki'] = df[col_gender].astype(str).str.lower().str.contains('l') | df[col_gender].astype(str).str.lower().str.contains('male') | df[col_gender].astype(str).str.lower().str.contains('lk')
            df['_perempuan'] = df[col_gender].astype(str).str.lower().str.contains('p') | df[col_gender].astype(str).str.lower().str.contains('female') | df[col_gender].astype(str).str.lower().str.contains('pr')
            g = df.groupby(col_date).agg(
                total=(col_date,'size'),
                laki=('_laki','sum'),
                perempuan=('_perempuan','sum')
            )
            g['prop_laki'] = g['laki']/g['total']
            g['prop_perempuan'] = g['perempuan']/g['total']
            daily = daily.merge(g[['prop_laki','prop_perempuan']], left_index=True, right_index=True, how='left')
        else:
            daily['prop_laki'] = np.nan
            daily['prop_perempuan'] = np.nan

        # Ensure daily has a daily datetime index
        daily.index = pd.to_datetime(daily.index)
        daily = daily.asfreq('D').fillna(method='ffill').fillna(0)

        st.subheader('Ringkasan Harian (sample)')
        st.dataframe(daily.head(20))

        # -----------------------------
        # GRAFIK
        # -----------------------------
        st.subheader('Grafik Analisis')
        col1, col2 = st.columns(2)

        with col1:
            if col_region and col_region in df.columns:
                st.markdown('**Top wilayah**')
                top_reg = df[col_region].value_counts().nlargest(20)
                fig, ax = plt.subplots(figsize=(6,5))
                ax.barh(top_reg.index, top_reg.values)
                ax.set_xlabel('Jumlah kasus')
                ax.set_ylabel('Wilayah')
                st.pyplot(fig)
            else:
                st.info('Kolom wilayah tidak ditemukan.')

            if col_gender and col_gender in df.columns:
                st.markdown('**Proporsi Jenis Kelamin**')
                jk = df[col_gender].value_counts()
                fig2, ax2 = plt.subplots(figsize=(5,5))
                ax2.pie(jk.values, labels=jk.index, autopct='%1.1f%%')
                ax2.axis('equal')
                st.pyplot(fig2)
            else:
                st.info('Kolom jenis kelamin tidak ditemukan.')

        with col2:
            if col_drug and col_drug in df.columns:
                st.markdown('**Top Jenis Narkoba**')
                top_drug = df[col_drug].value_counts().nlargest(20)
                fig3, ax3 = plt.subplots(figsize=(6,5))
                ax3.barh(top_drug.index, top_drug.values)
                ax3.set_xlabel('Jumlah kasus')
                ax3.set_ylabel('Jenis Narkoba')
                st.pyplot(fig3)
            else:
                st.info('Kolom jenis narkoba tidak ditemukan.')

            st.markdown('**Tren Bulanan**')
            try:
                monthly = df.groupby(df[col_date].dt.to_period('M')).size()
                monthly.index = monthly.index.to_timestamp()
                fig4, ax4 = plt.subplots(figsize=(6,3))
                ax4.plot(monthly.index, monthly.values, marker='o')
                ax4.set_xlabel('Bulan')
                ax4.set_ylabel('Jumlah kasus')
                st.pyplot(fig4)
            except Exception:
                st.info('Tidak cukup data untuk membuat tren bulanan.')

        st.markdown('**Heatmap: Wilayah x Jenis Narkoba (jumlah kasus)**')
        if col_region and col_region in df.columns and col_drug and col_drug in df.columns:
            try:
                # safer: count by groupby then unstack
                heat = df.groupby([col_region, col_drug]).size().unstack(fill_value=0)
                top_w = df[col_region].value_counts().nlargest(20).index
                top_d = df[col_drug].value_counts().nlargest(10).index
                heat = heat.reindex(index=top_w, columns=top_d, fill_value=0)
                if heat.shape[0] == 0 or heat.shape[1] == 0:
                    st.info('Pivot heatmap kosong — tidak cukup kombinasi wilayah/jenis narkoba.')
                else:
                    fig5, ax5 = plt.subplots(figsize=(12,6))
                    sns.heatmap(heat, annot=True, fmt='d', cmap='YlOrRd', ax=ax5)
                    ax5.set_xlabel('Jenis Narkoba')
                    ax5.set_ylabel('Wilayah')
                    st.pyplot(fig5)
            except Exception:
                st.info('Gagal membuat heatmap (periksa ukuran pivot).')
        else:
            st.info('Butuh kolom wilayah dan jenis narkoba untuk heatmap ini.')

        # -----------------------------
        # HEATMAP TAMBAHAN (empat varian)
        # -----------------------------
        # 1) Heatmap: Jumlah Kasus × Jenis Narkoba × Wilayah (lebih lengkap)
        st.subheader("Heatmap: Jumlah Kasus × Jenis Narkoba × Wilayah (detail)")

        if col_region and col_region in df.columns and col_drug and col_drug in df.columns:
            try:
                # REPLACED: pivot_table with values=col_drug (which could be string) -> use groupby size
                heat_case = df.groupby([col_region, col_drug]).size().unstack(fill_value=0)
                if heat_case.shape[0] == 0 or heat_case.shape[1] == 0:
                    st.info("Data tidak cukup untuk heatmap jumlah kasus (detail).")
                else:
                    fig_hc, ax_hc = plt.subplots(figsize=(12, 8))
                    sns.heatmap(heat_case, annot=True, fmt="d", cmap="YlOrRd", ax=ax_hc)
                    ax_hc.set_xlabel("Jenis Narkoba")
                    ax_hc.set_ylabel("Wilayah")
                    ax_hc.set_title("Jumlah Kasus per Jenis Narkoba")
                    st.pyplot(fig_hc)
            except Exception:
                st.info("Gagal membuat heatmap jumlah kasus.")
        else:
            st.info("Kolom wilayah/jenis narkoba tidak lengkap untuk heatmap jumlah kasus.")

        # 2) Heatmap: Rata-rata Berat × Jenis Narkoba × Wilayah
        st.subheader("Heatmap: Rata-rata Berat × Jenis Narkoba × Wilayah")

        if col_region and col_region in df.columns and col_drug and col_drug in df.columns and col_weight and col_weight in df.columns:
            try:
                heat_avg = df.pivot_table(
                    index=col_region,
                    columns=col_drug,
                    values=col_weight,
                    aggfunc="mean",
                    fill_value=0
                )
                if heat_avg.shape[0] == 0 or heat_avg.shape[1] == 0:
                    st.info("Tidak cukup data untuk heatmap rata-rata berat.")
                else:
                    fig_ha, ax_ha = plt.subplots(figsize=(12, 8))
                    sns.heatmap(heat_avg, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax_ha)
                    ax_ha.set_xlabel("Jenis Narkoba")
                    ax_ha.set_ylabel("Wilayah")
                    ax_ha.set_title("Rata-Rata Berat Narkoba per Wilayah")
                    st.pyplot(fig_ha)
            except Exception:
                st.info("Gagal membuat heatmap rata-rata berat.")
        else:
            st.warning("Kolom 'BERAT' tidak ditemukan atau kolom wilayah/jenis narkoba kurang lengkap.")

        # 3) Heatmap: Tren Bulanan Jenis Narkoba
        st.subheader("Heatmap: Tren Bulanan Jenis Narkoba")

        if col_drug and col_date and col_drug in df.columns:
            try:
                df['_tmp_date_for_month'] = pd.to_datetime(df[col_date], errors='coerce')
                df['BULAN'] = df['_tmp_date_for_month'].dt.to_period("M").astype(str)
                # REPLACED: pivot_table using values=col_drug -> safer to use groupby size
                trend_monthly = df.groupby(["BULAN", col_drug]).size().unstack(fill_value=0)
                if trend_monthly.shape[0] == 0 or trend_monthly.shape[1] == 0:
                    st.info("Tidak cukup data untuk membuat heatmap tren bulanan.")
                else:
                    # convert index to something seaborn likes (string labels)
                    fig_tm, ax_tm = plt.subplots(figsize=(14, 7))
                    sns.heatmap(trend_monthly, annot=True, fmt="d", cmap="Oranges", ax=ax_tm)
                    ax_tm.set_xlabel("Jenis Narkoba")
                    ax_tm.set_ylabel("Bulan")
                    ax_tm.set_title("Tren Bulanan Jenis Narkoba")
                    st.pyplot(fig_tm)
            except Exception:
                st.info("Gagal membuat heatmap tren bulanan.")
            finally:
                df.drop(columns=['_tmp_date_for_month','BULAN'], inplace=True, errors='ignore')
        else:
            st.info("Kolom tanggal atau jenis narkoba tidak lengkap untuk heatmap tren bulanan.")

        # 4) Heatmap: Total Berat Narkoba × Wilayah × Jenis
        st.subheader("Heatmap: Total Berat Narkoba × Wilayah × Jenis")

        if col_region and col_region in df.columns and col_drug and col_drug in df.columns and col_weight and col_weight in df.columns:
            try:
                rank_heat = df.pivot_table(
                    index=col_region,
                    columns=col_drug,
                    values=col_weight,
                    aggfunc="sum",
                    fill_value=0
                )
                if rank_heat.shape[0] == 0 or rank_heat.shape[1] == 0:
                    st.info("Tidak cukup data untuk heatmap total berat.")
                else:
                    fig_rh, ax_rh = plt.subplots(figsize=(12, 8))
                    sns.heatmap(rank_heat, annot=True, fmt=".1f", cmap="Reds", ax=ax_rh)
                    ax_rh.set_xlabel("Jenis Narkoba")
                    ax_rh.set_ylabel("Wilayah")
                    ax_rh.set_title("Total Berat Narkoba per Wilayah")
                    st.pyplot(fig_rh)
            except Exception:
                st.info("Gagal membuat heatmap total berat.")
        else:
            st.warning("Kolom 'BERAT' tidak ditemukan, heatmap total berat tidak dapat ditampilkan.")

        # ------------------------------------------------------
        # FORECAST (diletakkan di dalam blok uploaded processed)
        # ------------------------------------------------------
        st.subheader(f"Forecast {periods} Hari Ke Depan")

        model_choice = st.selectbox("Pilih metode", ["Prophet (jika tersedia)", "ARIMA"])
        use_prophet_final = model_choice.startswith("Prophet") and PROPHET_AVAILABLE and use_prophet


        # Ensure 'daily' exists and has the expected columns
        if 'jumlah_kasus' not in daily.columns or daily.shape[0] == 0:
            st.warning("Data harian tidak cukup untuk forecasting.")
        else:

            # ------------------------------------------------------
            # FORECAST — JUMLAH KASUS
            # ------------------------------------------------------
            try:
                if use_prophet_final:
                    try:
                        total_fc = forecast_prophet(daily["jumlah_kasus"], periods)
                        method_used = "Prophet"
                    except Exception:
                        total_fc = forecast_arima(daily["jumlah_kasus"], periods)
                        method_used = "ARIMA"
                else:
                    total_fc = forecast_arima(daily["jumlah_kasus"], periods)
                    method_used = "ARIMA"
            except Exception as e:
                st.error(f"Terjadi kesalahan saat forecasting jumlah kasus: {e}")
                total_fc = None
                method_used = "Error"

            st.info(f"Metode forecasting yang digunakan: **{method_used}**")

            # ------------------------------------------------------
            # FORECAST — USIA
            # ------------------------------------------------------
            age_fc = None
            if 'rata2_umur' in daily.columns and not daily['rata2_umur'].isna().all():
                try:
                    age_fc = forecast_prophet(daily["rata2_umur"], periods) if use_prophet_final else forecast_arima(daily["rata2_umur"], periods)
                except Exception:
                    age_fc = forecast_arima(daily["rata2_umur"], periods)

            # ------------------------------------------------------
            # FORECAST — PROPORSI LAKI-LAKI
            # ------------------------------------------------------
            male_fc = None
            if 'prop_laki' in daily.columns and not daily['prop_laki'].isna().all():
                try:
                    male_fc = forecast_prophet(daily["prop_laki"], periods) if use_prophet_final else forecast_arima(daily["prop_laki"], periods)
                except Exception:
                    male_fc = forecast_arima(daily["prop_laki"], periods)

            # ------------------------------------------------------
            # FORECAST — PROPORSI PEREMPUAN
            # ------------------------------------------------------
            female_fc = None
            if 'prop_perempuan' in daily.columns and not daily['prop_perempuan'].isna().all():
                try:
                    female_fc = forecast_prophet(daily["prop_perempuan"], periods) if use_prophet_final else forecast_arima(daily["prop_perempuan"], periods)
                except Exception:
                    female_fc = forecast_arima(daily["prop_perempuan"], periods)

            # ------------------------------------------------------
            # PLOT — JUMLAH KASUS
            # ------------------------------------------------------
            if total_fc is not None:
                st.subheader("Prediksi Jumlah Kasus")
                fig1, ax1 = plt.subplots(figsize=(10,4))
                ax1.plot(daily.index, daily["jumlah_kasus"], label="Aktual")
                ax1.plot(total_fc.index, total_fc.values, '--', label="Forecast")
                ax1.legend()
                st.pyplot(fig1)

            # ------------------------------------------------------
            # PLOT — USIA
            # ------------------------------------------------------
            if age_fc is not None:
                st.subheader("Prediksi Rata-rata Umur")
                fig2, ax2 = plt.subplots(figsize=(10,4))
                ax2.plot(daily.index, daily["rata2_umur"], label="Aktual")
                ax2.plot(age_fc.index, age_fc.values, '--', label="Forecast")
                ax2.legend()
                st.pyplot(fig2)

            # ------------------------------------------------------
            # PLOT — LAKI-LAKI
            # ------------------------------------------------------
            if male_fc is not None:
                st.subheader("Prediksi Proporsi Laki-laki")
                fig3, ax3 = plt.subplots(figsize=(10,4))
                ax3.plot(daily.index, daily["prop_laki"], label="Aktual")
                ax3.plot(male_fc.index, male_fc.values, '--', label="Forecast")
                ax3.legend()
                st.pyplot(fig3)

            # ------------------------------------------------------
            # PLOT — PEREMPUAN
            # ------------------------------------------------------
            if female_fc is not None:
                st.subheader("Prediksi Proporsi Perempuan")
                fig4, ax4 = plt.subplots(figsize=(10,4))
                ax4.plot(daily.index, daily["prop_perempuan"], label="Aktual")
                ax4.plot(female_fc.index, female_fc.values, '--', label="Forecast")
                ax4.legend()
                st.pyplot(fig4)

            # =============================
            # RINGKASAN PREDIKSI
            # =============================
            st.subheader("Ringkasan Prediksi")
            st.json({
                "mean_jumlah_kasus": float(total_fc.mean()) if total_fc is not None else None,
                "tren_jumlah_kasus": (
                    "naik" if (total_fc is not None and total_fc.iloc[-1] > total_fc.iloc[0])
                    else "turun"
                ) if total_fc is not None else None,
                "mean_umur": float(age_fc.mean()) if age_fc is not None else None,
                "mean_prop_laki": float(male_fc.mean()) if male_fc is not None else None,
                "mean_prop_perempuan": float(female_fc.mean()) if female_fc is not None else None,
            })
    
            # =============================
            # TABEL HASIL PREDIKSI
            # =============================
            forecast_df = pd.DataFrame({
                "Tanggal": total_fc.index if total_fc is not None else [],
                "Prediksi_Jumlah_Kasus": total_fc.values if total_fc is not None else [],
            })
    
            if age_fc is not None:
                forecast_df["Prediksi_Rata2_Umur"] = age_fc.values
            if male_fc is not None:
                forecast_df["Proporsi_Laki_Laki"] = male_fc.values
            if female_fc is not None:
                forecast_df["Proporsi_Perempuan"] = female_fc.values
    
            forecast_df["Tanggal"] = pd.to_datetime(forecast_df["Tanggal"])
    
            # simpan ke session_state (PENTING)
            st.session_state["forecast_df"] = forecast_df
    
            st.subheader("Tabel Hasil Prediksi")
            st.dataframe(forecast_df, use_container_width=True)
    
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
    
    else:
        st.info("Silakan unggah file Excel untuk memulai analisis.")
    
    # =============================
    # DOWNLOAD EXCEL (DI LUAR TRY)
    # =============================
    if "forecast_df" in st.session_state:
    
        def convert_df_to_excel(df):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Forecast")
            return output.getvalue()
    
        excel_data = convert_df_to_excel(st.session_state["forecast_df"])
    
        st.download_button(
            label="📥 Download Hasil Prediksi (Excel)",
            data=excel_data,
            file_name="hasil_prediksi_kasus_narkoba.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
