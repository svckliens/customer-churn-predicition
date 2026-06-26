import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# LOAD MODEL & INFO
# ──────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    """Load model, scaler, dan metadata."""
    model_dir = "model"
    artifacts = {}

    model_path  = os.path.join(model_dir, "best_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    info_path   = os.path.join(model_dir, "model_info.json")
    results_path = os.path.join(model_dir, "results_all_models.csv")

    if not os.path.exists(model_path):
        return None

    artifacts['model']  = joblib.load(model_path)
    artifacts['scaler'] = joblib.load(scaler_path)

    if os.path.exists(info_path):
        with open(info_path) as f:
            artifacts['info'] = json.load(f)

    if os.path.exists(results_path):
        artifacts['results_df'] = pd.read_csv(results_path)

    return artifacts


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Customer Churn Prediction")
    st.markdown("---")
    st.markdown("### Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini memprediksi kemungkinan seorang pelanggan akan **churn** (berhenti berlangganan)
    berdasarkan data perilaku dan transaksi mereka.

    **Dataset:** Sales & Marketing Dataset (Kaggle)
    """)

    st.markdown("---")
    artifacts = load_model_artifacts()

    if artifacts and 'info' in artifacts:
        info = artifacts['info']
        st.markdown("### Model Terbaik")
        st.markdown(f"**{info.get('model_name', 'N/A')}**")
        st.markdown(f"Skenario: `{info.get('scenario', 'N/A')}`")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{info.get('accuracy', 0)*100:.1f}%")
            st.metric("Recall", f"{info.get('recall', 0)*100:.1f}%")
        with col2:
            st.metric("F1-Score", f"{info.get('f1_score', 0):.4f}")
            st.metric("Precision", f"{info.get('precision', 0)*100:.1f}%")

    st.markdown("---")
    st.markdown("### Developer")
    st.markdown("**Aditya Rendy Setyawan**")
    st.markdown("NIM: A11.2023.15189")
    st.markdown("UAS Bengkel Koding Data Science")


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;"> <h1>Customer Churn Prediction</h1>
    <p>Prediksi kemungkinan pelanggan berhenti berlangganan menggunakan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Prediksi Churn", "Performa Model", "Informasi Fitur"])

# ══════════════════════════════════════════════
# TAB 1: PREDIKSI
# ══════════════════════════════════════════════
with tab1:
    if artifacts is None:
        st.warning("""
        ⚠️ **Model belum tersedia!**

        Silakan jalankan notebook `CUSTOMER_CHURN_PREDICTION.ipynb` terlebih dahulu
        untuk melatih dan menyimpan model ke folder `model/`.

        ```bash
        jupyter notebook CUSTOMER_CHURN_PREDICTION.ipynb
        ```

        **Jika sudah deploy ke Streamlit Cloud:** Pastikan folder `model/` (berisi `best_model.pkl`, `scaler.pkl`, `model_info.json`, `results_all_models.csv`) sudah ter-push ke repository GitHub Anda.
        """)
    else:
        info = artifacts.get('info', {})
        features_needed = info.get('features', info.get('all_features', []))

        st.subheader("Input Data Pelanggan")

        # ── Form Input ──────────────────────────
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            input_data = {}

            # Column 1: Profil & Penggunaan
            with col1:
                st.markdown("### 👤 Profil & Penggunaan")
                input_data['age'] = st.number_input("Usia (tahun)", min_value=18, max_value=90, value=35, help="Usia dari pelanggan saat ini")
                
                sub_opts = {'Basic': 0, 'Pro': 1, 'Premium': 2}
                sub_sel = st.selectbox("Tipe Langganan", list(sub_opts.keys()), help="Kategori paket keanggotaan pelanggan")
                input_data['subscription_type'] = sub_opts[sub_sel]
                
                input_data['total_visits'] = st.number_input("Total Kunjungan", min_value=0, max_value=500, value=25, help="Berapa kali pelanggan mengunjungi platform")
                input_data['avg_session_time'] = st.slider("Rata-rata Waktu Sesi (menit)", 1.0, 120.0, 20.0, help="Durasi rata-rata yang dihabiskan dalam sekali kunjungan")

            # Column 2: Finansial & Kepuasan
            with col2:
                st.markdown("### 💰 Finansial & Kepuasan")
                input_data['total_spent'] = st.number_input("Total Pengeluaran (USD)", min_value=0.0, max_value=10000.0, value=350.0, help="Akumulasi biaya yang sudah dibelanjakan pelanggan")
                input_data['support_tickets'] = st.number_input("Jumlah Tiket Keluhan (Support)", min_value=0, max_value=50, value=1, help="Jumlah laporan keluhan yang dikirim pelanggan")
                input_data['satisfaction_score'] = st.slider("Skor Kepuasan (1-5)", 1.0, 5.0, 3.5, help="Skor kepuasan pelanggan terhadap layanan")

                input_data['last_3_month_purchase_freq'] = st.number_input("Frekuensi Pembelian 3 Bulan Terakhir", min_value=0, max_value=100, value=3, help="Jumlah pembelian dalam 3 bulan terakhir")
                
                discount_opts = {'Tidak': 0, 'Ya': 1}
                discount_sel = st.selectbox("Menggunakan Diskon?", list(discount_opts.keys()), help="Apakah pelanggan pernah menggunakan diskon signifikan?")
                input_data['discount_used'] = discount_opts[discount_sel]
                
                refund_opts = {'Tidak': 0, 'Ya': 1}
                refund_sel = st.selectbox("Pernah Meminta Refund?", list(refund_opts.keys()), help="Apakah pelanggan pernah meminta pengembalian dana?")
                input_data['refund_requested'] = refund_opts[refund_sel]

            # Define default values for non-UI features to ensure model compatibility
            default_values = {
                'is_premium_user': 1 if sub_sel in ['Pro', 'Premium'] else 0,
                'gender': 0, # Default: Male
                'device_type': 1, # Default: Mobile
                'payment_method': 1, # Default: Credit Card
                'acquisition_channel': 0, # Default: Ads
                'country': 1, # Default: Indonesia/Region 1
                'city': 0,
                'pages_per_session': 6.0,
                'email_open_rate': 0.25,
                'email_click_rate': 0.08,
                'avg_order_value': input_data['total_spent'] / max(1, input_data['total_visits']),
                'lifetime_value': input_data['total_spent'] * 1.2,
                'marketing_spend_per_user': 25.0,
                'delivery_delay_days': 1,
                'nps_score': int((input_data['satisfaction_score'] - 3.0) * 40) # Estimate NPS based on satisfaction
            }

            # Merge UI inputs and default values
            for feat, val in default_values.items():
                input_data[feat] = val

            # Submit button
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Prediksi Churn Sekarang")

        # ── PREDICTION ──────────────────────────
        if submitted:
            try:
                model  = artifacts['model']
                scaler = artifacts['scaler']

                # Build input dataframe in correct feature order
                input_df = pd.DataFrame([input_data])

                # Align columns with what the model expects
                all_features = info.get('all_features', features_needed)
                for col in all_features:
                    if col not in input_df.columns:
                        input_df[col] = 0  # default 0 for missing features

                input_df = input_df[all_features]

                # Scale
                input_scaled = scaler.transform(input_df)

                # Select top features if model was trained on subset
                if info.get('scenario') == 'Tuning':
                    top_feats = info.get('top_features', all_features)
                    idx = [all_features.index(f) for f in top_feats if f in all_features]
                    input_scaled = input_scaled[:, idx]

                # Predict
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None

                # ── Result Display ───────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("### Hasil Prediksi")

                res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
                with res_col2:
                    if prediction == 1:
                        prob_churn = proba[1] if proba is not None else 0.8
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #ff416c, #ff4b2b); border-radius: 20px; padding: 2rem; text-align: center; box-shadow: 0 15px 40px rgba(255, 65, 108, 0.4);">
                            <div style="font-size: 3rem; margin-bottom: 0.5rem;">⚠️</div>
                            <p style="color: white; font-size: 1.5rem; font-weight: 700; margin: 0;">PELANGGAN BERISIKO CHURN</p>
                            <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-top: 0.5rem;">Probabilitas Churn: <strong>{prob_churn*100:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        prob_no_churn = proba[0] if proba is not None else 0.8
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #11998e, #38ef7d); border-radius: 20px; padding: 2rem; text-align: center; box-shadow: 0 15px 40px rgba(17, 153, 142, 0.4);">
                            <div style="font-size: 3rem; margin-bottom: 0.5rem;">✅</div>
                            <p style="color: white; font-size: 1.5rem; font-weight: 700; margin: 0;">PELANGGAN AMAN (TIDAK CHURN)</p>
                            <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-top: 0.5rem;">Probabilitas Bertahan: <strong>{prob_no_churn*100:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)

                # ── Probability Gauge ────────────
                if proba is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    prob_churn = proba[1]

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prob_churn * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Probabilitas Churn (%)", 'font': {'size': 18, 'color': 'white'}},
                        number={'font': {'color': 'white', 'size': 40}},
                        delta={'reference': 50, 'increasing': {'color': "#FF4136"}, 'decreasing': {'color': "#2ECC40"}, 'font': {'color': 'white'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
                            'bar': {'color': "#FF4136" if prob_churn > 0.5 else "#2ECC40"},
                            'bgcolor': "rgba(255,255,255,0.05)",
                            'borderwidth': 2,
                            'bordercolor': "rgba(255,255,255,0.2)",
                            'steps': [
                                {'range': [0, 30], 'color': 'rgba(46,204,64,0.3)'},
                                {'range': [30, 60], 'color': 'rgba(255,220,0,0.3)'},
                                {'range': [60, 100], 'color': 'rgba(255,65,54,0.3)'}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': 'white'},
                        height=320
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ── Recommendations ─────────────
                st.markdown("<br>", unsafe_allow_html=True)
                if prediction == 1:
                    st.markdown("### Rekomendasi Tindakan")
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.error("**Tindakan Segera:**")
                        st.markdown("""
                        - Hubungi pelanggan untuk survei kepuasan
                        - Berikan penawaran retensi khusus
                        - Review dan tangani support tickets yang tertunda
                        - Tawarkan upgrade atau perpanjangan dengan diskon
                        """)
                    with col_r2:
                        st.warning("**Faktor Risiko:**")
                        st.markdown("""
                        - Periksa skor kepuasan pelanggan
                        - Evaluasi frekuensi pembelian terakhir
                        - Monitor email engagement rate
                        - Tinjau riwayat refund dan keluhan
                        """)
                else:
                    st.success("**✅ Pelanggan tampak puas!** Pertahankan dengan program loyalty.")

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")
                st.info("Pastikan notebook sudah dijalankan dan model sudah tersimpan di folder `model/`.")


# ══════════════════════════════════════════════
# TAB 2: PERFORMA MODEL
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Perbandingan Performa 9 Model")

    if artifacts and 'results_df' in artifacts:
        results_df = artifacts['results_df']

        # Summary table
        st.dataframe(
            results_df.style
                .background_gradient(subset=['F1-Score', 'Accuracy', 'Recall'], cmap='RdYlGn')
                .format({'Accuracy': '{:.4f}', 'Precision': '{:.4f}',
                         'Recall': '{:.4f}', 'F1-Score': '{:.4f}'}),
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # F1-Score chart
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_f1 = px.bar(
                results_df, x='Model', y='F1-Score', color='Scenario',
                barmode='group', title='Perbandingan F1-Score per Model & Skenario',
                color_discrete_map={'Direct': '#FF7043', 'Preprocessing': '#42A5F5', 'Tuning': '#66BB6A'}
            )
            fig_f1.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(tickangle=15),
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        with col_c2:
            fig_acc = px.bar(
                results_df, x='Model', y='Accuracy', color='Scenario',
                barmode='group', title='Perbandingan Accuracy per Model & Skenario',
                color_discrete_map={'Direct': '#FF7043', 'Preprocessing': '#42A5F5', 'Tuning': '#66BB6A'}
            )
            fig_acc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(tickangle=15),
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        # Radar chart - best model metrics
        if artifacts and 'info' in artifacts:
            info = artifacts['info']
            categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            values = [info.get('accuracy', 0), info.get('precision', 0),
                      info.get('recall', 0), info.get('f1_score', 0)]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.3)',
                line=dict(color='#667eea', width=2),
                name=info.get('model_name', 'Best Model')
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1]),
                    angularaxis=dict()
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title=f"Metrik Model Terbaik: {info.get('model_name', '')}",
                showlegend=True
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    else:
        st.info("Jalankan notebook terlebih dahulu untuk melihat hasil performa model.")
        # Show placeholder image if available
        if os.path.exists("rangkuman_9model.png"):
            st.image("rangkuman_9model.png", caption="Rangkuman 9 Model", use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3: INFORMASI FITUR
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Penjelasan Fitur Dataset")

    feature_info = {
        "customer_id": ("int64", "ID unik setiap pelanggan"),
        "gender": ("object", "Jenis kelamin pelanggan (Male/Female)"),
        "age": ("float64", "Usia pelanggan (tahun)"),
        "country": ("object", "Negara asal pelanggan"),
        "city": ("object", "Kota pelanggan"),
        "signup_date": ("object", "Tanggal pelanggan mendaftar"),
        "last_purchase_date": ("object", "Tanggal transaksi terakhir"),
        "acquisition_channel": ("object", "Sumber akuisisi pelanggan (Email, Ads, dll)"),
        "device_type": ("object", "Jenis perangkat yang digunakan"),
        "subscription_type": ("object", "Jenis langganan (Basic/Premium/Pro)"),
        "is_premium_user": ("int64", "Status pengguna premium (0=Tidak, 1=Ya)"),
        "total_visits": ("int64", "Total kunjungan pelanggan ke platform"),
        "avg_session_time": ("float64", "Rata-rata waktu sesi penggunaan (menit)"),
        "pages_per_session": ("float64", "Rata-rata halaman per sesi"),
        "email_open_rate": ("float64", "Persentase email yang dibuka (0-1)"),
        "email_click_rate": ("float64", "Persentase klik pada email (0-1)"),
        "total_spent": ("float64", "Total pengeluaran pelanggan (USD)"),
        "avg_order_value": ("float64", "Rata-rata nilai transaksi (USD)"),
        "discount_used": ("int64", "Penggunaan diskon (0=Tidak, 1=Ya)"),
        "coupon_code": ("object", "Kode kupon yang digunakan"),
        "support_tickets": ("int64", "Jumlah tiket dukungan yang diajukan"),
        "refund_requested": ("int64", "Permintaan refund (0=Tidak, 1=Ya)"),
        "delivery_delay_days": ("int64", "Keterlambatan pengiriman (hari)"),
        "payment_method": ("object", "Metode pembayaran yang digunakan"),
        "satisfaction_score": ("float64", "Skor kepuasan pelanggan (1-5)"),
        "nps_score": ("int64", "Net Promoter Score (-100 s/d 100)"),
        "marketing_spend_per_user": ("float64", "Biaya pemasaran per pelanggan (USD)"),
        "lifetime_value": ("float64", "Nilai total pelanggan selama berlangganan (USD)"),
        "last_3_month_purchase_freq": ("int64", "Frekuensi pembelian 3 bulan terakhir"),
        "churn": ("int64", "TARGET: Status churn (0=Tidak, 1=Ya)")
    }

    feat_df = pd.DataFrame([
        {"No": i+1, "Kolom": k, "Tipe": v[0], "Keterangan": v[1]}
        for i, (k, v) in enumerate(feature_info.items())
    ])

    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("### Insight Utama")
        st.markdown("""
        - **`total_spent`** dan **`satisfaction_score`** adalah fitur paling berpengaruh terhadap churn
        - Dataset mengalami **class imbalance** (~85% tidak churn vs ~15% churn)
        - Pelanggan dengan **banyak support tickets** dan **satisfaction rendah** lebih berisiko churn
        - **Email engagement** (open_rate, click_rate) mencerminkan tingkat keterlibatan pelanggan
        """)
    with col_i2:
        st.markdown("### Preprocessing yang Dilakukan")
        st.markdown("""
        - **Missing Value**: Numerik diisi median, kategorik diisi mode
        - **Outlier**: Diatasi dengan IQR clipping
        - **Encoding**: Label Encoding untuk fitur kategorik
        - **Scaling**: StandardScaler (fit pada train set saja)
        - **Class Imbalance**: Diatasi dengan `class_weight='balanced'`
        """)


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:0.85rem; padding: 1rem 0;">
    Customer Churn Prediction | UAS Bengkel Koding Data Science |
    Aditya Rendy Setyawan (A11.2023.15189)
</div>
""", unsafe_allow_html=True)
