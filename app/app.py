import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ==========================================
# 0. PAGE CONFIG & HCI CSS
# ==========================================
st.set_page_config(page_title="Spotify Hit Predictor", page_icon="🎵", layout="wide")

st.markdown(
    """
    <style>
    .main { background: linear-gradient(135deg, #F8FAFC 0%, #EEF6F9 100%); }
    .hero-card { background: rgba(255, 255, 255, 0.92); border-radius: 24px; padding: 28px 32px; box-shadow: 0 18px 45px rgba(31, 41, 55, 0.08); margin-bottom: 24px; }
    .result-positive { background: linear-gradient(135deg, #D5F5E3 0%, #B8E0D2 100%); border-radius: 22px; padding: 24px; text-align: center;}
    .result-warning { background: linear-gradient(135deg, #FDECEC 0%, #F6C6C6 100%); border-radius: 22px; padding: 24px; text-align: center;}
    .result-neutral { background: linear-gradient(135deg, #EAECEE 0%, #D5D8DC 100%); border-radius: 22px; padding: 24px; text-align: center;}
    h1, h2, h3 { color: #2C3E50; }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 1. LOAD DATA BOUNDS & MODELS (CACHED)
# ==========================================
@st.cache_data
def load_dataset_bounds_v2():
    """dataset.csv üzerinden HCI kuralı gereği validasyon (min/max) sınırlarını belirler."""
    df = pd.read_csv("data/processed/dataset_cleaned.csv")
    bounds = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        bounds[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean())
        }
    return bounds, df

@st.cache_resource
def load_ml_assets_v2():
    with open("models/final_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

bounds, df_raw = load_dataset_bounds_v2()
model, scaler = load_ml_assets_v2()

# ==========================================
# 2. UI HEADER & SIDEBAR
# ==========================================
st.markdown('<div class="hero-card">', unsafe_allow_html=True)
st.title("🎵 Profesyonel Spotify Popülarite (Hit) Tahmincisi")
st.write("Bu uygulama, seçtiğiniz müzikal özelliklerin Spotify ekosisteminde ne kadar **popüler** olabileceğini saniyeler içinde tahmin eder. Değerleri `dataset.csv` havuzunun güvenlik sınırları içerisinde değiştirin.")
st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.header("⚙️ Model Ayarları & Bilgi")
st.sidebar.info("Model: **XGBoost Regressor**\n\nHata Sapması (MAE): **~16 Puan**\n\nBu model bir parçanın Hit olma potansiyelini öngörmek amaçlı bir eleme filtresi olarak yapılandırılmıştır.")

# Örnek Veri ile Doldurma Kısayolu (Shneiderman Kural 2)
if st.sidebar.button("🎲 Örnek Veriyle Doldur (Rastgele Şarkı)"):
    sample_row = df_raw.sample(1).iloc[0]
    for col in bounds.keys():
        if col != 'popularity':
            st.session_session_state_col = sample_row[col]
            st.session_state[col] = sample_row[col]

# ==========================================
# 3. INPUT FORMS (HCI - Bilişsel Yükü Azaltma)
# ==========================================
col1, col2, col3 = st.columns(3)

def create_slider(column_name, label, col_block):
    # Eğer session state'te varsa onu kullan, yoksa meane eşitle
    val = st.session_state.get(column_name, bounds[column_name]['mean'])
    return col_block.slider(
        label,
        min_value=bounds[column_name]['min'],
        max_value=bounds[column_name]['max'],
        value=float(val),
        help=f"Dataset Sınırları: {bounds[column_name]['min']:.2f} - {bounds[column_name]['max']:.2f}"
    )

with col1:
    st.subheader("🎼 Ritim ve Akustik")
    danceability = create_slider('danceability', "Dans Edilebilirlik", col1)
    energy = create_slider('energy', "Enerji", col1)
    tempo = create_slider('tempo', "Tempo (BPM)", col1)
    acousticness = create_slider('acousticness', "Akustiklik", col1)

with col2:
    st.subheader("🎧 Teknik Ses Özellikleri")
    loudness = create_slider('loudness', "Ses Yüksekliği (dB)", col2)
    instrumentalness = create_slider('instrumentalness', "Enstrümantallik", col2)
    liveness = create_slider('liveness', "Canlılık (Liveness)", col2)
    speechiness = create_slider('speechiness', "Konuşma Oranı (Speechiness)", col2)

with col3:
    st.subheader("🎹 Diğer Metrikler")
    duration_min = create_slider('duration_min', "Müzik Süresi (Dakika)", col3)
    valence = create_slider('valence', "Valence (Pozitif Hissiyat)", col3)
    key = col3.selectbox("Müzikal Notalar (Key)", options=list(range(0, 12)), index=int(bounds['key']['mean']))
    mode = col3.selectbox("Mod (0: Minör, 1: Majör)", options=[0, 1], index=int(bounds['mode']['mean']))
    time_signature = col3.selectbox("Ölçü (Time Signature)", options=[1, 3, 4, 5], index=2)
    is_explicit = col3.checkbox("Explicit (Küfür/Argo içeriyor mu?)", value=False)

# ==========================================
# 4. PREDICTION ENGINE (Feature Engineering Uygulaması)
# ==========================================
st.markdown("---")
if st.button("🚀 Şarkının Popülaritesini Tahmin Et", use_container_width=True):
    with st.spinner("Yapay Zeka Müzikal DNA'yı Analiz Ediyor..."):
        
        # DataPrep'in Feature Engineering hesaplamalarını canlı yapıyoruz (artık duration input'u dakika)
        energy_loudness_ratio = energy / (loudness + 60 + 1e-5)
        danceability_squared = danceability ** 2
        
        # Tempo encode
        if tempo <= 90:
            tempo_encoded = 0 # Yavaş
        elif tempo <= 130:
            tempo_encoded = 1 # Orta
        else:
            tempo_encoded = 2 # Hızlı

        is_explicit_val = 1 if is_explicit else 0
        
        # Modele girecek olan sıraya göre X_input oluştur (Milisaniye kaldırıldı, sadece Dakika)
        feature_order = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                         'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 
                         'duration_min', 'energy_loudness_ratio', 'danceability_squared', 'tempo_encoded', 'is_explicit']
        
        input_data = pd.DataFrame([[danceability, energy, key, loudness, mode, speechiness, 
                                    acousticness, instrumentalness, liveness, valence, tempo, time_signature, 
                                    duration_min, energy_loudness_ratio, danceability_squared, tempo_encoded, is_explicit_val]],
                                  columns=feature_order)
        
        # Model tahmini
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        # Güven aralığı ve sınırlama (Log mantığı)
        final_prediction = max(0, min(100, prediction)) # 0-100 dışına çıkmasını engelle (HCI: Hata önleme)
        
        # Prediction Card Rendering
        if final_prediction >= 70:
            card_class = "result-positive"
            text_desc = "🌟 Hit Potansiyeli Yüksek! A&R Listelerine Alınabilir."
        elif final_prediction >= 40:
            card_class = "result-neutral"
            text_desc = "📈 Standart Performans. Ortalama dinleyiciye ulaşacaktır."
        else:
            card_class = "result-warning"
            text_desc = "Niche / Düşük Potansiyel. Pazarlama bütçesi ayrılmadan önce düşünülmeli."
            
        st.markdown(
            f'''
            <div class="{card_class}">
                <h2 style="margin-bottom:0;">Tahmin Edilen Popülarite Puanı</h2>
                <h1 style="font-size:3rem; margin-top:0;">{final_prediction:.1f} / 100</h1>
                <p><strong>Değerlendirme:</strong> {text_desc}</p>
                <p style="font-size:0.8rem; color:#555;">(Model Hata Payı: ±16 Puan. Bu değer {max(0, final_prediction-16):.1f} ile {min(100, final_prediction+16):.1f} aralığında dalgalanabilir.)</p>
            </div>
            ''', unsafe_allow_html=True
        )