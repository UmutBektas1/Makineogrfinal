# 🚀 Spotify Popülarite Predictor - Deployment Dokümantasyonu

Bu klasör ve dosyalar, `Model Expert` tarafından seçilen Lider Makine Öğrenmesi (XGBoost) modelinin son kullanıcılarla buluştuğu **Arayüz (UI)** ve **Deployment (Yayına Alma)** ortamıdır.

## 🛠️ Sistem Mimarisi & HCI Standartları
Uygulama, Ben Shneiderman'ın 8 Altın Kuralı ve Nielsen Kullanılabilirlik ilkeleri baz alınarak **Streamlit** üzerinden tasarlanmıştır:
1. **Güvenlik & Sınırlar:** Kullanıcıların girdiği veriler (Acousticness, Energy vb.) `dataset_cleaned.csv` tablosunun gerçek Min/Max değerleri arasına dinamik çekilerek sıkıştırılmış, uygulama çökmeleri engellenmiştir.
2. **Kısayol Kullanımı:** Kullanıcıyı veri girmekten kurtarmak için `🎲 Örnek Veriyle Doldur` kısayolu eklenmiştir.
3. **Gerçek Zamanlı Feature Engineering:** Kullanıcı uygulamaya baz değerleri girer; `duration_min`, `danceability_squared` gibi Feature Engineering türevleri saniyeler içinde arka planda hesaplanıp modele öyle aktarılır.
4. **Tahmin Kartı (Prediction Card):** Tahmin sonucunun renkleri (Yeşil, Gri, Kırmızı) çıkacak olan popülarite puanına göre değişmektedir. Ayrıca yöneticiler için "A&R bütçesi ayrılmalı mı?" iş çıkarımı puana entegre edilmiştir.

## 💻 Nasıl Çalıştırılır?

**1.** Terminali klasör konumunda açın.
**2.** Eğer henüz kurmadıysanız paketleri indirin: 
```bash
pip install -r requirements.txt
```
**3.** Uygulamayı Başlatın:
```bash
streamlit run app/app.py
```

Uygulamanız tarayıcınızda (localhost:8501) harika bir tasarımla çalışmaya başlayacaktır!