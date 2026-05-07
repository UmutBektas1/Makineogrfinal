# 📝 DATA PREP HANDOFF (Görev Paylaşım Belgesi)

Bu belge EDA Expert analizleri sonucunda **DataPrep Expert** ajanına aktarılan iş paketlerini içerir.

### 1. Eksik Veriler
- `artists`, `album_name`, `track_name` sütunlarındaki kayıp veriler sadece 1 adet. Satırları *drop* ediniz.

### 2. Gereksiz Veriler
- `track_id` index olarak ayarlanmalı veya model eğitimi için X setinden çıkarılmalıdır.

### 3. Hedef Değişken ve Outlier
- `popularity` 0 olan ciddi bir kitle var. Problem türümüz regressyon ise bu outlier'lar özel bir dönüşüme tabi tutulmalı veya Tree-based modeller için müdahalesiz bırakılmalı.
