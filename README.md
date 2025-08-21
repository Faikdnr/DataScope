[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b)](https://streamlit.io/) [![Python](https://img.shields.io/badge/Python-3.x-3776AB)](https://www.python.org/)

DataScope; Streamlit tabanlı, kolay kullanımlı bir veri analiz uygulamasıdır. CSV/XLSX veri setlerini yükleyip özellik seçimi yaparak kümeleme (K-Means, DBSCAN, Hiyerarşik) ve sınıflandırma (Decision Tree, Random Forest, SVM, XGBoost, LightGBM) modellerini hızlıca kurup metrikler, karışıklık matrisi, ROC eğrisi ve özellik önemleriyle birlikte görselleştirir.

## Anahtar Kelimeler (SEO)
- veri analizi, makine öğrenmesi, ML, streamlit, python
- kümeleme, k-means, dbscan, hiyerarşik kümeleme
- sınıflandırma, decision tree, random forest, svm, xgboost, lightgbm
- ön işleme, aykırı değer, ölçeklendirme, roc, karışıklık matrisi, özellik önemi

## DataScope | Veri Analiz Platformu

Modern, etkileşimli bir Streamlit uygulaması ile CSV/XLSX veri setlerinizi hızlıca yükleyin, özellik seçin ve iki farklı analiz türünde sonuçları görselleştirin:

- **Kümeleme**: K-Means, DBSCAN, Hiyerarşik Kümeleme
- **Sınıflandırma**: Decision Tree, Random Forest, SVM, XGBoost, LightGBM

### Özellikler
- **Kolay veri yükleme**: CSV veya Excel
- **Özellik seçimi**: Sayısal sütunları seçerek analiz
- **Kümeleme**: 2D görselleştirme ve Silhouette skoru
- **Sınıflandırma**: Esnek etiket seçimi (label sütun adını özgürce seç), sayısal etiketleri kategorize etme
- **Ön işleme**: Eksik veri doldurma, aykırı değer temizleme, ölçeklendirme
- **Raporlama**: Doğruluk, sınıflandırma raporu, karışıklık matrisi, ROC eğrisi, özellik önemi
- **Model kaydetme**: Eğitilmiş modeli `joblib` ile dışa aktar

### Ekran Görüntüsü
`docs/` klasörüne ekran görüntüleri ekleyebilirsiniz.

### Proje Yapısı
```
Bt/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ static/
   └─ style.css (opsiyonel)
```

## Kurulum

### 1) Ortamı Hazırlama (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Uygulamayı Çalıştırma
```powershell
streamlit run app.py
```
Varsayılan adres: `http://localhost:8501`

## Kullanım
1. Sol menüden veri dosyanızı (CSV/XLSX) yükleyin
2. Analiz türünü seçin: Kümeleme veya Sınıflandırma
3. Özellik seçimlerini yapın (en az iki sayısal sütun)
4. Sonuçları inceleyin: grafikler, raporlar, metrikler
5. Gerekirse modeli kaydedin

## Gereksinimler
`requirements.txt` dosyasında listelenmiştir. Pandas ile XLSX okumak için `openpyxl` dahil edilmiştir.

## Profesyonel Şekilde GitHub’a Yükleme

### Yöntem A — Komut Satırı ile
1. Git’i kurun ve yapılandırın (tek seferlik):
   ```powershell
   git config --global user.name "Ad Soyad"
   git config --global user.email "mail@ornek.com"
   ```
2. Depoyu başlatın ve ilk commit’i yapın:
   ```powershell
   cd "C:\Users\Faik Döner\Desktop\Bt"
   git init
   git add .
   git commit -m "feat: initial commit DataScope app"
   ```
3. GitHub’da boş bir repo oluşturun (ör: `datascope`). Ardından uzaktan bağlayıp gönderin:
   ```powershell
   git remote add origin https://github.com/<kullanici-adi>/datascope.git
   git branch -M main
   git push -u origin main
   ```

### Yöntem B — GitHub Desktop ile
- Uygulamayı açın, “Add local repository” ile `Bt` klasörünü seçin, “Publish repository” ile GitHub’a gönderin.

### İyi Uygulamalar
- Anlamlı commit mesajları kullanın (Conventional Commits önerilir)
- Issue ve Pull Request şablonları ekleyin (ileride)
- `docs/` klasöründe görseller ve detaylı kullanım senaryoları tutun

## Lisans
- MIT Lisansı: bkz. [LICENSE](LICENSE)
- Türkçe gayriresmî çeviri: [LICENSE.tr.md](LICENSE.tr.md)

## İletişim
- Geliştirici: Faik Döner — LinkedIn: https://www.linkedin.com/in/faik-döner

---
DataScope ile verilerinizi zahmetsizce keşfedin ve modelleyin.
