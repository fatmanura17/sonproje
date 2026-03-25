# 🧬 Kodon-X: Hastalık Yapıcı Genetik Mutasyonların Tespiti İçin Yapay Zeka
**TEKNOFEST Sağlıkta Yapay Zeka Yarışması | Takım: SISANKA** 🚀

Kodon-X, BRCA1 ve BRCA2 genlerindeki bilinmeyen varyantların (VUS - Variant of Uncertain Significance) patojenik (hastalık yapıcı) veya benign (zararsız) olup olmadığını yüksek isabetle sınıflandıran, açıklanabilir bir klinik karar destek sistemidir.

Proje, sadece istatistiksel verilere dayanmakla kalmaz; mutasyonları biyokimyasal özelliklerine (Hidrofobiklik, Moleküler Ağırlık, Polarite), evrimsel korunmuşluk geçmişine (BLOSUM62) ve proteomik komşuluk ilişkilerine (UniProt API) göre analiz ederek klinik şeffaflık sunar.

---

## 🌟 Öne Çıkan Özellikler

* **🔬 Biyokimyasal Feature Engineering:** Amino asit değişimlerindeki hidrofobiklik, ağırlık ve polarite farklarının matematiksel modeli.
* **🌐 Dinamik Proteomik Komşuluk (UniProt API):** İsviçre UniProt veritabanına anlık bağlanarak her mutasyonun etrafındaki `+/- 5 amino asitlik` dizilimi çekme ve 11 ayrı sütuna (özniteliğe) bölme.
* **🦕 Evrimsel Zeka (BLOSUM62):** Milyonlarca yıllık evrimsel veriye dayalı In-Silico risk skorlaması.
* **🧠 Optimize XGBoost & SHAP:** GridSearch ile hiperparametreleri optimize edilmiş XGBoost algoritması ve doktorlara kararın *nedenini* açıklayan SHAP entegrasyonu.
* **⚖️ Özel Klinik Karar Eşiği (Threshold):** Patojenik vakaların kaçırılmaması için eşik değeri `0.30` olarak optimize edilmiş ve recall (duyarlılık) maksimize edilmiştir.
* **⚡ M4 İşlemci Optimizasyonu:** `tracemalloc` ve `time` modülleri ile milisaniye bazında RAM ve çıkarım (inference) süresi testleri.

---

## 📊 Model Performansı

CatBoost, Random Forest ve Logistic Regression gibi modellerle yapılan çapraz kıyaslamalar (5-Fold CV) sonucunda, eksik verilere olan direnci ve SHAP uyumu nedeniyle **XGBoost** nihai model olarak seçilmiştir.

| Metrik | Değer |
| :--- | :--- |
| **Model** | XGBoost (GridSearch Optimized) |
| **Duyarlılık (Recall)** | **%91.2** *(0.30 Eşik Değeri İle)* |
| **Açıklanabilirlik** | %100 SHAP Uyumlu (Ağaç Tabanlı) |
| **İterasyon Başı Süre** | ~X.XX ms (Apple M4) |

---

## 🛠️ Kullanılan Teknolojiler

* **Veri İşleme & API:** `pandas`, `numpy`, `requests`, `re`
* **Makine Öğrenmesi:** `xgboost`, `catboost`, `scikit-learn`
* **Görselleştirme & Açıklanabilirlik:** `matplotlib`, `shap`
* **Eksik Veri Tamamlama:** `KNNImputer`

---

## ⚙️ Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

**1. Repoyu Klonlayın:**
```bash
git clone [https://github.com/kullaniciadiniz/Kodon-X.git](https://github.com/kullaniciadiniz/Kodon-X.git)
cd Kodon-X

##
