import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_recall_curve, auc, f1_score

print("🧬 KODON-X V5.2: KLİNİK METRİKLER VE PANEL ANALİZİ BAŞLATILIYOR...\n")

# 1. Verileri Yükle
df_kimya = pd.read_excel("YapayZeka_Hazir_Veri.xlsx")
df_evrim = pd.read_excel("YapayZeka_Evrimsel_Veri.xlsx")
df_final = pd.merge(df_kimya, df_evrim[['Mutasyon_Adi', 'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']], on='Mutasyon_Adi', how='left')

# Encoding
le_gen = LabelEncoder()
df_final['Gen_Kodu'] = le_gen.fit_transform(df_final['Gen'].astype(str))

def guvenli_dizilim(x):
    x_str = str(x)
    if x_str != 'nan' and len(x_str) == 11: return x_str
    return "XXXXXXXXXXX"

df_final['Komsuluk_Dizilimi'] = df_final['Komsuluk_Dizilimi'].apply(guvenli_dizilim)

for i in range(11):
    kolon_adi = f'Komsu_{i-5}'
    df_final[kolon_adi] = LabelEncoder().fit_transform(df_final['Komsuluk_Dizilimi'].str[i])

# Özellikler ve Hedef
X = df_final.drop(columns=['Mutasyon_Adi', 'Gen', 'Komsuluk_Dizilimi', 'ETIKET']).fillna(0)
y = df_final['ETIKET']

# --- KRİTİK: TEST SETİ AYIRMA ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modeller (Senin Optimize Parametrelerinle)
modeller = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=150, learning_rate=0.05, depth=7, random_state=42, verbose=0),
    "🏆 OPTİMİZE XGBOOST": xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42, eval_metric='logloss')
}

# 2. METRİK HESAPLAMA DÖNGÜSÜ
print(f"{'Algoritma':<20} | {'Acc':<7} | {'Recall':<7} | {'ROC-AUC':<7} | {'PR-AUC':<7}")
print("-" * 65)

for isim, model in modeller.items():
    model.fit(X_train, y_train)
    
    # Tahminler
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # AUC'ler için olasılık şart!
    
    # Metrikler
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred) # Bu senin DUYARLILIK (Sensitivity) değerin!
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # PR-AUC (Precision-Recall Curve altındaki alan)
    precision, rec_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec_curve, precision)
    
    print(f"{isim:<20} | %{acc*100:<6.1f} | %{recall*100:<6.1f} | {roc_auc:<7.3f} | {pr_auc:<7.3f}")

# --- EKSTRA: PANEL BAZLI RAPORLAMA (TÜM METRİKLERLE) ---
print("\n🔍 PANEL BAZLI ANALİZ (OPTİMİZE XGBOOST İÇİN)...")

# Başlığa PR-AUC da eklendi, tablo resmen Şampiyonlar Ligi oldu!
print(f"{'Gen Paneli':<12} | {'Örnek':<6} | {'Acc':<6} | {'Recall':<7} | {'F1':<6} | {'ROC-AUC':<8} | {'PR-AUC':<8}")
print("-" * 75)

test_indices = X_test.index
df_test_gen = df_final.loc[test_indices, 'Gen']

# XGBoost tahminleri ve olasılıkları (Probability)
xgb_final = modeller["🏆 OPTİMİZE XGBOOST"]
y_test_proba = xgb_final.predict_proba(X_test)[:, 1] # Olasılıkları alıyoruz
y_test_pred = (y_test_proba >= 0.30).astype(int)     # 0.30 EŞİĞİNİ BURAYA ÇAKTIK KAPTAN!

for gen in df_test_gen.unique():
    mask = (df_test_gen == gen)
    if mask.sum() > 5: # Yetersiz verisi olan genleri atlıyoruz
        y_true_subset = y_test[mask]
        y_pred_subset = y_test_pred[mask]
        y_proba_subset = y_test_proba[mask]
        
        # 1. Temel Metrikler
        gen_acc = accuracy_score(y_true_subset, y_pred_subset)
        gen_rec = recall_score(y_true_subset, y_pred_subset, zero_division=0)
        gen_f1 = f1_score(y_true_subset, y_pred_subset, zero_division=0)
        
        # 2. İleri Seviye Metrikler (ROC-AUC ve PR-AUC)
        try:
            # Eğer test setinde o gen için hem sağlıklı(0) hem hasta(1) varsa hesaplar
            gen_roc = roc_auc_score(y_true_subset, y_proba_subset)
            gen_roc_str = f"{gen_roc:.3f}"
            
            precision_sub, recall_sub, _ = precision_recall_curve(y_true_subset, y_proba_subset)
            gen_pr = auc(recall_sub, precision_sub)
            gen_pr_str = f"{gen_pr:.3f}"
        except ValueError:
            # Test setine tesadüfen sadece tek bir sınıf düştüyse hata vermez, N/A yazar
            gen_roc_str = "N/A"
            gen_pr_str = "N/A"
            
        print(f"{gen:<12} | {mask.sum():<6} | %{gen_acc*100:<5.1f} | %{gen_rec*100:<6.1f} | {gen_f1:<6.3f} | {gen_roc_str:<8} | {gen_pr_str:<8}")

print("-" * 75)
print("🚀 Kod ile rapor artık %100 senkronize Kaptan!")

# --- JÜRİNİN İSTEDİĞİ: THRESHOLD (EŞİK) ANALİZİ ---
print("\n⚖️ KARAR EŞİĞİNİN (THRESHOLD) PERFORMANSA ETKİSİ (XGBOOST)")
print(f"{'Eşik Değeri':<12} | {'Doğruluk (Acc)':<15} | {'Duyarlılık (Recall)':<20} | {'F1-Skoru':<10}")
print("-" * 65)

# XGBoost modelimizin olasılık tahminlerini alıyoruz
y_proba_xgb = xgb_final.predict_proba(X_test)[:, 1]

# Deneyeceğimiz farklı eşik değerleri
esik_degerleri = [0.30, 0.40, 0.50, 0.60]

for esik in esik_degerleri:
    # Kendi eşiğimizi uyguluyoruz: Olasılık eşikten büyükse 1, değilse 0 yap
    y_pred_kisisel = (y_proba_xgb >= esik).astype(int)
    
    # Yeni eşiğe göre metrikleri hesapla
    acc_esik = accuracy_score(y_test, y_pred_kisisel)
    rec_esik = recall_score(y_test, y_pred_kisisel, zero_division=0)
    f1_esik = f1_score(y_test, y_pred_kisisel, zero_division=0)
    
    # Eğer eşik 0.50 ise (Varsayılan) yanına yıldız koyalım ki belli olsun
    etiket = f"{esik:.2f} (Varsayılan)" if esik == 0.50 else f"{esik:.2f}"
    
    print(f"{etiket:<12} | %{acc_esik*100:<13.1f} | %{rec_esik*100:<18.1f} | {f1_esik:<10.3f}")

print("-" * 65)

import time
import tracemalloc

print("\n⏳ M4 İŞLEMCİ VE RAM PERFORMANS TESTİ BAŞLIYOR...")

# 1. RAM Ölçümü Başlat
tracemalloc.start()

# 2. Eğitim Süresi Ölçümü
baslangic_zamani = time.time()
# Optimize XGBoost modelimizi yeniden eğitiyoruz (sadece süre tutmak için)
xgb_final.fit(X_train, y_train)
egitim_suresi = time.time() - baslangic_zamani

# RAM Ölçümünü Bitir ve Zirve (Peak) Değerini Al
guncel_bellek, peak_bellek = tracemalloc.get_traced_memory()
tracemalloc.stop()

# 3. Çıkarım (Inference) Süresi Ölçümü - Toplu (Tüm Test Seti)
test_baslangic = time.time()
xgb_final.predict(X_test)
toplu_cikarim_suresi = time.time() - test_baslangic

# 4. Çıkarım (Inference) Süresi Ölçümü - Tekil (1 Hasta)
tek_hasta = X_test.iloc[[0]] # Sadece ilk satırı alıyoruz
tek_baslangic = time.time()
xgb_final.predict(tek_hasta)
tek_cikarim_suresi = time.time() - tek_baslangic

# --- GERÇEK SONUÇLARI EKRANA YAZDIR ---
print("-" * 65)
print(f"💻 Zirve (Peak) RAM Kullanımı: {peak_bellek / 1024 / 1024:.2f} MB")
print(f"⏱️ Toplam Eğitim Süresi (150 İterasyon): {egitim_suresi:.4f} Saniye")
print(f"⚙️ İterasyon Başına Süre: {(egitim_suresi / 150) * 1000:.4f} Milisaniye (ms)")
print(f"⚡ Toplu Çıkarım Süresi (Tüm Test Seti): {toplu_cikarim_suresi:.4f} Saniye")
print(f"⚡ Tekil Çıkarım Süresi (1 Hasta): {tek_cikarim_suresi * 1000:.4f} Milisaniye (ms)")
print("-" * 65)