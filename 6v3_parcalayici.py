import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import shap

print("🚀 V3.1: JÜRİ ONAYLI, SIZINTISIZ YAPAY ZEKA BAŞLATILIYOR...")

# 1. Verileri Yükle
df_kimya = pd.read_excel("YapayZeka_Hazir_Veri.xlsx")
df_evrim = pd.read_excel("YapayZeka_Evrimsel_Veri.xlsx")
df_final = pd.merge(df_kimya, df_evrim[['Mutasyon_Adi', 'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']], on='Mutasyon_Adi', how='left')
df_final = df_final.drop(columns=['Prior_Skoru', 'Align_GVGD_Skoru'])

# 2. Encoding
le_gen = LabelEncoder()
df_final['Gen_Kodu'] = le_gen.fit_transform(df_final['Gen'].astype(str))

# --- HATA DÜZELTİLDİ: FLOAT/NAN KORUMASI EKLENDİ ---
komsuluk_sutunlari = []

def guvenli_dizilim(x):
    x_str = str(x)
    if x_str != 'nan' and len(x_str) == 11:
        return x_str
    return "XXXXXXXXXXX"

df_final['Komsuluk_Dizilimi'] = df_final['Komsuluk_Dizilimi'].apply(guvenli_dizilim)

print("✂️ Dizilimler kesiliyor, 11 yeni mikroskobik özellik yaratılıyor...")
for i in range(11):
    pozisyon = i - 5 
    kolon_adi = f'Komsu_{pozisyon}'
    komsuluk_sutunlari.append(kolon_adi)
    
    df_final[kolon_adi] = df_final['Komsuluk_Dizilimi'].str[i]
    df_final[kolon_adi] = LabelEncoder().fit_transform(df_final[kolon_adi])

# 3. YENİ BÜYÜK X TABLOMUZ (Toplam 18 özellik!)
temel_ozellikler = ['Gen_Kodu', 'Popülasyon_Frekansi', 'Hidrofobiklik_Farki', 
                    'Molekuler_Agirlik_Farki', 'Polarite_Degisimi',
                    'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']

X = df_final[temel_ozellikler + komsuluk_sutunlari]
y = df_final['ETIKET']

# ---------------------------------------------------------
# 4. KASAYA KİLİTLENEN BAĞIMSIZ TEST SETİ (%20) [ALTIN VURUŞ BURASI]
# ---------------------------------------------------------
print("\n🔒 Verinin %20'si 'Bağımsız Final Sınavı' için kasaya kilitleniyor...")
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------
# 5. EĞİTİM KÜMESİNDE (%80) 5-FOLD İDMANI
# ---------------------------------------------------------
print("🔄 Kalan %80'lik eğitim verisi üzerinde 5-Fold İdmanı Başlıyor...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = []

model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=7, random_state=42, eval_metric='logloss')

fold_no = 1
for train_index, val_index in skf.split(X_train_val, y_train_val):
    X_fold_train, X_fold_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
    y_fold_train, y_fold_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
    
    model.fit(X_fold_train, y_fold_train)
    preds = model.predict(X_fold_val)
    acc = accuracy_score(y_fold_val, preds)
    cv_accuracies.append(acc)
    print(f"   İdman Fold {fold_no} Sonucu: %{acc*100:.2f}")
    fold_no += 1

print(f"🏆 5-Fold İdman (Eğitim/Doğrulama) Ortalaması: %{np.mean(cv_accuracies)*100:.2f}")

# ---------------------------------------------------------
# 6. BÜYÜK FİNAL: KASADAKİ %20 İLE GERÇEK DÜNYA SINAVI
# ---------------------------------------------------------
print("\n🔥 BÜYÜK FİNAL: Model, o hiç görmediği kilitli %20'lik test setine giriyor!")
# Modeli idman yaptığımız %80'in tamamıyla son bir kez harika bir şekilde eğitiyoruz
model.fit(X_train_val, y_train_val)

# Hiç görmediği test setiyle sınav yapıyoruz
y_tahmin = model.predict(X_test)
gercek_dogruluk = accuracy_score(y_test, y_tahmin)

print(f"🎯 JÜRİYE SUNULACAK NET VE SARSILMAZ DOĞRULUK ORANI: %{gercek_dogruluk * 100:.2f}")
print("\n--- 📋 Detaylı Performans Raporu ---")
print(classification_report(y_test, y_tahmin))
print("\n--- 🎯 Hata Matrisi ---")
print(confusion_matrix(y_test, y_tahmin))

# ---------------------------------------------------------
# 7. YENİ MODEL İÇİN Feature Importance & SHAP ANALİZİ
# ---------------------------------------------------------
print("\n📊 XGBoost Feature Importance Grafiği Çiziliyor...")
# Biyolojik analizleri raporlamak için modeli tüm veriyle son bir kez kavratıyoruz
model.fit(X, y) 

plt.figure(figsize=(10, 8))
xgb.plot_importance(model, max_num_features=10, title="Hangi Özellik Ne Kadar Önemli? (V3.1)", xlabel="Önem Skoru (F-score)", ylabel="Özellikler")
plt.tight_layout()
plt.savefig("Feature_Importance_V3.png", dpi=300)
print("✅ Feature_Importance_V3.png klasöre kaydedildi!")

print("\n🧠 XGBoost SHAP Analizi Yapılıyor...")
explainer = shap.Explainer(model)
shap_values = explainer(X)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("SHAP_Analizi_V3.png", dpi=300, bbox_inches='tight')
print("✅ SHAP_Analizi_V3.png klasöre kaydedildi!")

# ---------------------------------------------------------
# 6. BÜYÜK FİNAL: KASADAKİ %20 İLE GERÇEK DÜNYA SINAVI (0.30 EŞİĞİYLE!)
# ---------------------------------------------------------
print("\n🔥 BÜYÜK FİNAL: Model, o hiç görmediği kilitli %20'lik test setine giriyor!")
# Modeli idman yaptığımız %80'in tamamıyla son bir kez harika bir şekilde eğitiyoruz
model.fit(X_train_val, y_train_val)

# --- İŞTE 0.30 EŞİĞİ BURADA DEVREYE GİRİYOR ---
y_proba = model.predict_proba(X_test)[:, 1]         # Önce olasılıkları alıyoruz
y_tahmin = (y_proba >= 0.30).astype(int)            # 0.30'u geçen her şeye "1 (Patojenik)" diyoruz!
# ----------------------------------------------

gercek_dogruluk = accuracy_score(y_test, y_tahmin)

print(f"🎯 JÜRİYE SUNULACAK 0.30 EŞİKLİ DOĞRULUK ORANI: %{gercek_dogruluk * 100:.2f}")
print("\n--- 📋 Detaylı Performans Raporu (Recall Şov!) ---")
print(classification_report(y_test, y_tahmin))
print("\n--- 🎯 Hata Matrisi ---")
print(confusion_matrix(y_test, y_tahmin))