import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import shap

print("🚀 V4.0: OTOMATİK HİPERPARAMETRE AVCISI BAŞLATILIYOR...")

# 1. Verileri Yükle
df_kimya = pd.read_excel("YapayZeka_Hazir_Veri.xlsx")
df_evrim = pd.read_excel("YapayZeka_Evrimsel_Veri.xlsx")
df_final = pd.merge(df_kimya, df_evrim[['Mutasyon_Adi', 'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']], on='Mutasyon_Adi', how='left')
df_final = df_final.drop(columns=['Prior_Skoru', 'Align_GVGD_Skoru'])

# 2. Encoding
le_gen = LabelEncoder()
df_final['Gen_Kodu'] = le_gen.fit_transform(df_final['Gen'].astype(str))

# --- FLOAT/NAN KORUMASI VE 11 PARÇAYA BÖLME ---
komsuluk_sutunlari = []

def guvenli_dizilim(x):
    x_str = str(x)
    if x_str != 'nan' and len(x_str) == 11:
        return x_str
    return "XXXXXXXXXXX"

df_final['Komsuluk_Dizilimi'] = df_final['Komsuluk_Dizilimi'].apply(guvenli_dizilim)

for i in range(11):
    pozisyon = i - 5 
    kolon_adi = f'Komsu_{pozisyon}'
    komsuluk_sutunlari.append(kolon_adi)
    
    df_final[kolon_adi] = df_final['Komsuluk_Dizilimi'].str[i]
    df_final[kolon_adi] = LabelEncoder().fit_transform(df_final[kolon_adi])

# 3. YENİ BÜYÜK X TABLOMUZ (18 Özellik)
temel_ozellikler = ['Gen_Kodu', 'Popülasyon_Frekansi', 'Hidrofobiklik_Farki', 
                    'Molekuler_Agirlik_Farki', 'Polarite_Degisimi',
                    'Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']

X = df_final[temel_ozellikler + komsuluk_sutunlari]
y = df_final['ETIKET']

# 4. KASAYA KİLİTLENEN BAĞIMSIZ TEST SETİ (%20)
print("\n🔒 Verinin %20'si 'Bağımsız Final Sınavı' için kasaya kilitlendi.")
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------
# 5. GRID SEARCH İLE EN İYİ AYARLARI BULMA (M4 İşlemci Şov Yapıyor!)
# ---------------------------------------------------------
print("⚙️ XGBoost için yüzlerce farklı ayar deneniyor... Lütfen bekleyin...")

# XGBoost'un deneyeceği ayar kombinasyonları (Arama Uzayı)
param_grid = {
    'n_estimators': [100, 150, 200],       # Ağaç sayısı
    'learning_rate': [0.01, 0.05, 0.1],    # Öğrenme hızı
    'max_depth': [3, 5, 7],                # Ağaç derinliği
    'subsample': [0.8, 1.0],               # Her ağaçta kullanılacak veri oranı
    'colsample_bytree': [0.8, 1.0]         # Her ağaçta kullanılacak özellik oranı
}

# Temel modelimiz
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

# 5-Fold düzenimizi GridSearch'e veriyoruz
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearch kurulumu
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           cv=skf, scoring='accuracy', n_jobs=-1, verbose=1)

# Aramayı başlat!
grid_search.fit(X_train_val, y_train_val)

print("\n🌟 EUREKA! EN İYİ AYARLAR BULUNDU:")
print(grid_search.best_params_)

# En iyi modeli alıyoruz
en_iyi_model = grid_search.best_estimator_

# ---------------------------------------------------------
# 6. BÜYÜK FİNAL: KASADAKİ %20 İLE GERÇEK DÜNYA SINAVI
# ---------------------------------------------------------
print("\n🔥 BÜYÜK FİNAL: 'En İyi Model' o hiç görmediği kilitli %20'lik test setine giriyor!")

# Hiç görmediği test setiyle sınav yapıyoruz (0.30 EŞİĞİ DEVREDE!)
y_proba = en_iyi_model.predict_proba(X_test)[:, 1]   # En iyi modelin olasılıklarını al
y_tahmin = (y_proba >= 0.30).astype(int)             # 0.30 eşiğini acımadan yapıştır!

gercek_dogruluk = accuracy_score(y_test, y_tahmin)

print(f"🎯 JÜRİYE SUNULACAK NET, SARSILMAZ VE OPTİMİZE DOĞRULUK ORANI: %{gercek_dogruluk * 100:.2f}")
print("\n--- 📋 Detaylı Performans Raporu ---")
print(classification_report(y_test, y_tahmin))
print("\n--- 🎯 Hata Matrisi ---")
print(confusion_matrix(y_test, y_tahmin))

# ---------------------------------------------------------
# 7. SHAP VE FEATURE IMPORTANCE
# ---------------------------------------------------------
print("\n📊 XGBoost Feature Importance Grafiği Çiziliyor...")
en_iyi_model.fit(X, y) # Raporlama için tüm veriyle son bir eğitim

plt.figure(figsize=(10, 8))
xgb.plot_importance(en_iyi_model, max_num_features=10, title="Hangi Özellik Ne Kadar Önemli? (V4.0)", xlabel="Önem Skoru (F-score)", ylabel="Özellikler")
plt.tight_layout()
plt.savefig("Feature_Importance_V4.png", dpi=300)

print("\n🧠 XGBoost SHAP Analizi Yapılıyor...")
explainer = shap.Explainer(en_iyi_model)
shap_values = explainer(X)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("SHAP_Analizi_V4.png", dpi=300, bbox_inches='tight')
print("✅ Tüm işlemler başarıyla tamamlandı!")