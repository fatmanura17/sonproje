import pandas as pd
import numpy as np
import requests
import re
import warnings
warnings.filterwarnings('ignore')

print("🚀 VUS VERİ FABRİKASI BAŞLATILIYOR (Evrimsel Zeka + Kimya + API) ...")

# 1. BOŞ VUS DOSYASINI YÜKLE
try:
    df_vus = pd.read_excel("-Bos_Etiketli_Veriler.xlsx")
    # Kod hata vermesin diye kolon isimlerini standartlaştırıyoruz (GEN -> Gen)
    df_vus.rename(columns={'GEN': 'Gen', 'MUTASYON_ADI': 'Mutasyon_Adi'}, inplace=True)
    print("✅ VUS verisi başarıyla yüklendi!")
except FileNotFoundError:
    print("❌ HATA: -Bos_Etiketli_Veriler.xlsx bulunamadı!")
    exit()

# 2. EVRİMSEL SKOR VE RİSK HESAPLAMA (BLOSUM62)
blosum62 = {
    ('A', 'V'): 0, ('A', 'D'): -2, ('A', 'E'): -1, ('A', 'G'): 0, ('A', 'P'): -1,
    ('R', 'G'): -2, ('R', 'H'): 0, ('R', 'K'): 2, ('R', 'W'): -3, ('R', 'C'): -3,
    ('N', 'D'): 1, ('N', 'H'): 1, ('N', 'S'): 1, ('N', 'T'): 0, ('N', 'K'): 0,
    ('D', 'E'): 2, ('D', 'N'): 1, ('D', 'G'): -1, ('D', 'V'): -3, ('D', 'A'): -2,
    ('C', 'Y'): -2, ('C', 'R'): -3, ('C', 'S'): -1, ('C', 'G'): -3, ('C', 'W'): -2,
    ('E', 'D'): 2, ('E', 'K'): 1, ('E', 'Q'): 2, ('E', 'A'): -1, ('E', 'V'): -2,
    ('Q', 'R'): 1, ('Q', 'K'): 1, ('Q', 'E'): 2, ('Q', 'P'): -1, ('Q', 'H'): 0,
    ('G', 'A'): 0, ('G', 'S'): 0, ('G', 'D'): -1, ('G', 'R'): -2, ('G', 'C'): -3,
    ('H', 'Y'): 2, ('H', 'N'): 1, ('H', 'Q'): 0, ('H', 'R'): 0, ('H', 'P'): -2,
    ('I', 'L'): 2, ('I', 'V'): 3, ('I', 'M'): 1, ('I', 'F'): 0, ('I', 'T'): -1,
    ('L', 'I'): 2, ('L', 'V'): 1, ('L', 'M'): 2, ('L', 'F'): 0, ('L', 'P'): -3,
    ('K', 'R'): 2, ('K', 'Q'): 1, ('K', 'E'): 1, ('K', 'T'): -1, ('K', 'A'): -1,
    ('M', 'L'): 2, ('M', 'I'): 1, ('M', 'V'): 1, ('M', 'T'): -1, ('M', 'R'): -1,
    ('F', 'Y'): 3, ('F', 'W'): 1, ('F', 'L'): 0, ('F', 'I'): 0, ('F', 'S'): -2,
    ('P', 'A'): -1, ('P', 'S'): -1, ('P', 'T'): -1, ('P', 'L'): -3, ('P', 'R'): -2,
    ('S', 'T'): 1, ('S', 'A'): 1, ('S', 'N'): 1, ('S', 'G'): 0, ('S', 'P'): -1,
    ('T', 'S'): 1, ('T', 'A'): 0, ('T', 'I'): -1, ('T', 'M'): -1, ('T', 'V'): 0,
    ('W', 'Y'): 2, ('W', 'F'): 1, ('W', 'R'): -3, ('W', 'C'): -2, ('W', 'L'): -2,
    ('Y', 'F'): 3, ('Y', 'W'): 2, ('Y', 'H'): 2, ('Y', 'C'): -2, ('Y', 'S'): -2,
    ('V', 'I'): 3, ('V', 'L'): 1, ('V', 'M'): 1, ('V', 'A'): 0, ('V', 'T'): 0
}
aa_cevirici = {'Ala':'A', 'Arg':'R', 'Asn':'N', 'Asp':'D', 'Cys':'C', 'Glu':'E', 'Gln':'Q', 'Gly':'G', 'His':'H', 'Ile':'I', 'Leu':'L', 'Lys':'K', 'Met':'M', 'Phe':'F', 'Pro':'P', 'Ser':'S', 'Thr':'T', 'Trp':'W', 'Tyr':'Y', 'Val':'V'}

def evrimsel_skor_hesapla(mutasyon_str):
    if pd.isna(mutasyon_str): return pd.Series([None, None])
    mutasyon_str = str(mutasyon_str).replace('p.', '')
    match = re.match(r"([A-Za-z]+)(\d+)([A-Za-z]+)", mutasyon_str)
    if not match: return pd.Series([0, 0])
    ilk_aa_kod, son_aa_kod = match.group(1), match.group(3)
    ilk_aa = aa_cevirici.get(ilk_aa_kod, ilk_aa_kod)
    son_aa = aa_cevirici.get(son_aa_kod, son_aa_kod)
    skor = blosum62.get((ilk_aa, son_aa), -4)
    risk_skoru = abs(skor) * 1.5 if skor < 0 else 0.5
    return pd.Series([skor, risk_skoru])

print("⚙️ Mutasyonların evrimsel geçmişi ve in-silico skorları hesaplanıyor...")
df_vus[['Evrimsel_Korunmusluk_Skoru', 'InSilico_Risk_Skoru']] = df_vus['Mutasyon_Adi'].apply(evrimsel_skor_hesapla)

# 3. KİMYASAL HESAPLAMALAR
aa_ozellikleri = {
    'A': {'isim': 'Alanine', 'hidrofobiklik': 1.8, 'agirlik': 89.1, 'polarite': 'Nonpolar'},
    'R': {'isim': 'Arginine', 'hidrofobiklik': -4.5, 'agirlik': 174.2, 'polarite': 'Basic polar'},
    'N': {'isim': 'Asparagine', 'hidrofobiklik': -3.5, 'agirlik': 132.1, 'polarite': 'Polar'},
    'D': {'isim': 'Aspartic acid', 'hidrofobiklik': -3.5, 'agirlik': 133.1, 'polarite': 'Acidic polar'},
    'C': {'isim': 'Cysteine', 'hidrofobiklik': 2.5, 'agirlik': 121.2, 'polarite': 'Nonpolar'},
    'E': {'isim': 'Glutamic acid', 'hidrofobiklik': -3.5, 'agirlik': 147.1, 'polarite': 'Acidic polar'},
    'Q': {'isim': 'Glutamine', 'hidrofobiklik': -3.5, 'agirlik': 146.2, 'polarite': 'Polar'},
    'G': {'isim': 'Glycine', 'hidrofobiklik': -0.4, 'agirlik': 75.1, 'polarite': 'Nonpolar'},
    'H': {'isim': 'Histidine', 'hidrofobiklik': -3.2, 'agirlik': 155.2, 'polarite': 'Basic polar'},
    'I': {'isim': 'Isoleucine', 'hidrofobiklik': 4.5, 'agirlik': 131.2, 'polarite': 'Nonpolar'},
    'L': {'isim': 'Leucine', 'hidrofobiklik': 3.8, 'agirlik': 131.2, 'polarite': 'Nonpolar'},
    'K': {'isim': 'Lysine', 'hidrofobiklik': -3.9, 'agirlik': 146.2, 'polarite': 'Basic polar'},
    'M': {'isim': 'Methionine', 'hidrofobiklik': 1.9, 'agirlik': 149.2, 'polarite': 'Nonpolar'},
    'F': {'isim': 'Phenylalanine', 'hidrofobiklik': 2.8, 'agirlik': 165.2, 'polarite': 'Nonpolar'},
    'P': {'isim': 'Proline', 'hidrofobiklik': -1.6, 'agirlik': 115.1, 'polarite': 'Nonpolar'},
    'S': {'isim': 'Serine', 'hidrofobiklik': -0.8, 'agirlik': 105.1, 'polarite': 'Polar'},
    'T': {'isim': 'Threonine', 'hidrofobiklik': -0.7, 'agirlik': 119.1, 'polarite': 'Polar'},
    'W': {'isim': 'Tryptophan', 'hidrofobiklik': -0.9, 'agirlik': 204.2, 'polarite': 'Nonpolar'},
    'Y': {'isim': 'Tyrosine', 'hidrofobiklik': -1.3, 'agirlik': 181.2, 'polarite': 'Polar'},
    'V': {'isim': 'Valine', 'hidrofobiklik': 4.2, 'agirlik': 117.1, 'polarite': 'Nonpolar'}
}

def mutasyon_cozucu(mutasyon_str):
    if pd.isna(mutasyon_str): return pd.Series([None, None, None])
    mutasyon_str = str(mutasyon_str).replace('p.', '')
    match = re.match(r"([A-Za-z]+)(\d+)([A-Za-z]+)", mutasyon_str)
    if not match: return pd.Series([None, None, None])
    
    ilk_aa = aa_cevirici.get(match.group(1), match.group(1))
    son_aa = aa_cevirici.get(match.group(3), match.group(3))
    
    if ilk_aa in aa_ozellikleri and son_aa in aa_ozellikleri:
        ozellik_ilk = aa_ozellikleri[ilk_aa]
        ozellik_son = aa_ozellikleri[son_aa]
        hidro_fark = ozellik_son['hidrofobiklik'] - ozellik_ilk['hidrofobiklik']
        agirlik_fark = ozellik_son['agirlik'] - ozellik_ilk['agirlik']
        polarite_degisimi = 1 if ozellik_ilk['polarite'] != ozellik_son['polarite'] else 0
        return pd.Series([round(hidro_fark, 2), round(agirlik_fark, 2), polarite_degisimi])
    return pd.Series([None, None, None])

print("🧬 Biyokimyasal farklar (Hidrofobiklik, Ağırlık, Polarite) laboratuvarda analiz ediliyor...")
df_vus[['Hidrofobiklik_Farki', 'Molekuler_Agirlik_Farki', 'Polarite_Degisimi']] = df_vus['Mutasyon_Adi'].apply(mutasyon_cozucu)

# 4. PROTEOMİK KOMŞULUK (+/- 5 AMİNO ASİT) MODÜLÜ
print("🌐 İsviçre UniProt API'sine bağlanılıyor ve komşuluk dizilimleri çekiliyor...")
uniprot_id = {'BRCA1': 'P38398', 'BRCA2': 'P51587'}
protein_dizilimleri = {}

for gen, id_kod in uniprot_id.items():
    url = f"https://rest.uniprot.org/uniprotkb/{id_kod}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_satirlari = response.text.split('\n')[1:]
        protein_dizilimleri[gen] = "".join(fasta_satirlari)
        print(f"✅ {gen} dizilimi başarıyla çekildi!")

def komsuluk_bulucu(row):
    gen = str(row['Gen']).upper() # BRCA1 / BRCA2 küçük harfle yazılmışsa düzelt
    mutasyon_str = str(row['Mutasyon_Adi']).replace('p.', '')
    match = re.match(r"([A-Za-z]+)(\d+)([A-Za-z]+)", mutasyon_str)
    if not match or gen not in protein_dizilimleri: return None
    
    pozisyon = int(match.group(2)) - 1
    sekans = protein_dizilimleri[gen]
    baslangic = max(0, pozisyon - 5)
    bitis = min(len(sekans), pozisyon + 6)
    return sekans[baslangic:bitis]

print("🔍 11'li Komşuluk sekansları parçalanıyor...")
df_vus['Komsuluk_Dizilimi'] = df_vus.apply(komsuluk_bulucu, axis=1)

# 5. DOSYAYI KAYDET
dosya_adi = "VUS_Dolu_Veriler.xlsx"
df_vus.to_excel(dosya_adi, index=False)

print(f"\n🎉 VUS VERİLERİ LABORATUVARDAN ÇIKTI!")
print(f"📁 İçi dopdolu yeni dosyan '{dosya_adi}' adıyla kaydedildi.")