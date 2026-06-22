# 🧠 CIFAR-10 CNN Görüntü Sınıflandırıcı

**Kısa Proje Açıklaması**

Bu proje, **CIFAR‑10** veri setindeki 10 farklı nesneyi sınıflandıran bir **Convolutional Neural Network (CNN)** modeli içerir. Kullanıcı bir görüntü yükleyerek modelin tahminini anında görebilir; ayrıca eğitim sırasında toplanan doğruluk ve kayıp grafiklerini sidebar’da inceleyebilir.

---

## 🚀 Kurulum ve Çalıştırma

1. **Bağımlılıkları kur**
   ```bash
   pip install -r requirements.txt
   ```

2. **Modeli eğit (tek seferlik)**
   ```bash
   python train.py
   ```
   Eğitim yaklaşık **15‑20 dakika** sürer. Başarılı olduğunda `model/cifar10_cnn.keras` dosyası oluşturulur ve `model/training_history.json` içinde eğitim geçmişi saklanır.

3. **Uygulamayı başlat**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Streamlit Cloud Deploy

1. Bu repoyu GitHub’a **model/cifar10_cnn.keras** dahil ederek push edin.
2. Streamlit Cloud’da **New app** oluşturun ve GitHub reposunu seçin.
3. **Main file:** `app.py`  →  **Deploy**

---

## 📁 Proje Yapısı
```
<cifar10-streamlit-app>/
├── app.py               # Streamlit arayüzü
├── train.py             # Model eğitimi, kaydetme
├── model/
│   ├── .gitkeep        # Model klasörü (model dosyası burada)
│   └── cifar10_cnn.keras
├── utils/
│   ├── __init__.py
│   ├── preprocess.py   # Görüntü ön‑işleme
│   └── visualize.py    # Eğitim grafikleri & tahmin barı
├── assets/
│   └── sample_images/
│       └── .gitkeep    # Örnek görüntüler (isteğe bağlı)
├── requirements.txt
├── .gitignore
├── README.md
└── .streamlit/
    └── config.toml     # Tema ayarları
```

---

## 🎯 Sınıflar

| Sınıf No | Sınıf Adı |
|----------|-----------|
| 0 | uçak |
| 1 | araba |
| 2 | kuş |
| 3 | kedi |
| 4 | geyik |
| 5 | köpek |
| 6 | kurbağa |
| 7 | at |
| 8 | gemi |
| 9 | kamyon |

---

**Not:** Model dosyasını `.gitignore` dosyasına eklemeyin; GitHub’a doğrudan push edilmelidir.
