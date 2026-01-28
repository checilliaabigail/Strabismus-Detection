# 👁️ Strabismus Screening System (Computer Vision Based)

Sistem **skrining strabismus (mata juling)** berbasis **computer vision** yang diimplementasikan dalam bentuk **website**.  
Pengguna dapat mengunggah foto wajah, kemudian sistem akan melakukan **deteksi wajah dan mata**, **ekstraksi pupil**, serta **klasifikasi posisi mata** (normal / esotropia / exotropia) berdasarkan perbedaan posisi pupil.

Proyek ini dikembangkan sebagai bagian dari **skripsi** dengan judul:

> **Computer Vision for Strabismus Detection**

---

## 🔍 Fitur Utama
- Upload gambar wajah melalui website
- Deteksi wajah dan mata menggunakan **Haar Cascade**
- Deteksi pupil otomatis menggunakan **Hough Circle Transform**
- Perhitungan perbedaan posisi horizontal pupil (Δx)
- Klasifikasi:
  - **Normal**
  - **Esotropia** (mata mengarah ke dalam)
  - **Exotropia** (mata mengarah ke luar)
- Visualisasi hasil deteksi (bounding box & pupil)
- Berbasis **client-side (JavaScript)** - tidak perlu server

---

## 🧠 Metode yang Digunakan
1. **Preprocessing**
   - Grayscale conversion
2. **Face Detection**
   - Haar Cascade Classifier (`haarcascade_frontalface_default.xml`)
3. **Eye Detection**
   - Haar Cascade Eye Detector (`haarcascade_eye.xml`)
4. **Pupil Detection**
   - Hough Circle Transform (auto detection)
5. **Strabismus Classification**
   - Berdasarkan selisih posisi pupil kiri dan kanan (Δx)
   - Threshold ditentukan dari analisis statistik data

---

## 📁 Struktur Proyek

```
Strabismus-Detection/
├── index.html                              # Halaman utama website
├── script.js                               # JavaScript logic (deteksi & klasifikasi)
├── haarcascade_frontalface_default.xml     # Model deteksi wajah
├── haarcascade_eye.xml                     # Model deteksi mata
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/checilliaabigail/Strabismus-Detection.git
cd Strabismus-Detection
```

### 2. Download Dataset (Opsional - untuk analisis)
Dataset yang digunakan tersedia di Kaggle:  
👉 **[STRABISMUS Dataset by Anantha Moorthy A](https://www.kaggle.com/datasets/ananthamoorthya/strabismus)**

**Cara download:**
```bash
# Menggunakan Kaggle API
pip install kaggle
kaggle datasets download -d ananthamoorthya/strabismus
unzip strabismus.zip -d data/
```

### 3. Jalankan Website
Buka file `index.html` menggunakan browser (disarankan **Google Chrome** atau **Firefox**)

**Atau gunakan local server:**
```bash
# Python 3
python -m http.server 8000

# Akses di browser: http://localhost:8000
```

### 4. Upload Gambar
- Pilih foto wajah (tampak depan)
- Klik "Detect" atau "Analyze"
- Lihat hasil deteksi dan klasifikasi

---

## 📊 Output Sistem
- ✅ Deteksi wajah dan mata
- ✅ Titik pusat pupil kiri dan kanan
- ✅ Nilai perbedaan posisi pupil (Δx)
- ✅ Hasil klasifikasi:
  - **Normal**: Δx ≈ 0
  - **Esotropia**: Δx > threshold (positif)
  - **Exotropia**: Δx < threshold (negatif)

### Visualisasi
- Bounding box wajah dan mata
- Marker pada posisi pupil
- Label klasifikasi hasil deteksi

⚠️ **Catatan**: Sistem ini bukan alat diagnosis medis, melainkan **alat skrining awal**.

---

## 🧪 Dataset & Threshold

### Dataset
- **Sumber**: [Kaggle - STRABISMUS Dataset](https://www.kaggle.com/datasets/ananthamoorthya/strabismus)
- **Author**: Anantha Moorthy A
- **Kategori**: Normal, Esotropia, Exotropia

### Analisis Statistik
Threshold klasifikasi ditentukan berdasarkan:
- Uji normalitas (Shapiro-Wilk Test)
- Analisis distribusi data per kategori
- Perhitungan mean dan standar deviasi

| Kelompok | N | Mean (μ) | Std Dev (σ) | Min | Max |
|----------|---|----------|-------------|-----|-----|
| **Exotropia** | 57 | -0.1193 | 0.1016 | -0.3543 | 0.1476 |
| **Normal** | 63 | 0.0021 | 0.0609 | -0.1647 | 0.1310 |
| **Esotropia** | 47 | 0.2210 | 0.0826 | 0.0210 | 0.3652 |

---

## ⚠️ Keterbatasan
Sistem sensitif terhadap:
- ❌ Pencahayaan (terlalu terang/gelap)
- ❌ Sudut wajah (tidak frontal)
- ❌ Resolusi gambar rendah
- ❌ Mata tertutup atau tertutup sebagian
- ❌ Tidak mendeteksi strabismus vertikal (hanya horizontal)

**Penting**: Sistem ini **tidak menggantikan** pemeriksaan klinis oleh dokter mata spesialis.

---

## 🛠️ Teknologi yang Digunakan

### Frontend
- HTML5, CSS3, JavaScript (ES6+)
- OpenCV.js (Computer Vision library)
- Haar Cascade Classifiers

### Analisis Data (Python)
- Pandas, NumPy
- Matplotlib, Seaborn
- SciPy (statistical tests)

---

## 👩‍🎓 Author

**Checillia Jaqueline Abighail**  
Physics – Medical & Computational Physics  
Parahyangan Catholic University

- 🌐 GitHub: @checilliaabigail(https://github.com/checilliaabigail)
- 📧 Email: checilliaabigail@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/checilliaabigail/

---

## 📄 Lisensi

Proyek ini dikembangkan untuk keperluan **akademik dan penelitian**.  
Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgments

- **Dosen Pembimbing**: Reinard Primulando, Ph.D., Drs. Janto Vincent Sulungbudi
- **Universitas**: Universitas Katolik Parahyangan
- **Dataset**: [Anantha Moorthy A - Kaggle](https://www.kaggle.com/datasets/ananthamoorthya/strabismus)
- OpenCV.js Documentation
- Haar Cascade Models from OpenCV

---
