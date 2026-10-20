# ACLE-Net: Deep Learning for HCI & Data Visualization Evaluation

## 📌 Deskripsi Proyek
Proyek Deep Learning ini dikembangkan berdasarkan resume sesi **Human-Computer Interaction (HCI) & Data Visualization**. 
Model **ACLE-Net (Automated Cognitive Load Evaluator Network)** menggunakan **Convolutional Neural Network (CNN)** dalam PyTorch untuk secara otomatis mengevaluasi gambar antarmuka (UI) dan visualisasi data, mendeteksi tingkat **Extraneous Cognitive Load** (Beban Mental Kognitif Pengguna) serta memberikan rekomendasi perbaikan desain antarmuka berbasis prinsip IMK & UCD.

---

## 💡 Konsep IMK & Visualisasi Data yang Diterapkan
1. **Beban Kognitif & Memori Manusia:** *Working memory* manusia terbatas ($7 \pm 2$ item). Model mengklasifikasikan layout UI menjadi `low_load`, `medium_load`, dan `high_load`.
2. **User-Centered Design (UCD):** Menggunakan filosofi *"Fail early, fail cheap"* untuk melakukan evaluasi cepat pada prototipe awal (*low-fidelity*).
3. **Preattentive Visual Attributes:** Memastikan elemen visual penting langsung ditangkap mata dalam < 200 ms.
4. **Prinsip Visualisasi Edward Tufte & Shneiderman:**
   - Menghindari *Chartjunk* (elemen dekoratif berlebihan).
   - Menerapkan mantra: *"Overview first, zoom and filter, then details-on-demand"*.
5. **Aksesibilitas & WCAG:** Mengutamakan antarmuka yang ramah disabilitas dan kontras tinggi.

---

## 📁 Struktur Folder Proyek
```
acle_net_deeplearning_project/
├── data/                       # Dataset sampel antarmuka & grafik sintetis
│   ├── cognitive_load/          # Folder kelas low_load, medium_load, high_load
│   └── chart_type/              # Folder kelas bar_chart, line_chart, scatter_plot
├── models/                     # Tempat menyimpan weights model (.pth)
├── config.py                   # Konfigurasi hyperparameter & path
├── dataset.py                  # PyTorch Custom Dataset & DataLoader
├── model.py                    # Arsitektur Convolutional Neural Network (ACLE-Net)
├── train.py                    # Script training loop
├── evaluate.py                 # Script evaluasi metrik performa
├── predict.py                  # Script inference & rekomendasi IMK/UCD
├── requirements.txt            # Dependensi library
└── README.md                   # Dokumentasi proyek
```

---

## 🚀 Cara Menjalankan Proyek

1. **Install Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Training Model:**
   ```bash
   python train.py
   ```

3. **Evaluasi Model:**
   ```bash
   python evaluate.py
   ```

4. **Inference / Prediksi Gambar Antarmuka Baru:**
   ```bash
   python predict.py
   ```
