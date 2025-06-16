# Deteksi Jenis Biji Kopi dengan YOLOv8 ☕️

Aplikasi berbasis Streamlit untuk mendeteksi jenis biji kopi: **Arabika, Robusta, dan Liberika** dari gambar menggunakan model YOLOv8.

## 🚀 Cara Menggunakan (Online di Hugging Face Spaces)
1. Klik tombol **"Upload gambar..."**
2. Aplikasi otomatis akan menampilkan:
   - Gambar asli
   - Gambar hasil deteksi
   - Statistik deteksi: jumlah dan persentase tiap jenis

## 🔍 Fitur
- Deteksi otomatis 3 kelas kopi (arabika, liberika, robusta)
- Visualisasi bounding box
- Statistik deteksi yang informatif
- Model akan otomatis diunduh saat pertama kali dijalankan

## 🧠 Model
Model YOLOv8 custom dilatih untuk klasifikasi biji kopi. File model (`sken4.pt`) akan diunduh dari Google Drive saat pertama kali dijalankan.

## 📦 Dibangun dengan:
- [Streamlit](https://streamlit.io/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
