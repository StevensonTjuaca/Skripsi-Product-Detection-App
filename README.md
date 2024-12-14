# Skripsi - Product Detection App

## Deskripsi Proyek
Aplikasi **Product Detection App** dirancang sebagai bagian dari proyek skripsi untuk mendeteksi produk tertentu menggunakan teknologi computer vision. Aplikasi ini memanfaatkan **TensorFlow** untuk klasifikasi gambar, **MediaPipe** untuk deteksi tangan, dan **Tkinter** sebagai antarmuka pengguna (GUI).


## Fitur Utama
- 📂 **Mode File**: Unggah file gambar untuk mendeteksi produk di dalamnya.
- 📸 **Mode Kamera**: Tangkap gambar langsung dari kamera untuk deteksi secara real-time.
- 🔍 **Pengolahan Gambar Otomatis**: Deteksi tangan untuk menentukan area gambar yang relevan.
- 💻 **Antarmuka Pengguna**: Antarmuka grafis yang ramah pengguna dibangun menggunakan Tkinter.

## Teknologi yang Digunakan
- **Python**: Bahasa pemrograman utama.
- **TensorFlow**: Untuk pelatihan dan prediksi model deep learning.
- **MediaPipe**: Untuk deteksi tangan berbasis landmark.
- **OpenCV**: Untuk manipulasi gambar dan kamera.
- **Tkinter**: Untuk antarmuka pengguna grafis.

## 📥 Cara Mendapatkan File Besar
Beberapa file besar seperti folder `_internal` tidak diunggah ke GitHub karena keterbatasan ukuran. Silakan unduh file berikut dari Google Drive:
- [Folder _internal](https://drive.google.com/drive/folders/1zyFqBbZv1z9fXSConalnXiVDJ0QYrt4c?usp=sharing)
Setelah mengunduh file tersebut, letakkan di folder sesuai struktur proyek berikut:


## 📂 Struktur Proyek
Skripsi-Product-Detection-App/ │ ├── README.md # Penjelasan lengkap tentang project ├── src/ # Folder untuk source code │ ├── Skripsi.py # Source code utama aplikasi │ ├── model_mobilenet_fixed1.h5 # File model │ ├── Logo.png # File logo │ ├── background.jpg # Background ├── build/ # Folder untuk file .exe │ ├── ProductDetectionApp.exe │ └── _internal/ # Folder internal yang berisi dependensi .dll ├── manual/ # Folder untuk manual penggunaan │ └── UserManual.pdf # Manual penggunaan dalam PDF ├── LICENSE # File lisensi

## 📦 Cara Menjalankan Aplikasi

### 1. Menggunakan File `.exe`
1. Navigasikan ke folder `build/`.
2. Jalankan file `ProductDetectionApp.exe` dengan klik dua kali.

### 2. Menjalankan dari Source Code
1. Pastikan Python 3.10 atau versi kompatibel telah diinstal.
2. Instal semua dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt
3. Jalankan aplikasi dengan perintah:
    python src/main.py

🛠️ Panduan Penggunaan
### Mode File
- Pilih tombol File di menu utama.
- Klik tombol Upload Gambar untuk mengunggah file gambar.
- Gambar yang berhasil diunggah akan ditampilkan di layar.
- Klik tombol Mulai Pengenalan untuk mendeteksi produk dalam gambar.
- Hasil pengenalan akan muncul di bawah gambar.

### Mode Kamera
- Pilih tombol Camera di menu utama.
- Kamera akan menyala dan menampilkan umpan langsung.
- Klik tombol Capture untuk mengambil gambar.
- Klik tombol Mulai Pengenalan untuk mendeteksi produk dalam gambar yang diambil.
- Hasil pengenalan akan muncul di layar.

💻 Spesifikasi Minimum
- OS: Windows 10 atau lebih baru
- RAM: Minimal 4GB (Direkomendasikan 8GB)
- Prosesor: Intel Core i5 atau setara
- GPU (Opsional): Untuk akselerasi TensorFlow

⚠️ Masalah Umum
1. Error: File tidak ditemukan
Pastikan file model dan folder _internal sudah diletakkan pada lokasi yang sesuai.

2. Error saat mendeteksi produk
Periksa pencahayaan atau pastikan produk terlihat jelas di gambar.

3. File .exe tidak bisa dijalankan
Pastikan semua file dependensi sudah disertakan (misalnya, folder _internal).

📜 Lisensi
Proyek ini dilisensikan di bawah MIT License.

👤 Kontributor
- Nama: Stevenson Tjuaca
- Email: tjuacastevenson@gmail.com

📘 Manual Penggunaan
Manual penggunaan aplikasi tersedia dalam folder:
manual/UserManual.pdf.
