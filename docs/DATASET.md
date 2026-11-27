# Dataset

## Sumber dan lisensi

Proyek menggunakan SEA-VL Indonesian subset (`KORIKA-AI/sea-vl_crowdsourcing_id`), turunan `SEACrowd/sea-vl_crowdsourcing`. Atribusi lengkap dan sitasi tersedia di [ATTRIBUTION.md](../ATTRIBUTION.md). Dataset berlisensi CC BY-SA 4.0. Akses bundel yang digunakan untuk tugas kuliah diberikan melalui LMS; repository publik ini tidak mendistribusikan bundel.

## Berkas yang dibutuhkan

Notebook M1 dan M2 membaca `train.csv` dan `val.csv`; M1 juga membaca citra sesuai kolom file. Notebook menggunakan kolom `file`, `caption_id`, `caption_en`, `category`, dan `loc`. Audit artefak melaporkan 2.072 baris train dan 275 baris validation. Pastikan struktur bundel yang diberikan dosen berisi CSV dan citra yang sesuai sebelum menjalankan notebook. Konfigurasi global mendukung citra di `images/` maupun langsung di folder data.

Contoh struktur lokal:

```text
dataset-bundle/
├── train.csv
├── val.csv
└── images/
```

Notebook historis M2/M3 masih menyimpan konfigurasi Google Drive di dalam notebook. Pada Colab, mount Drive dan sesuaikan path `DATA_DIR` sesuai bundel aktual. Untuk konfigurasi bersama yang baru, atur `MM_DATA_DIR` ke folder yang berisi CSV.

```powershell
$env:MM_DATA_DIR = 'D:\path\to\dataset-bundle'
```

```python
import os
os.environ["MM_DATA_DIR"] = "/content/drive/MyDrive/MULTIMODAL/bundle-mahasiswa/bundle-mahasiswa"
```

Jangan menambahkan dataset mentah, materi LMS, test set privat, atau prediksi test privat ke repository publik. Gunakan hanya salinan yang memang diizinkan untuk tugas ini dan pertahankan atribusi pada distribusi turunannya.
