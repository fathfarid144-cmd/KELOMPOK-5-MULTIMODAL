# Reproduksibilitas

## Status dan batas bukti

Artefak hasil historis M1–M3 tersedia di folder milestone masing-masing. Dokumentasi ini tidak menyatakan full rerun telah berhasil. Notebook M1–M3 memiliki setup/path yang ditulis secara lokal di masing-masing notebook; konfigurasi global baru di root belum diimpor oleh notebook historis agar output dan kode historis tetap utuh.

Metadata hasil yang tersedia tidak menunjukkan satu environment konsisten lintas milestone: JSON M1 mencatat Python 3.14.0 dan PyTorch 2.9.1+cpu; metadata M3 mencatat Python 3.13.18, sentence-transformers 5.7.0, torch 2.14.1+cpu, NumPy 2.4.6, pandas 2.2.3, Pillow 11.3.0, dan transformers 5.22.1. Karena itu `requirements-lock.txt` belum memuat versi pin yang mengklaim mewakili seluruh eksperimen.

## Instalasi lokal

Gunakan environment Python yang kompatibel dengan paket proyek, lalu:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Versi Python dan paket harus dicatat saat eksekusi. Setelah satu environment diverifikasi untuk rerun yang dituju, ekspor lock nyata dengan:

```bash
python -m pip freeze > requirements-lock.txt
```

Jangan menyebut hasil rerun reproducible penuh hanya berdasarkan instalasi dependensi atau pemeriksaan statis.

## Path dan konfigurasi

`project_config.py` menggunakan `pathlib.Path`. Default lokal menunjuk ke `data/`, `cache/`, dan `results/`; path dapat diubah tanpa mengedit kode:

| Variabel | Default | Isi |
|---|---|---|
| `MM_DATA_DIR` | `<repo>/data` | Folder `train.csv`, `val.csv`, `images/` |
| `MM_CACHE_DIR` | `<repo>/cache` | Cache embedding |
| `MM_RESULTS_DIR` | `<repo>/results` | Hasil run baru |
| `MM_SEED` | `0` | Seed |
| `MM_DEVICE` | deteksi CUDA, fallback CPU | Perangkat komputasi |

API tersedia melalui `validate_dataset()` dan `prepare_directories(milestone)`. Notebook historis belum menggunakan API tersebut dan tetap memerlukan path setup di sel masing-masing.

## Urutan eksekusi dan ketergantungan

1. M0: proposal dan baseline referensi; artefak di `reports/M0/` dan `experiments/M0/`.
2. M1: baseline retrieval; notebook di `src/M1/`, hasil serta cache embedding di luar repository atau di lokasi yang dikonfigurasi notebook.
3. M2: membaca embedding/cache M1 dengan asumsi split, urutan baris, encoder, dan normalisasi sesuai. Notebook M2 saat ini mengharapkan nama cache `emb_train.npz` dan `emb_val.npz` pada Drive. Periksa metadata dan urutan CSV sebelum menggunakan kembali cache.
4. M3: protokol kasus sulit dan evaluasi; notebook menggunakan manifest dan path Google Drive yang tertanam. Hasil historis M3 tercatat di `experiments/M3/`.
5. M4: belum ada notebook/protokol/hasil M4 terverifikasi di working tree.

M1 dan M2 melaporkan seed 0. Metrik utama M1/M2 dan ringkasan hard-case M3 dicatat dalam README dengan sumber artefak masing-masing. Gunakan CPU untuk konfigurasi yang tercatat; hasil pada perangkat/pustaka berbeda perlu dibandingkan sebagai run baru.

## Colab dan Google Drive

Notebook resmi yang tersedia ditautkan di [README](../README.md). Open in Colab abre notebook; notebook harus tetap disimpan di repository publik dan dataset/cache di Drive pengguna. Untuk M1, ikuti sel pemilihan atau ekstraksi bundle. M2/M3 memuat path Drive yang dapat berbeda dari akun pengguna, sehingga perbarui path pada sel konfigurasi. Default path contoh di README adalah petunjuk historis, bukan validasi bahwa bundel pengguna berada di sana.

## Membandingkan run dan keterbatasan

Simpan output baru di direktori hasil terpisah, catat commit kode, versi Python/paket, seed, device, nama model, split, hash/metadata cache, dan waktu eksekusi. Bandingkan metrik baru dengan JSON/CSV historis tanpa menimpa artefak lama. Jangan memakai test privat untuk pemilihan metode, debugging, atau tuning.

Validasi statis format notebook/JSON/CSV tidak setara dengan smoke test atau full execution. Full rerun memerlukan dataset resmi dan model yang dapat diakses; keduanya tidak disertakan dalam repository publik.
