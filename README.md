# Proyek Akhir Multimodal — Kelompok 05
IF25-40304 Pembelajaran Mesin Multimodal, Ganjil 2026/2027

## Anggota

| # | Nama | NIM | Peran |
|---|---|---|---|
| 1 | Muhamad Rafi Ilham | 123140173 | Data dan Etika Data |
| 2 | Cikal Galih Nur Arifin | 123140109 | Baseline dan Pipeline|
| 3 | Raditya Alrasyid Nugroho | 123140125 | 3 · Eksperimen Penyejajaran/Fusi |
| 4 | Jana Rohman Wasiso | 123140046 | Evaluasi dan Uji Kasus Sulit |
| 5 | Muhammad Fatahillah Farid | 123140203 | ntegrasi dan Koordinasi|

Peran terkunci sejak M0, perubahan butuh persetujuan Dosen Pengampu.

## Isi folder
- `src/` — kode pipeline (Peran 2) dan eksperimen (Peran 3)
- `experiments/` — hasil angka dari eksekusi (csv/json), bukan data mentah
- `reports/` — laporan tiap milestone
- `data/`, `cache/` — TIDAK di-commit (lihat .gitignore). Taro train.csv/val.csv/images/
  di data/, dan hasil embedding .npz di cache/, secara lokal masing-masing.

## Cara menjalankan
pip install numpy torch transformers sentence-transformers pillow pandas
python src/m1_baseline.py --seed 0 --data data

## Status
- M0: proposal + reproduksi baseline — selesai, dikumpulkan.
- M1: pipeline end-to-end + evaluasi 4 arah + audit data — selesai, dikumpulkan.
- M2: rancangan varian prompt ensemble ID (lihat reports/rancangan_varian_m2_peran3.docx) —
  kode di src/m2_prompt_ensemble.py, menunggu cache/emb_{train,val}.npz untuk dijalankan.

## Atribusi
Data SEA-VL (Cahyawijaya et al., 2025, ACL), lisensi CC BY-SA 4.0. Lihat ATTRIBUTION.md.