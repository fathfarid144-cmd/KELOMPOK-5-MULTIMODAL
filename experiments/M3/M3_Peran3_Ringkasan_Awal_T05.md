# Ringkasan eksperimen Peran 3 — T05

Hasil eksekusi pada 10 query, 5 pasangan; V1 tetap sistem utama.

- V0: 10/10 query, 5/5 pasangan; mean margin 0.029946; perbaikan vs V0 0, regresi 0; encoding 0.347 detik.
- V1: 10/10 query, 5/5 pasangan; mean margin 0.030692; perbaikan vs V0 0, regresi 0; encoding 0.629 detik.
- V2: 10/10 query, 5/5 pasangan; mean margin 0.034848; perbaikan vs V0 0, regresi 0; encoding 3.209 detik.

Verifikasi V1: belum tersedia

Limitasi: kandidat hanya dua, softmax skala 100 tidak terkalibrasi; atribut kasus berubah bersamaan; V1/V2 berbeda jumlah dan komposisi templat; V2 eksploratif. Tidak ada klaim generalisasi.

Data: SEA-VL Indonesian subset, Cahyawijaya et al. (2025), CC BY-SA 4.0 — https://creativecommons.org/licenses/by-sa/4.0/.