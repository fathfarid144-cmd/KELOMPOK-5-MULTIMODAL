from pathlib import Path

folder_gambar = Path(__file__).resolve().parent / "images"

daftar_gambar = [
    file for file in folder_gambar.rglob("*")
    if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
]

print("Jumlah file gambar:", len(daftar_gambar))

import hashlib

hash_gambar = {}

duplikat = []

for gambar in daftar_gambar:
    with open(gambar, "rb") as file:
        hash_file = hashlib.file_digest(file, "sha256").hexdigest()

    if hash_file in hash_gambar:
        duplikat.append((hash_gambar[hash_file], gambar))
    else:
        hash_gambar[hash_file] = gambar

print("Jumlah pasangan file gambar duplikat:", len(duplikat))

for gambar_asli, gambar_duplikat in duplikat:
    print("Duplikat:", gambar_asli.name, "dan", gambar_duplikat.name)

import csv

folder_dataset = Path(__file__).resolve().parent

for nama_csv in ["train.csv", "val.csv"]:
    jumlah_hilang = 0

    with open(folder_dataset / nama_csv, encoding="utf-8", newline="") as f:
        data = csv.DictReader(f)

        for baris in data:
            lokasi_gambar = folder_dataset / baris["file"]

            if not lokasi_gambar.is_file():
                jumlah_hilang += 1

    print(f"{nama_csv} - Gambar tidak ditemukan: {jumlah_hilang}")