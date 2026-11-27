#!/usr/bin/env python3
"""Audit reproduksibel untuk Peran A, M2 T05.

Memeriksa pemisahan metadata train/val, kesamaan caption lintas split,
subset train 500 baris untuk seleksi V1, serta konsistensi artefak JSON
terhadap notebook. TIDAK menjalankan ulang encoder dan TIDAK membuktikan
kemiripan visual file gambar.

Cara pakai:
    python audit_leakage_m2_peran_A.py --root .
Dependensi: pip install pandas numpy rapidfuzz
File yang dibutuhkan pada --root:
    train.csv, val.csv, M2_PromptEnsemble_v2.ipynb,
    templates_terpilih_v2.json, hasil_m2_prompt_ensemble_v2.json
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz  # pip install rapidfuzz


def norm(x: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(x).lower())).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit leakage M2 T05 oleh Peran A")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve() if args.out else root / "hasil_audit_m2_peran_A"
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(root / "train.csv")
    val = pd.read_csv(root / "val.csv")
    nb = json.loads((root / "M2_PromptEnsemble_v2.ipynb").read_text(encoding="utf-8"))
    chosen = json.loads((root / "templates_terpilih_v2.json").read_text(encoding="utf-8"))
    result = json.loads((root / "hasil_m2_prompt_ensemble_v2.json").read_text(encoding="utf-8"))

    for label, df in (("train", train), ("validation", val)):
        required = {"image_id", "file", "caption_id", "caption_en", "split"}
        if missing := required.difference(df.columns):
            raise ValueError(f"{label} kehilangan kolom {sorted(missing)}")
    if not (train["split"] == "train").all() or not (val["split"] == "val").all():
        raise AssertionError("Kolom split tidak cocok dengan berkas CSV")

    counts = {
        "train": int(len(train)), "val": int(len(val)),
        "train_image_id_duplicated": int(train["image_id"].duplicated().sum()),
        "val_image_id_duplicated": int(val["image_id"].duplicated().sum()),
        "cross_split_image_id": len(set(train["image_id"]) & set(val["image_id"])),
        "cross_split_file_path": len(set(train["file"]) & set(val["file"])),
    }

    for df in (train, val):
        for col in ("caption_id", "caption_en"):
            df[col + "_norm"] = df[col].map(norm)
    id_full = val["caption_id_norm"].isin(set(train["caption_id_norm"]))
    en_full = val["caption_en_norm"].isin(set(train["caption_en_norm"]))
    match_full = id_full | en_full

    seed = int(chosen["seed"])
    n_sub = int(chosen["n_sub"])
    if chosen.get("split") != "train":
        raise AssertionError("templates_terpilih_v2.json: split harus train")
    if n_sub > len(train):
        raise ValueError("n_sub melebihi jumlah train")
    idx = np.sort(np.random.default_rng(seed).choice(len(train), size=n_sub, replace=False))
    train_subset = train.iloc[idx]
    id_sub = val["caption_id_norm"].isin(set(train_subset["caption_id_norm"]))
    en_sub = val["caption_en_norm"].isin(set(train_subset["caption_en_norm"]))
    match_sub = id_sub | en_sub

    code = "\n".join(c.get("source", "") if isinstance(c.get("source", ""), str)
                     else "".join(c.get("source", []))
                     for c in nb["cells"] if c.get("cell_type") == "code")
    cell12 = nb["cells"][12]["source"]
    cell12 = cell12 if isinstance(cell12, str) else "".join(cell12)
    source_checks = {
        "seleksi_greedy_cell12_tanpa_referensi_val": not re.search(r"\b(?:val|va|cap_id_val|cap_en_val)\b", cell12) and "va[" not in cell12.lower(),
        "seleksi_train_dipanggil": 'captions("train")' in code,
        "subset_train_dipakai_greedy": 'tr["img"][idx]' in code,
        "seed_rng_train": "np.random.default_rng(SEED)" in code,
        "val_dipakai_evaluasi": 'captions("val")' in code,
        "cache_validation_dimuat_sebelum_seleksi": 'va = load("val")' in code,
        "hasil_V2_diproses": 'TEMPLATES_ID)' in code and 'Ti_v2' in code,
    }
    # Pemeriksaan sumber bersifat statis; bukan bukti semua riwayat eksperimen.
    if not all(source_checks.values()):
        raise AssertionError(f"Ada pola alur notebook yang berubah: {source_checks}")
    config_ok = (
        chosen["chosen"] == result["_meta"]["templat_V1"]
        and result["_meta"]["templat_V2"] == ["{}", "foto {}", "sebuah foto {}", "gambar {}",
           "foto budaya Indonesia: {}", "dokumentasi budaya Indonesia: {}", "{} di Indonesia",
           "sebuah foto yang menunjukkan {}", "potret {}", "ini adalah foto {}"]
        and result["_meta"]["seed"] == seed
        and result["_meta"]["split"] == "val"
    )
    if not config_ok:
        raise AssertionError("JSON konfigurasi V1/V2 tidak konsisten")

    # Caption MIRIP: string similarity karakter, bukan ukuran kemiripan makna.
    # Ambang 90 dari 100 adalah aturan skrining eksploratif, bukan ground truth.
    near_id, near_en = set(), set()
    near_rows = []
    for col, target_set in (("caption_id", near_id), ("caption_en", near_en)):
        pool = train[col + "_norm"].tolist()
        for i, q in enumerate(val[col + "_norm"].tolist()):
            hit = process.extractOne(q, pool, scorer=fuzz.ratio, score_cutoff=90)
            if hit is None or hit[0] == q:
                continue  # hanya pasangan MIRIP tetapi tidak IDENTIK
            txt_train, score, j = hit
            target_set.add(i)
            near_rows.append({
                "val_image_id": val.iloc[i]["image_id"],
                "train_image_id": train.iloc[j]["image_id"],
                "bahasa": col, "kemiripan_karakter_persen": round(float(score), 2),
                "caption_val": val.iloc[i][col], "caption_train": train.iloc[j][col],
            })
    pd.DataFrame(near_rows).to_csv(
        out / "bukti_caption_mirip_lintas_split_heuristik.csv",
        index=False, encoding="utf-8-sig")

    overlaps = val.loc[match_full, ["image_id", "caption_id", "caption_en"]].copy()
    overlaps["cocok_id_dengan_train"] = id_full.loc[match_full].values
    overlaps["cocok_en_dengan_train"] = en_full.loc[match_full].values
    overlaps["cocok_dengan_subtrain_500"] = match_sub.loc[match_full].values
    overlaps.to_csv(out / "bukti_caption_val_cocok_train.csv", index=False, encoding="utf-8-sig")

    summary = {
        "dataset": counts,
        "caption_normalisasi": "lowercase, hapus tanda baca, rapikan spasi; mengikuti m1_analisis.py",
        "caption_val_cocok_train": {
            "ID": int(id_full.sum()), "EN": int(en_full.sum()),
            "baris_unik": int(match_full.sum()),
        },
        "caption_val_cocok_subtrain_seleksi_V1": {
            "seed": seed, "n_sub": n_sub,
            "ID": int(id_sub.sum()), "EN": int(en_sub.sum()),
            "baris_unik": int(match_sub.sum()),
            "image_id_val": val.loc[match_sub, "image_id"].tolist(),
        },
        "caption_mirip_nonidentik_heuristik_90": {
            "metode": "rapidfuzz fuzz.ratio pada caption normalisasi, ambang >=90/100, pasangan terbaik, bukan anotasi semantik",
            "ID": len(near_id), "EN": len(near_en),
            "baris_val_unik": len(near_id | near_en),
        },
        "audit_statis_notebook": source_checks,
        "json_konsisten_dengan_konfigurasi": config_ok,
        "kronologi_dikonfirmasi_pelaksana": [
            "V1 dipilih di train melalui greedy selection",
            "hasil validation V1 dilihat",
            "V2 10 templat dirancang sesudah melihat hasil val V1",
            "V2 kemudian diuji pada val yang sama",
        ],
        "klasifikasi": "V1 hasil utama (berdasarkan kronologi); V2 eksploratif/adaptif, bukan uji independen",
        "keterbatasan": [
            "Keputusan V2 berasal dari konfirmasi pelaksana; tidak dapat disimpulkan hanya dari source code",
            "Kemiripan visual gambar tidak dapat diuji tanpa file gambar; hasil hash M1 tidak diuji ulang",
            "Kesamaan caption bukan dengan sendirinya bukti kebocoran data",
            "Kode dievaluasi secara statis; encoder dan metrik M2 tidak dijalankan ulang",
        ],
    }
    (out / "hasil_audit_m2_peran_A.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    txt = f"""BUKTI AUDIT LEAKAGE M2 — KELOMPOK T05 — PERAN A
Sumber: train.csv, val.csv, M2_PromptEnsemble_v2.ipynb, templates_terpilih_v2.json,
        hasil_m2_prompt_ensemble_v2.json, konfirmasi kronologi dari pelaksana eksperimen.

A. PEMISAHAN DATA
Train = {len(train)}; validation = {len(val)}.
Duplikat image_id internal train/val = {counts['train_image_id_duplicated']}/{counts['val_image_id_duplicated']}.
Image ID tumpang tindih antar-split = {counts['cross_split_image_id']}.
Path gambar tumpang tindih antar-split = {counts['cross_split_file_path']}.

B. CAPTION LINTAS SPLIT
Caption ID validation cocok dengan train = {int(id_full.sum())} baris.
Caption EN validation cocok dengan train = {int(en_full.sum())} baris.
Total baris validation unik yang cocok minimal satu bahasa = {int(match_full.sum())}.
Dengan subset 500 train yang dipilih seed {seed}, baris validation unik yang cocok = {int(match_sub.sum())}.
Kemiripan caption nonidentik (rasio >=90): ID={len(near_id)}; EN={len(near_en)}; baris val unik={len(near_id | near_en)}.
Ini hanya indikasi berdasarkan kemiripan karakter; perlu peninjauan manusia.
Perhatian: kesamaan/kemiripan caption bukan bukti langsung kebocoran gambar/model.

C. ALUR NOTEBOOK DAN KONFIGURASI
Seleksi greedy V1 memakai caption train dan tr['img'][idx], seed={seed}, n_sub={n_sub}.
JSON V1 chosen = {chosen['chosen']}.
V2 menggunakan {len(result['_meta']['templat_V2'])} templat Indonesia.
Cache validation dimuat di cell sebelum seleksi, tetapi tidak ditemukan sebagai input greedy V1.
Validation dipakai untuk evaluasi; konsistensi JSON varian = {config_ok}.

D. TEMUAN KRITIS (BERDASARKAN KONFIRMASI RADIT)
V2 ditentukan SETELAH melihat hasil validation V1, dan diuji pada validation yang sama.
Ini adalah pemilihan/desain adaptif menggunakan informasi evaluasi (risiko validation leakage).
TIDAK bisa dipulihkan menjadi evaluasi independen hanya dengan mengulang notebook.
V1 dapat dijelaskan sebagai hasil utama berdasarkan pemilihan di train sebelum eval val;
V2 WAJIB diberi status eksploratif dalam laporan. Tidak boleh mengubah kronologi.

E. BATAS PEMERIKSAAN
Tidak menjalankan ulang encoder/hasil retrieval; tidak menilai near-duplicate secara visual.
Pemeriksaan MD5/SHA-256 dari M1 dapat dirujuk, tetapi tidak dihitung ulang tanpa gambar asli.
Hasil audit tidak membuktikan bahwa seluruh eksperimen bebas dari leakage lain.
"""
    (out / "bukti_audit_leakage_m2.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print("Bukti tersimpan di:", out)


if __name__ == "__main__":
    main()
