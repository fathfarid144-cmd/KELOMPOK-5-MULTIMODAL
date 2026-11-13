"""M1 - Audit data dan analisis tambahan (dijalankan setelah m1_baseline.py).

Menghasilkan hasil_m1/audit_data.json dan hasil_m1/analisis_val.json.
Audit memakai train.csv + val.csv (metadata, bukan pelatihan). Analisis kinerja per kelompok
memakai ranking val yang sudah dihitung m1_baseline.py; tidak ada hiperparameter yang disetel.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(s).lower())).strip()


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(here.parent.parent / "bundle-mahasiswa" / "bundle-mahasiswa"))
    ap.add_argument("--out", default=str(here / "hasil_m1"))
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.out)
    tr = pd.read_csv(root / "train.csv")
    va = pd.read_csv(root / "val.csv")
    al = pd.concat([tr, va], ignore_index=True)
    R = {}

    R["n"] = {"train": len(tr), "val": len(va), "total": len(al)}
    R["file_hilang"] = int(sum(not (root / f).exists() for f in al["file"]))

    # sebaran
    R["kategori"] = {s: d["category"].value_counts().to_dict() for s, d in (("train", tr), ("val", va))}
    R["kategori_val_pct"] = (va["category"].value_counts(normalize=True) * 100).round(1).to_dict()
    R["native_lang"] = al["native_lang"].value_counts().to_dict()
    R["native_lang_val"] = va["native_lang"].value_counts().to_dict()

    # wilayah (kolom loc berisi daftar negara)
    locs = al["loc"].fillna("").map(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    cnt = Counter(x for l in locs for x in l)
    R["loc_negara"] = dict(cnt.most_common())
    R["loc_hanya_indonesia"] = int((locs.map(len) == 1).sum())
    R["loc_multi_negara"] = int((locs.map(len) > 1).sum())
    R["loc_rata_negara"] = round(float(locs.map(len).mean()), 2)
    R["loc_ge4_negara"] = int((locs.map(len) >= 4).sum())

    # duplikat
    R["dup_image_id"] = int(al["image_id"].duplicated().sum())
    hashes = {f: md5(root / f) for f in al["file"]}
    al["md5"] = al["file"].map(hashes)
    R["dup_md5"] = int(al["md5"].duplicated().sum())
    for col in ("caption_id", "caption_en"):
        al[col + "_n"] = al[col].map(norm)
    trn, van = al[al.split == "train"], al[al.split == "val"]
    R["dup_caption_dalam_train"] = {c: int(trn[c + "_n"].duplicated().sum()) for c in ("caption_id", "caption_en")}
    R["dup_caption_dalam_val"] = {c: int(van[c + "_n"].duplicated().sum()) for c in ("caption_id", "caption_en")}
    inter = {}
    for c in ("caption_id", "caption_en"):
        s = set(trn[c + "_n"])
        inter[c] = int(van[c + "_n"].isin(s).sum())
    R["caption_val_sama_dengan_train"] = inter
    R["baris_val_caption_sama_train"] = int((van["caption_id_n"].isin(set(trn["caption_id_n"])) | van["caption_en_n"].isin(set(trn["caption_en_n"]))).sum())
    R["caption_id_sama_caption_en"] = int((al["caption_id_n"] == al["caption_en_n"]).sum())

    # caption val yang kembar di dalam val -> mengacaukan ground truth satu-satu
    dup_val_id = van["caption_id_n"].duplicated(keep=False)
    R["val_caption_id_kembar_dalam_val_baris"] = int(dup_val_id.sum())

    # panjang caption
    for c in ("caption_id", "caption_en"):
        w = al[c].astype(str).str.split().map(len)
        R[f"panjang_{c}_kata"] = {"min": int(w.min()), "median": float(w.median()), "mean": round(float(w.mean()), 1), "max": int(w.max())}
    R["caption_id_le3_kata"] = int((al["caption_id"].astype(str).str.split().map(len) <= 3).sum())

    # kecurigaan
    sus = {}
    sus["caption_en_artefak_terjemahan_mesin"] = al.loc[al["caption_en"].str.contains(r"\d+\s*/\s*[\d,]+\s", regex=True), "image_id"].tolist()
    sus["caption_en_template_a_photo_of"] = int(al["caption_en"].str.lower().str.startswith("a photo of").sum())
    sus["nol_nilai_skala_rusak_relevance_gt5"] = int((al["relevance"] > 5).sum())
    sus["caption_fit_lt_0.9"] = int((al["caption_fit"] < 0.9).sum())
    sus["img_quality_lt_0.9"] = int((al["img_quality"] < 0.9).sum())
    sus["relevance_min_max"] = [float(al["relevance"].min()), float(al["relevance"].max())]
    sus["n_annotator"] = al["n_annotator"].value_counts().to_dict()
    sus["lokasi_non_indonesia_pada_caption"] = al.loc[
        al["caption_en"].str.contains(r"Singapore|Fullerton|Merlion|Malaysia|Thailand|Vietnam", case=False, regex=True)
        & ~al["caption_en"].str.contains("Indonesia", case=False), "image_id"
    ].tolist()
    jav = al["native_lang"].str.contains("Javanese|Sundanese|Minang|Balin", regex=True)
    sus["native_non_indonesia"] = int(jav.sum())
    R["mencurigakan"] = sus
    R["ukuran_gambar"] = {"w_min": int(al.width.min()), "w_max": int(al.width.max()), "h_min": int(al.height.min()), "h_max": int(al.height.max())}
    R["gambar_kecil_lt_400"] = int(((al.width < 400) | (al.height < 400)).sum())

    # kategori 'kuliner' vs kata kunci makanan yang berada di kategori lain (keyword-based label bising)
    food = r"nasi|soto|bakso|mie |mi |sate|bubur|goreng|kue|jajan|sambal|sayur|es |opor|rendang|gado"
    nonfood = al[(al.category != "kuliner") & al["caption_en"].str.contains(food, case=False, regex=True)]
    R["kategori_bukan_kuliner_tapi_caption_makanan"] = {"n": int(len(nonfood)), "kategori": nonfood["category"].value_counts().to_dict(), "contoh": nonfood["caption_en"].head(5).tolist()}

    (out / "audit_data.json").write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- analisis kinerja per kelompok pada val ----
    rk = pd.read_csv(out / "rank_per_item_val.csv").merge(va[["image_id", "category", "native_lang", "caption_id", "caption_en"]], on="image_id")
    rk["len_id"] = rk["caption_id"].str.split().map(len)
    A = {}
    d = {
        "i2t_id": "rank_img2txt_id", "i2t_en": "rank_img2txt_en",
        "t2i_id": "rank_txt2img_id", "t2i_en": "rank_txt2img_en",
    }
    def grp(col):
        o = {}
        for g, s in rk.groupby(col):
            o[str(g)] = {"n": int(len(s)), **{k: round(float((s[v] <= 1).mean()), 3) for k, v in d.items()}}
        return o
    A["per_kategori_R@1"] = grp("category")
    A["per_native_R@1"] = grp("native_lang")
    rk["len_bin"] = pd.cut(rk["len_id"], [0, 5, 10, 20, 1000], labels=["<=5", "6-10", "11-20", ">20"])
    A["per_panjang_caption_R@1"] = grp("len_bin")

    # kasus: ID gagal tapi EN berhasil (language gap) dan sebaliknya
    A["t2i_EN_bisa_ID_gagal_R@1"] = int(((rk.rank_txt2img_en == 1) & (rk.rank_txt2img_id > 1)).sum())
    A["t2i_ID_bisa_EN_gagal_R@1"] = int(((rk.rank_txt2img_id == 1) & (rk.rank_txt2img_en > 1)).sum())
    A["i2t_EN_bisa_ID_gagal_R@1"] = int(((rk.rank_img2txt_en == 1) & (rk.rank_img2txt_id > 1)).sum())
    A["i2t_ID_bisa_EN_gagal_R@1"] = int(((rk.rank_img2txt_id == 1) & (rk.rank_img2txt_en > 1)).sum())
    A["median_rank"] = {k: float(rk[v].median()) for k, v in d.items()}
    # acak: R@1 = 1/N
    A["pembanding_acak_R@1"] = round(1 / len(va), 4)
    A["pembanding_acak_R@5"] = round(5 / len(va), 4)
    A["pembanding_acak_R@10"] = round(10 / len(va), 4)

    # kata Jawa pada caption_id (heuristik): kata yang sering muncul di caption berlabel Javanese
    jv = rk[rk.native_lang.str.contains("Javanese")]
    A["val_jawa_n"] = int(len(jv))
    worst = rk.sort_values("rank_txt2img_id", ascending=False).head(12)[["image_id", "category", "caption_id", "caption_en", "rank_txt2img_id", "rank_txt2img_en"]]
    A["terburuk_t2i_id"] = worst.to_dict("records")
    (out / "analisis_val.json").write_text(json.dumps(A, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"audit": R, "analisis": {k: v for k, v in A.items() if k != "terburuk_t2i_id"}}, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
