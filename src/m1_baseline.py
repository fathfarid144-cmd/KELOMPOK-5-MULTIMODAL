"""M1 - Baseline pencarian lintas modal (SEA-VL Indonesia).

Pipeline end-to-end, encoder beku, tanpa pelatihan:
  gambar      -> sentence-transformers/clip-ViT-B-32
  teks EN     -> sentence-transformers/clip-ViT-B-32                (encoder teks CLIP asli)
  teks ID     -> sentence-transformers/clip-ViT-B-32-multilingual-v1 (siswa multibahasa, ruang CLIP)
Semua vektor dinormalisasi L2, kesamaan = kosinus (dot product), peringkat = urutan menurun.

Aturan leakage (instruksi M1): val.csv HANYA dipakai untuk menghitung metrik akhir.
Contoh query dipilih dari train.csv dan galerinya adalah gambar train.

Pemakaian:
  python m1_baseline.py --seed 0
  python m1_baseline.py --seed 0 --data ../../bundle-mahasiswa/bundle-mahasiswa --device cpu
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

IMG_MODEL = "sentence-transformers/clip-ViT-B-32"
TXT_ID_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
DIRECTIONS = ["img2txt_id", "img2txt_en", "txt2img_id", "txt2img_en"]
LABEL = {
    "img2txt_id": "gambar -> teks ID",
    "img2txt_en": "gambar -> teks EN",
    "txt2img_id": "teks ID -> gambar",
    "txt2img_en": "teks EN -> gambar",
}


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(here.parent.parent / "bundle-mahasiswa" / "bundle-mahasiswa"))
    p.add_argument("--out", default=str(here / "hasil_m1"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=10, help="K untuk mAP@K")
    p.add_argument("--batch-img", type=int, default=32)
    p.add_argument("--batch-txt", type=int, default=64)
    p.add_argument("--n-examples", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-cache", action="store_true", help="abaikan cache sematan")
    return p.parse_args()


def load_images(root, files):
    out = []
    for f in files:
        with Image.open(root / f) as im:
            out.append(im.convert("RGB"))
    return out


def embed_split(name, df, root, img_model, txt_id_model, args, cache_dir):
    """Kembalikan (E_img, T_id, T_en) ter-normalisasi. Di-cache ke .npz per split."""
    cache = cache_dir / f"emb_{name}.npz"
    if cache.exists() and not args.no_cache:
        z = np.load(cache)
        return z["img"], z["id"], z["en"], 0.0, 0.0, 0.0
    t0 = time.perf_counter()
    imgs = load_images(root, df["file"])
    E = img_model.encode(imgs, batch_size=args.batch_img, normalize_embeddings=True, convert_to_numpy=True)
    t_img = time.perf_counter() - t0
    t0 = time.perf_counter()
    Ti = txt_id_model.encode(list(df["caption_id"]), batch_size=args.batch_txt, normalize_embeddings=True, convert_to_numpy=True)
    t_id = time.perf_counter() - t0
    t0 = time.perf_counter()
    Te = img_model.encode(list(df["caption_en"]), batch_size=args.batch_txt, normalize_embeddings=True, convert_to_numpy=True)
    t_en = time.perf_counter() - t0
    np.savez(cache, img=E, id=Ti, en=Te)
    return E, Ti, Te, t_img, t_id, t_en


def ranks_of_truth(Q, G):
    """Peringkat (1 = terbaik) item benar (indeks diagonal) untuk tiap query. Seri dihitung pesimis."""
    S = Q @ G.T
    diag = np.diag(S)[:, None]
    return 1 + (S > diag).sum(axis=1)


def metrics(rank, k_map):
    r = {f"R@{k}": float((rank <= k).mean()) for k in (1, 5, 10)}
    # satu item relevan per query -> AP@K = 1/rank bila rank <= K, selain itu 0
    r[f"mAP@{k_map}"] = float(np.where(rank <= k_map, 1.0 / rank, 0.0).mean())
    r["median_rank"] = float(np.median(rank))
    r["n_query"] = int(len(rank))
    return r


def evaluate(E, Ti, Te, k_map):
    ranks = {
        "img2txt_id": ranks_of_truth(E, Ti),
        "img2txt_en": ranks_of_truth(E, Te),
        "txt2img_id": ranks_of_truth(Ti, E),
        "txt2img_en": ranks_of_truth(Te, E),
    }
    return {d: metrics(r, k_map) for d, r in ranks.items()}, ranks


def make_examples(df, E, Ti, Te, n, seed, top=3):
    """n query ID acak dari TRAIN; galeri = gambar TRAIN. Ditampilkan apa adanya (tanpa seleksi).
    rank_en_query_setara = peringkat gambar benar bila caption Inggris pasangannya yang dipakai (diagnostik language gap)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False)
    S = Ti[idx] @ E.T
    S_en = Te[idx] @ E.T
    rows = []
    for row, i in enumerate(idx):
        order = np.argsort(-S[row])
        rank = int(np.where(order == i)[0][0]) + 1
        rank_en = int(1 + (S_en[row] > S_en[row, i]).sum())
        rows.append(
            {
                "no": row + 1,
                "train_index": int(i),
                "query_id": df.loc[i, "caption_id"],
                "image_id_benar": df.loc[i, "image_id"],
                "rank_benar": rank,
                "rank_en_query_setara": rank_en,
                "benar_top1": rank == 1,
                "top3_image_id": [df.loc[j, "image_id"] for j in order[:top]],
                "top3_caption_id": [df.loc[j, "caption_id"] for j in order[:top]],
                "top3_skor": [round(float(S[row, j]), 3) for j in order[:top]],
                "kategori_benar": df.loc[i, "category"],
                "kategori_top3": [df.loc[j, "category"] for j in order[:top]],
            }
        )
    return rows


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.data)
    out = Path(args.out)
    cache_dir = out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(root / "train.csv")
    val = pd.read_csv(root / "val.csv")
    print(f"train={len(train)} val={len(val)} device={args.device}")

    need_models = args.no_cache or not all((cache_dir / f"emb_{s}.npz").exists() for s in ("train", "val"))
    t_load = time.perf_counter()
    img_model = txt_id_model = None
    if need_models:  # model baru dimuat/diunduh bila cache sematan belum ada
        img_model = SentenceTransformer(IMG_MODEL, device=args.device)
        txt_id_model = SentenceTransformer(TXT_ID_MODEL, device=args.device)
    t_load = time.perf_counter() - t_load

    timing = {"load_model_s": t_load}
    emb = {}
    for name, df in (("train", train), ("val", val)):
        E, Ti, Te, t_img, t_id, t_en = embed_split(name, df, root, img_model, txt_id_model, args, cache_dir)
        emb[name] = (E, Ti, Te)
        timing[f"{name}_embed_img_s"] = t_img
        timing[f"{name}_embed_txt_id_s"] = t_id
        timing[f"{name}_embed_txt_en_s"] = t_en
        print(f"[{name}] img {E.shape} id {Ti.shape} en {Te.shape}  ({t_img:.1f}s/{t_id:.1f}s/{t_en:.1f}s)")

    # ---- metrik val (satu-satunya penggunaan val) ----
    res_val, ranks_val = evaluate(*emb["val"], args.k)
    # ---- diagnostik di train (boleh dilihat; tidak dipakai menyetel apa pun) ----
    res_train, _ = evaluate(*emb["train"], args.k)

    gap = {}
    for a, b in (("img2txt_id", "img2txt_en"), ("txt2img_id", "txt2img_en")):
        gap[f"{a}_vs_{b}"] = {m: res_val[a][m] - res_val[b][m] for m in ("R@1", "R@5", "R@10", f"mAP@{args.k}")}
        gap[f"{a}_vs_{b}_relatif_R@1"] = (res_val[a]["R@1"] - res_val[b]["R@1"]) / max(res_val[b]["R@1"], 1e-9)

    examples = make_examples(train, *emb["train"], args.n_examples, args.seed)

    timing["total_s"] = sum(v for k, v in timing.items() if k.endswith("_s") and k != "total_s")
    meta = {
        "seed": args.seed,
        "k_map": args.k,
        "device": args.device,
        "cpu": platform.processor(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "models": {"image": IMG_MODEL, "text_en": IMG_MODEL, "text_id": TXT_ID_MODEL},
        "batch": {"image": args.batch_img, "text": args.batch_txt},
        "n_train": len(train),
        "n_val": len(val),
        "timing": timing,
    }
    (out / "results_val.json").write_text(json.dumps({"meta": meta, "val": res_val, "train_diagnostik": res_train, "gap": gap}, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([{"arah": LABEL[d], **res_val[d]} for d in DIRECTIONS]).to_csv(out / "results_val.csv", index=False)
    pd.DataFrame(examples).to_csv(out / "contoh_query_id.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        {"image_id": val["image_id"], **{f"rank_{d}": ranks_val[d] for d in DIRECTIONS}}
    ).to_csv(out / "rank_per_item_val.csv", index=False)

    print("\n=== VAL (N=%d, seed=%d, mAP@%d) ===" % (len(val), args.seed, args.k))
    for d in DIRECTIONS:
        r = res_val[d]
        print(f"{LABEL[d]:20s} R@1={r['R@1']:.3f} R@5={r['R@5']:.3f} R@10={r['R@10']:.3f} mAP@{args.k}={r[f'mAP@{args.k}']:.3f} medR={r['median_rank']:.0f}")
    print("\n=== Contoh query ID (train) ===")
    for e in examples:
        print(e["no"], "OK " if e["benar_top1"] else "GAGAL", f"rank={e['rank_benar']}", "|", e["query_id"][:80])
    print("\nwaktu (s):", {k: round(v, 1) for k, v in timing.items()})


if __name__ == "__main__":
    main()
