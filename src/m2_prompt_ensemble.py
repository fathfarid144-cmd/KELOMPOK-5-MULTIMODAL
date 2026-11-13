"""
m2_prompt_ensemble.py  --  Peran 3 (Eksperimen Penyejajaran/Fusi), pendekatan (b)
Prompt ensemble berbahasa Indonesia di atas encoder beku. Tidak ada parameter yang dilatih.

Data: SEA-VL Indonesian subset (KORIKA-AI/sea-vl_crowdsourcing_id), CC BY-SA 4.0.
Cahyawijaya et al. (2025), "Crowdsource, Crawl, or Generate? Creating SEA-VL", ACL.

Antarmuka dengan Peran 2 (cache embedding .npy, sudah dinormalisasi, urutan baris = urutan CSV):
    cache/{split}_img.npy      gambar               (N, d)   clip-ViT-B-32
    cache/{split}_cap_id.npy   caption ID polos     (N, d)   clip-ViT-B-32-multilingual-v1
    cache/{split}_cap_en.npy   caption EN polos     (N, d)   clip-ViT-B-32
Gambar TIDAK perlu di-encode ulang: ensemble hanya mengubah sisi teks.

Aturan leakage (instruksi M1): memilih templat/hiperparameter HANYA boleh memakai train.csv.
`val` hanya dipakai untuk melaporkan angka akhir, satu kali per konfigurasi yang sudah dikunci.

Pemakaian:
    python m2_prompt_ensemble.py select --seed 0     # pilih templat di train (subset), simpan templates_terpilih.json
    python m2_prompt_ensemble.py eval   --seed 0     # laporkan 4 arah di val: baseline vs ensemble
    python m2_prompt_ensemble.py selftest            # uji logika dengan encoder tiruan (tanpa model/data)
"""
import argparse, json, time
import numpy as np

# ---------------------------------------------------------------- templat
# "{}" = caption polos (identik dengan baseline M1). Selalu ikut sebagai kandidat.
TEMPLATES_ID = [
    "{}",
    "foto {}",
    "sebuah foto {}",
    "gambar {}",
    "foto budaya Indonesia: {}",
    "dokumentasi budaya Indonesia: {}",
    "{} di Indonesia",
    "sebuah foto yang menunjukkan {}",
    "potret {}",
    "ini adalah foto {}",
]
# Kontrol: templat EN setara, untuk memisahkan "efek ensemble" dari "efek bahasa".
TEMPLATES_EN = [
    "{}",
    "a photo of {}",
    "a picture of {}",
    "a photo showing {}",
    "a photo of Indonesian culture: {}",
    "documentation of Indonesian culture: {}",
    "{} in Indonesia",
    "this is a photo of {}",
]


# ---------------------------------------------------------------- util
def l2norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def apply_template(tpl, text, lower_first=True):
    if lower_first and tpl != "{}" and text and text[0].isupper() and not text[:2].isupper():
        text = text[0].lower() + text[1:]
    return tpl.format(text)


def encode_ensemble(encoder, texts, templates, batch_size=64, lower_first=True):
    """Rata-rata embedding teks di bawah tiap templat, lalu dinormalisasi ulang. -> (n, d)
    `encoder.encode(list[str], batch_size=..., normalize_embeddings=True)` -> ndarray (API sentence-transformers)."""
    outs = []
    for tpl in templates:
        wrapped = [apply_template(tpl, t, lower_first) for t in texts]
        outs.append(np.asarray(encoder.encode(wrapped, batch_size=batch_size, normalize_embeddings=True)))
    return l2norm(np.mean(outs, axis=0))


def fuse(e_a, e_b, w):
    """Fusi lambat pada level embedding: w*a + (1-w)*b (mis. ID-ensemble + terjemahan EN)."""
    return l2norm(w * e_a + (1.0 - w) * e_b)


# ---------------------------------------------------------------- metrik (cadangan; pakai evaluate() Peran 4 bila sudah ada)
def retrieval_metrics(Q, G, q_group=None, g_group=None, ks=(1, 5, 10), map_k=10):
    """Q: (nq,d) query, G: (ng,d) galeri, keduanya ternormalisasi. Kesamaan kosinus.
    q_group/g_group: id kelompok; item relevan = galeri dengan g_group == q_group[q].
    Default satu-satu (query i <-> galeri i). Mengembalikan R@k dan mAP@map_k (AP@K dirata-rata atas query,
    AP@K = sum_k P(k)*rel(k) / min(#relevan, K))."""
    nq, ng = len(Q), len(G)
    q_group = np.arange(nq) if q_group is None else np.asarray(q_group)
    g_group = np.arange(ng) if g_group is None else np.asarray(g_group)
    order = np.argsort(-(Q @ G.T), axis=1, kind="stable")
    rel = (g_group[order] == q_group[:, None])                       # (nq, ng) bool
    n_rel = (g_group[None, :] == q_group[:, None]).sum(1)
    out = {f"R@{k}": float(rel[:, :k].any(1).mean()) for k in ks}
    topk = rel[:, :map_k].astype(float)
    prec = np.cumsum(topk, 1) / np.arange(1, map_k + 1)
    out[f"mAP@{map_k}"] = float(((prec * topk).sum(1) / np.minimum(n_rel, map_k)).mean())
    return out


def four_directions(E, Ti, Te, **kw):
    return {
        "gambar->teks ID": retrieval_metrics(E, Ti, **kw),
        "gambar->teks EN": retrieval_metrics(E, Te, **kw),
        "teks ID->gambar": retrieval_metrics(Ti, E, **kw),
        "teks EN->gambar": retrieval_metrics(Te, E, **kw),
    }


# ---------------------------------------------------------------- pemilihan templat (HANYA train)
def greedy_select(encoder, texts_train, img_train, candidates, max_k=5, batch_size=64, tol=0.0):
    """Seleksi maju (greedy forward): mulai dari '{}', tambah templat yang paling menaikkan rata-rata
    R@1 dua arah ID pada subset train; berhenti bila tak ada kenaikan > tol atau mencapai max_k.
    Embedding per-templat di-cache sehingga tiap templat hanya di-encode sekali."""
    per = {t: np.asarray(encoder.encode([apply_template(t, x) for x in texts_train],
                                        batch_size=batch_size, normalize_embeddings=True))
           for t in candidates}

    def score(sel):
        Ti = l2norm(np.mean([per[t] for t in sel], axis=0))
        a = retrieval_metrics(Ti, img_train)["R@1"]
        b = retrieval_metrics(img_train, Ti)["R@1"]
        return (a + b) / 2

    chosen, best, trace = ["{}"], None, []
    best = score(chosen)
    trace.append((list(chosen), best))
    while len(chosen) < max_k:
        gains = [(score(chosen + [t]), t) for t in candidates if t not in chosen]
        if not gains:
            break
        s, t = max(gains)
        if s - best <= tol:
            break
        chosen.append(t); best = s; trace.append((list(chosen), s))
    return chosen, trace


# ---------------------------------------------------------------- CLI
def load(split):
    """Baca cache Peran 2: hasil_m1/cache/emb_{split}.npz dengan key img/id/en.
    Dikembalikan dengan nama kunci internal skrip ini (img, cap_id, cap_en)
    supaya sisa kode (select/eval) tidak perlu diubah."""
    z = np.load(f"hasil_m1/cache/emb_{split}.npz")
    return {"img": z["img"], "cap_id": z["id"], "cap_en": z["en"]}


def captions(split):
    import pandas as pd
    df = pd.read_csv(f"{split}.csv")
    return list(df["caption_id"]), list(df["caption_en"])


def get_encoders():
    from sentence_transformers import SentenceTransformer
    return (SentenceTransformer("sentence-transformers/clip-ViT-B-32"),
            SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1"))


def cmd_select(a):
    rng = np.random.default_rng(a.seed)
    cap_id, _ = captions("train")
    tr = load("train")
    idx = np.sort(rng.choice(len(cap_id), size=min(a.n_sub, len(cap_id)), replace=False))
    _, txt_id = get_encoders()
    t0 = time.time()
    chosen, trace = greedy_select(txt_id, [cap_id[i] for i in idx], tr["img"][idx], TEMPLATES_ID, max_k=a.max_k)
    json.dump({"seed": a.seed, "n_sub": int(len(idx)), "split": "train", "chosen": chosen,
               "trace": [{"set": s, "R@1_train_rata2_dua_arah_ID": r} for s, r in trace]},
              open("templates_terpilih.json", "w"), ensure_ascii=False, indent=2)
    print("terpilih:", chosen, f"({time.time()-t0:.0f} dtk)")


def cmd_eval(a):
    chosen = json.load(open("templates_terpilih.json"))["chosen"]
    va, (cap_id, cap_en) = load("val"), captions("val")
    img_enc, txt_id = get_encoders()
    t0 = time.time()
    Ti_ens = encode_ensemble(txt_id, cap_id, chosen)
    Te_ens = encode_ensemble(img_enc, cap_en, TEMPLATES_EN[:len(chosen)])  # kontrol: jumlah templat sama
    res = {"V0 baseline M1": four_directions(va["img"], va["cap_id"], va["cap_en"]),
           "V1 ensemble ID": four_directions(va["img"], Ti_ens, va["cap_en"]),
           "K1 kontrol ensemble EN": four_directions(va["img"], va["cap_id"], Te_ens)}
    res["_meta"] = {"split": "val", "seed": a.seed, "templat_ID": chosen, "waktu_dtk": round(time.time() - t0, 1)}
    json.dump(res, open("hasil_m2_prompt_ensemble.json", "w"), ensure_ascii=False, indent=2)
    for name, r in res.items():
        if name.startswith("_"): continue
        print("==", name)
        for d, m in r.items(): print(f"  {d:18s}", {k: round(v, 4) for k, v in m.items()})


# ---------------------------------------------------------------- uji logika (tanpa model/data)
class _FakeEnc:
    """Encoder tiruan deterministik: kata -> vektor acak tetap; kalimat = jumlah vektor kata.
    HANYA untuk menguji kode, angkanya tidak bermakna."""
    def __init__(self, d=64): self.d = d
    def _w(self, w):
        import zlib
        return np.random.default_rng(zlib.crc32(w.encode())).normal(size=self.d)
    def encode(self, texts, batch_size=0, normalize_embeddings=True):
        return l2norm(np.stack([sum(self._w(w) for w in t.lower().split()) for t in texts]))


def cmd_selftest(_):
    enc, N = _FakeEnc(), 200
    caps = [f"penari {i} memakai kostum {i % 7} di panggung {i % 11}" for i in range(N)]
    img = enc.encode(caps)                                     # gambar = teks polos -> baseline sempurna
    # 1) metrik: identitas harus R@1 = 1, mAP = 1
    m = retrieval_metrics(img, img); assert m["R@1"] == 1.0 and abs(m["mAP@10"] - 1.0) < 1e-9, m
    # 2) metrik: acak ~ 1/N
    rnd = l2norm(np.random.default_rng(0).normal(size=(N, 64)))
    assert retrieval_metrics(rnd, img)["R@1"] < 0.05
    # 3) multi-caption: dua caption per gambar
    G = np.concatenate([img, img]); gg = np.concatenate([np.arange(N), np.arange(N)])
    m2 = retrieval_metrics(img, G, g_group=gg); assert m2["R@1"] == 1.0 and abs(m2["mAP@10"] - 1.0) < 1e-9, m2
    # 4) ensemble: bentuk, norma satu, templat "{}" saja == baseline
    e1 = encode_ensemble(enc, caps, ["{}"]); assert np.allclose(e1, img)
    e2 = encode_ensemble(enc, caps, TEMPLATES_ID); assert e2.shape == img.shape and np.allclose(np.linalg.norm(e2, axis=1), 1)
    # 5) fusi
    f = fuse(e1, e2, 0.5); assert np.allclose(np.linalg.norm(f, axis=1), 1)
    # 6) seleksi greedy berjalan dan selalu memuat '{}'
    ch, tr = greedy_select(enc, caps, img, TEMPLATES_ID, max_k=3)
    assert ch[0] == "{}" and len(ch) <= 3
    print("selftest OK | templat terpilih (encoder tiruan, tidak bermakna):", ch)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["select", "eval", "selftest"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_sub", type=int, default=500, help="ukuran subset train untuk seleksi templat")
    p.add_argument("--max_k", type=int, default=5)
    a = p.parse_args()
    {"select": cmd_select, "eval": cmd_eval, "selftest": cmd_selftest}[a.cmd](a)
