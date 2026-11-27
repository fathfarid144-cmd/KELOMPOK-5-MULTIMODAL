"""Eksekusi hasil utama M3 Peran 4 dengan konfigurasi V1 M2 yang telah dibekukan.

Sumber data: SEA-VL Indonesian subset, Cahyawijaya et al. (2025), CC BY-SA 4.0.
Tidak membaca split test dan tidak melakukan pemilihan model/templat baru.
"""
from __future__ import annotations

import json
import platform
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).parent
MANIFEST_PATH = ROOT / "M3_Peran4_Protokol_Kasus_Sulit_T05.json"
ZIP_PATH = Path(r"C:\Users\Pongo\Downloads\images-20261112T145641Z-1-001.zip")
OUT_PATH = ROOT / "M3_Peran4_Hasil_Uji_Kasus_Sulit_T05.json"

MODEL_IMG = "sentence-transformers/clip-ViT-B-32"
MODEL_TXT_ID = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
TEMPLATES_V1 = ["{}", "ini adalah foto {}"]
SOFTMAX_SCALE = 100.0


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def apply_template(template: str, text: str) -> str:
    if template != "{}" and text and text[0].isupper() and not text[:2].isupper():
        text = text[0].lower() + text[1:]
    return template.format(text)


def encode_v1(encoder: SentenceTransformer, texts: list[str]) -> np.ndarray:
    per_template = [
        np.asarray(encoder.encode([apply_template(t, text) for text in texts],
                                  batch_size=64, normalize_embeddings=True))
        for t in TEMPLATES_V1
    ]
    return l2norm(np.mean(per_template, axis=0))


def softmax_pair(query: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cosine = candidates @ query
    z = SOFTMAX_SCALE * cosine
    z = z - z.max()
    exp = np.exp(z)
    return exp / exp.sum(), cosine


def load_images(required_ids: set[str]) -> dict[str, Image.Image]:
    loaded: dict[str, Image.Image] = {}
    with ZipFile(ZIP_PATH) as zf:
        for image_id in sorted(required_ids):
            with Image.open(BytesIO(zf.read(image_id))) as image:
                loaded[image_id] = image.convert("RGB")
    return loaded


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    rows = [row for case in manifest["kasus"] for row in case["query"]]
    image_ids = {row[key] for row in rows for key in ("benar", "pengecoh")}

    print("Memuat encoder gambar dan encoder teks Indonesia V1 …", flush=True)
    encoder_img = SentenceTransformer(MODEL_IMG, device="cpu")
    encoder_txt = SentenceTransformer(MODEL_TXT_ID, device="cpu")

    images = load_images(image_ids)
    ordered_ids = sorted(images)
    image_emb = np.asarray(encoder_img.encode([images[i] for i in ordered_ids],
                                               batch_size=16, normalize_embeddings=True))
    emb_by_id = dict(zip(ordered_ids, image_emb, strict=True))
    query_emb = encode_v1(encoder_txt, [row["teks"] for row in rows])

    result_rows = []
    for row, q_emb in zip(rows, query_emb, strict=True):
        candidate_ids = [row["benar"], row["pengecoh"]]
        probs, cos = softmax_pair(q_emb, np.vstack([emb_by_id[i] for i in candidate_ids]))
        result_rows.append({
            "id": row["id"],
            "query": row["teks"],
            "correct_image": row["benar"],
            "distractor_image": row["pengecoh"],
            "cosine_correct": round(float(cos[0]), 8),
            "cosine_distractor": round(float(cos[1]), 8),
            "p_correct": round(float(probs[0]), 8),
            "p_distractor": round(float(probs[1]), 8),
            "pass": bool(probs[0] > probs[1]),
        })
        print(f"{row['id']}: P benar={probs[0]:.4f}, P pengecoh={probs[1]:.4f}", flush=True)

    output = {
        "title": "Hasil utama M3 Peran 4 — Uji Kasus Sulit",
        "status": "Hasil eksekusi V1, bukan angka simulasi.",
        "source_data_attribution": "SEA-VL Indonesian subset, Cahyawijaya et al. (2025), CC BY-SA 4.0.",
        "source_manifest": MANIFEST_PATH.name,
        "system": {
            "variant": "V1 ensemble ID (greedy, UTAMA)",
            "image_encoder": MODEL_IMG,
            "text_encoder": MODEL_TXT_ID,
            "templates": TEMPLATES_V1,
            "text_aggregation": "mean then L2 normalization",
            "candidate_probability": "softmax(100 * cosine) over correct and distractor",
            "pass_rule": "p_correct > p_distractor",
        },
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(), "device": "CPU"},
        "results": result_rows,
    }
    OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Disimpan: {OUT_PATH}")


if __name__ == "__main__":
    main()
