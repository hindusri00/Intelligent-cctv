"""
evaluate_identity_methods.py
-----------------------------
Benchmark harness for objective #10: compares face-only, body-Re-ID-only,
and the fused method (modules/fusion_module.py) on identity-matching
accuracy, so your report has a real precision/recall comparison instead of
just a demo.

WHAT YOU NEED TO PROVIDE
-------------------------
A "pair manifest" CSV with one row per comparison you want scored:

    image_a,image_b,same_identity
    data/crowdhuman/crop_0001.jpg,data/crowdhuman/crop_0002.jpg,1
    data/crowdhuman/crop_0001.jpg,data/crowdhuman/crop_0099.jpg,0
    ...

- image_a / image_b: paths to two person crops (ideally containing a
  visible face for at least some pairs, so the face-only method has
  something to score - CCTV crops where no face is visible will simply
  score that pair as "no decision" for the face-only method, which is
  itself a useful thing to report on).
- same_identity: 1 if both crops are the same person, 0 otherwise.

This script does NOT ship with CrowdHuman itself (it's a detection dataset,
not an identity-matching one - you'll need to derive pairs from tracks in
your own annotated footage, or from a re-ID dataset such as Market-1501 /
MSMT17 if your report allows using a public re-ID benchmark instead).

WHAT IT MEASURES
-----------------
For each method (face-only, body-only, fused), and a sweep of thresholds:
  - precision, recall, F1 at each threshold
  - the best-F1 threshold and its confusion counts
  - AUC-ish separation: mean similarity for genuine vs impostor pairs

The "fused" method reuses IdentityResolver.score_pair(), so it is exactly
the same weighting/logic your live system uses - not a re-implementation
that could drift from production behaviour.

USAGE
------
    python evaluate_identity_methods.py pairs.csv --out results.csv
"""

import argparse
import csv
import sys

import cv2
import numpy as np

from modules.face_module import FaceRecognizer
from modules.reid_module import PersonReIdentifier
from modules.fusion_module import IdentityResolver


THRESHOLDS = np.arange(0.05, 0.96, 0.05)


def load_pairs(manifest_path):
    pairs = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            pairs.append((row["image_a"], row["image_b"], int(row["same_identity"])))
    return pairs


def extract_features(path, face_model, body_model):
    image = cv2.imread(path)
    if image is None:
        print(f"  [warn] could not read {path}", file=sys.stderr)
        return None, None
    face_emb = face_model.extract_embedding(image)
    body_emb = body_model.extract_embedding(image)
    return face_emb, body_emb


def score_pairs(pairs, face_model, body_model, resolver):
    """
    Returns three lists of (similarity, label) - one per method - skipping
    a pair for a given method only when that method genuinely has nothing
    to compare (e.g. no face detected on either side).
    """
    face_scores, body_scores, fused_scores = [], [], []
    cache = {}

    for path_a, path_b, label in pairs:
        if path_a not in cache:
            cache[path_a] = extract_features(path_a, face_model, body_model)
        if path_b not in cache:
            cache[path_b] = extract_features(path_b, face_model, body_model)

        face_a, body_a = cache[path_a]
        face_b, body_b = cache[path_b]

        if face_a is not None and face_b is not None:
            face_scores.append((face_model.calculate_similarity(face_a, face_b), label))
        if body_a is not None and body_b is not None:
            body_scores.append((body_model.calculate_similarity(body_a, body_b), label))

        fused, _, _ = resolver.score_pair(face_a, body_a, face_b, body_b)
        if fused is not None:
            fused_scores.append((fused, label))

    return face_scores, body_scores, fused_scores


def evaluate_at_threshold(scored, threshold):
    tp = fp = tn = fn = 0
    for score, label in scored:
        predicted_same = score >= threshold
        if predicted_same and label == 1:
            tp += 1
        elif predicted_same and label == 0:
            fp += 1
        elif not predicted_same and label == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"threshold": round(float(threshold), 2), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
             "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def summarize(name, scored):
    if not scored:
        print(f"\n{name}: no comparable pairs (feature missing on one side for every pair)")
        return None

    genuine = [s for s, l in scored if l == 1]
    impostor = [s for s, l in scored if l == 0]
    print(f"\n{name}  (n={len(scored)}, genuine={len(genuine)}, impostor={len(impostor)})")
    if genuine:
        print(f"  mean genuine similarity : {np.mean(genuine):.4f}")
    if impostor:
        print(f"  mean impostor similarity: {np.mean(impostor):.4f}")

    rows = [evaluate_at_threshold(scored, t) for t in THRESHOLDS]
    best = max(rows, key=lambda r: r["f1"])
    print(f"  best F1 = {best['f1']:.4f} at threshold {best['threshold']} "
          f"(precision {best['precision']:.4f}, recall {best['recall']:.4f})")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help="CSV of image_a,image_b,same_identity")
    parser.add_argument("--out", default="eval_results.csv", help="Where to write the full per-threshold table")
    args = parser.parse_args()

    pairs = load_pairs(args.manifest)
    print(f"Loaded {len(pairs)} labeled pairs from {args.manifest}")

    face_model = FaceRecognizer()
    body_model = PersonReIdentifier()
    resolver = IdentityResolver(db=None, face_recognizer=face_model, reid=body_model)

    face_scores, body_scores, fused_scores = score_pairs(pairs, face_model, body_model, resolver)

    all_rows = []
    for name, scored in [("Face-only", face_scores), ("Body Re-ID-only", body_scores), ("Fused (face+body)", fused_scores)]:
        rows = summarize(name, scored)
        if rows:
            for r in rows:
                all_rows.append({"method": name, **r})

    if all_rows:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nFull per-threshold results written to {args.out}")


if __name__ == "__main__":
    main()
