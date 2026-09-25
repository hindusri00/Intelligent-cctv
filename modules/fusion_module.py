"""
fusion_module.py
-----------------
This is the piece your project's second objective is actually about:
"improve person identification reliability by combining multiple sources
of visual and temporal information."

Before this module, identity decisions were made by ONE signal at a time:
- normal mode: whichever body Re-ID embedding was most similar (OSNet only)
- target-lock mode: face OR body, whichever cleared its own threshold first

IdentityResolver instead scores every candidate identity using:
  1. Face similarity   (when a face was visible)      - strongest signal
  2. Body similarity    (OSNet appearance)              - always-available fallback
  3. Attribute agreement (shirt/pant/hair/shoe/mask/...) - a tiebreaker, not primary evidence
  4. Temporal continuity (hysteresis on the same track)  - resists frame-to-frame flicker

IMPORTANT: this module does not extract any features itself. It only
*decides*, given embeddings/attributes handed to it by face_module.py,
reid_module.py and attribute_module.py, which identity they belong to.
Keeping extraction and decision-making separate is what lets you swap or
retrain any one signal later without touching this file.
"""

import time


class IdentityResolver:
    # How much each modality contributes to the combined embedding score.
    # These are *relative* weights: if only one modality is available for a
    # given comparison, its weight is renormalized to 1.0 rather than the
    # comparison being penalized for the missing modality.
    FACE_WEIGHT = 0.55
    BODY_WEIGHT = 0.45

    # Attribute agreement is blended IN ON TOP of the embedding score, not
    # treated as an equal-weight third signal - two strangers can easily
    # wear the same color shirt, so attributes should nudge a close call,
    # not decide it outright.
    ATTRIBUTE_WEIGHT = 0.15

    # A combined score below this is not a match - register a new identity.
    MATCH_THRESHOLD = 0.55

    # Don't even bother pulling a candidate's attribute profile from the DB
    # if its embedding score isn't remotely close - keeps the per-frame
    # query cost bounded regardless of gallery size.
    CANDIDATE_FLOOR = 0.25
    TOP_K_FOR_ATTRIBUTES = 3

    # Temporal hysteresis: once a track has been assigned an identity, a
    # DIFFERENT identity must beat it by more than this margin to take over.
    # Without this, a track can flicker between two similar-looking gallery
    # entries frame to frame purely from embedding noise.
    HYSTERESIS_MARGIN = 0.06
    STICKY_WINDOW_SECONDS = 4.0

    ATTRIBUTE_FIELDS = [
        "shirt_color", "pant_color", "hair_color", "shoe_color",
        "mask", "spectacles", "accessory",
    ]

    def __init__(self, db, face_recognizer, reid):
        self.db = db
        self.face = face_recognizer
        self.reid = reid
        # track_id -> {"person_id": str, "last_update": float}
        self.track_state = {}

    # ------------------------------------------------------------------
    # Reusable pairwise scoring (also used directly for target-lock, where
    # there's no gallery to scan - just one known target to compare against)
    # ------------------------------------------------------------------

    def score_pair(self, face_a, body_a, face_b, body_b):
        """
        Combine a face-similarity and a body-similarity comparison into one
        score, redistributing weight to whichever modality is actually
        available on BOTH sides of the comparison.

        Returns (combined_score | None, face_similarity | None, body_similarity | None).
        combined_score is None only when neither modality could be compared.
        """
        face_sim = None
        if face_a is not None and face_b is not None:
            face_sim = self.face.calculate_similarity(face_a, face_b)

        body_sim = None
        if body_a is not None and body_b is not None:
            body_sim = self.reid.calculate_similarity(body_a, body_b)

        if face_sim is None and body_sim is None:
            return None, None, None

        weight_total = 0.0
        score = 0.0
        if face_sim is not None:
            score += face_sim * self.FACE_WEIGHT
            weight_total += self.FACE_WEIGHT
        if body_sim is not None:
            score += body_sim * self.BODY_WEIGHT
            weight_total += self.BODY_WEIGHT

        return score / weight_total, face_sim, body_sim

    def _attribute_score(self, attributes, person_id):
        """
        Fraction of comparable attribute fields that agree between the
        current observation and this identity's typical (most-common-value)
        profile. Returns None when there isn't enough data to compare
        (brand-new identity, or every field is "Unknown" on one side).
        """
        if not attributes:
            return None
        profile = self.db.get_identity_attribute_profile(person_id)
        if not profile:
            return None

        matched, comparable = 0, 0
        for field in self.ATTRIBUTE_FIELDS:
            current = attributes.get(field)
            typical = profile.get(field)
            if current in (None, "Unknown", "None") or typical in (None, "Unknown", "None"):
                continue
            comparable += 1
            if str(current).lower() == str(typical).lower():
                matched += 1

        return None if comparable == 0 else matched / comparable

    # ------------------------------------------------------------------
    # Open-set resolution against the whole identity gallery
    # ------------------------------------------------------------------

    def resolve(self, track_id, face_embedding, body_embedding, attributes=None):
        """
        Decide which known identity (or a brand-new one) this track's
        current face/body embeddings belong to.

        Returns:
            {
                "person_id": str,
                "confidence": float,              # final fused score, 0-1
                "face_similarity": float | None,  # sub-score, for the UI breakdown
                "body_similarity": float | None,
                "attribute_score": float | None,
                "is_new": bool,
                "sticky": bool,   # True if temporal hysteresis overrode a higher-scoring candidate
            }
        """
        if face_embedding is None and body_embedding is None:
            return self._unknown_result()

        candidates = self.db.get_all_identity_embeddings()

        shortlist = []
        for cand in candidates:
            combined, face_sim, body_sim = self.score_pair(
                face_embedding, body_embedding,
                cand["face_embedding"], cand["body_embedding"],
            )
            if combined is None or combined < self.CANDIDATE_FLOOR:
                continue
            shortlist.append({
                "person_id": cand["person_id"],
                "embedding_score": combined,
                "face_similarity": face_sim,
                "body_similarity": body_sim,
            })

        shortlist.sort(key=lambda c: c["embedding_score"], reverse=True)
        top = shortlist[: self.TOP_K_FOR_ATTRIBUTES]

        refined = []
        for cand in top:
            attr_score = self._attribute_score(attributes, cand["person_id"])
            final = cand["embedding_score"] if attr_score is None else (
                (1 - self.ATTRIBUTE_WEIGHT) * cand["embedding_score"]
                + self.ATTRIBUTE_WEIGHT * attr_score
            )
            refined.append({**cand, "attribute_score": attr_score, "final_score": final})

        refined.sort(key=lambda c: c["final_score"], reverse=True)
        best = refined[0] if refined else None

        sticky = False
        best = self._apply_hysteresis(track_id, best, refined)
        if best is not None and best.get("_stuck"):
            sticky = True

        if best is None or best["final_score"] < self.MATCH_THRESHOLD:
            person_id = self._next_person_id()
            self.db.upsert_identity(person_id, face_embedding=face_embedding, body_embedding=body_embedding)
            self.track_state[track_id] = {"person_id": person_id, "last_update": time.time()}
            return {
                "person_id": person_id,
                "confidence": 1.0,
                "face_similarity": None,
                "body_similarity": None,
                "attribute_score": None,
                "is_new": True,
                "sticky": False,
            }

        self.track_state[track_id] = {"person_id": best["person_id"], "last_update": time.time()}
        self.db.upsert_identity(best["person_id"], face_embedding=face_embedding, body_embedding=body_embedding)

        return {
            "person_id": best["person_id"],
            "confidence": round(float(best["final_score"]), 3),
            "face_similarity": _round_or_none(best["face_similarity"]),
            "body_similarity": _round_or_none(best["body_similarity"]),
            "attribute_score": _round_or_none(best["attribute_score"]),
            "is_new": False,
            "sticky": sticky,
        }

    def _apply_hysteresis(self, track_id, best, refined):
        prev = self.track_state.get(track_id)
        if not prev or (time.time() - prev["last_update"]) > self.STICKY_WINDOW_SECONDS:
            return best
        if best is None or best["person_id"] == prev["person_id"]:
            return best

        prev_entry = next((c for c in refined if c["person_id"] == prev["person_id"]), None)
        if prev_entry is None:
            return best

        if best["final_score"] - prev_entry["final_score"] < self.HYSTERESIS_MARGIN:
            prev_entry = dict(prev_entry)
            prev_entry["_stuck"] = True
            return prev_entry

        return best

    def _next_person_id(self):
        n = self.db.count_identities(exclude_prefixes=("TARGET", "Target"))
        return f"Person_{n + 1:03d}"

    @staticmethod
    def _unknown_result():
        return {
            "person_id": "Unknown", "confidence": 0.0,
            "face_similarity": None, "body_similarity": None,
            "attribute_score": None, "is_new": False, "sticky": False,
        }


def _round_or_none(value):
    return None if value is None else round(float(value), 3)
