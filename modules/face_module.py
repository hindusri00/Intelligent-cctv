"""
face_module.py
---------------
Facial identity extraction, mirroring the role modules/reid_module.py plays
for body appearance - but for faces.

Design choice: this module does NOT keep its own identity gallery (unlike
PersonReIdentifier, which registers "Person_001", "Person_002", ...). Faces
and bodies are two *signals* about the same underlying person, and if each
kept a separate numbering scheme you'd end up with two disconnected identity
spaces (Person_003 from the body Re-ID gallery vs. Face_007 from a face
gallery) with no guaranteed link between them. Instead, FaceRecognizer only
detects faces and extracts/compares embeddings; a single identity manager
(added in the SQL/persistence layer) owns the actual person_id -> {face
embedding, body embedding} mapping and does the fusion.

Uses InsightFace's buffalo_l model pack, which bundles a RetinaFace-based
detector with an ArcFace-style recognition model in one pass - convenient
for CCTV crops where you don't already know a face is present.

Install:
    pip install insightface onnxruntime          # CPU
    pip install insightface onnxruntime-gpu       # GPU (CUDA)
"""

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "insightface is required for face_module.py. "
        "Install with: pip install insightface onnxruntime"
    ) from e


class FaceRecognizer:
    def __init__(
        self,
        model_name="buffalo_l",
        det_size=(640, 640),
        device=None,
        similarity_threshold=0.45,
        min_det_score=0.55,
    ):
        """
        model_name: InsightFace model pack. buffalo_l is the standard
            accurate pack (detector + recognizer together).
        det_size: detector input resolution. Larger = better on small/far
            faces (typical CCTV), slower. 640x640 is a reasonable default;
            drop to (320, 320) if you need more speed than accuracy.
        similarity_threshold: cosine similarity above which two face
            embeddings are considered the same person. ArcFace-style
            embeddings typically need ~0.35-0.5 depending on pose/lighting;
            0.45 is a conservative middle ground - tune against your own
            footage rather than trusting this blindly.
        min_det_score: discard detections below this confidence. CCTV faces
            are frequently partial/blurry/angled; a low threshold here lets
            garbage detections corrupt the embedding gallery.
        """
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cpu":
            providers = ["CPUExecutionProvider"]

        self.app = FaceAnalysis(name=model_name, providers=providers)
        # ctx_id=0 uses GPU if a CUDA provider loaded successfully, falls
        # back to CPU automatically otherwise.
        self.app.prepare(ctx_id=0, det_size=det_size)

        self.similarity_threshold = similarity_threshold
        self.min_det_score = min_det_score

        print(f"[Face] InsightFace ({model_name}) initialized")

    def detect_faces(self, image_bgr):
        """
        Run detection + embedding extraction on a BGR image (a full frame
        OR a person crop both work - InsightFace does its own detection).

        Returns a list of dicts, each:
            {
                "bbox": [x1, y1, x2, y2],   # coords in image_bgr's space
                "embedding": np.ndarray,     # 512-d, L2-normalized
                "det_score": float,
                "landmark": np.ndarray | None,
                "pose": (yaw, pitch, roll) | None
            }
        Empty list if no usable face found.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        faces = self.app.get(image_bgr)
        results = []
        for f in faces:
            if f.det_score < self.min_det_score:
                continue
            embedding = f.normed_embedding  # already L2-normalized by insightface
            results.append({
                "bbox": [int(v) for v in f.bbox],
                "embedding": embedding,
                "det_score": float(f.det_score),
                "landmark": getattr(f, "kps", None),
                "pose": getattr(f, "pose", None),
            })
        return results

    def extract_embedding(self, person_crop):
        """
        Convenience wrapper for the common case: given a person (or head)
        crop, return the embedding of the largest/most confident face found,
        or None if no usable face is visible.

        CCTV person crops are frequently side-on, occluded, or too low-res
        for a usable face - returning None here (rather than a garbage
        embedding) is the expected common case, not an error. Callers
        should fall back to body Re-ID / attributes when this is None.
        """
        faces = self.detect_faces(person_crop)
        if not faces:
            return None

        # Prefer the highest-confidence face if several were found
        # (e.g. a crop that accidentally includes a second person).
        best = max(faces, key=lambda f: f["det_score"])
        return best["embedding"]

    def calculate_similarity(self, embedding_a, embedding_b):
        """Cosine similarity between two face embeddings, in [-1, 1]."""
        if embedding_a is None or embedding_b is None:
            return 0.0

        a = np.asarray(embedding_a, dtype=np.float32)
        b = np.asarray(embedding_b, dtype=np.float32)

        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    def is_match(self, embedding_a, embedding_b):
        """True if two embeddings are similar enough to call the same person."""
        return self.calculate_similarity(embedding_a, embedding_b) >= self.similarity_threshold

    @staticmethod
    def crop_head_region(person_crop, top_fraction=0.30):
        """
        Optional helper: restrict face detection to the top of a person
        crop (where the head is), which is faster and cuts false positives
        from patterned clothing when you already have a person bbox from
        YOLO/DeepSort and just want to check for a face without paying full
        image-detection cost every frame.
        """
        if person_crop is None or person_crop.size == 0:
            return person_crop
        h = person_crop.shape[0]
        return person_crop[0:int(h * top_fraction), :]
