import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from modules.attribute_module import AttributeAnalyzer
from modules.reid_module import PersonReIdentifier
from modules.face_module import FaceRecognizer
from modules.fusion_module import IdentityResolver
import time


class RealTimeTracker:
    def __init__(self, db, model_path="yolo11n.pt"):
        """
        db: a modules.database.Database instance. Required now - the
        identity gallery lives in SQLite (see database.py), and the fusion
        layer (fusion_module.IdentityResolver) needs it to score candidates
        and to persist embeddings for newly-registered identities.
        """
        # Auto-detect: use GPU if available, otherwise fall back to CPU.
        # Hardcoding device=0 / half=True breaks on any CPU-only machine,
        # since FP16 ("half") inference is a CUDA-only feature.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_half = self.device == "cuda"
        print(f"[Tracker] Running on device: {self.device}")

        self.db = db
        self.model = YOLO(model_path)
        self.attribute_analyzer = AttributeAnalyzer()
        self.reid = PersonReIdentifier()
        self.face = FaceRecognizer(device=self.device)
        self.resolver = IdentityResolver(db=self.db, face_recognizer=self.face, reid=self.reid)

        self.frame_count = 0
        self.reid_cache = {}
        self.attribute_cache = {}
        self.face_cache = {}
        self.target_embedding = None
        self.target_face_embedding = None
        # Fused face+body confidence, not a raw OSNet-only similarity, so
        # this threshold sits in the same 0-1 range as the resolver's
        # MATCH_THRESHOLD rather than needing its own separate calibration.
        self.target_similarity_threshold = 0.55
        self.target_cache = {}
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.2,
            embedder="mobilenet",
            half=self.use_half
        )

    def set_target(self, embedding, face_embedding=None):
        """
        face_embedding is optional: pass it whenever a usable face was
        found in the reference photo (see app.py's /api/target handler).
        Matching is now a single fused score (see fusion_module.score_pair)
        rather than "face OR body" - a track with only a weak body match
        and no face at all will correctly score low instead of being
        accepted on a lucky partial signal.
        """
        self.target_embedding = embedding
        self.target_face_embedding = face_embedding
        self.target_cache = {}
        print("[Target] Target person set successfully")

    def process_frame(self, frame):
        start_time = time.time()
        self.frame_count += 1

        results = self.model(
            frame,
            verbose=False,
            device=self.device,
            half=self.use_half,
            imgsz=640
        )
        detections = []
        carried_accessories = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id == 0 and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], conf, 'person'))
                elif cls_id in [24, 26] and conf > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    carried_accessories.append((x1, y1, x2, y2, self.model.names[cls_id]))

        tracks = self.tracker.update_tracks(detections, frame=frame)
        active_targets = []

        for track in tracks:
            print(f"[TRACK STATE] id={track.track_id}, confirmed={track.is_confirmed()}, age={track.age}, hits={track.hits}", flush=True)
            if not track.is_confirmed():
                print(f"[UNCONFIRMED TRACK] {track.track_id}", flush=True)
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            person_crop = frame[y1:y2, x1:x2]
            bbox_height = y2 - y1

            # --- Face embedding (every 10 frames - detection+recognition
            # together cost more than body Re-ID alone, and a face doesn't
            # change between frames as fast as pose/lighting does) ---
            if track_id not in self.face_cache or self.frame_count % 10 == 0:
                head_region = FaceRecognizer.crop_head_region(person_crop)
                face_embedding = self.face.extract_embedding(head_region)
                self.face_cache[track_id] = face_embedding
            face_embedding = self.face_cache[track_id]

            # --- Attributes (every 10 frames). Computed BEFORE identity
            # resolution now, because the fusion layer uses the current
            # attribute snapshot as a tiebreaker between close candidates. ---
            if track_id not in self.attribute_cache or self.frame_count % 10 == 0:
                self.attribute_cache[track_id] = (
                    self.attribute_analyzer.extract_all_attributes(
                        person_crop,
                        bbox_height
                    )
                )
            attr = self.attribute_cache[track_id]
            print(f"[TARGET EMBEDDINGS] body={self.target_embedding is not None}, face={self.target_face_embedding is not None}",flush=True)
            # --- Identity resolution ---
            if self.target_embedding is not None or self.target_face_embedding is not None:
                # Target-lock mode: score this track against the ONE known
                # target using the same fused face+body scoring the open-set
                # resolver uses, rather than a separate ad-hoc rule.
                print(f"[TRACK CHECK] Track {track_id} frame {self.frame_count}", flush=True)
                if track_id not in self.target_cache or self.frame_count % 15 == 0:
                    print(f"[TARGET CHECK] Track {track_id} frame {self.frame_count}", flush=True)
                    body_embedding = self.reid.extract_embedding(person_crop)
                    combined, face_sim, body_sim = self.resolver.score_pair(
                        face_embedding, body_embedding,
                        self.target_face_embedding, self.target_embedding,
                    )
                    self.target_cache[track_id] = {
                        "is_target": (
                            combined is not None
                            and combined >= self.target_similarity_threshold
                            and (
                                self.target_face_embedding is None
                                or face_sim is not None
                            )
                        ),
                        "confidence": combined or 0.0,
                        "face_similarity": face_sim,
                        "body_similarity": body_sim,
                    }
                    print(f"[TARGET SCORE] Track {track_id}: combined={combined:.3f} face={face_sim} body={body_sim}", flush=True)
                target_result = self.target_cache[track_id]

                # Ignore everyone who does not match the target
                if not target_result["is_target"]:
                    continue

                person_id = "TARGET"
                confidence = target_result["confidence"]
                face_similarity = target_result["face_similarity"]
                body_similarity = target_result["body_similarity"]
                attribute_score = None
                is_new = False

            else:
                # Open-set mode: fuse face + body + attributes + temporal
                # continuity into one decision (see fusion_module.py).
                if track_id not in self.reid_cache or self.frame_count % 5 == 0:
                    body_embedding = self.reid.extract_embedding(person_crop)
                    self.reid_cache[track_id] = self.resolver.resolve(
                        track_id=track_id,
                        face_embedding=face_embedding,
                        body_embedding=body_embedding,
                        attributes=attr,
                    )

                resolution = self.reid_cache[track_id]
                person_id = resolution["person_id"]
                confidence = resolution["confidence"]
                face_similarity = resolution["face_similarity"]
                body_similarity = resolution["body_similarity"]
                attribute_score = resolution["attribute_score"]
                is_new = resolution["is_new"]

            has_bag = "None"
            for ax1, ay1, ax2, ay2, item_type in carried_accessories:
                if not (x2 < ax1 or x1 > ax2 or y2 < ay1 or y1 > ay2):
                    has_bag = item_type
                    break

            # Build Dynamic Tag List (Shows Active Detections Only)
            detected_tags = [
                f"Track #{track_id}",
                f"Person ID: {person_id}",
            ]

            if not is_new:
                detected_tags.append(f"Conf: {confidence:.0%}")
                breakdown = []
                if face_similarity is not None:
                    breakdown.append(f"F {face_similarity:.0%}")
                if body_similarity is not None:
                    breakdown.append(f"B {body_similarity:.0%}")
                if attribute_score is not None:
                    breakdown.append(f"A {attribute_score:.0%}")
                if breakdown:
                    detected_tags.append("(" + " ".join(breakdown) + ")")
            if face_embedding is not None:
                detected_tags.append("Face: detected")

            if attr["estimated_height"] != "Unknown":
                detected_tags.append(f"Ht: {attr['estimated_height']}")
            if attr["hair_color"] != "Unknown":
                detected_tags.append(f"Hair: {attr['hair_color']}")
            if attr["shirt_color"] != "Unknown":
                detected_tags.append(f"Shirt: {attr['shirt_color']}")
            if attr["pant_color"] != "Unknown":
                detected_tags.append(f"Pant: {attr['pant_color']}")
            if attr["shoe_color"] != "Unknown":
                detected_tags.append(f"Shoes: {attr['shoe_color']}")

            if attr["mask"] == "Yes":
                detected_tags.append(f"Mask: {attr['mask_color']}")
            if attr["spectacles"] == "Yes":
                detected_tags.append("Glasses")
            if attr["wristband"] == "Yes":
                detected_tags.append("Wristband")
            if attr["tattoo"] == "Yes":
                detected_tags.append("Tattoo")
            if has_bag != "None":
                detected_tags.append(f"Bag: {has_bag}")

            active_targets.append({
                "track_id": track_id,
                "person_id": person_id,
                "confidence": confidence,
                "face_similarity": face_similarity,
                "body_similarity": body_similarity,
                "attribute_score": attribute_score,
                "is_new": is_new,
                "has_face": face_embedding is not None,
                "face_embedding": face_embedding,   # consumed by app.py for DB logging; strip before JSON-serializing
                "bbox": [x1, y1, x2, y2],
                **attr,
                "accessory": has_bag
            })

            # Scale box/text thickness to the frame's actual resolution.
            # A fixed 2px line / 0.45 font-scale is fine on a 720p frame but
            # becomes sub-pixel and invisible once a 4K frame is downscaled
            # for display, even though detection/tracking is working fine.
            scale = frame.shape[1] / 1280
            box_thickness = max(2, round(3 * scale))
            font_scale = max(0.5, 0.6 * scale)
            text_thickness = max(1, round(2 * scale))

            box_color = (0, 200, 255) if person_id == "TARGET" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

            label_text = " | ".join(detected_tags)
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
            )
            label_y = max(text_h + 10, y1 - 10)

            # Filled backing box so the label stays legible over busy backgrounds
            cv2.rectangle(
                frame,
                (x1, label_y - text_h - baseline - 4),
                (x1 + text_w + 8, label_y + baseline - 4),
                (0, 100, 0),
                -1
            )
            cv2.putText(
                frame, label_text, (x1 + 4, label_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness
            )
        total_time = time.time() - start_time
        print(f"[PERF] Frame: {total_time:.3f}s")
        return frame, active_targets
