"""
database.py
------------
SQLite persistence for the Sentry system.

Replaces two things that currently live only in process memory and are lost
on every restart:
  1. app.py's `detection_logs` list (per-frame sightings + attributes)
  2. PersonReIdentifier's `self.gallery` dict (identity -> body embedding)

and adds a home for face embeddings, which didn't exist before this module.

Design notes:
  - One identity row per person_id, holding BOTH a face embedding and a body
    embedding (nullable independently - CCTV often gives you only one).
    This is the shared identity space that modules/face_module.py's
    docstring refers to: face_module.py itself stays gallery-free, and this
    is where "the same person_id owns both signals" actually happens.
  - Embeddings are stored as raw float32 BLOBs (np.ndarray.tobytes()),
    not as JSON - much smaller and avoids float-precision round-tripping
    issues.
  - Every detection is stored on its own row rather than only keeping the
    latest sighting, so /api/search and report generation can look back
    over the full history of a person, not just "right now".
  - check_same_thread=False + a short-lived connection per call. Flask's
    dev server serves the MJPEG generator and normal API requests on
    different threads; opening a fresh connection per operation sidesteps
    SQLite's thread-affinity rules without needing a connection pool for
    what is, for a student/demo-scale project, a modest write volume.
    WAL mode is enabled so reads (dashboard polling) don't block writes
    (the video stream logging detections).
"""

import sqlite3
import time
import threading
import numpy as np
from contextlib import contextmanager


SCHEMA = """
CREATE TABLE IF NOT EXISTS identities (
    person_id       TEXT PRIMARY KEY,
    face_embedding  BLOB,
    body_embedding  BLOB,
    observations    INTEGER NOT NULL DEFAULT 0,
    first_seen      REAL NOT NULL,
    last_seen       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id        TEXT NOT NULL,
    track_id         TEXT,
    camera_id        TEXT DEFAULT 'default',
    video_source     TEXT,
    video_timestamp  REAL,          -- seconds into the source video/stream
    created_at       REAL NOT NULL, -- wall-clock time the row was written
    shirt_color      TEXT,
    pant_color       TEXT,
    hair_color       TEXT,
    shoe_color       TEXT,
    mask             TEXT,
    mask_color       TEXT,
    spectacles       TEXT,
    wristband        TEXT,
    tattoo           TEXT,
    accessory        TEXT,
    estimated_height TEXT,
    reid_similarity  REAL,      -- body-similarity sub-score
    face_similarity  REAL,      -- face-similarity sub-score
    attribute_score  REAL,      -- attribute-agreement sub-score (fusion_module.py)
    fused_confidence REAL,      -- final combined confidence (fusion_module.py)
    has_face         INTEGER DEFAULT 0,
    snapshot_path    TEXT,
    FOREIGN KEY (person_id) REFERENCES identities(person_id)
);

CREATE INDEX IF NOT EXISTS idx_detections_person ON detections(person_id);
CREATE INDEX IF NOT EXISTS idx_detections_created ON detections(created_at);

CREATE TABLE IF NOT EXISTS cases (
    case_id          TEXT PRIMARY KEY,
    target_person_id TEXT NOT NULL,
    opened_at        REAL NOT NULL,
    closed_at        REAL,
    status            TEXT NOT NULL DEFAULT 'open',   -- 'open' | 'closed'
    notes             TEXT,
    report_pdf_path   TEXT,
    report_csv_path   TEXT
);

CREATE TABLE IF NOT EXISTS case_sources (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id      TEXT NOT NULL,
    camera_id    TEXT,
    source       TEXT,
    mode         TEXT,          -- 'footage' | 'live'
    started_at   REAL,
    ended_at     REAL,
    FOREIGN KEY (case_id) REFERENCES cases(case_id)
);

CREATE INDEX IF NOT EXISTS idx_case_sources_case ON case_sources(case_id);
"""


def _blob(embedding):
    if embedding is None:
        return None
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _unblob(blob):
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


class Database:
    def __init__(self, db_path="sentry.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn):
        """
        Lightweight forward-migration: add any columns introduced after a
        database file already existed. CREATE TABLE IF NOT EXISTS in SCHEMA
        only helps on a brand-new sentry.db - an existing file from before
        the fusion layer was added won't otherwise pick up the new columns.
        """
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(detections)")}
        new_columns = {
            "attribute_score": "REAL",
            "fused_confidence": "REAL",
        }
        for name, col_type in new_columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE detections ADD COLUMN {name} {col_type}")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Identities
    # ------------------------------------------------------------------

    def upsert_identity(self, person_id, face_embedding=None, body_embedding=None):
        """
        Create the identity if it doesn't exist, else update its embeddings
        via a running average (same averaging behaviour PersonReIdentifier
        used to do in memory) and bump its observation count / last_seen.

        Passing only one of face_embedding/body_embedding leaves the other
        untouched - a track without a visible face this frame shouldn't
        erase a face embedding captured earlier.
        """
        now = time.time()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM identities WHERE person_id = ?", (person_id,)
            ).fetchone()

            if row is None:
                conn.execute(
                    """INSERT INTO identities
                       (person_id, face_embedding, body_embedding, observations, first_seen, last_seen)
                       VALUES (?, ?, ?, 1, ?, ?)""",
                    (person_id, _blob(face_embedding), _blob(body_embedding), now, now),
                )
                return

            observations = row["observations"]
            new_face = self._running_average(row["face_embedding"], face_embedding, observations)
            new_body = self._running_average(row["body_embedding"], body_embedding, observations)

            conn.execute(
                """UPDATE identities
                   SET face_embedding = ?, body_embedding = ?,
                       observations = observations + 1, last_seen = ?
                   WHERE person_id = ?""",
                (_blob(new_face), _blob(new_body), now, person_id),
            )

    @staticmethod
    def _running_average(old_blob, new_embedding, observations):
        if new_embedding is None:
            return _unblob(old_blob)
        new_embedding = np.asarray(new_embedding, dtype=np.float32)
        if old_blob is None:
            return new_embedding
        old = _unblob(old_blob)
        averaged = (old * observations + new_embedding) / (observations + 1)
        norm = np.linalg.norm(averaged)
        return averaged / norm if norm > 1e-12 else averaged

    def get_identity(self, person_id):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM identities WHERE person_id = ?", (person_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "person_id": row["person_id"],
            "face_embedding": _unblob(row["face_embedding"]),
            "body_embedding": _unblob(row["body_embedding"]),
            "observations": row["observations"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
        }

    def get_all_identities(self):
        """Lightweight summary list (no embeddings) for a gallery/roster view."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT person_id, observations, first_seen, last_seen,
                          face_embedding IS NOT NULL AS has_face,
                          body_embedding IS NOT NULL AS has_body
                   FROM identities ORDER BY last_seen DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_identity_embeddings(self):
        """
        Every identity's face/body embeddings (as numpy arrays, or None),
        for fusion_module.IdentityResolver to scan. NOTE: linear scan over
        the whole gallery on every call - fine at student-project scale
        (tens to low hundreds of identities); a large deployment would want
        a vector index (e.g. FAISS) here instead.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT person_id, face_embedding, body_embedding FROM identities"
            ).fetchall()
        return [
            {
                "person_id": row["person_id"],
                "face_embedding": _unblob(row["face_embedding"]),
                "body_embedding": _unblob(row["body_embedding"]),
            }
            for row in rows
        ]

    def get_identity_attribute_profile(self, person_id, limit=20):
        """
        The identity's "typical" attribute values - the most common value
        seen for each field across its most recent sightings. Used by the
        fusion layer as a tiebreaker, and useful on its own for a
        person-summary view ("usually wears a dark shirt").
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT shirt_color, pant_color, hair_color, shoe_color,
                          mask, spectacles, accessory
                   FROM detections WHERE person_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (person_id, limit),
            ).fetchall()

        if not rows:
            return {}

        from collections import Counter
        fields = ["shirt_color", "pant_color", "hair_color", "shoe_color",
                  "mask", "spectacles", "accessory"]
        profile = {}
        for field in fields:
            values = [r[field] for r in rows if r[field] not in (None, "Unknown", "None")]
            if values:
                profile[field] = Counter(values).most_common(1)[0][0]
        return profile

    def count_identities(self, exclude_prefixes=()):
        with self._connect() as conn:
            rows = conn.execute("SELECT person_id FROM identities").fetchall()
        return sum(
            1 for r in rows
            if not any(r["person_id"].startswith(p) for p in exclude_prefixes)
        )

    def find_best_match(self, embedding, kind, threshold, calculate_similarity):
        """
        Compare `embedding` against every stored identity's `kind`
        embedding ('face_embedding' or 'body_embedding') using the caller's
        similarity function (so this module doesn't need to know cosine
        math belongs to FaceRecognizer vs PersonReIdentifier).

        Returns (person_id, similarity) or (None, best_similarity_seen).
        NOTE: for a large gallery this linear scan is the first thing to
        replace with a vector index (e.g. FAISS) - fine at student-project
        scale (tens to low hundreds of identities).
        """
        with self._connect() as conn:
            rows = conn.execute(f"SELECT person_id, {kind} FROM identities").fetchall()

        best_id, best_sim = None, -1.0
        for row in rows:
            candidate = _unblob(row[kind])
            if candidate is None:
                continue
            sim = calculate_similarity(embedding, candidate)
            if sim > best_sim:
                best_sim, best_id = sim, row["person_id"]

        if best_id is not None and best_sim >= threshold:
            return best_id, best_sim
        return None, best_sim

    # ------------------------------------------------------------------
    # Detections
    # ------------------------------------------------------------------

    def log_detection(self, person_id, track_id, attributes, camera_id="default",
                       video_source=None, video_timestamp=None,
                       reid_similarity=None, face_similarity=None,
                       attribute_score=None, fused_confidence=None,
                       has_face=False, snapshot_path=None):
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO detections
                   (person_id, track_id, camera_id, video_source, video_timestamp,
                    created_at, shirt_color, pant_color, hair_color, shoe_color,
                    mask, mask_color, spectacles, wristband, tattoo, accessory,
                    estimated_height, reid_similarity, face_similarity,
                    attribute_score, fused_confidence, has_face, snapshot_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    person_id, track_id, camera_id, video_source, video_timestamp,
                    time.time(),
                    attributes.get("shirt_color"), attributes.get("pant_color"),
                    attributes.get("hair_color"), attributes.get("shoe_color"),
                    attributes.get("mask"), attributes.get("mask_color"),
                    attributes.get("spectacles"), attributes.get("wristband"),
                    attributes.get("tattoo"), attributes.get("accessory"),
                    attributes.get("estimated_height"),
                    reid_similarity, face_similarity, attribute_score,
                    fused_confidence, int(has_face), snapshot_path,
                ),
            )

    def search_detections(self, attribute=None, value=None, limit=500):
        """
        attribute: one of shirt_color/pant_color/hair_color/mask/accessory/
            spectacles/tattoo, or None to match across all of them.
        value: substring to match (case-insensitive), applied the same way
            app.py's /api/search endpoint already tokenizes queries.
        Returns one row per distinct person_id (most recent sighting).
        """
        with self._connect() as conn:
            if attribute:
                query = f"""
                    SELECT * FROM detections
                    WHERE lower({attribute}) LIKE ?
                    ORDER BY created_at DESC LIMIT ?
                """
                rows = conn.execute(query, (f"%{value.lower()}%", limit)).fetchall()
            else:
                query = """
                    SELECT * FROM detections
                    WHERE lower(shirt_color || ' ' || pant_color || ' ' ||
                                hair_color || ' ' || mask || ' ' || accessory) LIKE ?
                    ORDER BY created_at DESC LIMIT ?
                """
                rows = conn.execute(query, (f"%{value.lower()}%", limit)).fetchall()

        seen, unique = set(), []
        for row in rows:
            if row["person_id"] in seen:
                continue
            seen.add(row["person_id"])
            unique.append(dict(row))
        return unique

    def get_recent_detections(self, limit=20):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM detections ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_person_history(self, person_id, limit=500):
        """Full sighting history for one identity - the basis of a case report."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM detections WHERE person_id = ?
                   ORDER BY created_at ASC LIMIT ?""",
                (person_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Cases (a "case" bounds one target-lock investigation: the target was
    # set, some footage/cameras were reviewed against it, and it ends with
    # an official case report before the console resets for the next one)
    # ------------------------------------------------------------------

    def open_case(self, case_id, target_person_id):
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO cases (case_id, target_person_id, opened_at, status)
                   VALUES (?, ?, ?, 'open')""",
                (case_id, target_person_id, now),
            )
        return now

    def log_case_source(self, case_id, camera_id, source, mode, started_at, ended_at=None):
        """Record one video/camera source reviewed during a case, so the
        final report can list exactly what evidence was checked."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO case_sources (case_id, camera_id, source, mode, started_at, ended_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (case_id, camera_id, source, mode, started_at, ended_at),
            )

    def close_case_source(self, case_id, camera_id):
        """Stamp ended_at on the most recent still-open source row for this
        case/camera (called when a stream stops or the case closes)."""
        now = time.time()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """SELECT id FROM case_sources
                   WHERE case_id = ? AND camera_id = ? AND ended_at IS NULL
                   ORDER BY started_at DESC LIMIT 1""",
                (case_id, camera_id),
            ).fetchone()
            if row:
                conn.execute("UPDATE case_sources SET ended_at = ? WHERE id = ?", (now, row["id"]))

    def get_case_sources(self, case_id):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM case_sources WHERE case_id = ? ORDER BY started_at ASC",
                (case_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def close_case(self, case_id, report_pdf_path=None, report_csv_path=None, notes=None):
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE cases SET status = 'closed', closed_at = ?,
                       report_pdf_path = ?, report_csv_path = ?, notes = ?
                   WHERE case_id = ?""",
                (now, report_pdf_path, report_csv_path, notes, case_id),
            )
        return now

    def get_case(self, case_id):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        return dict(row) if row else None

    def get_all_cases(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM cases ORDER BY opened_at DESC").fetchall()
        return [dict(r) for r in rows]
