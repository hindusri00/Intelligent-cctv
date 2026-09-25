import os
import time
import uuid
import cv2
from flask import Flask, Response, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from modules.tracking_module import RealTimeTracker
from modules.database import Database
from modules.report_generator import generate_csv_report, generate_pdf_report, generate_case_report
import numpy as np

app = Flask(__name__, static_folder='frontend', template_folder='frontend')
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EVIDENCE_FOLDER'] = 'evidence'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EVIDENCE_FOLDER'], exist_ok=True)

db = Database(db_path="sentry.db")

target_person_id = "Target"
# Kept so a tracker created AFTER a target was set (e.g. a camera added
# mid-session) still gets the target applied - see get_tracker().
target_embeddings = {"body": None, "face": None}

# --- CASE LIFECYCLE -------------------------------------------------------
# A "case" bounds one target-lock investigation: it opens the moment a
# target is set, spans every camera/footage source reviewed against that
# target, and ends when the operator explicitly closes it from the UI -
# which snapshots everything into an official PDF (modules/report_generator
# .generate_case_report) before the console resets so a new case can start
# clean (see /api/case/close below).
current_case = None  # {"case_id": str, "opened_at": float} | None

# Configurable User Blacklist Rules
active_blacklist = {
    "mask": True,
    "tattoo": False,
    "spectacles": False,
    "wristband": False,
    "bag": False
}

# --- MULTI-CAMERA TRACKER REGISTRY ---------------------------------------
# One RealTimeTracker per camera_id. Each camera gets its own track-id space
# (a person can be "track #3" on cam1 and "track #7" on cam2 at the same
# time), but they all resolve identities against the SAME database, which is
# what makes "the same person recognized on a different camera" work: the
# fusion layer (modules/fusion_module.py) scans the whole identities table
# regardless of which camera produced the current embedding.
trackers = {}
camera_sources = {}  # camera_id -> {"source": str, "mode": "footage"|"live", "started_at": float}


def get_tracker(camera_id):
    tracker = trackers.get(camera_id)
    if tracker is None:
        tracker = RealTimeTracker(db=db, model_path="yolo11n.pt")
        if target_embeddings["body"] is not None or target_embeddings["face"] is not None:
            tracker.set_target(target_embeddings["body"], target_embeddings["face"])
        trackers[camera_id] = tracker
    return tracker


# --- FRONTEND ROUTING ---
@app.route('/')
def serve_frontend():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory('frontend', path)

# --- CONFIGURABLE BLACKLIST ENDPOINTS ---
@app.route('/api/blacklist', methods=['GET', 'POST'])
def manage_blacklist():
    global active_blacklist
    if request.method == 'POST':
        active_blacklist = request.json.get('blacklist', active_blacklist)
        return jsonify({"status": "updated", "blacklist": active_blacklist})
    return jsonify({"blacklist": active_blacklist})

@app.route('/api/target', methods=['POST'])
def set_target():
    global current_case

    if 'image' not in request.files:
        return jsonify({"error": "No target image uploaded"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No target image selected"}), 400

    image_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    # Any existing tracker (there may be none yet, or several - one per
    # active camera) can extract the reference embeddings; they all share
    # the same face/reid models, so which one does it is arbitrary.
    reference_tracker = next(iter(trackers.values()), None) or get_tracker("default")

    body_embedding = reference_tracker.reid.extract_embedding(image)
    # Try the full reference photo for a face first (most target photos are
    # framed with the face visible even if it's not strictly a headshot).
    face_embedding = reference_tracker.face.extract_embedding(image)
    print(
    f"[TARGET EXTRACT] body={body_embedding is not None}, face={face_embedding is not None}",
    flush=True)

    if body_embedding is None and face_embedding is None:
        return jsonify({"error": "Could not extract target features (no usable body or face found)"}), 400

    target_embeddings["body"] = body_embedding
    target_embeddings["face"] = face_embedding

    # Apply to every camera currently running, not just one.
    for tracker in trackers.values():
        tracker.set_target(body_embedding, face_embedding)

    # Persist the target as a named identity so it survives a restart and
    # shows up in /api/identities like any other person.
    db.upsert_identity(target_person_id, face_embedding=face_embedding, body_embedding=body_embedding)

    # Setting (or re-setting) a target opens the case if one isn't already
    # running. Re-locking onto a new reference photo mid-case does NOT open
    # a second case - that's still the same investigation.
    if current_case is None:
        case_id = time.strftime("CASE-%Y%m%d-") + uuid.uuid4().hex[:6].upper()
        opened_at = db.open_case(case_id, target_person_id)
        current_case = {"case_id": case_id, "opened_at": opened_at}
        # Any camera already streaming when the target locks counts as a
        # source for this case too.
        for cam_id, info in camera_sources.items():
            db.log_case_source(
                case_id, camera_id=cam_id, source=info.get("source"),
                mode=info.get("mode"), started_at=info.get("started_at"),
            )

    return jsonify({
        "status": "success",
        "target_id": target_person_id,
        "has_face": face_embedding is not None,
        "has_body": body_embedding is not None,
        "case_id": current_case["case_id"],
    })

# --- FLEXIBLE NATURAL LANGUAGE SEARCH ---
@app.route('/api/search', methods=['POST'])
def search_person():
    raw_query = request.json.get('query', '').lower().strip()
    # Normalize hyphens to spaces so "t-shirt" tokenizes the same as "tshirt".
    normalized_query = raw_query.replace('-', ' ')

    attribute_map = {
        "shirt": "shirt_color",
        "tshirt": "shirt_color",
        "pant": "pant_color",
        "pants": "pant_color",
        "trouser": "pant_color",
        "trousers": "pant_color",
        "jeans": "pant_color",
        "hair": "hair_color",
        "mask": "mask",
        "bag": "accessory",
        "glasses": "spectacles",
        "spectacles": "spectacles",
        "tattoo": "tattoo"
    }

    requested_attribute = None
    for keyword, attribute in attribute_map.items():
        if keyword in normalized_query:
            requested_attribute = attribute
            break

    # Every noun this search understands (plus common synonyms/variants) has
    # to be stripped out of the query too, or it survives into search_value
    # and gets LIKE-matched against a plain color field like "Brown" -
    # which it will never match. Previously "tshirt"/"t-shirt" weren't in
    # here, so "brown tshirt" searched for the literal substring
    # "brown tshirt" instead of just "brown".
    stop_words = {
        "person", "people", "man", "woman", "guy",
        "with", "wearing", "a", "an", "the",
        "carrying", "shirt", "shirts", "tshirt", "tshirts", "tee", "tees",
        "pant", "pants", "trouser", "trousers", "jean", "jeans",
        "hair", "mask", "bag", "glasses",
        "spectacles", "tattoo", "color", "colour"
    }

    query_tokens = [w for w in normalized_query.split() if w not in stop_words]
    search_value = " ".join(query_tokens) if query_tokens else raw_query

    rows = db.search_detections(attribute=requested_attribute, value=search_value)

    results = [{
        "timestamp": f"{row['video_timestamp']}s" if row['video_timestamp'] is not None else "—",
        "person_id": row["person_id"],
        "shirt": row["shirt_color"],
        "pant": row["pant_color"],
        "hair": row["hair_color"],
        "accessory": row["accessory"],
        "mask": row["mask"],
        "confidence": row["fused_confidence"],
    } for row in rows]

    return jsonify({"query": raw_query, "results": results})

# --- DYNAMIC ALERTS ENDPOINT ---
@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    alerts = []

    for row in db.get_recent_detections(limit=20):
        triggered_reasons = []

        if row["person_id"] == target_person_id.upper() or row["person_id"] == "TARGET":
            triggered_reasons.append("Target Located")

        if active_blacklist.get("mask") and row["mask"] == "Yes":
            triggered_reasons.append("Mask Worn")
        if active_blacklist.get("tattoo") and row["tattoo"] == "Yes":
            triggered_reasons.append("Tattoo Visible")
        if active_blacklist.get("spectacles") and row["spectacles"] == "Yes":
            triggered_reasons.append("Glasses Worn")
        if active_blacklist.get("wristband") and row.get("wristband") == "Yes":
            triggered_reasons.append("Wristband Worn")
        if active_blacklist.get("bag") and row["accessory"] not in (None, "None"):
            triggered_reasons.append(f"Carrying Bag ({row['accessory']})")

        if triggered_reasons:
            alerts.append({
                "timestamp": f"{row['video_timestamp']}s" if row['video_timestamp'] is not None else "—",
                "person_id": row["person_id"],
                "track_id": row["track_id"],
                "camera_id": row.get("camera_id"),
                "reason": ", ".join(triggered_reasons),
                "confidence": row["fused_confidence"],
                "face_similarity": row["face_similarity"],
                "body_similarity": row["reid_similarity"],
            })

    unique_alerts = list({(a['person_id'], a['reason']): a for a in alerts}.values())
    return jsonify({"alerts": unique_alerts})

# --- IDENTITY ROSTER ---
@app.route('/api/identities', methods=['GET'])
def list_identities():
    return jsonify({"identities": db.get_all_identities()})

# --- ACTIVE CAMERAS (new: multi-camera support) --------------------------
@app.route('/api/cameras', methods=['GET'])
def list_cameras():
    return jsonify({"cameras": [
        {"camera_id": cam_id, **info} for cam_id, info in camera_sources.items()
    ]})

# --- CASE REPORT / EVIDENCE EXPORT (new) ---------------------------------
@app.route('/api/report/<person_id>', methods=['GET'])
def get_report(person_id):
    fmt = request.args.get('format', 'pdf').lower()
    try:
        if fmt == 'csv':
            path = generate_csv_report(db, person_id)
            return send_file(path, as_attachment=True, download_name=f"{person_id}_report.csv")
        path = generate_pdf_report(db, person_id)
        return send_file(path, as_attachment=True, download_name=f"{person_id}_report.pdf")
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


def _save_evidence_snapshot(frame, track, camera_id):
    """
    Crop and persist the region a detection came from, so a case report can
    show the actual frame a sighting was pulled from rather than just a
    confidence number. Returns a path to store in detections.snapshot_path,
    or None if there's no bbox to crop (defensive: this assumes
    tracking_module attaches a "bbox": [x1, y1, x2, y2] key to each track
    dict - adjust the key name below if yours differs).
    """
    bbox = track.get("bbox")
    if not bbox:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = frame[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size == 0:
            return None
        person_dir = os.path.join(app.config['EVIDENCE_FOLDER'], camera_id, str(track["person_id"]))
        os.makedirs(person_dir, exist_ok=True)
        filename = f"{int(time.time() * 1000)}_track{track.get('track_id')}.jpg"
        path = os.path.join(person_dir, filename)
        cv2.imwrite(path, crop)
        return path
    except Exception:
        return None


def generate_video_stream(video_path, camera_id):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = 0

    source_label = str(video_path)
    tracker_system = get_tracker(camera_id)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp_sec = round(frame_count / fps, 2)
        annotated_frame, targets = tracker_system.process_frame(frame)

        for t in targets:
            t.pop("face_embedding", None)  # not JSON-serializable

            # Only snapshot occasionally (roughly once every couple of
            # seconds per track), not on every processed frame - keeps disk
            # usage and per-frame latency bounded on longer footage.
            snapshot_path = None
            if frame_count % max(int(fps), 1) == 0:
                snapshot_path = _save_evidence_snapshot(frame, t, camera_id)

            db.log_detection(
                person_id=t["person_id"],
                track_id=t["track_id"],
                camera_id=camera_id,
                attributes=t,
                video_source=source_label,
                video_timestamp=timestamp_sec,
                reid_similarity=t.get("body_similarity"),
                face_similarity=t.get("face_similarity"),
                attribute_score=t.get("attribute_score"),
                fused_confidence=t.get("confidence"),
                has_face=t.get("has_face", False),
                snapshot_path=snapshot_path,
            )

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    camera_sources.pop(camera_id, None)
    if current_case is not None:
        db.close_case_source(current_case["case_id"], camera_id)

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['video']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    return jsonify({"status": "success", "video_path": file_path})

@app.route('/api/stream')
def video_stream():
    video_path = request.args.get('path', default=0)
    camera_id = request.args.get('camera_id', default='default')
    started_at = time.time()
    camera_sources[camera_id] = {
        "source": str(video_path) if video_path != 0 else "live camera",
        "mode": "live" if video_path == 0 else "footage",
        "started_at": started_at,
    }
    # If a case is already open (a target is locked), every source that
    # starts streaming from here on is evidence reviewed for that case.
    if current_case is not None:
        db.log_case_source(
            current_case["case_id"], camera_id=camera_id,
            source=camera_sources[camera_id]["source"],
            mode=camera_sources[camera_id]["mode"], started_at=started_at,
        )
    return Response(generate_video_stream(video_path, camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --- CASE MANAGEMENT (open/close/report/list) ----------------------------
@app.route('/api/case/status', methods=['GET'])
def case_status():
    if current_case is None:
        return jsonify({"open": False})
    return jsonify({
        "open": True,
        "case_id": current_case["case_id"],
        "opened_at": current_case["opened_at"],
        "target_id": target_person_id,
    })


@app.route('/api/case/close', methods=['POST'])
def close_case():
    """
    Close the active case: generate the official case report (PDF, with
    saved evidence frames, the target's sighting timeline, and every
    video/camera source reviewed), persist it under /reports/cases, then
    reset all in-memory tracking state (trackers, target lock, per-camera
    caches) so the console is ready to start a brand-new case from scratch.
    """
    global current_case, target_embeddings, trackers, camera_sources

    if current_case is None:
        return jsonify({"error": "No case is currently open."}), 400

    case_id = current_case["case_id"]

    # Close out any still-streaming sources so the report's "ended" column
    # isn't blank for cameras that were live right up to case close.
    for cam_id in list(camera_sources.keys()):
        db.close_case_source(case_id, cam_id)

    # Build the case dict the report generator needs BEFORE writing
    # closed_at to the DB, so the same closed_at timestamp is used both in
    # the report and in the stored record (a single source of truth).
    case = db.get_case(case_id)
    case["closed_at"] = time.time()
    sources = db.get_case_sources(case_id)

    try:
        pdf_path = generate_case_report(db, case, target_person_id, sources)
    except Exception as e:
        # No sightings/identity at all - still close the case, just without
        # a populated report.
        db.close_case(case_id)
        current_case = None
        return jsonify({"error": f"Could not generate case report: {e}"}), 500

    db.close_case(case_id, report_pdf_path=pdf_path)
    # close_case() stamps its own closed_at = now(); re-read so the response
    # reflects exactly what was persisted.
    case = db.get_case(case_id)

    notes = {
        "case_id": case_id,
        "opened_at": case["opened_at"],
        "closed_at": case["closed_at"],
        "sightings": len(db.get_person_history(target_person_id, limit=2000)),
        "sources_reviewed": len(sources),
        "report_url": f"/api/cases/{case_id}/report",
    }

    # --- Reset for the next case ---------------------------------------
    target_embeddings = {"body": None, "face": None}
    trackers = {}          # drop every RealTimeTracker (and its per-track caches)
    camera_sources = {}
    current_case = None
    # The target identity itself stays in the database (historical record);
    # only the live working state resets, so the next case starts clean.

    return jsonify({"status": "closed", "case": notes})


@app.route('/api/cases', methods=['GET'])
def list_cases():
    return jsonify({"cases": db.get_all_cases(), "active_case": current_case})


@app.route('/api/cases/<case_id>/report', methods=['GET'])
def download_case_report(case_id):
    case = db.get_case(case_id)
    if case is None:
        return jsonify({"error": "Unknown case"}), 404
    if not case.get("report_pdf_path") or not os.path.isfile(case["report_pdf_path"]):
        return jsonify({"error": "No report has been generated for this case"}), 404
    return send_file(case["report_pdf_path"], as_attachment=True,
                      download_name=f"{case_id}_case_report.pdf")


if __name__ == '__main__':
    app.run(debug=True, port=5000)
