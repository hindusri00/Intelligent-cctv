import os
import cv2
from flask import Flask, Response, request, jsonify, send_from_directory
from flask_cors import CORS
from modules.tracking_module import RealTimeTracker
import numpy as np
app = Flask(__name__, static_folder='frontend', template_folder='frontend')
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

tracker_system = RealTimeTracker(model_path="yolo11n.pt")
detection_logs = []

target_embedding = None
target_person_id = "Target"

# Configurable User Blacklist Rules
active_blacklist = {
    "mask": True,
    "tattoo": False,
    "spectacles": False,
    "wristband": False,
    "bag": False
}

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
    global target_embedding

    if 'image' not in request.files:
        return jsonify({"error": "No target image uploaded"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No target image selected"}), 400

    image_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    embedding = tracker_system.reid.extract_embedding(image)

    if embedding is None:
        return jsonify({"error": "Could not extract target features"}), 400

    target_embedding = embedding
    tracker_system.set_target(embedding)

    return jsonify({
        "status": "success",
        "target_id": target_person_id
    })
# --- FLEXIBLE NATURAL LANGUAGE SEARCH ---
@app.route('/api/search', methods=['POST'])
def search_person():
    raw_query = request.json.get('query', '').lower().strip()

    matches = []

    # Detect requested attribute from the query
    attribute_map = {
        "shirt": "shirt_color",
        "pant": "pant_color",
        "pants": "pant_color",
        "hair": "hair_color",
        "mask": "mask",
        "bag": "accessory",
        "glasses": "spectacles",
        "spectacles": "spectacles",
        "tattoo": "tattoo"
    }

    requested_attribute = None

    for keyword, attribute in attribute_map.items():
        if keyword in raw_query:
            requested_attribute = attribute
            break

    # Remove common words
    stop_words = {
        "person", "people", "man", "woman", "guy",
        "with", "wearing", "a", "an", "the",
        "carrying", "shirt", "pant", "pants",
        "hair", "mask", "bag", "glasses",
        "spectacles", "tattoo", "color"
    }

    query_tokens = [
        word for word in raw_query.split()
        if word not in stop_words
    ]

    for log in detection_logs:
        attr = log["attributes"]

        shirt = str(attr.get("shirt_color", "")).lower()
        pant = str(attr.get("pant_color", "")).lower()
        hair = str(attr.get("hair_color", "")).lower()
        mask = str(attr.get("mask", "")).lower()
        accessory = str(attr.get("accessory", "")).lower()

        # If an attribute was explicitly mentioned,
        # search only inside that attribute.
        if requested_attribute == "shirt_color":
            searchable_text = shirt

        elif requested_attribute == "pant_color":
            searchable_text = pant

        elif requested_attribute == "hair_color":
            searchable_text = hair

        elif requested_attribute == "mask":
            searchable_text = mask

        elif requested_attribute == "accessory":
            searchable_text = accessory

        elif requested_attribute == "spectacles":
            searchable_text = str(attr.get("spectacles", "")).lower()

        elif requested_attribute == "tattoo":
            searchable_text = str(attr.get("tattoo", "")).lower()

        else:
            searchable_text = (
                f"{shirt} {pant} {hair} {mask} {accessory}"
            )

        if query_tokens and all(
            token in searchable_text
            for token in query_tokens
        ):
            matches.append({
                "timestamp": f"{log['timestamp']}s",
                "person_id": log["person_id"],
                "shirt": attr.get("shirt_color"),
                "pant": attr.get("pant_color"),
                "hair": attr.get("hair_color"),
                "accessory": attr.get("accessory"),
                "mask": attr.get("mask")
            })

    unique_matches = list({
        m["person_id"]: m
        for m in matches
    }.values())

    return jsonify({
        "query": raw_query,
        "results": unique_matches
    })

# --- DYNAMIC ALERTS ENDPOINT ---
@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    alerts = []
    
    for log in detection_logs[-20:]:
        attr = log["attributes"]
        triggered_reasons = []

        # A target-lock sighting should always be surfaced, regardless of
        # which blacklist rules happen to be toggled on - previously this
        # only showed up in /api/alerts when a blacklist rule ALSO matched,
        # so "target found" silently disappeared with rules like mask off.
        if log.get("person_id") == "TARGET":
            triggered_reasons.append("Target Located")

        if active_blacklist.get("mask") and attr.get("mask") == "Yes":
            triggered_reasons.append("Mask Worn")
        if active_blacklist.get("tattoo") and attr.get("tattoo") == "Yes":
            triggered_reasons.append("Tattoo Visible")
        if active_blacklist.get("spectacles") and attr.get("spectacles") == "Yes":
            triggered_reasons.append("Glasses Worn")
        if active_blacklist.get("wristband") and attr.get("wristband") == "Yes":
            triggered_reasons.append("Wristband Worn")
        if active_blacklist.get("bag") and attr.get("accessory") != "None":
            triggered_reasons.append(f"Carrying Bag ({attr.get('accessory')})")

        if triggered_reasons:
            alerts.append({
                "timestamp": f"{log['timestamp']}s",
                "person_id": log["person_id"],
                "track_id": log["track_id"],
                "reason": ", ".join(triggered_reasons)
            })
            
    unique_alerts = list({
        (a['person_id'], a['reason']): a
        for a in alerts
    }.values())

    return jsonify({"alerts": unique_alerts})

def generate_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp_sec = round(frame_count / fps, 2)
        annotated_frame, targets = tracker_system.process_frame(frame)

        for t in targets:
            detection_logs.append({
                "timestamp": timestamp_sec,
                "track_id": t["track_id"],
                "person_id": t["person_id"],
                "reid_similarity": t["reid_similarity"],
                "attributes": t
            })

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

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
    return Response(generate_video_stream(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)