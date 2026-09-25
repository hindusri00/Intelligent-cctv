"""
report_generator.py
--------------------
Case-file / evidence report generation for the Sentry system.

This closes the "no evidence management or report generation" gap: given a
person_id already in the database (modules/database.py), it produces:

  - a CSV of every sighting (timestamp, camera, track, attributes,
    confidence sub-scores) - for spreadsheet analysis / appendices, and
  - a PDF case report - a readable document for a supervisor/report
    appendix, including the identity summary, its "typical" attribute
    profile, a sightings table, and (when snapshot_path was recorded on a
    detection) the evidence images themselves.

Nothing here talks to Flask directly - app.py just calls these functions
and sends the resulting file back with send_file(). That keeps this module
testable/runnable on its own (see the __main__ block).
"""

import os
import csv
import time
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
)

REPORTS_DIR = "reports"


def _ensure_reports_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    return REPORTS_DIR


def _fmt_ts(unix_seconds):
    if not unix_seconds:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(unix_seconds))


# ----------------------------------------------------------------------
# CSV export
# ----------------------------------------------------------------------

CSV_FIELDS = [
    "created_at", "video_source", "camera_id", "video_timestamp", "track_id",
    "shirt_color", "pant_color", "hair_color", "shoe_color", "mask",
    "mask_color", "spectacles", "wristband", "tattoo", "accessory",
    "estimated_height", "face_similarity", "reid_similarity",
    "attribute_score", "fused_confidence", "has_face", "snapshot_path",
]


def generate_csv_report(db, person_id, limit=1000):
    """
    Write every recorded sighting of `person_id` to a CSV file and return
    its path. Raises ValueError if the identity has no history at all.
    """
    history = db.get_person_history(person_id, limit=limit)
    if not history:
        raise ValueError(f"No detections on record for '{person_id}'")

    out_dir = _ensure_reports_dir()
    path = os.path.join(out_dir, f"{person_id}_report.csv")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            row = dict(row)
            row["created_at"] = _fmt_ts(row.get("created_at"))
            writer.writerow(row)

    return path


# ----------------------------------------------------------------------
# PDF case report
# ----------------------------------------------------------------------

def generate_pdf_report(db, person_id, limit=500, max_snapshots=6):
    """
    Build a PDF case report for `person_id`: identity summary, typical
    attribute profile, a sightings table, and (if available) a strip of
    evidence snapshots. Returns the output file path.
    """
    identity = db.get_identity(person_id)
    history = db.get_person_history(person_id, limit=limit)
    if identity is None and not history:
        raise ValueError(f"No identity or detections on record for '{person_id}'")

    profile = db.get_identity_attribute_profile(person_id, limit=limit) or {}

    out_dir = _ensure_reports_dir()
    path = os.path.join(out_dir, f"{person_id}_report.pdf")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SentryTitle", parent=styles["Title"], fontSize=18, spaceAfter=4,
    )
    meta_style = ParagraphStyle(
        "SentryMeta", parent=styles["Normal"], textColor=colors.grey, fontSize=9,
    )
    h2 = ParagraphStyle("SentryH2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)

    doc = SimpleDocTemplate(
        path, pagesize=A4,
        topMargin=20 * mm, bottomMargin=18 * mm,
        leftMargin=18 * mm, rightMargin=18 * mm,
    )
    story = []

    story.append(Paragraph(f"Sentry case report &mdash; {person_id}", title_style))
    story.append(Paragraph(f"Generated {_fmt_ts(time.time())}", meta_style))
    story.append(Spacer(1, 10))

    # --- Identity summary -------------------------------------------------
    story.append(Paragraph("Identity summary", h2))
    summary_rows = [["Field", "Value"]]
    if identity:
        summary_rows += [
            ["First seen", _fmt_ts(identity["first_seen"])],
            ["Last seen", _fmt_ts(identity["last_seen"])],
            ["Total observations", str(identity["observations"])],
            ["Face embedding on file", "Yes" if identity["face_embedding"] is not None else "No"],
            ["Body embedding on file", "Yes" if identity["body_embedding"] is not None else "No"],
        ]
    else:
        summary_rows += [["Total sightings logged", str(len(history))]]

    summary_table = Table(summary_rows, colWidths=[55 * mm, 110 * mm])
    summary_table.setStyle(_table_style(header=True))
    story.append(summary_table)

    # --- Typical attribute profile -----------------------------------------
    if profile:
        story.append(Paragraph("Typical attributes", h2))
        attr_rows = [["Attribute", "Most common value"]]
        for field, value in profile.items():
            attr_rows.append([field.replace("_", " ").title(), str(value)])
        attr_table = Table(attr_rows, colWidths=[55 * mm, 110 * mm])
        attr_table.setStyle(_table_style(header=True))
        story.append(attr_table)

    # --- Evidence snapshots -------------------------------------------------
    snapshot_rows = [h for h in history if h.get("snapshot_path") and os.path.isfile(h["snapshot_path"])]
    if snapshot_rows:
        story.append(Paragraph("Evidence snapshots", h2))
        thumbs = []
        for row in snapshot_rows[:max_snapshots]:
            try:
                img = RLImage(row["snapshot_path"], width=38 * mm, height=38 * mm)
                thumbs.append(img)
            except Exception:
                continue
        if thumbs:
            # Wrap thumbnails 3 to a row.
            grid = [thumbs[i:i + 3] for i in range(0, len(thumbs), 3)]
            snap_table = Table(grid, colWidths=[40 * mm] * 3)
            snap_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            story.append(snap_table)

    # --- Sightings table ------------------------------------------------
    story.append(Paragraph(f"Sightings ({len(history)})", h2))
    sighting_rows = [["Time", "Camera", "Track", "Confidence", "Shirt", "Mask"]]
    for row in history[-200:]:  # cap rows so the PDF stays a reasonable length
        sighting_rows.append([
            _fmt_ts(row.get("created_at")),
            row.get("camera_id") or "—",
            str(row.get("track_id") or "—"),
            f"{row['fused_confidence']:.2f}" if row.get("fused_confidence") is not None else "—",
            row.get("shirt_color") or "—",
            row.get("mask") or "—",
        ])
    sightings_table = Table(
        sighting_rows,
        colWidths=[32 * mm, 24 * mm, 18 * mm, 24 * mm, 24 * mm, 24 * mm],
        repeatRows=1,
    )
    sightings_table.setStyle(_table_style(header=True, small=True))
    story.append(sightings_table)

    doc.build(story)
    return path


def _fmt_duration(seconds):
    if not seconds or seconds < 0:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ----------------------------------------------------------------------
# Official case report (target-lock investigation, closed from the UI)
# ----------------------------------------------------------------------

def generate_case_report(db, case, person_id, sources, max_snapshots=12):
    """
    Build the official, downloadable case-closure PDF for a target-lock
    investigation ("case"). Unlike generate_pdf_report() (a general
    per-identity dossier), this is scoped to ONE case: only the sightings
    and sources logged between the case's opened_at/closed_at timestamps,
    plus a dedicated "Sources Reviewed" section so the report documents
    exactly which footage/cameras were checked, not just the outcome.

    `case`: dict from Database.get_case() (case_id, opened_at, closed_at, ...)
    `person_id`: the target identity's person_id (usually "Target")
    `sources`: list of dicts from Database.get_case_sources(case_id)

    Returns the output PDF path.
    """
    identity = db.get_identity(person_id)
    full_history = db.get_person_history(person_id, limit=2000)

    # Scope sightings to this case's time window so a report only reflects
    # what actually happened during THIS investigation, not the target's
    # entire lifetime history across earlier cases.
    opened_at = case["opened_at"]
    closed_at = case.get("closed_at") or time.time()
    history = [h for h in full_history if opened_at <= (h.get("created_at") or 0) <= closed_at]

    profile = db.get_identity_attribute_profile(person_id, limit=len(history) or 20) or {}

    out_dir = _ensure_reports_dir()
    cases_dir = os.path.join(out_dir, "cases")
    os.makedirs(cases_dir, exist_ok=True)
    path = os.path.join(cases_dir, f"{case['case_id']}_case_report.pdf")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SentryCaseTitle", parent=styles["Title"], fontSize=19, spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "SentryCaseSubtitle", parent=styles["Normal"], fontSize=11,
        textColor=colors.HexColor("#ff9f43"), spaceAfter=4,
    )
    meta_style = ParagraphStyle(
        "SentryCaseMeta", parent=styles["Normal"], textColor=colors.grey, fontSize=9,
    )
    h2 = ParagraphStyle("SentryCaseH2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("SentryCaseBody", parent=styles["Normal"], fontSize=10, leading=14)

    doc = SimpleDocTemplate(
        path, pagesize=A4,
        topMargin=20 * mm, bottomMargin=18 * mm,
        leftMargin=18 * mm, rightMargin=18 * mm,
        title=f"Sentry Case Report {case['case_id']}",
    )
    story = []

    # --- Header / official case identification -----------------------
    story.append(Paragraph("SENTRY &mdash; OFFICIAL CASE REPORT", title_style))
    story.append(Paragraph(f"Case ID: {case['case_id']}", subtitle_style))
    story.append(Paragraph(
        f"Generated {_fmt_ts(time.time())} &middot; Status: CLOSED", meta_style,
    ))
    story.append(Spacer(1, 10))

    case_rows = [["Field", "Value"]]
    case_rows += [
        ["Case opened", _fmt_ts(opened_at)],
        ["Case closed", _fmt_ts(closed_at)],
        ["Investigation duration", _fmt_duration(closed_at - opened_at)],
        ["Target identity", person_id],
        ["Total sightings this case", str(len(history))],
        ["Sources reviewed", str(len(sources))],
    ]
    case_table = Table(case_rows, colWidths=[55 * mm, 110 * mm])
    case_table.setStyle(_table_style(header=True))
    story.append(case_table)

    # --- Target identity summary --------------------------------------
    story.append(Paragraph("Target identity summary", h2))
    summary_rows = [["Field", "Value"]]
    if identity:
        summary_rows += [
            ["First seen (all time)", _fmt_ts(identity["first_seen"])],
            ["Last seen (all time)", _fmt_ts(identity["last_seen"])],
            ["Total observations (all time)", str(identity["observations"])],
            ["Face embedding on file", "Yes" if identity["face_embedding"] is not None else "No"],
            ["Body embedding on file", "Yes" if identity["body_embedding"] is not None else "No"],
        ]
    else:
        summary_rows += [["Total sightings logged", str(len(history))]]
    summary_table = Table(summary_rows, colWidths=[55 * mm, 110 * mm])
    summary_table.setStyle(_table_style(header=True))
    story.append(summary_table)

    # --- Target attributes (as observed during this case) --------------
    if profile:
        story.append(Paragraph("Target attributes (observed)", h2))
        attr_rows = [["Attribute", "Most common value"]]
        for field, value in profile.items():
            attr_rows.append([field.replace("_", " ").title(), str(value)])
        attr_table = Table(attr_rows, colWidths=[55 * mm, 110 * mm])
        attr_table.setStyle(_table_style(header=True))
        story.append(attr_table)

    # --- Sources reviewed ------------------------------------------------
    story.append(Paragraph("Sources reviewed", h2))
    if sources:
        src_rows = [["Camera", "Source", "Mode", "Started", "Ended"]]
        for s in sources:
            src_rows.append([
                s.get("camera_id") or "—",
                os.path.basename(str(s.get("source"))) if s.get("source") else "—",
                s.get("mode") or "—",
                _fmt_ts(s.get("started_at")),
                _fmt_ts(s.get("ended_at")) if s.get("ended_at") else "in progress at close",
            ])
        src_table = Table(src_rows, colWidths=[24 * mm, 62 * mm, 20 * mm, 32 * mm, 27 * mm], repeatRows=1)
        src_table.setStyle(_table_style(header=True, small=True))
        story.append(src_table)
    else:
        story.append(Paragraph("No footage or camera sources were logged for this case.", body_style))

    # --- Target sighting timeline with evidence frames -------------------
    story.append(Paragraph(f"Target sighting timeline ({len(history)})", h2))
    if not history:
        story.append(Paragraph("No sightings of the target were logged during this case.", body_style))
    else:
        timeline_rows = [["Time", "Camera", "Video ts", "Track", "Confidence", "F / B"]]
        for row in history:
            conf = f"{row['fused_confidence']:.2f}" if row.get("fused_confidence") is not None else "—"
            face = f"{row['face_similarity']:.2f}" if row.get("face_similarity") is not None else "—"
            body = f"{row['reid_similarity']:.2f}" if row.get("reid_similarity") is not None else "—"
            timeline_rows.append([
                _fmt_ts(row.get("created_at")),
                row.get("camera_id") or "—",
                f"{row['video_timestamp']}s" if row.get("video_timestamp") is not None else "—",
                str(row.get("track_id") or "—"),
                conf,
                f"{face} / {body}",
            ])
        timeline_table = Table(
            timeline_rows,
            colWidths=[30 * mm, 20 * mm, 20 * mm, 18 * mm, 22 * mm, 25 * mm],
            repeatRows=1,
        )
        timeline_table.setStyle(_table_style(header=True, small=True))
        story.append(timeline_table)

    # --- Evidence frames (saved snapshot images) --------------------------
    snapshot_rows = [h for h in history if h.get("snapshot_path") and os.path.isfile(h["snapshot_path"])]
    if snapshot_rows:
        story.append(Paragraph(f"Evidence frames ({len(snapshot_rows)} saved, showing up to {max_snapshots})", h2))
        thumbs = []
        for row in snapshot_rows[:max_snapshots]:
            try:
                img = RLImage(row["snapshot_path"], width=42 * mm, height=42 * mm)
                vt = row.get("video_timestamp")
                vt_label = f"{vt}s" if vt is not None else "—"
                caption = Paragraph(
                    f"<font size=7>{_fmt_ts(row.get('created_at'))}<br/>cam {row.get('camera_id') or '—'} "
                    f"&middot; {vt_label}</font>",
                    body_style,
                )
                thumbs.append([img, caption])
            except Exception:
                continue
        if thumbs:
            for img, caption in thumbs:
                cell = Table([[img], [caption]], colWidths=[42 * mm])
                cell.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
                story.append(cell)
                story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No evidence frames were saved during this case.", body_style))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "This report was generated automatically by the Sentry identification console "
        "and reflects data logged between case open and case close. It is intended as a "
        "case-file artifact for internal review.",
        meta_style,
    ))

    doc.build(story)
    return path


def _table_style(header=False, small=False):
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8ccd6")),
        ("FONTSIZE", (0, 0), (-1, -1), 8 if small else 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#171c26")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    return TableStyle(style)


if __name__ == "__main__":
    # Quick smoke test against a throwaway in-memory-ish database, so this
    # module can be sanity-checked without the rest of the Sentry stack.
    import numpy as np
    from database import Database

    db = Database(db_path="/tmp/_report_gen_smoke.db")
    db.upsert_identity("Person_001", face_embedding=np.random.rand(512), body_embedding=np.random.rand(512))
    for i in range(5):
        db.log_detection(
            person_id="Person_001", track_id=1, camera_id="cam1",
            attributes={"shirt_color": "Blue", "mask": "Yes"},
            video_source="demo.mp4", video_timestamp=i * 2.0,
            reid_similarity=0.8, face_similarity=0.7,
            attribute_score=0.9, fused_confidence=0.78, has_face=True,
        )

    csv_path = generate_csv_report(db, "Person_001")
    pdf_path = generate_pdf_report(db, "Person_001")
    print("CSV ->", csv_path)
    print("PDF ->", pdf_path)
