import json
import shutil
from pathlib import Path

# ---------------------------------------------------------
# CrowdHuman → YOLO Dataset Preparation
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "crowdhuman"
RAW_DIR = DATASET_DIR / "raw"
IMAGE_DIR = DATASET_DIR / "images"

# Original extracted images are currently directly inside images/
SOURCE_IMAGES = IMAGE_DIR

TRAIN_ANNOTATION = RAW_DIR / "annotation_train.odgt"
VAL_ANNOTATION = RAW_DIR / "annotation_val.odgt"

TRAIN_IMAGES = IMAGE_DIR / "train"
VAL_IMAGES = IMAGE_DIR / "val"

TRAIN_LABELS = DATASET_DIR / "labels" / "train"
VAL_LABELS = DATASET_DIR / "labels" / "val"

for directory in [TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS]:
    directory.mkdir(parents=True, exist_ok=True)


def convert_box_to_yolo(box, image_width, image_height):
    """
    CrowdHuman fbox format:
        [x, y, width, height]

    YOLO format:
        class x_center y_center width height

    All coordinates are normalized to 0-1.
    """

    x, y, w, h = box

    # Convert from x,y,w,h to x_center,y_center,w,h
    x_center = x + (w / 2)
    y_center = y + (h / 2)

    # Normalize
    x_center /= image_width
    y_center /= image_height
    w /= image_width
    h /= image_height

    return x_center, y_center, w, h


def process_split(annotation_file, image_destination, label_destination, split_name):
    print(f"\nProcessing {split_name}...")

    processed = 0
    skipped = 0
    boxes_written = 0

    with open(annotation_file, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            image_id = record["ID"]
            image_name = f"{image_id}.jpg"

            source_image = SOURCE_IMAGES / image_name

            # We only process images that we actually downloaded.
            if not source_image.exists():
                skipped += 1
                continue

            destination_image = image_destination / image_name
            label_file = label_destination / f"{image_id}.txt"

            # Copy image
            if not destination_image.exists():
                shutil.copy2(source_image, destination_image)

            # Read image dimensions
            try:
                import cv2

                image = cv2.imread(str(source_image))

                if image is None:
                    print(f"Warning: Could not read {image_name}")
                    skipped += 1
                    continue

                image_height, image_width = image.shape[:2]

            except Exception as error:
                print(f"Error reading {image_name}: {error}")
                skipped += 1
                continue

            yolo_lines = []

            for box_data in record.get("gtboxes", []):

                # We only want person annotations.
                if box_data.get("tag") != "person":
                    continue

                # Ignore explicitly ignored annotations.
                extra = box_data.get("extra", {})
                head_attr = box_data.get("head_attr", {})

                if extra.get("ignore", 0) == 1:
                    continue

                if head_attr.get("ignore", 0) == 1:
                    continue

                # Full-body bounding box
                fbox = box_data.get("fbox")

                if not fbox or len(fbox) != 4:
                    continue

                x, y, w, h = fbox

                # Skip invalid boxes
                if w <= 0 or h <= 0:
                    continue

                # Convert to YOLO format
                xc, yc, nw, nh = convert_box_to_yolo(
                    fbox,
                    image_width,
                    image_height
                )

                # Keep coordinates within valid YOLO range
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                yolo_lines.append(
                    f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
                )

            # Only create a label if at least one valid person exists.
            if yolo_lines:
                with open(label_file, "w", encoding="utf-8") as label:
                    label.write("\n".join(yolo_lines) + "\n")

                processed += 1
                boxes_written += len(yolo_lines)

            else:
                skipped += 1

    print(f"{split_name} processing complete.")
    print(f"Images processed : {processed}")
    print(f"Images skipped   : {skipped}")
    print(f"Person boxes     : {boxes_written}")


def main():
    print("=" * 60)
    print("CrowdHuman Dataset Preparation")
    print("=" * 60)

    process_split(
        TRAIN_ANNOTATION,
        TRAIN_IMAGES,
        TRAIN_LABELS,
        "TRAIN"
    )

    process_split(
        VAL_ANNOTATION,
        VAL_IMAGES,
        VAL_LABELS,
        "VALIDATION"
    )

    print("\nDataset preparation completed.")
    print(f"Train images : {TRAIN_IMAGES}")
    print(f"Train labels : {TRAIN_LABELS}")
    print(f"Val images   : {VAL_IMAGES}")
    print(f"Val labels   : {VAL_LABELS}")


if __name__ == "__main__":
    main()