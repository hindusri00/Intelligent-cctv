import cv2
import numpy as np

class AttributeAnalyzer:
    def __init__(self, camera_focal_length=800, real_camera_distance_cm=300):
        # Calibrated HSV Color Boundaries
        self.COLOR_RANGES = {
            "Red": [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
            "Blue": [((95, 80, 50), (130, 255, 255))],
            "Green": [((36, 50, 50), (89, 255, 255))],
            "Yellow": [((25, 50, 50), (35, 255, 255))],
            "Black": [((0, 0, 0), (180, 255, 25))],
            "White": [((0, 0, 130), (180, 45, 255))],  # Expanded threshold for shadowed/off-white fabrics
            "Grey": [((0, 0, 36), (180, 40, 160))],
            "Brown": [((10, 100, 20), (20, 255, 200))]
        }
        self.focal_length = camera_focal_length
        self.camera_distance = real_camera_distance_cm

    def _get_dominant_color(self, image_crop):
        name, _ratio = self._get_dominant_color_with_confidence(image_crop)
        return name

    def _get_dominant_color_with_confidence(self, image_crop):
        """
        Same color-bucket logic as before, but also returns what fraction
        of the crop's pixels actually matched that bucket. A crop that is
        mostly shadow/skin/hair can still end up with SOME bucket as the
        'largest', even if none of the buckets really explain the crop -
        the ratio lets callers require a real majority before trusting it.
        """
        if image_crop is None or image_crop.size == 0:
            return "Unknown", 0.0

        hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
        total_px = image_crop.shape[0] * image_crop.shape[1]
        max_pixels = 0
        dominant_color = "Unknown"

        for color_name, ranges in self.COLOR_RANGES.items():
            total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                total_mask = cv2.bitwise_or(total_mask, mask)

            count = cv2.countNonZero(total_mask)
            if count > max_pixels:
                max_pixels = count
                dominant_color = color_name

        ratio = max_pixels / total_px if total_px > 0 else 0.0
        return dominant_color, ratio

    def analyze_head_region(self, head_crop):
        if head_crop is None or head_crop.size == 0:
            return {"mask": "No", "mask_color": "N/A", "spectacles": "No"}

        h, w = head_crop.shape[:2]
        lower_face = head_crop[int(h * 0.45):h, int(w * 0.2):int(w * 0.8)]
        upper_face = cv2.cvtColor(head_crop[int(h * 0.15):int(h * 0.55), :], cv2.COLOR_BGR2GRAY) if h > 20 else None

        mask_status = "No"
        mask_color = "N/A"
        if lower_face.size > 0:
            detected_color, coverage = self._get_dominant_color_with_confidence(lower_face)
            # Require the color to actually cover most of the region, not
            # just be the largest of several small/noisy buckets - this is
            # what was causing skin/hair/shadow to be misread as a mask.
            if detected_color in ["White", "Blue", "Black"] and coverage >= 0.35:
                mask_status = "Yes"
                mask_color = detected_color

        has_specs = "No"
        if upper_face is not None and upper_face.size > 0:
            edges = cv2.Canny(upper_face, 100, 200)
            if cv2.countNonZero(edges) > (0.18 * upper_face.shape[0] * upper_face.shape[1]):
                has_specs = "Yes"

        return {"mask": mask_status, "mask_color": mask_color, "spectacles": has_specs}

    def analyze_wrists_and_tattoos(self, person_crop):
        if person_crop is None or person_crop.size == 0:
            return {"wristband": "No", "tattoo": "No"}

        h, w = person_crop.shape[:2]
        arm_region = person_crop[int(h * 0.35):int(h * 0.65), :]
        if arm_region.size == 0:
            return {"wristband": "No", "tattoo": "No"}

        hsv_arm = cv2.cvtColor(arm_region, cv2.COLOR_BGR2HSV)
        gray_arm = cv2.cvtColor(arm_region, cv2.COLOR_BGR2GRAY)

        sat_mask = cv2.inRange(hsv_arm, np.array([0, 150, 100]), np.array([180, 255, 255]))
        has_wristband = "Yes" if cv2.countNonZero(sat_mask) > (0.02 * arm_region.shape[0] * arm_region.shape[1]) else "No"

        laplacian_var = cv2.Laplacian(gray_arm, cv2.CV_64F).var()
        has_tattoo = "Yes" if laplacian_var > 350 else "No"

        return {"wristband": has_wristband, "tattoo": has_tattoo}

    def estimate_height(self, bbox_height_px):
        estimated_cm = (bbox_height_px * self.camera_distance) / self.focal_length
        return f"{int(np.clip(estimated_cm, 140, 200))} cm"

    def extract_all_attributes(self, person_crop, bbox_height_px):
        h, w, _ = person_crop.shape
        if h < 30 or w < 15:
            return {
                "hair_color": "Unknown", "shirt_color": "Unknown", "pant_color": "Unknown",
                "shoe_color": "Unknown", "mask": "No", "mask_color": "N/A", "spectacles": "No",
                "wristband": "No", "tattoo": "No", "estimated_height": "Unknown"
            }

        # 1. Head & Hair Slices (Top 20%)
        head_region = person_crop[0:int(h * 0.20), :]
        hair_slice = person_crop[
            int(h * 0.02):int(h * 0.14),
            int(w * 0.25):int(w * 0.75)
        ]

        # 2. Chest-Centered Shirt Crop (18% to 42% height, center 50% width)
        # Prevents long hair and counter/desk occlusions from corrupting shirt color
        shirt_crop = person_crop[int(h * 0.18):int(h * 0.42), int(w * 0.25):int(w * 0.75)]

        # 3. Lower Body & Feet Slices
        lower_region = person_crop[
            int(h * 0.45):int(h * 0.80),
            int(w * 0.20):int(w * 0.80)
        ]

        feet_region = person_crop[
            int(h * 0.82):h,
            int(w * 0.15):int(w * 0.85)
        ]

        head_info = self.analyze_head_region(head_region)
        body_marks = self.analyze_wrists_and_tattoos(person_crop)

        return {
            "hair_color": self._get_dominant_color(hair_slice),
            "shirt_color": self._get_dominant_color(shirt_crop),
            "pant_color": self._get_dominant_color(lower_region),
            "shoe_color": self._get_dominant_color(feet_region),
            "mask": head_info["mask"],
            "mask_color": head_info["mask_color"],
            "spectacles": head_info["spectacles"],
            "wristband": body_marks["wristband"],
            "tattoo": body_marks["tattoo"],
            "estimated_height": self.estimate_height(bbox_height_px)
        }