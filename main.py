from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from skimage.transform import ThinPlateSplineTransform, warp


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Face Reshape V1", layout="wide")

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_PATH = Path("models/face_landmarker.task")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# =========================================================
# LANDMARK GROUPS
# =========================================================
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109
]

LEFT_BROW = [70, 63, 105, 66, 107, 52, 65, 55]
RIGHT_BROW = [336, 296, 334, 293, 300, 282, 295, 285]

# Brow support groups for smoother eyebrow-height behavior
LEFT_BROW_INNER = [107, 66, 105]
LEFT_BROW_OUTER = [70, 63, 52]
RIGHT_BROW_INNER = [336, 296, 334]
RIGHT_BROW_OUTER = [300, 293, 282]

LEFT_EYE = [33, 160, 158, 133, 153, 144, 159, 145]
RIGHT_EYE = [362, 385, 387, 263, 373, 380, 386, 374]

LEFT_EYE_OUTER = [33, 130, 246, 161, 160, 159, 158, 157, 173]
RIGHT_EYE_OUTER = [263, 359, 466, 388, 387, 386, 385, 384, 398]

NOSE_BRIDGE = [6, 197, 195, 5, 4]
NOSE_TIP = [1, 2, 19, 94]
NOSE_LEFT = [64, 98, 97, 2, 5]
NOSE_RIGHT = [294, 327, 326, 2, 5]

MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
MOUTH_CORNERS = [61, 291]

# Lip-specific points
UPPER_LIP_CENTER = [0]
LOWER_LIP_CENTER = [17]
CUPID_BOW = [39, 0, 269]
UPPER_LIP_BULGE = [40, 39, 37, 0, 267, 269, 270]
LOWER_LIP_BULGE = [91, 181, 84, 17, 314, 405, 321]

CHIN_CORE = [152, 175, 199, 200, 17]
CHIN_ARC = [148, 176, 152, 377, 400]

JAW_LEFT = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
JAW_RIGHT = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]

CHEEK_LEFT = [50, 101, 205, 187, 147, 116, 117]
CHEEK_RIGHT = [280, 330, 425, 411, 376, 345, 346]

FOREHEAD = [10, 67, 109, 338, 297, 151, 9]
STABLE_MID = [6, 197, 195, 5, 4, 1, 168]
STABLE_ALL = list(dict.fromkeys(FOREHEAD + STABLE_MID))


# =========================================================
# DATA CLASSES
# =========================================================
@dataclass
class Landmarks:
    px: np.ndarray
    xyz: np.ndarray
    w: int
    h: int


# =========================================================
# DEFAULTS
# =========================================================
DEFAULTS = {
    "eyebrow_height": 0.0,
    "chin_length": 0.0,
    "mouth_width": 0.0,
    "nose_width": 0.0,
    "jaw_width": 0.0,
    "face_width": 0.0,
    "eye_size": 0.0,
    "eye_distance": 0.0,
    "lip_size": 0.0,
    "lip_width": 0.0,
    "lip_height": 0.0,
    "lip_peak": 0.0,
    "show_guides": False,
    "show_mask": False,
    "show_controls": False,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_all() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


# =========================================================
# HELPERS
# =========================================================
def unique(seq: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(seq))


def pts_center(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0)


def scale_group(points: np.ndarray, indices: list[int], sx: float, sy: float | None = None) -> None:
    if sy is None:
        sy = sx
    center = pts_center(points[indices])
    points[indices, 0] = center[0] + (points[indices, 0] - center[0]) * sx
    points[indices, 1] = center[1] + (points[indices, 1] - center[1]) * sy


def move_group(points: np.ndarray, indices: list[int], dx: float = 0.0, dy: float = 0.0) -> None:
    points[indices, 0] += dx
    points[indices, 1] += dy


def clip_points(points: np.ndarray, w: int, h: int) -> np.ndarray:
    out = points.copy()
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    return out


def build_border_anchors(w: int, h: int, margin: int = 6) -> np.ndarray:
    return np.array([
        [margin, margin],
        [w * 0.25, margin],
        [w * 0.50, margin],
        [w * 0.75, margin],
        [w - 1 - margin, margin],
        [w - 1 - margin, h * 0.25],
        [w - 1 - margin, h * 0.50],
        [w - 1 - margin, h * 0.75],
        [w - 1 - margin, h - 1 - margin],
        [w * 0.75, h - 1 - margin],
        [w * 0.50, h - 1 - margin],
        [w * 0.25, h - 1 - margin],
        [margin, h - 1 - margin],
        [margin, h * 0.75],
        [margin, h * 0.50],
        [margin, h * 0.25],
    ], dtype=np.float32)


def build_outer_background_anchors(w: int, h: int, lm_px: np.ndarray, spacing: int = 42) -> np.ndarray:
    spacing = max(24, int(spacing))
    hull = cv2.convexHull(np.round(lm_px[FACE_OVAL]).astype(np.float32))

    yy, xx = np.mgrid[0:h:spacing, 0:w:spacing]
    grid = np.vstack([xx.ravel(), yy.ravel()]).T.astype(np.float32)

    anchors = []
    for pt in grid:
        dist = cv2.pointPolygonTest(hull, (float(pt[0]), float(pt[1])), True)
        if dist < -18:
            anchors.append(pt)

    if not anchors:
        return np.empty((0, 2), dtype=np.float32)

    return np.asarray(anchors, dtype=np.float32)


def make_region_mask(lm: Landmarks, indices: list[int], pad: int = 22, blur: int = 41) -> np.ndarray:
    mask = np.zeros((lm.h, lm.w), dtype=np.uint8)
    pts = np.round(lm.px[indices]).astype(np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    if pad > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    if blur % 2 == 0:
        blur += 1

    alpha = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32) / 255.0
    return np.clip(alpha, 0.0, 1.0)


def make_lower_face_mask(lm: Landmarks, pad: int = 26, blur: int = 51) -> np.ndarray:
    mask = np.zeros((lm.h, lm.w), dtype=np.uint8)
    oval = np.round(lm.px[FACE_OVAL]).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(oval), 255)

    chin_y = float(np.mean(lm.px[CHIN_CORE, 1]))
    cutoff = int(round(chin_y - 0.26 * lm.h))
    mask[:max(cutoff, 0), :] = 0

    if pad > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    if blur % 2 == 0:
        blur += 1

    alpha = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32) / 255.0
    return np.clip(alpha, 0.0, 1.0)


def make_full_face_mask(lm: Landmarks, pad: int = 18, blur: int = 41) -> np.ndarray:
    mask = np.zeros((lm.h, lm.w), dtype=np.uint8)
    oval = np.round(lm.px[FACE_OVAL]).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(oval), 255)

    if pad > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    if blur % 2 == 0:
        blur += 1

    alpha = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32) / 255.0
    return np.clip(alpha, 0.0, 1.0)


def combine_masks(masks: list[np.ndarray], h: int, w: int) -> np.ndarray:
    if not masks:
        return np.zeros((h, w), dtype=np.float32)
    out = np.zeros((h, w), dtype=np.float32)
    for m in masks:
        out = np.maximum(out, m)
    return np.clip(out, 0.0, 1.0)


# =========================================================
# MEDIAPIPE
# =========================================================
def download_model() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


@st.cache_resource
def load_landmarker():
    download_model()

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def detect_landmarks(image_rgb: np.ndarray, landmarker) -> Landmarks | None:
    h, w = image_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    face = result.face_landmarks[0]
    xyz = np.array([[lm.x, lm.y, lm.z] for lm in face], dtype=np.float32)
    px = np.empty((xyz.shape[0], 2), dtype=np.float32)
    px[:, 0] = xyz[:, 0] * w
    px[:, 1] = xyz[:, 1] * h

    return Landmarks(px=px, xyz=xyz, w=w, h=h)


# =========================================================
# DEFORMATION LOGIC
# =========================================================
def apply_deformations(lm: Landmarks) -> tuple[np.ndarray, np.ndarray]:
    moved = lm.px.copy()
    masks: list[np.ndarray] = []

    # 1) Eyebrow Height - fixed
    eyebrow_height = float(st.session_state.eyebrow_height)
    if abs(eyebrow_height) > 1e-6:
        dy = -1.55 * eyebrow_height
        tilt_support = -0.35 * eyebrow_height

        move_group(moved, LEFT_BROW, dy=dy)
        move_group(moved, RIGHT_BROW, dy=dy)

        move_group(moved, LEFT_BROW_INNER, dy=tilt_support)
        move_group(moved, LEFT_BROW_OUTER, dy=0.18 * dy)
        move_group(moved, RIGHT_BROW_INNER, dy=tilt_support)
        move_group(moved, RIGHT_BROW_OUTER, dy=0.18 * dy)

        move_group(moved, LEFT_EYE_OUTER, dy=0.10 * dy)
        move_group(moved, RIGHT_EYE_OUTER, dy=0.10 * dy)

        masks.append(make_region_mask(
            lm,
            unique(
                LEFT_BROW + RIGHT_BROW +
                LEFT_BROW_INNER + LEFT_BROW_OUTER +
                RIGHT_BROW_INNER + RIGHT_BROW_OUTER +
                LEFT_EYE_OUTER + RIGHT_EYE_OUTER +
                FOREHEAD
            ),
            pad=24,
            blur=39,
        ))

    # 2) Chin Length
    chin_length = float(st.session_state.chin_length)
    if abs(chin_length) > 1e-6:
        move_group(moved, CHIN_CORE, dy=1.9 * chin_length)
        move_group(moved, CHIN_ARC, dy=1.2 * chin_length)
        move_group(moved, JAW_LEFT, dy=0.45 * chin_length)
        move_group(moved, JAW_RIGHT, dy=0.45 * chin_length)

        jaw_center_x = float(np.mean(lm.px[JAW_LEFT + JAW_RIGHT, 0]))
        jaw_scale = 1.0 + 0.004 * chin_length
        moved[JAW_LEFT, 0] = jaw_center_x + (moved[JAW_LEFT, 0] - jaw_center_x) * jaw_scale
        moved[JAW_RIGHT, 0] = jaw_center_x + (moved[JAW_RIGHT, 0] - jaw_center_x) * jaw_scale

        masks.append(make_lower_face_mask(lm, pad=28, blur=51))

    # 3) Mouth Width
    mouth_width = float(st.session_state.mouth_width)
    if abs(mouth_width) > 1e-6:
        mouth_center_x = float(np.mean(lm.px[MOUTH_OUTER, 0]))
        scale = 1.0 + 0.018 * mouth_width

        moved[MOUTH_OUTER, 0] = mouth_center_x + (moved[MOUTH_OUTER, 0] - mouth_center_x) * scale
        moved[UPPER_LIP, 0] = mouth_center_x + (moved[UPPER_LIP, 0] - mouth_center_x) * scale
        moved[LOWER_LIP, 0] = mouth_center_x + (moved[LOWER_LIP, 0] - mouth_center_x) * scale

        moved[MOUTH_CORNERS[0], 0] -= 0.8 * mouth_width
        moved[MOUTH_CORNERS[1], 0] += 0.8 * mouth_width

        masks.append(make_region_mask(
            lm,
            unique(MOUTH_OUTER + UPPER_LIP + LOWER_LIP + MOUTH_CORNERS),
            pad=26,
            blur=45,
        ))

    # 4) Nose Width
    nose_width = float(st.session_state.nose_width)
    if abs(nose_width) > 1e-6:
        nose_group = unique(NOSE_LEFT + NOSE_RIGHT + NOSE_TIP + NOSE_BRIDGE)
        nose_center_x = float(np.mean(lm.px[nose_group, 0]))
        scale = 1.0 + 0.015 * nose_width

        moved[NOSE_LEFT, 0] = nose_center_x + (moved[NOSE_LEFT, 0] - nose_center_x) * scale
        moved[NOSE_RIGHT, 0] = nose_center_x + (moved[NOSE_RIGHT, 0] - nose_center_x) * scale
        moved[NOSE_TIP, 0] = nose_center_x + (moved[NOSE_TIP, 0] - nose_center_x) * (1.0 + 0.004 * nose_width)

        masks.append(make_region_mask(lm, nose_group, pad=22, blur=41))

    # 5) Jaw Width
    jaw_width = float(st.session_state.jaw_width)
    if abs(jaw_width) > 1e-6:
        jaw_center_x = float(np.mean(lm.px[JAW_LEFT + JAW_RIGHT, 0]))
        scale = 1.0 + 0.016 * jaw_width

        moved[JAW_LEFT, 0] = jaw_center_x + (moved[JAW_LEFT, 0] - jaw_center_x) * scale
        moved[JAW_RIGHT, 0] = jaw_center_x + (moved[JAW_RIGHT, 0] - jaw_center_x) * scale

        masks.append(make_lower_face_mask(lm, pad=26, blur=49))

    # 6) Face Width
    face_width = float(st.session_state.face_width)
    if abs(face_width) > 1e-6:
        face_side_indices = unique(JAW_LEFT + JAW_RIGHT + CHEEK_LEFT + CHEEK_RIGHT)
        face_center_x = float(np.mean(lm.px[FACE_OVAL, 0]))
        scale = 1.0 + 0.012 * face_width

        moved[face_side_indices, 0] = face_center_x + (moved[face_side_indices, 0] - face_center_x) * scale

        masks.append(make_full_face_mask(lm, pad=16, blur=41))

    # 7) Eye Size
    eye_size = float(st.session_state.eye_size)
    if abs(eye_size) > 1e-6:
        sx = 1.0 + 0.012 * eye_size
        sy = 1.0 + 0.018 * eye_size

        scale_group(moved, LEFT_EYE_OUTER, sx=sx, sy=sy)
        scale_group(moved, RIGHT_EYE_OUTER, sx=sx, sy=sy)

        masks.append(make_region_mask(
            lm,
            unique(LEFT_EYE_OUTER + RIGHT_EYE_OUTER + LEFT_BROW + RIGHT_BROW),
            pad=20,
            blur=39,
        ))

    # 8) Eye Distance
    eye_distance = float(st.session_state.eye_distance)
    if abs(eye_distance) > 1e-6:
        move_group(moved, LEFT_EYE_OUTER, dx=-0.6 * eye_distance)
        move_group(moved, RIGHT_EYE_OUTER, dx=0.6 * eye_distance)
        move_group(moved, LEFT_BROW, dx=-0.35 * eye_distance)
        move_group(moved, RIGHT_BROW, dx=0.35 * eye_distance)

        masks.append(make_region_mask(
            lm,
            unique(LEFT_EYE_OUTER + RIGHT_EYE_OUTER + LEFT_BROW + RIGHT_BROW),
            pad=22,
            blur=41,
        ))

    # 9) Lip Size
    lip_size = float(st.session_state.lip_size)
    if abs(lip_size) > 1e-6:
        sx = 1.0 + 0.010 * lip_size
        sy = 1.0 + 0.022 * lip_size

        scale_group(moved, MOUTH_OUTER, sx=sx, sy=sy)
        scale_group(moved, UPPER_LIP, sx=sx, sy=sy)
        scale_group(moved, LOWER_LIP, sx=sx, sy=sy)

        masks.append(make_region_mask(
            lm,
            unique(MOUTH_OUTER + UPPER_LIP + LOWER_LIP),
            pad=24,
            blur=43,
        ))

    # 10) Lip Width
    lip_width = float(st.session_state.lip_width)
    if abs(lip_width) > 1e-6:
        mouth_center_x = float(np.mean(moved[MOUTH_OUTER, 0]))
        scale = 1.0 + 0.016 * lip_width

        moved[MOUTH_OUTER, 0] = mouth_center_x + (moved[MOUTH_OUTER, 0] - mouth_center_x) * scale
        moved[UPPER_LIP, 0] = mouth_center_x + (moved[UPPER_LIP, 0] - mouth_center_x) * scale
        moved[LOWER_LIP, 0] = mouth_center_x + (moved[LOWER_LIP, 0] - mouth_center_x) * scale
        moved[MOUTH_CORNERS[0], 0] -= 0.55 * lip_width
        moved[MOUTH_CORNERS[1], 0] += 0.55 * lip_width

        masks.append(make_region_mask(
            lm,
            unique(MOUTH_OUTER + UPPER_LIP + LOWER_LIP + MOUTH_CORNERS),
            pad=24,
            blur=43,
        ))

    # 11) Lip Height
    lip_height = float(st.session_state.lip_height)
    if abs(lip_height) > 1e-6:
        move_group(moved, UPPER_LIP_BULGE, dy=-0.45 * lip_height)
        move_group(moved, LOWER_LIP_BULGE, dy=0.45 * lip_height)
        move_group(moved, UPPER_LIP_CENTER, dy=-0.55 * lip_height)
        move_group(moved, LOWER_LIP_CENTER, dy=0.55 * lip_height)

        masks.append(make_region_mask(
            lm,
            unique(MOUTH_OUTER + UPPER_LIP + LOWER_LIP + UPPER_LIP_BULGE + LOWER_LIP_BULGE),
            pad=22,
            blur=41,
        ))

    # 12) Cupid Peak
    lip_peak = float(st.session_state.lip_peak)
    if abs(lip_peak) > 1e-6:
        move_group(moved, CUPID_BOW, dy=-0.55 * lip_peak)
        move_group(moved, [39], dx=-0.18 * lip_peak)
        move_group(moved, [269], dx=0.18 * lip_peak)

        masks.append(make_region_mask(
            lm,
            unique(UPPER_LIP + CUPID_BOW + MOUTH_OUTER),
            pad=20,
            blur=39,
        ))

    moved = clip_points(moved, lm.w, lm.h)
    alpha = combine_masks(masks, lm.h, lm.w)
    return moved, alpha


# =========================================================
# TPS
# =========================================================
def build_control_points(lm: Landmarks, moved_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    control_face = unique(
        FACE_OVAL +
        LEFT_BROW + RIGHT_BROW +
        LEFT_BROW_INNER + LEFT_BROW_OUTER + RIGHT_BROW_INNER + RIGHT_BROW_OUTER +
        LEFT_EYE_OUTER + RIGHT_EYE_OUTER +
        NOSE_LEFT + NOSE_RIGHT + NOSE_TIP + NOSE_BRIDGE +
        MOUTH_OUTER + UPPER_LIP + LOWER_LIP + MOUTH_CORNERS +
        UPPER_LIP_CENTER + LOWER_LIP_CENTER + CUPID_BOW +
        CHIN_CORE + CHIN_ARC + JAW_LEFT + JAW_RIGHT +
        CHEEK_LEFT + CHEEK_RIGHT +
        STABLE_ALL
    )

    src_face = lm.px[control_face]
    dst_face = moved_px[control_face]

    border = build_border_anchors(lm.w, lm.h, margin=6)
    outer_bg = build_outer_background_anchors(lm.w, lm.h, lm.px, spacing=max(lm.w, lm.h) // 18)

    src = np.vstack([src_face, border, outer_bg]).astype(np.float32)
    dst = np.vstack([dst_face, border, outer_bg]).astype(np.float32)

    return src, dst


def tps_warp(image_rgb: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    if src_pts.shape != dst_pts.shape:
        raise ValueError("src_pts and dst_pts must have the same shape.")
    if src_pts.shape[0] < 3:
        raise ValueError("Need at least 3 control points.")

    h, w = image_rgb.shape[:2]

    tps = ThinPlateSplineTransform()
    ok = tps.estimate(dst_pts, src_pts)
    if not ok:
        raise RuntimeError("ThinPlateSplineTransform estimation failed.")

    warped = warp(
        image_rgb.astype(np.float32),
        inverse_map=tps,
        output_shape=(h, w),
        preserve_range=True,
        mode="edge",
    )
    return np.clip(warped, 0, 255).astype(np.uint8)


# =========================================================
# VIS
# =========================================================
def alpha_compose(original_rgb: np.ndarray, warped_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None].astype(np.float32)
    out = warped_rgb.astype(np.float32) * a + original_rgb.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_points(image_rgb: np.ndarray, points: np.ndarray, color=(0, 255, 0), radius: int = 1) -> np.ndarray:
    out = image_rgb.copy()
    for x, y in points:
        cv2.circle(out, (int(round(x)), int(round(y))), radius, color, -1)
    return out


def draw_controls(image_rgb: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, max_lines: int = 250) -> np.ndarray:
    out = image_rgb.copy()
    n = min(len(src_pts), max_lines)
    for s, d in zip(src_pts[:n], dst_pts[:n]):
        sx, sy = int(round(s[0])), int(round(s[1]))
        dx, dy = int(round(d[0])), int(round(d[1]))
        cv2.circle(out, (sx, sy), 1, (255, 0, 0), -1)
        cv2.circle(out, (dx, dy), 1, (0, 255, 255), -1)
        cv2.line(out, (sx, sy), (dx, dy), (255, 0, 255), 1)
    return out


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("Face Reshape V1")

    st.markdown("### Face")
    st.slider("Eyebrow Height", -10.0, 10.0, key="eyebrow_height")
    st.slider("Chin Length", -10.0, 10.0, key="chin_length")
    st.slider("Mouth Width", -10.0, 10.0, key="mouth_width")
    st.slider("Nose Width", -10.0, 10.0, key="nose_width")
    st.slider("Jaw Width", -10.0, 10.0, key="jaw_width")
    st.slider("Face Width", -10.0, 10.0, key="face_width")
    st.slider("Eye Size", -10.0, 10.0, key="eye_size")
    st.slider("Eye Distance", -10.0, 10.0, key="eye_distance")

    st.markdown("### Lips")
    st.slider("Lip Size", -10.0, 10.0, key="lip_size")
    st.slider("Lip Width", -10.0, 10.0, key="lip_width")
    st.slider("Lip Height", -10.0, 10.0, key="lip_height")
    st.slider("Cupid Peak", -10.0, 10.0, key="lip_peak")

    st.divider()
    st.checkbox("Show guides", key="show_guides")
    st.checkbox("Show mask", key="show_mask")
    st.checkbox("Show controls", key="show_controls")
    st.button("Reset", on_click=reset_all, use_container_width=True)


# =========================================================
# MAIN
# =========================================================
st.title("Frontal Face Reshape")
st.caption("MediaPipe Tasks + TPS. Frontal face only.")

uploaded = st.file_uploader("Upload a frontal face image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not decode image.")
        st.stop()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    max_w = 900
    h0, w0 = image_rgb.shape[:2]
    if w0 > max_w:
        scale = max_w / float(w0)
        image_rgb = cv2.resize(
            image_rgb,
            (max_w, int(round(h0 * scale))),
            interpolation=cv2.INTER_AREA,
        )

    try:
        landmarker = load_landmarker()
        lm = detect_landmarks(image_rgb, landmarker)

        if lm is None:
            st.error("No face detected.")
            st.stop()

        moved_px, alpha = apply_deformations(lm)

        if np.max(alpha) < 1e-6:
            warped_rgb = image_rgb.copy()
            final_rgb = image_rgb.copy()
            src_pts = np.empty((0, 2), dtype=np.float32)
            dst_pts = np.empty((0, 2), dtype=np.float32)
        else:
            src_pts, dst_pts = build_control_points(lm, moved_px)
            warped_rgb = tps_warp(image_rgb, src_pts, dst_pts)
            final_rgb = alpha_compose(image_rgb, warped_rgb, alpha)

        if st.session_state.show_guides:
            final_rgb = draw_points(final_rgb, moved_px, color=(0, 255, 0), radius=1)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original")
            st.image(image_rgb, use_container_width=True)

        with col2:
            st.markdown("### Simulated")
            st.image(final_rgb, use_container_width=True)

        with st.expander("Debug"):
            d1, d2, d3 = st.columns(3)

            with d1:
                st.markdown("**Mask**")
                if st.session_state.show_mask:
                    st.image((alpha * 255).astype(np.uint8), use_container_width=True)
                else:
                    st.info("Enable 'Show mask'.")

            with d2:
                st.markdown("**Warped**")
                st.image(warped_rgb, use_container_width=True)

            with d3:
                st.markdown("**Controls**")
                if len(src_pts) > 0 and st.session_state.show_controls:
                    st.image(draw_controls(image_rgb, src_pts, dst_pts), use_container_width=True)
                else:
                    st.info("Enable 'Show controls'.")

    except Exception as e:
        st.exception(e)