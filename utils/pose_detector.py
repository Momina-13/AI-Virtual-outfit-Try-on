"""
pose_detector.py  (FIXED for MediaPipe 0.10+)
----------------------------------------------
MediaPipe removed mp.solutions.pose in 0.10+.
This version uses the new mediapipe.tasks API with a fallback
to the legacy API for older installs.
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    _mp_version = tuple(int(x) for x in mp.__version__.split(".")[:2])
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    _mp_version = (0, 0)


# ── Helper: download the pose-landmarker model if needed ──────────────────────
def _ensure_model(model_path: str = "/tmp/pose_landmarker.task") -> str:
    import os, urllib.request
    if not os.path.exists(model_path):
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/latest/"
            "pose_landmarker_lite.task"
        )
        print(f"Downloading MediaPipe pose model → {model_path} …")
        urllib.request.urlretrieve(url, model_path)
        print("✅ Model downloaded.")
    return model_path


class PoseDetector:
    """
    Unified pose detector that works with both:
      • MediaPipe < 0.10  (legacy mp.solutions.pose)
      • MediaPipe ≥ 0.10  (new mediapipe.tasks PoseLandmarker)
    """

    # Landmark indices (same in both APIs)
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP       = 23
    RIGHT_HIP      = 24
    LEFT_ELBOW     = 13
    RIGHT_ELBOW    = 14
    LEFT_WRIST     = 15
    RIGHT_WRIST    = 16

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("mediapipe is required: pip install mediapipe")

        self._conf_detect  = min_detection_confidence
        self._conf_track   = min_tracking_confidence
        self._use_new_api  = _mp_version >= (0, 10)

        if self._use_new_api:
            self._init_new_api()
        else:
            self._init_legacy_api()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_new_api(self):
        """MediaPipe Tasks API (0.10+)."""
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        model_path = _ensure_model()
        base_opts  = mp_tasks.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            min_pose_detection_confidence=self._conf_detect,
            min_tracking_confidence=self._conf_track,
            output_segmentation_masks=False,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
        # For drawing we still use the drawing_utils from solutions if available
        try:
            self._mp_drawing        = mp.solutions.drawing_utils
            self._mp_drawing_styles = mp.solutions.drawing_styles
            self._mp_pose_conns     = mp.solutions.pose.POSE_CONNECTIONS
        except AttributeError:
            self._mp_drawing = None

    def _init_legacy_api(self):
        """Legacy MediaPipe API (< 0.10)."""
        self._mp_pose           = mp.solutions.pose
        self._mp_drawing        = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._pose_model        = self._mp_pose.Pose(
            min_detection_confidence=self._conf_detect,
            min_tracking_confidence=self._conf_track,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray):
        """
        Run pose detection on a BGR frame.

        Returns
        -------
        landmarks : list or NormalizedLandmarkList or None
        annotated_frame : np.ndarray  (BGR, with skeleton if drawing available)
        """
        if self._use_new_api:
            return self._detect_new(frame)
        else:
            return self._detect_legacy(frame)

    def _detect_new(self, frame: np.ndarray):
        import mediapipe as mp
        from mediapipe.framework.formats import landmark_pb2

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result    = self._landmarker.detect(mp_image)
        annotated = frame.copy()

        if not result.pose_landmarks:
            return None, annotated

        landmarks_list = result.pose_landmarks[0]   # first person

        # Draw skeleton if drawing utils are available
        if self._mp_drawing:
            proto = landmark_pb2.NormalizedLandmarkList()
            for lm in landmarks_list:
                proto.landmark.add(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
            self._mp_drawing.draw_landmarks(
                annotated,
                proto,
                self._mp_pose_conns,
            )

        return landmarks_list, annotated

    def _detect_legacy(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose_model.process(rgb)
        rgb.flags.writeable = True

        annotated  = frame.copy()
        landmarks  = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks
            self._mp_drawing.draw_landmarks(
                annotated,
                landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        return landmarks, annotated

    # ── Measurements ──────────────────────────────────────────────────────────

    def get_body_measurements(self, landmarks, frame_shape: tuple) -> dict:
        """Convert normalised landmarks → pixel measurements."""
        h, w = frame_shape[:2]

        def px(idx):
            lm = landmarks[idx] if self._use_new_api else landmarks.landmark[idx]
            return int(lm.x * w), int(lm.y * h)

        ls = px(self.LEFT_SHOULDER)
        rs = px(self.RIGHT_SHOULDER)
        lh = px(self.LEFT_HIP)
        rh = px(self.RIGHT_HIP)

        shoulder_width = abs(rs[0] - ls[0])
        torso_height   = abs(((lh[1] + rh[1]) // 2) - ((ls[1] + rs[1]) // 2))
        center_x       = (ls[0] + rs[0]) // 2
        shoulder_y     = (ls[1] + rs[1]) // 2
        hip_y          = (lh[1] + rh[1]) // 2

        return {
            "shoulder_width": shoulder_width,
            "torso_height":   torso_height,
            "center_x":       center_x,
            "shoulder_y":     shoulder_y,
            "hip_y":          hip_y,
            "left_shoulder":  ls,
            "right_shoulder": rs,
            "left_hip":       lh,
            "right_hip":      rh,
        }

    def is_pose_visible(self, landmarks, min_visibility: float = 0.5) -> bool:
        """True if all key torso landmarks meet min_visibility."""
        if landmarks is None:
            return False
        key_indices = [
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
            self.LEFT_HIP,      self.RIGHT_HIP,
        ]
        for idx in key_indices:
            lm = landmarks[idx] if self._use_new_api else landmarks.landmark[idx]
            if getattr(lm, "visibility", 1.0) < min_visibility:
                return False
        return True

    def close(self) -> None:
        if self._use_new_api:
            try:
                self._landmarker.close()
            except Exception:
                pass
        else:
            try:
                self._pose_model.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
