"""
pose_detector.py
----------------
Pose detection utilities using MediaPipe Pose.

Usage
-----
    from utils.pose_detector import PoseDetector

    detector = PoseDetector(min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
    landmarks, annotated_frame = detector.detect(frame)
    measurements = detector.get_body_measurements(landmarks, frame.shape)
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:  # pragma: no cover – mediapipe may be absent in test envs
    MEDIAPIPE_AVAILABLE = False


class PoseDetector:
    """Wraps MediaPipe Pose for real-time body-landmark detection.

    Parameters
    ----------
    min_detection_confidence : float
        Minimum confidence for initial person detection (0.0–1.0).
    min_tracking_confidence : float
        Minimum confidence to consider a landmark tracked (0.0–1.0).
    """

    # MediaPipe landmark indices used by this application
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "mediapipe is required. Install it with: pip install mediapipe"
            )
        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._pose = self._mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray):
        """Run pose detection on *frame* (BGR uint8).

        Returns
        -------
        landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList or None
            Detected landmarks, or *None* if no pose was found.
        annotated_frame : np.ndarray
            Copy of *frame* with pose skeleton drawn on it.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()
        landmarks = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks
            self._mp_drawing.draw_landmarks(
                annotated,
                landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        return landmarks, annotated

    def get_body_measurements(
        self, landmarks, frame_shape: tuple
    ) -> dict:
        """Extract pixel-space body measurements from normalised landmarks.

        Parameters
        ----------
        landmarks : NormalizedLandmarkList
            Output from :meth:`detect`.
        frame_shape : tuple
            ``(height, width, channels)`` of the source frame.

        Returns
        -------
        dict with keys:
            ``shoulder_width``, ``torso_height``, ``center_x``,
            ``shoulder_y``, ``hip_y``, ``left_shoulder``,
            ``right_shoulder``, ``left_hip``, ``right_hip``
        """
        h, w = frame_shape[:2]
        lm = landmarks.landmark

        def px(idx):
            p = lm[idx]
            return int(p.x * w), int(p.y * h)

        ls = px(self.LEFT_SHOULDER)
        rs = px(self.RIGHT_SHOULDER)
        lh = px(self.LEFT_HIP)
        rh = px(self.RIGHT_HIP)

        shoulder_width = abs(rs[0] - ls[0])
        torso_height = abs(((lh[1] + rh[1]) // 2) - ((ls[1] + rs[1]) // 2))
        center_x = (ls[0] + rs[0]) // 2
        shoulder_y = (ls[1] + rs[1]) // 2
        hip_y = (lh[1] + rh[1]) // 2

        return {
            "shoulder_width": shoulder_width,
            "torso_height": torso_height,
            "center_x": center_x,
            "shoulder_y": shoulder_y,
            "hip_y": hip_y,
            "left_shoulder": ls,
            "right_shoulder": rs,
            "left_hip": lh,
            "right_hip": rh,
        }

    def is_pose_visible(self, landmarks, min_visibility: float = 0.5) -> bool:
        """Return *True* if all key torso landmarks meet *min_visibility*."""
        if landmarks is None:
            return False
        lm = landmarks.landmark
        key_indices = [
            self.LEFT_SHOULDER,
            self.RIGHT_SHOULDER,
            self.LEFT_HIP,
            self.RIGHT_HIP,
        ]
        return all(lm[i].visibility >= min_visibility for i in key_indices)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
