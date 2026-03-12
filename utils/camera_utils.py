"""
camera_utils.py
---------------
Google Colab webcam capture helpers.

The Colab runtime runs in a browser-based environment where OpenCV's
``VideoCapture`` cannot directly access the user's webcam.  This module
bridges that gap by injecting a small JavaScript snippet that reads from
``navigator.mediaDevices.getUserMedia``, encodes each frame as a base64
JPEG, and passes it back to Python via the ``output.eval_js`` mechanism.

Usage
-----
    from utils.camera_utils import CameraUtils

    cam = CameraUtils(width=640, height=480)
    frame = cam.capture_frame()          # np.ndarray (BGR)
    cam.display_frame(frame)             # renders inside Colab output cell
    cam.release()
"""

import base64
import io
import time
import threading
from typing import Optional

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# JavaScript helpers injected into the Colab front-end
# ---------------------------------------------------------------------------

_JS_START_CAMERA = """
(async () => {
  if (window._virtualTryOnStream) return 'already_started';
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: %d, height: %d }
  });
  window._virtualTryOnStream = stream;
  const video = document.createElement('video');
  video.srcObject = stream;
  video.autoplay = true;
  video.playsInline = true;
  await new Promise(r => { video.onloadedmetadata = r; });
  video.play();
  window._virtualTryOnVideo = video;
  return 'started';
})()
"""

_JS_CAPTURE_FRAME = """
(async () => {
  const video = window._virtualTryOnVideo;
  if (!video) return null;
  const canvas = document.createElement('canvas');
  canvas.width  = video.videoWidth  || %d;
  canvas.height = video.videoHeight || %d;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.85);
})()
"""

_JS_STOP_CAMERA = """
(async () => {
  if (!window._virtualTryOnStream) return 'not_running';
  window._virtualTryOnStream.getTracks().forEach(t => t.stop());
  delete window._virtualTryOnStream;
  delete window._virtualTryOnVideo;
  return 'stopped';
})()
"""


class CameraUtils:
    """Manages webcam access inside a Google Colab notebook.

    Parameters
    ----------
    width : int
        Requested camera capture width in pixels.
    height : int
        Requested camera capture height in pixels.
    """

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._running = False
        self._last_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Colab-specific API
    # ------------------------------------------------------------------

    def start(self) -> str:
        """Inject the JS camera start snippet into the Colab front-end.

        Returns a status string ``'started'`` or ``'already_started'``.
        """
        try:
            from google.colab.output import eval_js  # type: ignore
        except ImportError:
            raise EnvironmentError(
                "CameraUtils.start() requires a Google Colab environment."
            )
        js = _JS_START_CAMERA % (self.width, self.height)
        status = eval_js(js)
        self._running = True
        return status

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the webcam via JavaScript.

        Returns
        -------
        np.ndarray or None
            BGR frame (uint8), or *None* if capture failed.
        """
        try:
            from google.colab.output import eval_js  # type: ignore
        except ImportError:
            raise EnvironmentError(
                "CameraUtils.capture_frame() requires a Google Colab environment."
            )
        js = _JS_CAPTURE_FRAME % (self.width, self.height)
        data_url = eval_js(js)
        if not data_url:
            return None
        frame = self._data_url_to_bgr(data_url)
        with self._lock:
            self._last_frame = frame
        return frame

    def release(self) -> str:
        """Stop the camera stream and release browser resources.

        Returns a status string ``'stopped'`` or ``'not_running'``.
        """
        try:
            from google.colab.output import eval_js  # type: ignore
        except ImportError:
            raise EnvironmentError(
                "CameraUtils.release() requires a Google Colab environment."
            )
        status = eval_js(_JS_STOP_CAMERA)
        self._running = False
        return status

    # ------------------------------------------------------------------
    # Display helpers (work in any IPython environment)
    # ------------------------------------------------------------------

    @staticmethod
    def display_frame(frame: np.ndarray, title: str = "") -> None:
        """Render a BGR frame inline in the current output cell.

        Parameters
        ----------
        frame : np.ndarray
            BGR image to display.
        title : str, optional
            Text label printed above the image.
        """
        from IPython.display import display, Image as IPImage  # type: ignore

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        if title:
            from IPython.display import HTML  # type: ignore
            display(HTML(f"<b>{title}</b>"))
        display(IPImage(data=buf.getvalue()))

    @staticmethod
    def frame_to_base64(frame: np.ndarray) -> str:
        """Encode a BGR frame as a base64 JPEG string (no data-URL prefix)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def save_frame(frame: np.ndarray, path: str) -> None:
        """Save *frame* (BGR) to *path* as a PNG file."""
        cv2.imwrite(path, frame)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _data_url_to_bgr(data_url: str) -> np.ndarray:
        """Convert a ``data:image/jpeg;base64,...`` string to a BGR ndarray."""
        header, encoded = data_url.split(",", 1)
        decoded = base64.b64decode(encoded)
        pil = Image.open(io.BytesIO(decoded)).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        return bgr

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True if the camera has been started and not yet released."""
        return self._running

    @property
    def last_frame(self) -> Optional[np.ndarray]:
        """Most recently captured frame, or *None*."""
        with self._lock:
            return self._last_frame

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.release()
