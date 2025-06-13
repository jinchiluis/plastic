import os
import json
import datetime as _dt
from typing import Union, List

import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
import joblib
from PIL import Image


class ModelRunner:
    """Load a trained Isolation‑Forest pipeline + MobileNetV2 feature extractor
    and produce anomaly predictions.  Every inference is automatically logged
    to ``logs/YYYY‑MM‑DD.jsonl``.

    **New in this version**
    -----------------------
    * ``anomaly_percentage`` is now the **percentile of the raw score wrt. the
      training distribution** (0 %= best training inliers, 100 %= worse than
      every training sample).  If reference scores are missing we fall back to
      the old heuristic so backwards compatibility is preserved.
    * Optional ``ref_dir`` parameter lets you point to the folder that contains
      the *normal* training images; the runner builds its reference scores the
      first time it is needed and caches them in memory (and to ``.npy`` on
      disk for fast reload).
    * Public API (constructor, ``score_image()``, etc.) remains unchanged – all
      upstream apps keep working.
    """

    # -------------------------- configuration defaults -------------------------
    LOG_DIR = "logs"
    ENABLE_LOGGING = True
    REF_SCORES_CACHE = "_ref_scores.npy"   # saved right next to the model file

    # -------------------------------------------------------------------------
    def __init__(self, model_path: str, *, enable_logging: bool = None,
                 log_dir: str | None = None, ref_dir: str | None = None):
        # 1) Load the Isolation‑Forest (or Pipeline)
        self.model_path = model_path
        self.model = self._load_model(model_path)

        # 2) Feature extractor
        base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
        self.feature_extractor = Model(inputs=base.input, outputs=base.output)

        # 3) Logging
        self._enable_logging = self.ENABLE_LOGGING if enable_logging is None else enable_logging
        self._log_dir = log_dir or self.LOG_DIR
        if self._enable_logging:
            os.makedirs(self._log_dir, exist_ok=True)

        # 4) Reference score distribution (for percentile mapping)
        self._ref_scores: np.ndarray | None = None
        self._ref_dir = ref_dir  # folder with *normal* training images (optional)
        self._maybe_load_cached_ref_scores()

    # -------------------------------------------------------------------------
    #                       persistence & initialisation
    # -------------------------------------------------------------------------
    def _load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        try:
            return joblib.load(path)
        except Exception as exc:
            raise RuntimeError(f"Cannot load model: {exc}")

    def _maybe_load_cached_ref_scores(self):
        cache_path = os.path.splitext(self.model_path)[0] + self.REF_SCORES_CACHE
        if os.path.exists(cache_path):
            try:
                self._ref_scores = np.load(cache_path)
            except Exception:
                self._ref_scores = None  # ignore corrupted cache

    # -------------------------------------------------------------------------
    #                          feature extraction helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _preprocess_pil(img: Image.Image) -> np.ndarray:
        img = img.convert("RGB").resize((224, 224))
        arr = image.img_to_array(img)[None, ...]
        return preprocess_input(arr)

    def extract_features(self, img_path: str | Image.Image) -> np.ndarray:
        if isinstance(img_path, Image.Image):
            arr = self._preprocess_pil(img_path)
        else:
            img = Image.open(img_path)
            arr = self._preprocess_pil(img)
        feats = self.feature_extractor.predict(arr, verbose=0)
        return feats.flatten()

    def _extract_features_from_folder(self, folder: str) -> np.ndarray:
        valid = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith(valid)]
        feats: List[np.ndarray] = []
        for fp in files:
            try:
                feats.append(self.extract_features(fp))
            except Exception:
                continue
        return np.asarray(feats)

    # -------------------------------------------------------------------------
    #                            logging utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def _now_iso() -> str:
        return _dt.datetime.now().isoformat(timespec="seconds")

    def _today_log_path(self) -> str:
        return os.path.join(self._log_dir, f"{_dt.date.today().isoformat()}.jsonl")

    def _log(self, payload: dict):
        if not self._enable_logging:
            return
        with open(self._today_log_path(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------------------
    #                         reference score handling
    # -------------------------------------------------------------------------
    def _ensure_ref_scores(self):
        """Build reference score distribution on first use if missing."""
        if self._ref_scores is not None:
            return
        if self._ref_dir and os.path.isdir(self._ref_dir):
            feats = self._extract_features_from_folder(self._ref_dir)
            if len(feats):
                self._ref_scores = self.model.score_samples(feats)
                # cache to disk for next run
                cache_path = os.path.splitext(self.model_path)[0] + self.REF_SCORES_CACHE
                try:
                    np.save(cache_path, self._ref_scores)
                except Exception:
                    pass  # non‑fatal

    # -------------------------------------------------------------------------
    #                               inference
    # -------------------------------------------------------------------------
    def score_image(self, image: Union[str, Image.Image]):
        # Accept path or PIL.Image and get temp path if needed
        temp_path = None
        if isinstance(image, str):
            img_path = image
        elif isinstance(image, Image.Image):
            temp_path = "_temp_score_image.jpg"
            image.save(temp_path)
            img_path = temp_path
        else:
            raise ValueError("image must be file path or PIL.Image")

        try:
            feats = self.extract_features(img_path).reshape(1, -1)
            raw_score = float(self.model.score_samples(feats)[0])   # larger is better
            label = int(self.model.predict(feats)[0])               # 1 / -1

            # ---- percentile‑based anomaly percentage -----------------------
            self._ensure_ref_scores()
            if self._ref_scores is not None and len(self._ref_scores):
                frac_worse = float((self._ref_scores < raw_score).mean())
                anomaly_pct = (1.0 - frac_worse) * 100.0
            else:
                # fallback heuristic if ref_scores missing
                anomaly_pct = min(100.0, max(0.0, -raw_score * 200.0))

            result = {
                "anomaly_score": raw_score,
                "prediction": label,
                "is_defective": label == -1,
                "anomaly_percentage": anomaly_pct,
                "status": "defective" if label == -1 else "normal",
                "interpretation": self._interpret(anomaly_pct),
            }

            # logging --------------------------------------------------------
            self._log({
                "timestamp": self._now_iso(),
                "image_path": img_path if isinstance(image, str) else "(in‑memory)",
                **result,
            })
            return result
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    # -------------------------------------------------------------------------
    #                      human readable interpretation
    # -------------------------------------------------------------------------
    @staticmethod
    def _interpret(pct: float) -> str:
        if pct < 20:  return "excellent match"
        if pct < 40:  return "good match"
        if pct < 60:  return "borderline"
        if pct < 80:  return "suspect"
        return "likely defective"

    # -------------------------------------------------------------------------
    #                         metadata helper (unchanged)
    # -------------------------------------------------------------------------
    def get_model_info(self):
        return {
            "model_path": self.model_path,
            "model_type": "Isolation Forest",
            "feature_extractor": "MobileNetV2",
            "feature_dimensions": 1280,
            "n_estimators": getattr(self.model, "n_estimators", "N/A"),
            "contamination": getattr(self.model, "contamination", "N/A"),
            "ref_scores": "present" if self._ref_scores is not None else "absent",
        }
