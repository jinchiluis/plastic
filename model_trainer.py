import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.ensemble import IsolationForest
import joblib
from PIL import Image


class ModelTrainer:
    """Original MobileNet‑based trainer with a small patch:
    `save_model()` now writes a side‑car file `<model>.pkl_ref_scores.npy`
    that stores the raw scores of all *normal* training images.  This allows
    ModelRunner to convert raw scores into percentiles without ever loading the
    training folder again.  No public method names or signatures were changed.
    """

    def __init__(self):
        base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
        self.feature_extractor = Model(inputs=base.input, outputs=base.output)
        self.isolation_forest = None
        self.train_scores = None   # filled after training

    # ------------------------------------------------------------------
    #                       feature extraction
    # ------------------------------------------------------------------
    def extract_features(self, img_path: str) -> np.ndarray:
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = image.img_to_array(img)[None, ...]
        arr = preprocess_input(arr)
        feats = self.feature_extractor.predict(arr, verbose=0)
        return feats.flatten()

    # ------------------------------------------------------------------
    #                           training
    # ------------------------------------------------------------------
    def train_autoencoder(self, good_images, contamination: float = 0.1):
        """Fit IsolationForest on *good_images* (folder path or list of paths)."""
        if isinstance(good_images, str):
            paths = [
                os.path.join(good_images, f)
                for f in os.listdir(good_images)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
            ]
        else:
            paths = list(good_images)

        X = np.asarray([self.extract_features(p) for p in paths])

        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            n_jobs=-1,
            random_state=42,
        )
        self.isolation_forest.fit(X)

        # ---- NEW: keep raw training scores for percentile mapping ---------
        self.train_scores = self.isolation_forest.score_samples(X)
        return self.isolation_forest

    # ------------------------------------------------------------------
    #                           persistence
    # ------------------------------------------------------------------
    def save_model(self, model, path: str):
        """Save IsolationForest and side‑car reference scores."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not path.endswith(".pkl"):
            path += ".pkl"
        joblib.dump(model, path)

        # ---- NEW: side‑car file for percentile calibration ---------------
        if self.train_scores is not None:
            ref_path = os.path.splitext(path)[0] + "_ref_scores.npy"
            np.save(ref_path, self.train_scores)
        return path

    def load_model(self, path: str):
        self.isolation_forest = joblib.load(path)
        return self.isolation_forest

    # ------------------------------------------------------------------
    #                           inference
    # ------------------------------------------------------------------
    def predict(self, img_path: str):
        if self.isolation_forest is None:
            raise RuntimeError("Model not loaded or trained yet")
        feat = self.extract_features(img_path).reshape(1, -1)
        score = float(self.isolation_forest.score_samples(feat)[0])
        label = int(self.isolation_forest.predict(feat)[0])

        # fallback heuristic – runner will replace with percentile mapping
        anomaly_pct = min(100.0, max(0.0, -score * 200.0))
        return {
            "anomaly_score": score,
            "prediction": label,
            "is_defective": label == -1,
            "anomaly_percentage": anomaly_pct,
            "status": "defective" if label == -1 else "normal",
        }
