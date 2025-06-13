import os
import sys
import subprocess
import json
import tempfile
import uuid
from typing import Optional, Dict, Any, List

import joblib
import numpy as np


class PatchcoreRunner:
    """Lightweight wrapper around a *trained* PatchCore model directory.
    It provides the `predict()` interface expected by `model_creator_app.py`.
    Internally it shells out to the official CLI (`patchcore.predict`) so we
    don’t have to re‑implement scoring logic or worry about the FAISS index
    pickle‑ability.

    Parameters
    ----------
    model_dir : str
        Path that contains the PatchCore artefacts created by
        ``patchcore.train`` (e.g. ``config.yml``, ``memory_bank.faiss``,
        ``backbone.pt`` …).
    ref_scores : Optional[np.ndarray]
        Raw anomaly scores of *normal* training images.  When present, they are
        used to map a new sample’s score to a percentile, giving an intuitive
        “how anomalous?” number.
    """

    def __init__(self, model_dir: str, ref_scores: Optional[np.ndarray] = None):
        self.model_dir = model_dir
        self.ref_scores: Optional[np.ndarray] = None
        if ref_scores is not None and len(ref_scores):
            self.ref_scores = np.sort(ref_scores)  # ascending for np.searchsorted

    # ------------------------------------------------------------------
    #                        internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _run_patchcore_predict(model_dir: str, img_path: str) -> float:
        """Run the official CLI on *img_path* and return the image‑level score."""
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable,
                "-m",
                "patchcore.predict",
                "--model_dir",
                model_dir,
                "--image",
                img_path,
                "--out_dir",
                tmp,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Find *any* json file – PatchCore writes exactly one
            json_file = next(
                (os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".json")),
                None,
            )
            if json_file is None:
                raise RuntimeError("patchcore.predict did not create a JSON output")
            with open(json_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Official schema: {"image": "…", "anomaly_score": float, …}
            return float(data.get("anomaly_score", data.get("score", 0.0)))

    # ------------------------------------------------------------------
    #                           public API
    # ------------------------------------------------------------------
    def predict(self, img_path: str) -> Dict[str, Any]:
        score = self._run_patchcore_predict(self.model_dir, img_path)

        # Percentile mapping if we have reference scores
        if self.ref_scores is not None and len(self.ref_scores):
            # How many reference scores are <= current score?
            pct = 100.0 * np.searchsorted(self.ref_scores, score, side="right") / len(self.ref_scores)
        else:
            # Fallback: linear heuristic (PatchCore scores are already >0)
            pct = max(0.0, min(100.0, score * 100.0))

        # Simple threshold: flag as defective if score in top 5 % of normals
        is_defective = pct > 95.0
        return {
            "anomaly_score": score,
            "prediction": -1 if is_defective else 1,
            "is_defective": is_defective,
            "anomaly_percentage": pct,
            "status": "defective" if is_defective else "normal",
        }


class ModelTrainer:
    """PatchCore‑based *drop‑in* replacement for ``ModelTrainer`` used in
    *model_creator_app.py*.  All public method names and signatures are kept so
    the Streamlit UI works **unchanged**:

        >>> trainer = ModelTrainer()
        >>> model = trainer.train_autoencoder("data/train_good", contamination=0.01)
        >>> path = trainer.save_model(model, "models/patchcore_model.pkl")

    External requirements (GPU highly recommended):

    ```bash
    pip install patchcore-backend pytorch-lightning faiss-cpu
    ```
    """

    def __init__(self):
        # Will be set after training / loading
        self.model_dir: Optional[str] = None
        self.runner: Optional[PatchcoreRunner] = None
        self.train_scores: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    #                    feature extraction (NOT USED)
    # ------------------------------------------------------------------
    def extract_features(self, img_path: str):  # noqa: D401, E501 – kept for compatibility
        """PatchCore works on image *patches* and can’t expose a single 1 × N
        feature vector in a meaningful way.  If some part of your code calls
        this, it will raise immediately so the mismatch surfaces early."""
        raise NotImplementedError(
            "`extract_features()` is not supported in the PatchCore backend."
        )

    # ------------------------------------------------------------------
    #                             training
    # ------------------------------------------------------------------
    def _run_patchcore_train(
        self,
        data_dir: str,
        backbone: str = "resnet18",
        input_size: int = 224,
        mask_augment: float = 0.0,
    ) -> str:
        """Launch the official `patchcore.train` CLI and return the new model dir."""
        model_dir = os.path.join("models", f"patchcore_{uuid.uuid4().hex[:8]}")
        os.makedirs(model_dir, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "patchcore.train",
            "--data_dir",
            data_dir,
            "--save_dir",
            model_dir,
            "--backbone",
            backbone,
            "--input_size",
            str(input_size),
            "--mask_augment",
            str(mask_augment),
        ]
        subprocess.run(cmd, check=True)
        return model_dir

    def train_autoencoder(self, good_images: str, contamination: float = 0.1):  # noqa: D401, E501 – signature must match
        """Train PatchCore on *good_images* and return a ``PatchcoreRunner``.

        Notes
        -----
        * ``good_images`` **must** be a folder path (PatchCore expects a data
          directory).  Passing a list will raise ``ValueError``.
        * The *contamination* argument is accepted for API compatibility but is
          **ignored** – PatchCore does not use it.
        """
        if not isinstance(good_images, str):
            raise ValueError("PatchCore trainer expects *good_images* as a folder path.")

        # 1) train – this may take minutes to hours depending on data & GPU
        self.model_dir = self._run_patchcore_train(good_images)

        # 2) build the runtime wrapper
        self.runner = PatchcoreRunner(self.model_dir)

        # 3) Compute reference scores for percentile mapping (optional but nice)
        img_paths: List[str] = [
            os.path.join(good_images, f)
            for f in os.listdir(good_images)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
        ]
        scores: List[float] = []
        for p in img_paths:
            try:
                scores.append(self.runner._run_patchcore_predict(self.model_dir, p))
            except Exception as err:  # pragma: no cover – best‑effort
                print(f"[PatchCore] warning: failed to score {p}: {err}")
        self.train_scores = np.asarray(scores, dtype=np.float32)
        if len(self.train_scores):
            self.runner.ref_scores = self.train_scores
        return self.runner

    # ------------------------------------------------------------------
    #                            persistence
    # ------------------------------------------------------------------
    def save_model(self, model, path: str):  # noqa: D401 – signature must match
        """Persist ``PatchcoreRunner`` via :pymod:`joblib` + side‑car ``_ref_scores.npy``.

        We don’t try to pickle the full FAISS index; instead the wrapper just
        stores the *path* to the model directory, which is perfectly safe to
        serialise.  Make sure you keep the directory next to the «.pkl» file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not path.endswith(".pkl"):
            path += ".pkl"
        joblib.dump(model, path)

        if self.train_scores is not None and len(self.train_scores):
            ref_path = os.path.splitext(path)[0] + "_ref_scores.npy"
            np.save(ref_path, self.train_scores)
        return path

    def load_model(self, path: str):  # noqa: D401 – signature must match
        self.runner = joblib.load(path)
        ref_path = os.path.splitext(path)[0] + "_ref_scores.npy"
        if os.path.exists(ref_path):
            self.runner.ref_scores = np.load(ref_path)
        return self.runner

    # ------------------------------------------------------------------
    #                            inference
    # ------------------------------------------------------------------
    def predict(self, img_path: str):  # noqa: D401 – signature must match
        if self.runner is None:
            raise RuntimeError("Model not loaded or trained yet — call train_autoencoder() or load_model() first.")
        return self.runner.predict(img_path)
