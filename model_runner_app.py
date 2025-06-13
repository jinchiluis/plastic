import streamlit as st
import numpy as np
import cv2, glob, os
from datetime import datetime
from pathlib import Path
import tempfile
from PIL import Image
from model_runner import ModelRunner

# ---------------------------------------------------------------------
# helper: find newest .pkl in Models/
# ---------------------------------------------------------------------
def newest_model_in(folder: str = "Models") -> str:
    pkls = glob.glob(os.path.join(folder, "*.pkl"))
    if not pkls:
        return "models/forest.pkl"          # fallback
    newest = max(pkls, key=os.path.getmtime)
    return newest

# ---------------------------------------------------------------------
# Patch-saliency helper (unchanged apart from np.ptp)
# ---------------------------------------------------------------------
def patch_saliency(runner: ModelRunner, pil_img: Image.Image, grid: int = 14):
    """
    Blur-patch saliency: shows which regions push the Isolation-Forest score
    toward *defective*.  Uses unique temp files per patch (Windows-safe).
    """
    base_score = runner.score_image(pil_img)["anomaly_score"]

    arr = np.array(pil_img.resize((224, 224)))
    h, w = arr.shape[:2]
    ph, pw = h // grid, w // grid
    sal = np.zeros((grid, grid), np.float32)

    for i in range(grid):
        for j in range(grid):
            pert = arr.copy()
            y0, y1 = i * ph, (i + 1) * ph
            x0, x1 = j * pw, (j + 1) * pw
            region = pert[y0:y1, x0:x1]
            pert[y0:y1, x0:x1] = cv2.GaussianBlur(region, (9, 9), 0)

            # --- unique temp file -----------------------------------------
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                Image.fromarray(pert.astype(np.uint8)).save(tmp.name, "JPEG")
                score = runner.score_image(tmp.name)["anomaly_score"]
            os.remove(tmp.name)        # safe: file is closed
            # ----------------------------------------------------------------

            sal[i, j] = score - base_score        # +ve ⇒ patch hurts normality

    sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_CUBIC)
    sal = (sal - sal.min()) / (np.ptp(sal) + 1e-6)  # NumPy-2.0 safe
    heat = cv2.applyColorMap((sal * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(arr, 0.6, heat, 0.4, 0)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
st.set_page_config(page_title="Bag QA – Anomaly Checker", layout="centered")
st.title("👜 Bag QA – Isolation-Forest")

# ------------------------- sidebar -----------------------------------
with st.sidebar:
    st.header("Load model")

    default_path = newest_model_in()            # NEW line
    model_path = st.text_input("Path to trained model (.pkl)", default_path)

    if st.button("📦 Load Model"):
        st.session_state.runner = ModelRunner(model_path)
        st.success(f"Model loaded: {Path(model_path).name}")

# -------------------------- main pane --------------------------------
uploaded = st.file_uploader("Upload a bag photo", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original upload", width=320)

    if st.button("GO – Analyze Image"):
        runner: ModelRunner = st.session_state.runner
        res = runner.score_image(img)

        st.image(
            patch_saliency(runner, img),
            caption="Saliency overlay",
            width=320,
        )

        st.subheader("Prediction")
        if res["is_defective"]:
            st.error(f"DEFECTIVE – {res['anomaly_percentage']:.1f}% anomaly")
        else:
            st.success(f"OK – {res['anomaly_percentage']:.1f}% anomaly")

        label_val = res.get("label", res.get("prediction", "n/a"))
        with st.expander("📊 Technical details"):
            st.json(
                {
                    "Raw score": f"{res['anomaly_score']:.5f}",
                    "Anomaly %": f"{res['anomaly_percentage']:.2f}",
                    "Label": label_val,
                }
            )
