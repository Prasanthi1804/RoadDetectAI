from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from severity import calculate_severity


# ==================================================
# 1. APPLICATION CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent

WEIGHTS = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "pothole"
    / "pothole_img416_fixed"
    / "weights"
    / "best.pt"
)

st.set_page_config(
    page_title="Road DetectAI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# 2. PREMIUM DARK THEME
# ==================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

    :root {
        --bg: #070b16;
        --panel: #0f1829;
        --panel-2: #121f36;
        --border: #22314d;
        --border-glow: #3f5b86;
        --text: #eef3ff;
        --muted: #8ea3c4;
        --accent: #66b5ff;
        --accent-strong: #2379e8;
        --ok: #34d399;
        --warn: #fbbf24;
        --danger: #fb7185;
        --critical: #c084fc;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(1000px 500px at 0% -10%, rgba(35, 121, 232, 0.18), transparent 60%),
            radial-gradient(800px 480px at 100% 0%, rgba(192, 132, 252, 0.10), transparent 60%),
            var(--bg);
        color: var(--text);
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1829 0%, #0a1222 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] h1 {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        background: linear-gradient(90deg, #ffffff, #9fd0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 10px;
        padding: 6px 10px;
        transition: background 0.2s ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(102, 181, 255, 0.08);
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
        color: #f4f7ff;
    }

    h2, h3 {
        font-weight: 700;
    }

    [data-testid="stCaptionContainer"], .stCaption {
        color: var(--muted) !important;
    }

    hr {
        border-color: var(--border) !important;
        margin: 1.8rem 0 !important;
    }

    /* ---------- Hero banner ---------- */
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(120deg, rgba(20, 37, 68, 0.95) 0%, rgba(27, 61, 102, 0.95) 55%, rgba(31, 79, 134, 0.95) 100%);
        border: 1px solid #30476b;
        border-radius: 22px;
        padding: 34px 36px;
        margin-bottom: 28px;
        box-shadow:
            0 24px 48px -24px rgba(35, 121, 232, 0.55),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }

    .hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(rgba(102, 181, 255, 0.06) 1px, transparent 1px),
            linear-gradient(90deg, rgba(102, 181, 255, 0.06) 1px, transparent 1px);
        background-size: 34px 34px;
        mask-image: linear-gradient(120deg, transparent 30%, #000 100%);
        pointer-events: none;
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -80px;
        top: -80px;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(102, 181, 255, 0.35), transparent 70%);
        filter: blur(4px);
    }

    .hero > * {
        position: relative;
        z-index: 1;
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: #9fd0ff;
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        background: rgba(102, 181, 255, 0.12);
        border: 1px solid rgba(102, 181, 255, 0.30);
        padding: 6px 12px;
        border-radius: 999px;
        margin-bottom: 16px;
    }

    .hero-eyebrow::before {
        content: "";
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--ok);
        box-shadow: 0 0 10px var(--ok);
    }

    .hero h1 {
        color: #ffffff;
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1.15;
        margin: 0 0 10px 0;
        background: linear-gradient(90deg, #ffffff 0%, #c9e4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #c4d4ef;
        font-size: 1.04rem;
        margin: 0;
        max-width: 660px;
        line-height: 1.6;
    }

    /* ---------- Upload ---------- */
    [data-testid="stFileUploader"] section {
        background: linear-gradient(180deg, #0f1a2e 0%, #0c1626 100%);
        border: 1.5px dashed #345179;
        border-radius: 18px;
        transition: border-color 0.25s ease, background 0.25s ease, box-shadow 0.25s ease;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: var(--accent);
        background: #13223b;
        box-shadow: 0 0 0 4px rgba(102, 181, 255, 0.08);
    }

    /* ---------- Step cards ---------- */
    .step-card {
        position: relative;
        overflow: hidden;
        background: linear-gradient(180deg, var(--panel-2) 0%, var(--panel) 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px 22px 20px 22px;
        min-height: 150px;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }

    .step-card::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, var(--accent), var(--accent-strong));
    }

    .step-card:hover {
        transform: translateY(-3px);
        border-color: var(--border-glow);
        box-shadow: 0 16px 30px -20px rgba(35, 121, 232, 0.6);
    }

    .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(35, 121, 232, 0.25), rgba(102, 181, 255, 0.10));
        border: 1px solid rgba(102, 181, 255, 0.30);
        color: var(--accent);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 13px;
        margin-bottom: 14px;
    }

    .step-title {
        color: #ffffff;
        font-weight: 700;
        font-size: 1.08rem;
        margin-bottom: 6px;
    }

    .step-text {
        color: var(--muted);
        font-size: 0.93rem;
        line-height: 1.55;
    }

    /* ---------- Metric cards ---------- */
    .metric-card {
        position: relative;
        overflow: hidden;
        background: linear-gradient(180deg, var(--panel-2) 0%, var(--panel) 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px;
        min-height: 128px;
        box-shadow: 0 14px 28px -20px rgba(0, 0, 0, 0.8);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: var(--border-glow);
    }

    .metric-label {
        color: var(--muted);
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--accent);
        font-size: 34px;
        font-weight: 800;
        margin-top: 10px;
        letter-spacing: -0.02em;
        text-shadow: 0 0 22px rgba(102, 181, 255, 0.25);
    }

    .metric-sub {
        color: var(--muted);
        font-size: 12.5px;
        margin-top: 6px;
    }

    .badge {
        display: inline-block;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        padding: 5px 12px;
        border-radius: 999px;
        border: 1px solid currentColor;
        background: rgba(255, 255, 255, 0.03);
    }

    .sev-none     { color: var(--muted); }
    .sev-low      { color: var(--ok); }
    .sev-medium   { color: var(--warn); }
    .sev-high     { color: var(--danger); }
    .sev-critical { color: var(--critical); }

    /* ---------- Buttons ---------- */
    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(120deg, #2379e8, #4a9cf5);
        color: white;
        border: none;
        border-radius: 12px;
        min-height: 46px;
        font-weight: 600;
        letter-spacing: 0.01em;
        box-shadow: 0 12px 24px -14px rgba(35, 121, 232, 0.95);
        transition: transform 0.15s ease, box-shadow 0.25s ease, filter 0.25s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        filter: brightness(1.08);
        box-shadow: 0 16px 30px -12px rgba(35, 121, 232, 1);
        color: white;
        border: none;
    }

    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: translateY(1px) scale(0.99);
    }

    .stDownloadButton > button {
        background: rgba(20, 34, 58, 0.8);
        border: 1px solid #2f4a72;
        box-shadow: none;
        color: #cfe3ff;
    }

    .stDownloadButton > button:hover {
        background: #1a2c4a;
        border: 1px solid #4a6fa0;
        color: #ffffff;
        box-shadow: 0 10px 22px -16px rgba(102, 181, 255, 0.7);
    }

    /* ---------- Alerts, tables, images, code ---------- */
    [data-testid="stAlert"] {
        border-radius: 14px;
        backdrop-filter: blur(6px);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 14px 28px -22px rgba(0, 0, 0, 0.9);
    }

    [data-testid="stImage"] img {
        border-radius: 16px;
        border: 1px solid var(--border);
        box-shadow: 0 18px 32px -24px rgba(0, 0, 0, 0.9);
    }

    [data-testid="stCodeBlock"], .stCode, pre {
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
    }

    code {
        color: #9fd0ff !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    @keyframes rise {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .hero, .step-card, .metric-card {
        animation: rise 0.5s ease both;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 3. LOAD MODEL
# ==================================================

@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


# ==================================================
# 4. SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:
    st.title("🛣️ Road DetectAI")
    st.caption("AI-powered road inspection")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Model Information",
            "About Project",
        ],
    )

    st.divider()

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.05,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    st.caption(
        "Lower confidence may detect more defects "
        "but can also produce more false positives."
    )


# ==================================================
# 5. CHECK MODEL FILE
# ==================================================

if not WEIGHTS.is_file():
    st.error("Trained model file was not found.")

    st.code(str(WEIGHTS))

    st.write(
        "Check that best.pt exists in the expected "
        "folder before running detection."
    )

    st.stop()


# ==================================================
# 6. MODEL INFORMATION PAGE
# ==================================================

if page == "Model Information":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Model</div>
            <h1>🤖 Model Information</h1>
            <p>Details about the detector powering Road DetectAI.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("**Architecture:** YOLOv8n")
    st.write("**Inference device:** CPU")
    st.write("**Inference image size:** 320")
    st.write("**Training dataset:** Your configured pothole dataset")

    st.write("**Model weights path:**")
    st.code(str(WEIGHTS))

    st.success("Model weights file found.")

    st.stop()


# ==================================================
# 7. ABOUT PAGE
# ==================================================

if page == "About Project":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">About</div>
            <h1>About Road DetectAI</h1>
            <p>How the detector works and what its results mean.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        """
        Road DetectAI is an AI-based road damage
        detection application. It uses a trained YOLOv8
        model to identify defects in uploaded road images.
        """
    )

    st.write(
        """
        The dashboard displays predicted bounding boxes,
        confidence scores, detection counts, and estimated
        severity based on bounding-box area coverage.
        """
    )

    st.warning(
        "Image-area coverage is an estimate. It does not "
        "measure the real physical dimensions of a pothole."
    )

    st.stop()


# ==================================================
# 8. MAIN DASHBOARD
# ==================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">Road Inspection</div>
        <h1>AI-Powered Pothole Detection System</h1>
        <p>
            Detect potholes and road defects using your
            trained YOLOv8 model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 9. LOAD TRAINED MODEL
# ==================================================

try:
    model = load_model(str(WEIGHTS))
except Exception as exc:
    st.error(f"Unable to load the YOLO model: {exc}")
    st.stop()


# ==================================================
# 10. IMAGE UPLOAD
# ==================================================

st.subheader("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is None:
    st.info("Upload a road image to start detection.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-num">01</div>
                <div class="step-title">Upload</div>
                <div class="step-text">Choose an image of a road.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-num">02</div>
                <div class="step-title">Detect</div>
                <div class="step-text">Run your trained YOLOv8 model.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-num">03</div>
                <div class="step-title">Analyze</div>
                <div class="step-text">Review detections and severity.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()


# ==================================================
# 11. READ IMAGE
# ==================================================

try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception as exc:
    st.error(f"Unable to read the uploaded image: {exc}")
    st.stop()

image_width, image_height = image.size

left, right = st.columns(2)

with left:
    st.subheader("Original Image")
    st.image(image, use_container_width=True)


# ==================================================
# 12. RUN DETECTION
# ==================================================

if st.button(
    "🚀 Run Road Damage Detection",
    type="primary",
    use_container_width=True,
):

    with st.spinner("Analyzing the image..."):

        try:
            result = model.predict(
                source=image,
                conf=confidence,
                imgsz=320,
                device="cpu",
                verbose=False,
            )[0]

        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            st.stop()

    boxes = []
    rows = []

    if result.boxes is not None and len(result.boxes) > 0:

        coordinates = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()

        for coords, score, class_id in zip(
            coordinates, scores, classes
        ):

            boxes.append(coords)

            class_name = result.names.get(
                int(class_id),
                str(int(class_id)),
            )

            rows.append(
                {
                    "Class": class_name,
                    "Confidence (%)": round(score * 100, 2),
                    "X1": round(coords[0], 2),
                    "Y1": round(coords[1], 2),
                    "X2": round(coords[2], 2),
                    "Y2": round(coords[3], 2),
                }
            )

    # Calculate image coverage and severity.
    coverage, severity = calculate_severity(
        boxes,
        image_width,
        image_height,
    )

    detection_count = len(rows)

    # Display-only: map severity to a CSS color class.
    severity_class = f"sev-{str(severity).lower()}"

    # Generate the annotated image.
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]

    annotated_image = Image.fromarray(annotated_rgb)

    # Show the annotated image.
    with right:
        st.subheader("Detected Road Damage")
        st.image(
            annotated_image,
            use_container_width=True,
        )

    st.divider()

    # ==================================================
    # 13. DASHBOARD METRICS
    # ==================================================

    st.subheader("📊 Detection Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Total Detections
                </div>
                <div class="metric-value">
                    {detection_count}
                </div>
                <div class="metric-sub">
                    Potholes found at this threshold
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Damage Coverage
                </div>
                <div class="metric-value">
                    {coverage:.2f}%
                </div>
                <div class="metric-sub">
                    Share of image covered by boxes
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Severity
                </div>
                <div class="metric-value">
                    <span class="badge {severity_class}">{severity}</span>
                </div>
                <div class="metric-sub">
                    Based on coverage thresholds
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Coverage is calculated using the sum of predicted "
        "bounding-box areas divided by image area. Overlapping "
        "boxes may count the same area more than once."
    )

    # ==================================================
    # 14. DETECTION TABLE
    # ==================================================

    st.subheader("🔎 Detection Details")

    if rows:

        dataframe = pd.DataFrame(rows)

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = dataframe.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_data,
            file_name="road_detection_results.csv",
            mime="text/csv",
        )

    else:

        st.warning(
            "No defects detected at this confidence threshold. "
            "Try another image or a lower confidence value."
        )

    # ==================================================
    # 15. DOWNLOAD ANNOTATED IMAGE
    # ==================================================

    buffer = BytesIO()

    annotated_image.save(
        buffer,
        format="PNG",
    )

    st.download_button(
        "⬇️ Download Annotated Image",
        data=buffer.getvalue(),
        file_name="road_detection_annotated.png",
        mime="image/png",
    )

    st.success("Image analysis completed.")