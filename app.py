
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import subprocess
import imageio_ffmpeg
from ultralytics import YOLO
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="SafeVision AI | Smart Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# SESSION STATE
# =====================================================
defaults = {
    "processed_video": None,
    "processed_module": None,
    "alerts": [],
    "last_module": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =====================================================
# MODERN FRONTEND DESIGN
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
      radial-gradient(circle at 10% 5%, rgba(109,74,255,.18), transparent 30%),
      radial-gradient(circle at 90% 10%, rgba(56,189,248,.10), transparent 25%),
      linear-gradient(135deg,#080D1B,#10172B 60%,#17102E);
    color: #F5F7FF;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0B1224,#17132F);
    border-right: 1px solid rgba(139,92,246,.3);
}

[data-testid="stSidebar"] * {
    color: #E5E7F5;
}

h1,h2,h3,h4 {
    color: #FFFFFF !important;
    font-weight: 750 !important;
}

p,li,label {
    color: #C5CEE2;
}

.hero {
    padding: 38px;
    border-radius: 24px;
    background: linear-gradient(120deg,#182747,#282052);
    border: 1px solid rgba(167,139,250,.4);
    box-shadow: 0 15px 45px rgba(0,0,0,.25);
    margin-bottom: 25px;
}

.hero-label {
    color: #BCA7FF;
    letter-spacing: 2px;
    font-size: 13px;
    font-weight: 800;
}

.hero-title {
    font-size: clamp(32px,4vw,52px);
    font-weight: 800;
    color: white;
    line-height: 1.15;
    margin-top: 12px;
}

.hero-text {
    color: #C8D2ED;
    font-size: 16px;
    line-height: 1.7;
    max-width: 780px;
}

.card {
    background: linear-gradient(145deg,#17223D,#211C40);
    border: 1px solid rgba(139,92,246,.27);
    border-radius: 17px;
    padding: 22px;
    min-height: 145px;
    margin-bottom: 12px;
    transition: .25s;
}

.card:hover {
    border-color: #A78BFA;
    transform: translateY(-3px);
}

.card-icon {
    font-size: 30px;
    margin-bottom: 10px;
}

.card-title {
    color: white;
    font-size: 17px;
    font-weight: 750;
    margin-bottom: 7px;
}

.card-text {
    color: #AAB5D0;
    font-size: 13px;
    line-height: 1.6;
}

.tag {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 30px;
    background: rgba(139,92,246,.18);
    border: 1px solid rgba(167,139,250,.4);
    color: #D9CCFF;
    font-size: 12px;
    font-weight: 700;
    margin: 3px;
}

.stButton > button {
    background: linear-gradient(95deg,#6547E8,#9B5CFF);
    color: white;
    border: none;
    border-radius: 11px;
    font-weight: 750;
    min-height: 45px;
    transition: .2s;
}

.stButton > button:hover {
    background: linear-gradient(95deg,#805EFF,#B06CFF);
    box-shadow: 0 0 20px rgba(139,92,246,.4);
    color: white;
    transform: translateY(-2px);
}

.stDownloadButton > button {
    background: linear-gradient(95deg,#087E8B,#12A6B5);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg,#17223D,#211C40);
    border: 1px solid rgba(139,92,246,.3);
    border-radius: 16px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #AAB5D0 !important;
}

[data-testid="stMetricValue"] {
    color: white !important;
    font-weight: 800;
}

[data-testid="stFileUploader"] {
    background: rgba(23,34,61,.7);
    border: 1px dashed #9B7AFF;
    border-radius: 16px;
    padding: 15px;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: rgba(139,92,246,.25);
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# REUSABLE UI FUNCTIONS
# =====================================================
def section_title(label, title, description=""):
    st.markdown(
        f'<div style="color:#A78BFA;font-size:12px;'
        f'font-weight:800;letter-spacing:2px">{label}</div>',
        unsafe_allow_html=True
    )
    st.markdown(f"## {title}")
    if description:
        st.caption(description)


def feature_card(icon, title, description):
    st.markdown(f"""
    <div class="card">
        <div class="card-icon">{icon}</div>
        <div class="card-title">{title}</div>
        <div class="card-text">{description}</div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# LOAD YOLO MODEL
# =====================================================
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


# =====================================================
# VIDEO CONVERSION
# =====================================================
@st.cache_data(show_spinner=False)
def convert_uploaded_video(video_bytes):

    input_path = None
    output_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp:
            temp.write(video_bytes)
            input_path = temp.name

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp:
            output_path = temp.name

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg, "-y",
            "-i", input_path,
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_path
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr[-1500:])

        with open(output_path, "rb") as f:
            return f.read()

    finally:
        for path in [input_path, output_path]:
            if path and os.path.exists(path):
                os.remove(path)


# =====================================================
# VIDEO PROCESSING
# Existing detection rules retained
# =====================================================
def process_video(video_bytes, selected_module):

    input_path = None
    output_path = None
    final_path = None
    cap = None
    writer = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp:
            temp.write(video_bytes)
            input_path = temp.name

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError("Unable to open video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            raise ValueError("Invalid video dimensions.")

        if fps <= 0:
            fps = 25

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp:
            output_path = temp.name

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            raise ValueError("Unable to create output video.")

        model = None

        if selected_module != "Privacy Protection":
            model = load_model()

        # Load face detector only for privacy module
        face_detector = None

        if selected_module == "Privacy Protection":
            cascade_path = (
                cv2.data.haarcascades +
                "haarcascade_frontalface_default.xml"
            )

            face_detector = cv2.CascadeClassifier(cascade_path)

            if face_detector.empty():
                raise RuntimeError("Face detector could not be loaded.")

        x1 = int(width * 0.50)
        y1 = int(height * 0.15)
        x2 = int(width * 0.95)
        y2 = int(height * 0.85)

        crowd_limit = 5
        fall_ratio = 1.2
        movement_threshold = 40

        previous_centers = []

        frame_number = 0
        alert_frames = 0
        total_detected_people = 0
        privacy_faces = 0

        progress = st.progress(0)
        status = st.empty()

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1
            annotated = frame.copy()

            # PRIVACY PROTECTION
            if selected_module == "Privacy Protection":

                gray = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2GRAY
                )

                faces = face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                for (fx, fy, fw, fh) in faces:

                    face = annotated[
                        fy:fy + fh,
                        fx:fx + fw
                    ]

                    if face.size > 0:
                        annotated[
                            fy:fy + fh,
                            fx:fx + fw
                        ] = cv2.GaussianBlur(
                            face, (51, 51), 30
                        )

                    privacy_faces += 1

                cv2.putText(
                    annotated,
                    "PRIVACY PROTECTION ACTIVE",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            else:

                results = model(
                    frame,
                    classes=[0],
                    verbose=False
                )

                boxes = results[0].boxes
                centers = []
                person_count = len(boxes)

                total_detected_people += person_count

                # PERSON DETECTION
                if selected_module == "Person Detection":

                    annotated = results[0].plot()

                # RESTRICTED AREA
                elif selected_module == "Restricted Area":

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        annotated,
                        "RESTRICTED AREA",
                        (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    for box in boxes:

                        bx1, by1, bx2, by2 = map(
                            int, box.xyxy[0].tolist()
                        )

                        cx = (bx1 + bx2) // 2
                        cy = (by1 + by2) // 2

                        inside = (
                            x1 <= cx <= x2 and
                            y1 <= cy <= y2
                        )

                        color = (
                            (0, 0, 255)
                            if inside else (0, 255, 0)
                        )

                        cv2.rectangle(
                            annotated,
                            (bx1, by1),
                            (bx2, by2),
                            color,
                            2
                        )

                        if inside:

                            alert_frames += 1

                            cv2.putText(
                                annotated,
                                "RESTRICTED AREA ALERT!",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2
                            )

                # CROWD MONITORING
                elif selected_module == "Crowd Monitoring":

                    annotated = results[0].plot()

                    if person_count > crowd_limit:

                        alert_frames += 1

                        cv2.putText(
                            annotated,
                            "CROWD ALERT!",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3
                        )

                    else:

                        cv2.putText(
                            annotated,
                            "CROWD NORMAL",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                    cv2.putText(
                        annotated,
                        f"People: {person_count}",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                # FALL DETECTION
                elif selected_module == "Fall Detection":

                    for box in boxes:

                        bx1, by1, bx2, by2 = map(
                            int, box.xyxy[0].tolist()
                        )

                        box_width = bx2 - bx1
                        box_height = by2 - by1

                        ratio = box_width / max(box_height, 1)

                        possible_fall = ratio > fall_ratio

                        color = (
                            (0, 0, 255)
                            if possible_fall
                            else (0, 255, 0)
                        )

                        cv2.rectangle(
                            annotated,
                            (bx1, by1),
                            (bx2, by2),
                            color,
                            2
                        )

                        if possible_fall:
                            alert_frames += 1
                            label = "POSSIBLE FALL!"
                        else:
                            label = "PERSON"

                        cv2.putText(
                            annotated,
                            label,
                            (bx1, max(by1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )

                # UNUSUAL MOVEMENT
                elif selected_module == "Unusual Movement":

                    for box in boxes:

                        bx1, by1, bx2, by2 = map(
                            int, box.xyxy[0].tolist()
                        )

                        cx = (bx1 + bx2) // 2
                        cy = (by1 + by2) // 2

                        centers.append((cx, cy))

                        color = (0, 255, 0)
                        label = "PERSON"

                        if previous_centers:

                            distances = [
                                np.sqrt(
                                    (cx - px) ** 2 +
                                    (cy - py) ** 2
                                )
                                for px, py in previous_centers
                            ]

                            if min(distances) > movement_threshold:

                                color = (0, 0, 255)
                                label = "MOVEMENT ALERT!"
                                alert_frames += 1

                        cv2.rectangle(
                            annotated,
                            (bx1, by1),
                            (bx2, by2),
                            color,
                            2
                        )

                        cv2.putText(
                            annotated,
                            label,
                            (bx1, max(by1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )

                    previous_centers = centers

            writer.write(annotated)

            if total_frames > 0:
                progress.progress(
                    min(frame_number / total_frames, 1.0)
                )

            status.write(
                f"Processing frame: {frame_number}"
            )

        cap.release()
        cap = None

        writer.release()
        writer = None

        if frame_number == 0:
            raise ValueError("No video frames were processed.")

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp:
            final_path = temp.name

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg, "-y",
            "-i", output_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-an",
            "-movflags", "+faststart",
            final_path
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr[-1500:])

        with open(final_path, "rb") as f:
            output_bytes = f.read()

        return {
            "video": output_bytes,
            "frames": frame_number,
            "people": total_detected_people,
            "alert_frames": alert_frames,
            "faces": privacy_faces
        }

    finally:

        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()

        for path in [input_path, output_path, final_path]:

            if path and os.path.exists(path):
                os.remove(path)


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.markdown("""
<div style="padding:10px 3px 20px 3px">
    <div style="font-size:38px">🛡️</div>
    <div style="font-size:25px;font-weight:800;color:white">
        SafeVision <span style="color:#A78BFA">AI</span>
    </div>
    <div style="font-size:11px;color:#AAB5D0;letter-spacing:2px">
        SMART SECURITY PLATFORM
    </div>
</div>
""", unsafe_allow_html=True)

pages = [
    "Home",
    "Dashboard",
    "Person Detection",
    "Restricted Area",
    "Crowd Monitoring",
    "Fall Detection",
    "Unusual Movement",
    "Privacy Protection",
    "Alert History",
    "Reports & Analytics",
    "Technology Stack",
    "About Project"
]

page = st.sidebar.radio("MAIN MENU", pages)

st.sidebar.markdown("---")
st.sidebar.markdown("### SYSTEM STATUS")
st.sidebar.markdown(
    '<span class="tag">● Prototype Interface</span>',
    unsafe_allow_html=True
)
st.sidebar.caption("Human verification required.")
st.sidebar.caption("Use authorized footage only.")

# Prevent results from one module appearing in another
if st.session_state.last_module != page:

    if page in [
        "Person Detection",
        "Restricted Area",
        "Crowd Monitoring",
        "Fall Detection",
        "Unusual Movement",
        "Privacy Protection"
    ]:
        st.session_state.processed_video = None
        st.session_state.processed_module = None

    st.session_state.last_module = page


# =====================================================
# HOME PAGE
# =====================================================
if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-label">AI POWERED COMPUTER VISION PLATFORM</div>
        <div class="hero-title">
            Smarter Surveillance.<br>
            Safer Spaces. Privacy First.
        </div>
        <div class="hero-text">
            SafeVision AI is an AI-based smart security and privacy
            monitoring prototype. It analyses uploaded CCTV videos,
            highlights selected security events and supports privacy
            through automatic face blurring.
        </div>
        <br>
        <span class="tag">YOLO11</span>
        <span class="tag">OpenCV</span>
        <span class="tag">Python</span>
        <span class="tag">Streamlit</span>
        <span class="tag">Computer Vision</span>
    </div>
    """, unsafe_allow_html=True)

    section_title(
        "PLATFORM OVERVIEW",
        "Explore monitoring modules",
        "Select any module from the sidebar to start video analysis."
    )

    features = [
        ("👤", "Person Detection", "Detect people in uploaded CCTV footage."),
        ("🚧", "Restricted Area", "Highlight people inside a predefined zone."),
        ("👥", "Crowd Monitoring", "Compare people count with a threshold."),
        ("🧍", "Fall Detection", "Highlight possible horizontal body postures."),
        ("🏃", "Unusual Movement", "Identify basic frame-to-frame position changes."),
        ("🕶️", "Privacy Protection", "Detect and blur faces in video frames.")
    ]

    for start in range(0, 6, 3):
        cols = st.columns(3)

        for col, item in zip(cols, features[start:start + 3]):
            with col:
                feature_card(*item)

    st.markdown("---")

    section_title("PROJECT SNAPSHOT", "SafeVision at a glance")

    c1, c2, c3 = st.columns(3)

    c1.metric("Detection Modules", "5")
    c2.metric("Alert Records", len(st.session_state.alerts))
    c3.metric("Privacy Module", "Available")

    st.info(
        "Upload an authorized CCTV video, choose a monitoring module "
        "and review the processed output."
    )


# =====================================================
# DASHBOARD
# =====================================================
elif page == "Dashboard":

    section_title(
        "COMMAND CENTER",
        "Monitoring Dashboard",
        "Overview of the current application session."
    )

    if st.session_state.alerts:

        df = pd.DataFrame(st.session_state.alerts)

        total_alerts = len(df)
        total_frames = int(df["Alert Frames"].sum())
        modules_used = df["Module"].nunique()

    else:

        total_alerts = 0
        total_frames = 0
        modules_used = 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Alert Records", total_alerts)
    c2.metric("Alert Frames", total_frames)
    c3.metric("Modules Used", modules_used)
    c4.metric("System Status", "Ready")

    st.markdown("---")

    left, right = st.columns([1.5, 1])

    with left:

        st.subheader("Recent Activity")

        if st.session_state.alerts:

            st.dataframe(
                pd.DataFrame(st.session_state.alerts).tail(8),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("No alert records in this session yet.")

    with right:

        st.subheader("How to use")

        st.markdown("""
        **01** – Select a monitoring module.

        **02** – Upload CCTV video.

        **03** – Preview the video.

        **04** – Start AI processing.

        **05** – Review and download results.
        """)


# =====================================================
# TECHNOLOGY STACK
# =====================================================
elif page == "Technology Stack":

    section_title(
        "BUILT WITH",
        "Technology Stack",
        "Technologies used in the SafeVision AI project."
    )

    technologies = [
        ("🐍", "Python", "Main programming language."),
        ("🎯", "YOLO11", "Person and object detection."),
        ("🧩", "OpenCV", "Video processing and face blurring."),
        ("🌐", "Streamlit", "Interactive web application frontend."),
        ("🔢", "NumPy", "Numerical and array operations."),
        ("📊", "Pandas", "Alert data handling and analysis."),
        ("📈", "Matplotlib", "Analytics charts used during development."),
        ("🗃️", "SQLite", "Used in an earlier database prototype."),
        ("☁️", "Google Colab", "Model development and experimentation."),
        ("🐙", "GitHub", "Source-code management."),
        ("🚀", "Streamlit Cloud", "Application deployment."),
        ("🎞️", "FFmpeg", "Video format conversion.")
    ]

    for start in range(0, len(technologies), 3):

        cols = st.columns(3)

        for col, item in zip(cols, technologies[start:start + 3]):

            with col:
                feature_card(*item)

    st.caption(
        "Note: SQLite was used in an earlier prototype. "
        "The current deployed app stores alert history in session state."
    )


# =====================================================
# ABOUT PROJECT
# =====================================================
elif page == "About Project":

    section_title(
        "ABOUT THE PROJECT",
        "SafeVision AI",
        "AI-Based Smart Security & Privacy Monitoring System"
    )

    st.markdown("""
    <div class="hero">
        <div class="hero-label">PROJECT PURPOSE</div>
        <div class="hero-title" style="font-size:32px">
            Computer Vision for Safer Monitoring
        </div>
        <div class="hero-text">
            This project demonstrates how artificial intelligence
            and computer vision can help review surveillance footage
            by highlighting selected security-related events.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Project Objectives")

        st.markdown("""
        - Automate basic CCTV video analysis.
        - Detect people in video footage.
        - Monitor a predefined restricted area.
        - Identify crowd levels above a threshold.
        - Experiment with possible fall detection.
        - Highlight unusual frame-to-frame movement.
        - Apply face blurring for privacy support.
        """)

    with col2:

        st.subheader("Privacy & Responsible Use")

        st.markdown("""
        - Use only authorized video footage.
        - No facial recognition or identity identification.
        - Face blurring may miss some faces.
        - Alerts require human verification.
        - Do not treat alerts as proof of an incident.
        - Follow applicable privacy and security policies.
        """)

    st.warning(
        "SafeVision AI is an educational prototype, "
        "not a certified professional security system."
    )


# =====================================================
# ALERT HISTORY
# =====================================================
elif page == "Alert History":

    section_title(
        "EVENT LOG",
        "Alert History",
        "Records generated during the current app session."
    )

    if st.session_state.alerts:

        alert_df = pd.DataFrame(st.session_state.alerts)

        st.dataframe(
            alert_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Alert History CSV",
            alert_df.to_csv(index=False),
            "safevision_alerts.csv",
            "text/csv"
        )

    else:

        st.info("No alerts recorded yet. Run a detection module first.")

    st.caption(
        "Current alert history is session-based and is not permanent storage."
    )


# =====================================================
# REPORTS & ANALYTICS
# =====================================================
elif page == "Reports & Analytics":

    section_title(
        "INSIGHTS",
        "Reports & Analytics",
        "Summary of alert records in the current session."
    )

    if st.session_state.alerts:

        df = pd.DataFrame(st.session_state.alerts)

        summary = (
            df.groupby("Module")
            .size()
            .reset_index(name="Alert Records")
        )

        left, right = st.columns([1, 1.4])

        with left:

            st.subheader("Alert Summary")

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

        with right:

            st.subheader("Records by Module")

            st.bar_chart(
                summary.set_index("Module")
            )

    else:

        st.info("No detection records available yet.")


# =====================================================
# MONITORING MODULES
# =====================================================
else:

    descriptions = {
        "Person Detection":
            "Detect people in uploaded CCTV video using YOLO11.",

        "Restricted Area":
            "Highlight people inside the predefined restricted zone.",

        "Crowd Monitoring":
            "Compare detected people count with the configured threshold of 5.",

        "Fall Detection":
            "Highlight possible horizontal body postures using bounding-box shape.",

        "Unusual Movement":
            "Highlight large frame-to-frame person-centre position changes.",

        "Privacy Protection":
            "Detect faces and apply Gaussian blur using OpenCV."
    }

    section_title(
        "VIDEO ANALYSIS",
        page,
        descriptions.get(page, "")
    )

    st.markdown(
        '<span class="tag">VIDEO UPLOAD</span>',
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Upload CCTV Video",
        type=["mp4", "avi", "mov"],
        key="common_video_uploader"
    )

    if uploaded_video is not None:

        video_bytes = uploaded_video.getvalue()

        try:

            with st.spinner("Preparing video preview..."):

                preview_bytes = convert_uploaded_video(video_bytes)

            st.success("Video uploaded successfully!")

            preview_col, info_col = st.columns([1.5, 1])

            with preview_col:

                st.subheader("CCTV Video Preview")
                st.video(preview_bytes)

            with info_col:

                st.subheader("Selected Module")

                st.markdown(f"### {page}")

                st.write(descriptions.get(page, ""))

                st.caption(
                    "Tip: Start with a short video clip for faster processing."
                )

            if st.button(
                "▶ Start " + page,
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "AI is processing your CCTV video..."
                ):

                    result = process_video(
                        preview_bytes,
                        page
                    )

                st.session_state.processed_video = result["video"]
                st.session_state.processed_module = page

                st.success("Video processing completed!")

                st.markdown("---")

                section_title(
                    "ANALYSIS RESULT",
                    "Detection Summary"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Frames Processed",
                    result["frames"]
                )

                c2.metric(
                    "Person Detections",
                    result["people"]
                )

                c3.metric(
                    "Alert Frames",
                    result["alert_frames"]
                )

                if page == "Privacy Protection":

                    st.metric(
                        "Face Detections Processed",
                        result["faces"]
                    )

                    st.warning(
                        "Face detection may miss faces. "
                        "This prototype does not guarantee complete anonymization."
                    )

                elif page == "Fall Detection":

                    st.warning(
                        "Fall detection uses bounding-box shape. "
                        "Sitting, bending or lying down may cause false alerts."
                    )

                elif page == "Unusual Movement":

                    st.warning(
                        "Movement detection uses basic frame-to-frame position "
                        "changes. It is not reliable person tracking."
                    )

                elif page == "Restricted Area":

                    st.info(
                        "The restricted zone is a predefined rectangle "
                        "covering the right side of the video."
                    )

                elif page == "Crowd Monitoring":

                    st.info(
                        "Crowd alert threshold: more than 5 detected people."
                    )

                # Save alert record
                if result["alert_frames"] > 0:

                    st.session_state.alerts.append({
                        "Module": page,
                        "Alert Frames": result["alert_frames"],
                        "Timestamp": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    })

                elif page == "Privacy Protection":

                    st.session_state.alerts.append({
                        "Module": page,
                        "Alert Frames": 0,
                        "Timestamp": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    })

            # Show result only for the currently selected module
            if (
                st.session_state.processed_video is not None
                and st.session_state.processed_module == page
            ):

                st.markdown("---")

                section_title(
                    "OUTPUT",
                    "Processed Detection Video"
                )

                st.video(
                    st.session_state.processed_video
                )

                st.download_button(
                    "⬇️ Download Processed Video",
                    data=st.session_state.processed_video,
                    file_name=(
                        "safevision_" +
                        page.lower().replace(" ", "_") +
                        ".mp4"
                    ),
                    mime="video/mp4",
                    use_container_width=True
                )

        except Exception as error:

            st.error("Video processing failed.")
            st.exception(error)

    else:

        st.info("Please upload a CCTV video to start monitoring.")


# =====================================================
# FOOTER
# =====================================================
st.markdown("---")

st.markdown("""
<div style="text-align:center;padding:15px;color:#8491AE;font-size:13px">
    🛡️ <b>SafeVision AI</b><br>
    AI-Based Smart Security & Privacy Monitoring System<br>
    Smarter Surveillance. Safer Spaces. Privacy First.
    <br><br>
    Educational Prototype | Human Verification Required
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("SafeVision AI | Version 2.0")
