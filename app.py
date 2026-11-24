
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

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SafeVision AI",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================
# SESSION STATE
# ==========================================

if "processed_video" not in st.session_state:
    st.session_state.processed_video = None

if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "last_module" not in st.session_state:
    st.session_state.last_module = None

# ==========================================
# DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #111c32;
}
h1, h2, h3 {
    color: white;
}
.stButton button {
    background-color: #6c63ff;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD YOLO MODEL
# ==========================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

# ==========================================
# VIDEO CONVERSION
# ==========================================

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

# ==========================================
# VIDEO PROCESSING
# ==========================================

def process_video(video_bytes, selected_module):

    input_path = None
    output_path = None
    final_path = None

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

        # Face detector for privacy module
        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        # Restricted area coordinates
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

                # Person detection
                if selected_module == "Person Detection":

                    annotated = results[0].plot()

                # Restricted area detection
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

                        color = (0, 0, 255) if inside else (0, 255, 0)

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

                # Crowd monitoring
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

                # Possible fall detection
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

                # Unusual movement indicator
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
        writer.release()

        if frame_number == 0:
            raise ValueError("No video frames were processed.")

        # Convert output into browser-compatible MP4
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

        for path in [input_path, output_path, final_path]:

            if path and os.path.exists(path):
                os.remove(path)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🛡️ SafeVision AI")
st.sidebar.caption("Smart Security & Privacy Monitoring")

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Person Detection",
        "Restricted Area",
        "Crowd Monitoring",
        "Fall Detection",
        "Unusual Movement",
        "Privacy Protection",
        "Alert History",
        "Reports & Analytics"
    ]
)

# ==========================================
# HOME
# ==========================================

if page == "Home":

    st.title("🛡️ SafeVision AI")

    st.subheader(
        "AI-Based Smart Security & Privacy Monitoring System"
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Detection Modules", "6")
    col2.metric("Recorded Alerts", len(st.session_state.alerts))
    col3.metric("Privacy Protection", "Available")

    st.markdown("---")

    st.header("Welcome to SafeVision AI")

    st.write(
        "An AI-powered security monitoring prototype "
        "using YOLO and Computer Vision."
    )

    st.info(
        "Upload your CCTV video from any monitoring module "
        "and select the detection feature you want to run."
    )

    st.markdown("### Available Modules")

    st.write("""
    - Person Detection
    - Restricted Area Detection
    - Crowd Monitoring
    - Possible Fall Detection
    - Unusual Movement Detection
    - Privacy Protection
    """)

# ==========================================
# ALERT HISTORY
# ==========================================

elif page == "Alert History":

    st.title("📋 Alert History")

    if st.session_state.alerts:

        alert_df = pd.DataFrame(
            st.session_state.alerts
        )

        st.dataframe(
            alert_df,
            use_container_width=True
        )

        csv_data = alert_df.to_csv(index=False)

        st.download_button(
            "Download Alert History CSV",
            csv_data,
            "safevision_alerts.csv",
            "text/csv"
        )

    else:
        st.info("No alerts recorded yet. Run a detection module first.")

# ==========================================
# REPORTS AND ANALYTICS
# ==========================================

elif page == "Reports & Analytics":

    st.title("📊 Reports & Analytics")

    if st.session_state.alerts:

        df = pd.DataFrame(st.session_state.alerts)

        st.subheader("Alert Summary")

        summary = df.groupby("Module").size().reset_index(
            name="Alert Records"
        )

        st.dataframe(summary, use_container_width=True)

        st.bar_chart(
            summary.set_index("Module")
        )

    else:
        st.info("No detection records available yet.")

# ==========================================
# MONITORING MODULES
# ==========================================

else:

    st.title("🎥 " + page)

    st.write(
        "Upload one CCTV video and run the selected monitoring module."
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

                preview_bytes = convert_uploaded_video(
                    video_bytes
                )

            st.success("Video uploaded successfully!")

            st.subheader("CCTV Video Preview")

            st.video(preview_bytes)

            if st.button("Start " + page, type="primary"):

                with st.spinner(
                    "AI is processing your CCTV video..."
                ):

                    result = process_video(
                        preview_bytes,
                        page
                    )

                st.session_state.processed_video = result["video"]

                st.success("Detection completed!")

                st.subheader("Detection Summary")

                c1, c2, c3 = st.columns(3)

                c1.metric("Frames Processed", result["frames"])
                c2.metric("People Detected", result["people"])
                c3.metric("Alert Frames", result["alert_frames"])

                if page == "Privacy Protection":

                    st.metric(
                        "Face Detections Processed",
                        result["faces"]
                    )

                    st.warning(
                        "Face detection may miss some faces. "
                        "Do not treat this prototype as guaranteed anonymization."
                    )

                elif page == "Fall Detection":

                    st.warning(
                        "Possible fall is estimated using bounding-box shape. "
                        "Sitting, bending or lying down may cause false alerts."
                    )

                elif page == "Unusual Movement":

                    st.warning(
                        "Movement is estimated using frame-to-frame "
                        "position changes. This is not reliable person tracking."
                    )

                elif page == "Restricted Area":

                    st.info(
                        "The restricted zone is a predefined rectangle "
                        "covering the right side of the video."
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

            # Display saved output
            if st.session_state.processed_video is not None:

                st.markdown("---")

                st.subheader("🎯 Processed Detection Video")

                st.video(
                    st.session_state.processed_video
                )

                st.download_button(
                    "⬇️ Download Processed Video",
                    data=st.session_state.processed_video,
                    file_name="safevision_result.mp4",
                    mime="video/mp4"
                )

        except Exception as error:

            st.error("Video processing failed.")

            st.exception(error)

    else:

        st.info(
            "Please upload a CCTV video to start monitoring."
        )

# ==========================================
# FOOTER
# ==========================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "SafeVision AI | Security & Privacy Prototype"
)
