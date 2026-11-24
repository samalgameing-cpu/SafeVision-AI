
import streamlit as st
import cv2
import tempfile
import os
import subprocess
import imageio_ffmpeg
from ultralytics import YOLO

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="SafeVision AI",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------
# CUSTOM DESIGN
# -------------------------------

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
    border: none;
    padding: 10px 20px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD YOLO MODEL
# -------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

# -------------------------------
# VIDEO CONVERSION FUNCTION
# -------------------------------

def convert_video(input_path, output_path):

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
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
        raise RuntimeError(result.stderr[-2000:])

# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.title("🛡️ SafeVision AI")

st.sidebar.caption(
    "Smart Security & Privacy Monitoring"
)

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

# -------------------------------
# HOME DASHBOARD
# -------------------------------

if page == "Home":

    st.title("🛡️ SafeVision AI")

    st.subheader(
        "AI-Based Smart Security & Privacy Monitoring System"
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Detection Modules", "6")
    col2.metric("Sample Alerts", "5")
    col3.metric("Privacy Protection", "Enabled")

    st.markdown("---")

    st.header("Welcome to SafeVision AI")

    st.write(
        "SafeVision AI uses Computer Vision and "
        "Artificial Intelligence to monitor security "
        "events and support privacy protection."
    )

    st.info(
        "Select a monitoring module from the sidebar."
    )

# -------------------------------
# PERSON DETECTION
# -------------------------------

elif page == "Person Detection":

    st.title("👤 Person Detection")

    st.write(
        "Upload a CCTV video and detect people using YOLO AI."
    )

    uploaded_video = st.file_uploader(
        "Upload CCTV Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        input_file = None
        preview_file = None
        output_file = None
        final_file = None

        try:

            # Save uploaded video
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp:

                temp.write(uploaded_video.getvalue())
                input_file = temp.name

            # Convert for browser preview
            st.info("Preparing video preview...")

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp:

                preview_file = temp.name

            convert_video(input_file, preview_file)

            st.success("Video uploaded successfully!")

            st.subheader("Video Preview")

            with open(preview_file, "rb") as video:
                st.video(video.read())

            # Start detection
            if st.button("Start Person Detection"):

                st.info("Loading YOLO AI model...")

                model = load_model()

                cap = cv2.VideoCapture(preview_file)

                if not cap.isOpened():
                    st.error("Unable to open video.")
                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )
                height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                if fps <= 0:
                    fps = 25

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as temp:

                    output_file = temp.name

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                out = cv2.VideoWriter(
                    output_file,
                    fourcc,
                    fps,
                    (width, height)
                )

                progress = st.progress(0)

                status = st.empty()

                frame_number = 0

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    results = model(
                        frame,
                        classes=[0],
                        verbose=False
                    )

                    annotated_frame = results[0].plot()

                    out.write(annotated_frame)

                    frame_number += 1

                    if total_frames > 0:

                        progress.progress(
                            min(
                                frame_number / total_frames,
                                1.0
                            )
                        )

                    status.write(
                        f"Processing frame: {frame_number}"
                    )

                cap.release()
                out.release()

                st.success(
                    "Person detection completed successfully!"
                )

                # Convert detected video to browser format
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as temp:

                    final_file = temp.name

                convert_video(output_file, final_file)

                st.subheader("🎯 Detection Result")

                with open(final_file, "rb") as video:
                    video_bytes = video.read()

                st.video(video_bytes)

                st.download_button(
                    label="⬇️ Download Detected Video",
                    data=video_bytes,
                    file_name="safevision_detection.mp4",
                    mime="video/mp4"
                )

        except Exception as error:

            st.error("Something went wrong.")

            st.exception(error)

        finally:

            for file_path in [
                input_file,
                preview_file,
                output_file,
                final_file
            ]:

                if file_path and os.path.exists(file_path):
                    os.remove(file_path)

# -------------------------------
# OTHER MODULES
# -------------------------------

else:

    st.title(page)

    st.info(
        "This module will be connected to the "
        "SafeVision AI detection system in the next steps."
    )

# -------------------------------
# FOOTER
# -------------------------------

st.sidebar.markdown("---")

st.sidebar.caption(
    "SafeVision AI | Project Prototype"
)
