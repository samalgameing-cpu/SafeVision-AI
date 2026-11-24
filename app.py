
import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

# Page configuration
st.set_page_config(
    page_title="SafeVision AI",
    page_icon="🛡️",
    layout="wide"
)

# Load YOLO model once
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

# Custom styling
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
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
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

# Home Dashboard
if page == "Home":

    st.title("🛡️ SafeVision AI")
    st.subheader("AI-Based Smart Security & Privacy Monitoring System")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Detection Modules", "6")
    col2.metric("Sample Alerts", "5")
    col3.metric("Privacy Protection", "Enabled")

    st.markdown("---")

    st.header("Welcome to SafeVision AI")

    st.write(
        "SafeVision AI uses Computer Vision and Artificial "
        "Intelligence to monitor security events and support "
        "privacy protection."
    )

    st.info(
        "Select a monitoring module from the sidebar "
        "to explore the dashboard."
    )

# Person Detection Module
elif page == "Person Detection":

    st.title("👤 Person Detection")
    st.write("Upload a CCTV video to detect people using YOLO AI.")

    uploaded_video = st.file_uploader(
        "Upload CCTV Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("Start Person Detection"):

            with st.spinner("Loading YOLO model..."):

                model = load_model()

            temp_input = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            temp_input.write(uploaded_video.read())
            temp_input.close()

            cap = cv2.VideoCapture(temp_input.name)

            if not cap.isOpened():
                st.error("Unable to open uploaded video.")
                os.remove(temp_input.name)
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps <= 0:
                fps = 25

            output_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ).name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            progress = st.progress(0)
            status = st.empty()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
                        min(frame_number / total_frames, 1.0)
                    )

                status.write(
                    f"Processing frame: {frame_number}"
                )

            cap.release()
            out.release()

            os.remove(temp_input.name)

            st.success("Person detection completed!")

            st.subheader("Detection Result")

            with open(output_path, "rb") as video_file:
                video_bytes = video_file.read()

            st.video(video_bytes)

            st.download_button(
                "Download Detected Video",
                data=video_bytes,
                file_name="safevision_person_detection.mp4",
                mime="video/mp4"
            )

            os.remove(output_path)

# Other modules
else:

    st.title(page)

    st.info(
        "This module will be connected to the SafeVision AI "
        "detection system in the next development steps."
    )

st.sidebar.markdown("---")
st.sidebar.caption("SafeVision AI | Project Prototype")
