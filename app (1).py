
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SafeVision AI",
    page_icon="🛡️",
    layout="wide"
)

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

else:
    st.title(page)
    st.info(
        "This module will be connected to the SafeVision AI "
        "detection system in the next development steps."
    )

st.sidebar.markdown("---")
st.sidebar.caption("SafeVision AI | Project Prototype")
