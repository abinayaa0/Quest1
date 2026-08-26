"""
Streamlit Web Application — Video Dialogue Localization System
"""

import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Video Dialogue Localization System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Base API URL
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded video file to local output/ directory."""
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def main():
    st.title("🎬 Video Dialogue Localization System")
    st.markdown(
        "Locate exact video frames and timestamps corresponding to spoken dialogue quotes using "
        "**V2 Coarse-to-Fine ASR**."
    )

    st.sidebar.header("⚙️ Configuration")
    backend_mode = st.sidebar.radio(
        "Backend Connection:",
        ["FastAPI Backend (HTTP)", "Direct Local Pipeline"],
        index=0,
    )

    st.markdown("---")

    col_input, col_info = st.columns([2, 1])

    with col_input:
        st.subheader("1. Video Input Source")
        source_type = st.radio(
            "Select Video Source Type:",
            ["Video URL (OK.ru, YouTube, Direct MP4)", "Upload Local Video File"],
            horizontal=True,
        )

        video_source = ""
        if "URL" in source_type:
            video_source = st.text_input(
                "Enter Video URL or Path:",
                value="https://ok.ru/video/7869007661646",
                placeholder="https://ok.ru/video/... or output/sample.mp4",
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload Video File:",
                type=["mp4", "mkv", "avi", "mov", "webm"],
            )
            if uploaded_file is not None:
                saved_path = save_uploaded_file(uploaded_file)
                video_source = str(saved_path)
                st.success(f"Uploaded and saved: `{saved_path}`")

        st.subheader("2. Dialogue Query")
        dialogue_query = st.text_input(
            "Enter Spoken Dialogue Phrase to Locate:",
            value="the blue carbuncle",
            placeholder="Type quote here...",
        )

        find_button = st.button("🔍 Find Dialogue Frame", type="primary", use_container_width=True)

    with col_info:
        st.info(
            "**System Pipeline Flow:**\n"
            "1. Audio Extraction\n"
            "2. Coarse-to-Fine ASR\n"
            "3. RapidFuzz Dialogue Matching\n"
            "4. Sample-Accurate Frame Extraction"
        )

    if find_button:
        if not video_source or not video_source.strip():
            st.error("Please enter a Video URL or upload a Video File.")
            return

        if not dialogue_query or not dialogue_query.strip():
            st.error("Please enter a dialogue query phrase.")
            return

        with st.spinner("Processing video and locating dialogue quote..."):
            result_data = None
            error_msg = None

            if "FastAPI" in backend_mode:
                try:
                    resp = requests.post(
                        f"{API_URL}/localize",
                        json={"video_source": video_source.strip(), "dialogue_query": dialogue_query.strip()},
                        timeout=1800,
                    )
                    if resp.status_code == 200:
                        result_data = resp.json()
                    else:
                        error_msg = f"FastAPI Server returned status {resp.status_code}: {resp.text}"
                except Exception as e:
                    error_msg = f"Failed to connect to FastAPI Backend at {API_URL}: {e}"

            # Fallback to direct pipeline if API fails or direct mode selected
            if result_data is None:
                if error_msg and "FastAPI" in backend_mode:
                    st.warning(f"{error_msg} -> Falling back to Direct Local Pipeline.")
                try:

                    src_dir = Path(__file__).resolve().parent / "src"
                    if str(src_dir) not in sys.path:
                        sys.path.insert(0, str(src_dir))
                    from pipeline import localize_dialogue

                    res = localize_dialogue(
                        video_url_or_path=video_source.strip(),
                        dialogue_query=dialogue_query.strip(),
                        mode="v2",
                    )
                    if res.match_found:
                        ts_val = round(res.timestamp, 2) if res.timestamp is not None else 0.0
                        result_data = {
                            "match_found": True,
                            "timestamp": {
                                "seconds": ts_val,
                                "formatted": res.timestamp_hms,
                            },
                            "frame_number": res.frame_number,
                            "dialogue": res.extracted_dialogue_text or "",
                            "confidence": round(res.confidence, 1),
                            "frame_path": str(res.frame_image_path) if res.frame_image_path else "",
                            "width": res.width,
                            "height": res.height,
                            "pipeline_duration_seconds": round(res.pipeline_duration_seconds, 3),
                        }
                    else:
                        result_data = {"match_found": False, "reason": "dialogue_not_found"}
                except Exception as ex:
                    st.error(f"Pipeline Execution Error: {ex}")
                    return

            st.markdown("---")
            st.header("OUTPUT RESULT")

            if not result_data.get("match_found", False):
                st.warning("⚠️ No Dialogue Match Found in Video (`reason: dialogue_not_found`)")
            else:
                match = result_data
                ts_formatted = match.get("timestamp", {}).get("formatted", "N/A")
                ts_seconds = match.get("timestamp", {}).get("seconds", 0.0)
                frame_no = match.get("frame_number", "N/A")
                extracted_text = match.get("dialogue", "")
                confidence = match.get("confidence", 0.0)
                frame_path = match.get("frame_path", "")
                width = match.get("width", 1280)
                height = match.get("height", 720)
                duration = match.get("pipeline_duration_seconds", 0.0)

                # Required Terminal-Style Field Summary Display
                st.code(
                    f" * Dialogue Query:           \"{dialogue_query.strip()}\"\n"
                    f" * Match Found:              True\n"
                    f" * Timestamp of Frame (HMS): {ts_formatted}\n"
                    f" * Timestamp (Seconds):      {ts_seconds:.2f}s\n"
                    f" * Frame Number:             {frame_no}\n"
                    f" * Extracted Dialogue Text:  \"{extracted_text}\"\n"
                    f" * Confidence Score:         {confidence:.1f}%\n"
                    f" * Corresponding Frame Image:{frame_path}\n"
                    f" * Frame Resolution:         {width} x {height}\n"
                    f" * Pipeline Execution Time:  {duration:.3f} seconds\n",
                    language="yaml"
                )

                # Metrics Overview Cards
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Timestamp (HMS)", ts_formatted)
                m2.metric("Timestamp (Sec)", f"{ts_seconds:.2f}s")
                m3.metric("Frame Number", frame_no)
                m4.metric("Confidence Score", f"{confidence:.1f}%")

                st.markdown(f"**Extracted Dialogue:** *\"{extracted_text}\"*")
                st.markdown(f"**Resolution:** `{width} x {height}` | **Execution Latency:** `{duration:.3f}s`")

                # Frame Image Display Box
                st.subheader("🖼️ Corresponding Frame Image")
                if frame_path and Path(frame_path).exists():
                    st.image(
                        frame_path,
                        caption=f"Frame #{frame_no} at {ts_formatted} ({ts_seconds:.2f}s) — Confidence: {confidence:.1f}%",
                        use_container_width=True,
                    )
                else:
                    st.warning(f"Frame image file path not found locally: `{frame_path}`")


if __name__ == "__main__":
    main()
