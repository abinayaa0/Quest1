"""
Streamlit Web Application — Video Dialogue Localization System
================================================================
Direct Pipeline Streamlit Web UI.
"""

import sys
from pathlib import Path

import streamlit as st

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from pipeline import localize_dialogue

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Video Dialogue Localization System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
        "**V2 Optimization (Coarse-to-Fine ASR)**."
    )
    st.sidebar.header("📊 History Log")
    xlsx_path = Path("output/query_history.xlsx")
    csv_path = Path("output/query_history.csv")

    if xlsx_path.exists():
        with open(xlsx_path, "rb") as f:
            st.sidebar.download_button(
                label="📥 Download History (Excel .xlsx)",
                data=f,
                file_name="query_history.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    if csv_path.exists():
        with open(csv_path, "rb") as f:
            st.sidebar.download_button(
                label="📄 Download History (CSV .csv)",
                data=f,
                file_name="query_history.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.sidebar.header("📚 System Documentation")
    pdf_path = Path("Video_Dialogue_Localization_Documentation.pdf")
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            st.sidebar.download_button(
                label="📕 Download Combined PDF Docs",
                data=f,
                file_name="Video_Dialogue_Localization_Documentation.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    st.markdown("---")

    col_input, col_info = st.columns([2, 1])

    with col_input:
        st.subheader("1. Video Source")
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
                st.success(f"Uploaded & saved: `{saved_path}`")

        st.subheader("2. Dialogue Query")
        dialogue_query = st.text_input(
            "Enter Spoken Dialogue Phrase to Locate:",
            value="the blue carbuncle",
            placeholder="Type spoken quote here...",
        )

        find_button = st.button("🔍 Find Dialogue Frame", type="primary", use_container_width=True)

    with col_info:
        st.info(
            "**System Pipeline Architecture:**\n"
            "• **Ingestion**: `yt-dlp` / FFmpeg\n"
            "• **Stage 1**: Fast Coarse ASR (base)\n"
            "• **Stage 2**: Fuzzy Top-K Search\n"
            "• **Stage 3**: Fine Word-Timestamped ASR (small)\n"
            "• **Extraction**: Sample-Accurate Frame"
        )

    if find_button:
        if not video_source or not video_source.strip():
            st.error("Please enter a Video URL or upload a Video File.")
            return

        if not dialogue_query or not dialogue_query.strip():
            st.error("Please enter a dialogue query phrase.")
            return

        with st.spinner("Processing video and locating exact dialogue frame..."):
            try:
                res = localize_dialogue(
                    video_url_or_path=video_source.strip(),
                    dialogue_query=dialogue_query.strip(),
                    mode="v2",
                )
            except Exception as ex:
                st.error(f"Pipeline Execution Error: {ex}")
                return

            st.markdown("---")
            st.header("🎯 Localization Output Result")

            if not res.match_found:
                st.error("⚠️ No Dialogue Match Found in Video (`reason: dialogue_not_found`)")
            else:
                ts_formatted = res.timestamp_hms
                ts_seconds = round(res.timestamp, 2) if res.timestamp is not None else 0.0
                frame_no = res.frame_number if res.frame_number is not None else "N/A"
                extracted_text = res.extracted_dialogue_text or ""
                confidence = round(res.confidence, 1)
                frame_path = str(res.frame_image_path) if res.frame_image_path else ""
                width = res.width
                height = res.height
                duration = round(res.pipeline_duration_seconds, 3)

                # Quality tier label
                if confidence > 90.0:
                    quality = "Strong Match"
                elif confidence >= 80.0:
                    quality = "Acceptable"
                elif confidence >= 70.0:
                    quality = "Needs Review"
                else:
                    quality = "Reject"

                st.success(f"✅ **Match Located Successfully!** (Quality Tier: **{quality}**)")

                # 4 Clean Top Metric Cards (Non-redundant)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("⏱️ Timestamp (HMS)", ts_formatted, delta=f"{ts_seconds:.2f} seconds")
                m2.metric("🎞️ Frame Number", f"#{frame_no}")
                m3.metric("🎯 Confidence Score", f"{confidence:.1f}%", delta=quality)
                m4.metric("⚡ Pipeline Latency", f"{duration:.3f}s")

                st.markdown("---")

                # Dialogue Details Card & Frame Image Side-by-Side
                c_details, c_image = st.columns([1, 1])

                with c_details:
                    st.subheader("📝 Dialogue Details")
                    st.markdown(f"**Target Query:** `{dialogue_query.strip()}`")
                    st.markdown(f"**Extracted Spoken Text:**")
                    st.info(f"💬 *\"{extracted_text}\"*")
                    st.markdown(f"**Frame Resolution:** `{width} x {height}`")
                    st.markdown(f"**Image File Path:** `{frame_path}`")

                with c_image:
                    st.subheader("🖼️ Extracted Video Frame")
                    if frame_path and Path(frame_path).exists():
                        st.image(
                            frame_path,
                            caption=f"Frame #{frame_no} at {ts_formatted} ({ts_seconds:.2f}s) | Confidence: {confidence:.1f}%",
                            use_container_width=True,
                        )
                    else:
                        st.warning(f"Frame image file path not found locally: `{frame_path}`")

            # -------------------------------------------------------------
            # History Excel & CSV Logging & Interactive Preview
            # -------------------------------------------------------------
            try:
                from history_logger import log_query_result
                log_query_result(res, mode="v2")
            except Exception:
                pass

            st.markdown("---")
            st.subheader("📊 Saved Query History Log")

            xlsx_file = Path("output/query_history.xlsx")
            csv_file = Path("output/query_history.csv")

            if xlsx_file.exists():
                try:
                    import pandas as pd
                    df_hist = pd.read_excel(xlsx_file)
                    st.markdown(f"**Latest Logged Queries (Total Logged: `{len(df_hist)}` records in `output/query_history.xlsx`):**")
                    st.dataframe(df_hist.tail(5), use_container_width=True)
                except Exception as ex:
                    st.warning(f"Could not load Excel history preview: {ex}")

            c1, c2 = st.columns(2)
            if xlsx_file.exists():
                with open(xlsx_file, "rb") as f:
                    c1.download_button(
                        label="📥 Download History Spreadsheet (Excel .xlsx)",
                        data=f,
                        file_name="query_history.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="dl_xlsx_live",
                    )
            if csv_file.exists():
                with open(csv_file, "rb") as f:
                    c2.download_button(
                        label="📄 Download History Document (CSV .csv)",
                        data=f,
                        file_name="query_history.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_csv_live",
                    )


if __name__ == "__main__":
    main()
