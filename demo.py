"""
End-to-End Pipeline Demonstration Script
=========================================
Search for any spoken dialogue quote in the downloaded Sherlock Holmes video,
find its timestamp, and extract the corresponding video frame image!
"""

import sys
from pathlib import Path

# Add src/ directory to Python path
sys.path.insert(0, str(Path("src").resolve()))

from matching import match_dialogue
from frame_extraction import extract_frame


def locate_dialogue(target_quote: str):
    video_path = Path("output/248244667877.mp4")
    transcript_path = Path("output/248244667877_transcript_small.json")

    print("\n" + "=" * 60)
    print(f"SEARCHING FOR DIALOGUE: '{target_quote}'")
    print("=" * 60)

    if not video_path.exists() or not transcript_path.exists():
        print(f"ERROR: Required video or transcript file missing in output/ directory.")
        return

    # Phase 5: Match Dialogue to Timestamps
    result = match_dialogue(
        target_text=target_quote,
        transcript=transcript_path,
        confidence_threshold=75.0,
    )

    if not result.match_found:
        print(f"DIALOGUE NOT FOUND! Best match score was {result.confidence:.1f}%")
        return

    print(f"MATCH FOUND!")
    print(f"  * Matched Text:  '{result.matched_window_raw_text}'")
    print(f"  * Confidence:    {result.confidence:.1f}%")
    print(f"  * Start Time:    {result.start_time:.2f} seconds")
    print(f"  * End Time:      {result.end_time:.2f} seconds")

    # Phase 6: Extract Video Frame
    print(f"\nEXTRACTING VIDEO FRAME AT {result.start_time:.2f}s...")
    frame_res = extract_frame(
        video_path=video_path,
        timestamp=result.start_time,
        output_dir="output/frames",
    )

    print(f"FRAME EXTRACTED SUCCESSFULLY!")
    print(f"  * Saved Frame:   {frame_res.frame_path}")
    print(f"  * Resolution:    {frame_res.width} x {frame_res.height}")
    print(f"  * Extract Time:  {frame_res.extraction_duration_seconds}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test 1: Sherlock's famous line
    locate_dialogue("My mind rebels at stagnation")

    # Test 2: Opening narration line
    locate_dialogue("To Sherlock Holmes she was always the woman")
