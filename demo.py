"""
Video Dialogue Localization — End-to-End Execution Script
================================──────────────────────────
Given a target dialogue query, locates the exact timestamp, frame number,
extracted dialogue text, and outputs the video frame image.
"""

import sys
from pathlib import Path

# Add src/ directory to Python path
sys.path.insert(0, str(Path("src").resolve()))

from pipeline import localize_dialogue

if __name__ == "__main__":
    video_file = Path("output/248244667877.mp4")

    # Example 1: Sherlock's quote
    result1 = localize_dialogue(
        video_url_or_path=video_file,
        dialogue_query="My mind rebels at stagnation",
        model_size="small"
    )
    result1.display()

    # Example 2: Opening narration quote
    result2 = localize_dialogue(
        video_url_or_path=video_file,
        dialogue_query="To Sherlock Holmes she was always the woman",
        model_size="small"
    )
    result2.display()
