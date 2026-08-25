"""
Video Dialogue Localization — Demonstration Script comparing Small vs Base Models
"""

import sys
from pathlib import Path

# Add src/ directory to Python path
sys.path.insert(0, str(Path("src").resolve()))

from pipeline import localize_dialogue

if __name__ == "__main__":
    video_file = Path("output/248244667877.mp4")
    query = "My mind rebels at stagnation"

    print("\n" + "=" * 65)
    print("      COMPARING FASTER-WHISPER 'SMALL' vs 'BASE' MODELS")
    print("=" * 65)

    # Test 1: Using 'small' model
    print("\n--- [1] TESTING FASTER-WHISPER 'SMALL' MODEL ---")
    result_small = localize_dialogue(
        video_url_or_path=video_file,
        dialogue_query=query,
        model_size="small"
    )
    result_small.display()

    # Test 2: Using 'base' model
    print("\n--- [2] TESTING FASTER-WHISPER 'BASE' MODEL ---")
    result_base = localize_dialogue(
        video_url_or_path=video_file,
        dialogue_query=query,
        model_size="base"
    )
    result_base.display()
