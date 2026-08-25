"""
Video Dialogue Localization — Interactive CLI Tool
===================================================
Run dialogue localization by entering any Video URL (or file path) and target dialogue phrase.
"""

import argparse
import sys
from pathlib import Path

# Add src/ directory to Python path
sys.path.insert(0, str(Path("src").resolve()))

from pipeline import localize_dialogue


def main():
    parser = argparse.ArgumentParser(
        description="Video Dialogue Localization — Find exact frame for spoken dialogue."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Public video URL (OK.ru, YouTube, Vimeo) or local video file path.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Target spoken dialogue phrase to locate in video.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium"],
        help="Faster-Whisper model size (default 'small').",
    )

    args = parser.parse_args()

    url_or_path = args.url
    query = args.query
    model_size = args.model

    print("\n" + "=" * 65)
    print("      VIDEO DIALOGUE LOCALIZATION SYSTEM")
    print("=" * 65)

    # Interactive input prompt if arguments not passed via command line
    if not url_or_path:
        url_or_path = input("\n[1] Enter Video URL or File Path [Default: output/248244667877.mp4]: ").strip()
        if not url_or_path:
            url_or_path = "output/248244667877.mp4"

    if not query:
        query = input("\n[2] Enter Target Dialogue Phrase to Locate: ").strip()
        while not query:
            query = input("    Please enter a non-empty phrase: ").strip()

    print(f"\nProcessing Video:  '{url_or_path}'")
    print(f"Target Phrase:     '{query}'")
    print(f"ASR Model Size:    '{model_size}'")

    result = localize_dialogue(
        video_url_or_path=url_or_path,
        dialogue_query=query,
        output_dir="output",
        model_size=model_size,
    )

    result.display()

    if result.match_found:
        print(f"JSON Result Exported to: {Path('output') / 'localization_result.json'}")
        import json
        with open(Path("output") / "localization_result.json", "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
