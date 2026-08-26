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

    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["standard", "v1", "v2", "coarse_to_fine"],
        help="ASR pipeline mode: 'standard' (V1) or 'v2' (V2 optimization).",
    )

    args = parser.parse_args()

    url_or_path = args.url
    query = args.query
    model_size = args.model
    mode = args.mode

    print("\n" + "=" * 65)
    print("      VIDEO DIALOGUE LOCALIZATION SYSTEM")
    print("=" * 65)

    # Interactive input prompt if arguments not passed via command line
    if not url_or_path:
        url_or_path = input("\n[1] Enter Video URL or File Path [Default: output/248244667877.mp4]: ").strip().strip('"\'“”')
        if not url_or_path:
            url_or_path = "output/248244667877.mp4"
    else:
        url_or_path = url_or_path.strip().strip('"\'“”')

    if not query:
        query = input("\n[2] Enter Target Dialogue Phrase to Locate: ").strip().strip('"\'“”')
        while not query:
            query = input("    Please enter a non-empty phrase: ").strip().strip('"\'“”')
    else:
        query = query.strip().strip('"\'“”')

    if len(sys.argv) == 1:
        print("\n[3] Select ASR Pipeline Mode:")
        print("    1. Standard V1 (Full Audio ASR)")
        print("    2. V2 Optimization (Two-Stage Pipeline)")
        mode_choice = input("    Select option [1/2, default 1]: ").strip()
        if mode_choice == "2":
            mode = "v2"

    print(f"\nProcessing Video:  '{url_or_path}'")
    print(f"Target Phrase:     '{query}'")
    print(f"ASR Model Size:    '{model_size}'")
    print(f"ASR Mode:          '{mode}'")

    try:
        result = localize_dialogue(
            video_url_or_path=url_or_path,
            dialogue_query=query,
            output_dir="output",
            model_size=model_size,
            mode=mode,
        )
    except Exception as exc:
        print("\n" + "=" * 65)
        print("         VIDEO DIALOGUE LOCALIZATION ERROR RESULT")
        print("=" * 65)
        print(f" [ERROR] {exc}")
        print("=" * 65 + "\n")
        sys.exit(1)

    result.display()

    if result.match_found:
        print(f"JSON Result Exported to: {Path('output') / 'localization_result.json'}")
        import json
        with open(Path("output") / "localization_result.json", "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
