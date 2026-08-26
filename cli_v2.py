"""
Video Dialogue Localization — V2 Coarse-to-Fine Interactive CLI
================================================================
Runs V2 Coarse-to-Fine ASR optimization:
Stage 1: Fast coarse ASR (base) -> Stage 2: Candidate region detection -> Stage 3: Fine ASR (small)
"""

import argparse
import sys
from pathlib import Path

# Add src/ directory to Python path
sys.path.insert(0, str(Path("src").resolve()))

from pipeline import localize_dialogue


def main():
    parser = argparse.ArgumentParser(
        description="Video Dialogue Localization V2 — Coarse-to-Fine ASR Optimization CLI."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Public video URL (OK.ru, YouTube) or local video file path.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Target spoken dialogue phrase to locate in video.",
    )

    args = parser.parse_args()

    url_or_path = args.url
    query = args.query

    print("\n" + "=" * 65)
    print("   VIDEO DIALOGUE LOCALIZATION — V2 OPTIMIZATION")
    print("=" * 65)

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

    print(f"\nProcessing Video:  '{url_or_path}'")
    print(f"Target Phrase:     '{query}'")
    print("Pipeline Mode:     'v2' (Two-Stage Optimization)")

    try:
        result = localize_dialogue(
            video_url_or_path=url_or_path,
            dialogue_query=query,
            output_dir="output",
            model_size="small",
            mode="v2",
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
        print(f"JSON Result Exported to: {Path('output') / 'localization_result_v2.json'}")
        import json
        with open(Path("output") / "localization_result_v2.json", "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
