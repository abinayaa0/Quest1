"""
Query Result History Logger — Appends all localization results to Excel (.xlsx) and CSV (.csv).
"""

import csv
import datetime
import logging
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)

# CSV and Excel headers
HEADERS = [
    "Execution Timestamp",
    "Video Source",
    "Target Query",
    "Match Found",
    "Timestamp (HMS)",
    "Timestamp (Seconds)",
    "Frame Number",
    "Extracted Spoken Text",
    "Confidence Score (%)",
    "Quality Tier",
    "Frame Image Path",
    "Frame Resolution",
    "Pipeline Latency (s)",
    "Pipeline Mode",
]


def append_result_to_csv(row_data: list[Any], csv_path: Path):
    """Append a single localization result row to CSV file with retry on file locks."""
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    for attempt in range(3):
        try:
            with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(HEADERS)
                writer.writerow(row_data)
            logger.info(f"Appended query result to CSV: {csv_path}")
            return
        except PermissionError:
            import time
            time.sleep(0.2)
        except Exception as e:
            logger.warning(f"Failed to append query result to CSV: {e}")
            return


def append_result_to_excel(row_data: list[Any], xlsx_path: Path):
    """Append a single localization result row to Excel (.xlsx) file with retry on file locks."""
    for attempt in range(3):
        try:
            import pandas as pd
            import openpyxl

            if xlsx_path.exists() and xlsx_path.stat().st_size > 0:
                df_existing = pd.read_excel(xlsx_path)
                df_new = pd.DataFrame([row_data], columns=HEADERS)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = pd.DataFrame([row_data], columns=HEADERS)

            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df_combined.to_excel(writer, index=False, sheet_name="Query History")

            logger.info(f"Appended query result to Excel: {xlsx_path}")
            return
        except PermissionError:
            import time
            time.sleep(0.2)
        except Exception as e:
            logger.warning(f"Failed to append query result to Excel: {e}")
            return


def log_query_result(result: Any, mode: str = "v2", output_dir: Union[str, Path] = "output"):
    """
    Log localization result to both CSV and Excel (.xlsx) files in output_dir.

    Args:
        result: LocalizationResult object from pipeline.
        mode: Pipeline mode ('v2' or 'standard').
        output_dir: Output directory path.
    """
    try:
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ts_sec = round(result.timestamp, 2) if result.timestamp is not None else 0.0
        frame_no = result.frame_number if result.frame_number is not None else "N/A"
        frame_img = str(result.frame_image_path) if result.frame_image_path else ""
        res_str = f"{result.width}x{result.height}" if (result.width and result.height) else "N/A"

        row = [
            now_str,
            str(result.video_path),
            result.dialogue_query,
            result.match_found,
            result.timestamp_hms if result.match_found else "N/A",
            ts_sec if result.match_found else "N/A",
            frame_no if result.match_found else "N/A",
            result.extracted_dialogue_text if result.match_found else "N/A",
            round(result.confidence, 1) if result.match_found else 0.0,
            result.confidence_quality if result.match_found else "N/A",
            frame_img if result.match_found else "N/A",
            res_str if result.match_found else "N/A",
            round(result.pipeline_duration_seconds, 3),
            mode,
        ]

        csv_file = out_path / "query_history.csv"
        xlsx_file = out_path / "query_history.xlsx"

        append_result_to_csv(row, csv_file)
        append_result_to_excel(row, xlsx_file)

    except Exception as e:
        logger.error(f"Error logging query result history: {e}")
