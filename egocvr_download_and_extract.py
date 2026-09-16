#!/usr/bin/env python3
"""
EgoCVR Dataset: Download full videos from Ego4D and extract EgoCVR clips defined in a CSV.

CSV format expected:
    clip_name,narration_text,video_uid
    <video_uid>_<start>_<end>, ..., <video_uid>

Timestamp formats supported in clip_name:
  - Coarse (integer seconds):  d1d1b6da-..._971_980
  - Sub-second (decimal, dash-separated): f4390995-..._243-2_251-2
      → 243-2  means 243.2 s
      → 21-267 means 21.267 s  (arbitrary decimal digits)
  - Mixed decimals:  f54fd12b-..._515-088_523-088
      → 515-088 means 515.088 s

Usage
-----
    # 1. Install the Ego4D CLI (once):
    #    pip install ego4d

    # 2. Run:
    python egocvr_download_and_extract.py \
        --csv      clips.csv \
        --output   ./clips \
        --videos   ./full_videos \

Dependencies: ego4d CLI, ffmpeg (must be on PATH), pandas, tqdm
"""

import argparse
import csv
import logging
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


def parse_timestamp(token: str) -> float:
    """
    Convert a timestamp token from a clip_name into seconds (float).

    Supported formats
    -----------------
    "971"       → 971.0   (plain integer → whole seconds)
    "243-2"     → 243.2   (dash separates integer and first decimal digit)
    "130-7"     → 130.7
    "21-267"    → 21.267  (multiple decimal digits)
    "515-088"   → 515.088
    """
    if "-" in token:
        integer_part, decimal_part = token.split("-", 1)
        return float(f"{integer_part}.{decimal_part}")
    return float(token)


def parse_clip_name(clip_name: str):
    """
    Extract (video_uid, start_sec, end_sec) from a clip_name string.

    The clip_name is structured as:
        <video_uid>_<start_token>_<end_token>

    where video_uid itself contains hyphens AND underscores are forbidden
    inside a UUID, so we split on the LAST two underscores.

    Returns
    -------
    (video_uid: str, start: float, end: float)
    """
    # UUID-style: 8-4-4-4-12 hex characters separated by hyphens.
    # Everything after the UUID (+ separating underscore) is the timestamp pair.
    uuid_pattern = r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    m = re.match(uuid_pattern, clip_name, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse video_uid from clip_name: {clip_name!r}")

    video_uid = m.group(1)
    remainder = clip_name[len(video_uid) :]  # e.g.  "_971_980"  or "_21-267_29-267"

    # remainder starts with "_", then two underscore-separated timestamp tokens
    parts = remainder.lstrip("_").split("_")
    if len(parts) != 2:
        raise ValueError(
            f"Expected exactly 2 timestamp tokens after video_uid, "
            f"got {parts!r} in clip_name: {clip_name!r}"
        )

    start = parse_timestamp(parts[0])
    end = parse_timestamp(parts[1])
    return video_uid, start, end


# ---------------------------------------------------------------------------
# Ego4D download helper
# ---------------------------------------------------------------------------


def download_videos(
    video_uids: list[str], video_dir: Path, skip_confirmation: bool
) -> None:
    """
    Download a list of Ego4D full videos using the official CLI.

    The CLI reference:
        ego4d --output_directory <dir> --datasets full_scale \
              --video_uids uid1 uid2 ...

    Adjust `--datasets` / flags to match your Ego4D licence and needs.
    """
    video_dir.mkdir(parents=True, exist_ok=True)

    # Only download videos not already present
    missing = [uid for uid in video_uids if not (video_dir / f"{uid}.mp4").exists()]

    if not missing:
        log.info("All %d unique videos already downloaded.", len(video_uids))
        return

    log.info("Downloading %d missing video(s) via Ego4D CLI …", len(missing))

    cmd = [
        "ego4d",
        "--output_directory",
        str(video_dir),
        "--datasets",
        "full_scale",
        "--video_uids",
        *missing,
    ]

    if skip_confirmation:
        cmd.append("-y")

    log.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        log.error(
            "ego4d CLI exited with code %d. Check credentials / UIDs.",
            result.returncode,
        )
    else:
        log.info("Download complete.")


# ---------------------------------------------------------------------------
# FFmpeg clip extraction
# ---------------------------------------------------------------------------


def extract_clip(
    source_video: Path,
    out_path: Path,
    start: float,
    end: float,
    pad: float = 0.0,
) -> bool:
    """
    Use ffmpeg to extract [start, end] seconds from source_video → out_path.

    Parameters
    ----------
    pad   : optional padding in seconds added symmetrically around the clip
            (clamped to 0 at the lower bound).
    """
    if out_path.exists():
        log.debug("Clip already exists, skipping: %s", out_path.name)
        return True

    t_start = max(0.0, start - pad)
    duration = (end + pad) - t_start

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-ss",
        f"{t_start:.6f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration:.6f}",
        "-loglevel",
        "quiet",
        "-c:a",
        "ac3",
        "-c:v",
        "libx264",
        "--",
        str(out_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        log.warning(
            "ffmpeg failed for %s (start=%.3f, end=%.3f):\n%s",
            out_path.name,
            start,
            end,
            result.stderr.decode(errors="replace"),
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def load_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    log.info("Loaded %d rows from %s", len(rows), csv_path)
    return rows


def run(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    output_dir = Path(args.output)
    video_dir = Path(args.videos)

    video_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Parse CSV ──────────────────────────────────────────────────────
    rows = load_csv(csv_path)

    parsed_clips: list[tuple[str, str, float, float]] = (
        []
    )  # (clip_name, video_uid, start, end)
    for row in rows:
        clip_name = row["clip_name"].strip()
        csv_uid = row.get("video_uid", "").strip()
        try:
            vid_uid, start, end = parse_clip_name(clip_name)
        except ValueError as exc:
            log.warning("Skipping unparseable clip_name %r: %s", clip_name, exc)
            continue

        # Sanity-check against the explicit video_uid column when present
        if csv_uid and csv_uid != vid_uid:
            log.warning(
                "clip_name-derived UID %r differs from CSV column %r for %r — "
                "using CSV column value.",
                vid_uid,
                csv_uid,
                clip_name,
            )
            vid_uid = csv_uid

        parsed_clips.append((clip_name, vid_uid, start, end))

    unique_uids = sorted({uid for _, uid, _, _ in parsed_clips})
    log.info(
        "Found %d clips across %d unique video(s).", len(parsed_clips), len(unique_uids)
    )

    # ── 2. Download full videos ───────────────────────────────────────────
    download_videos(unique_uids, video_dir, args.yes)

    # # ── 3. Extract clips ──────────────────────────────────────────────────
    success = failed = skipped = 0

    for clip_name, video_uid, start, end in parsed_clips:
        source = video_dir / "v2" / "full_scale" / f"{video_uid}.mp4"
        if not source.exists():
            log.warning(
                "Source video not found, skipping clip %s: %s", clip_name, source
            )
            skipped += 1
            continue

        out_path_dir = output_dir / video_uid
        out_path_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_path_dir / f"{clip_name}.mp4"
        ok = extract_clip(source, out_path, start, end, pad=args.pad)
        if ok:
            success += 1
        else:
            failed += 1

    log.info(
        "Done. Extracted: %d  |  Failed: %d  |  Skipped (missing video): %d",
        success,
        failed,
        skipped,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download Ego4D full videos and extract clips defined in a CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True, help="Path to the input CSV file.")
    p.add_argument("--output", required=True, help="Directory to save extracted clips.")
    p.add_argument(
        "--videos", required=True, help="Directory to store / find full videos."
    )
    p.add_argument(
        "--pad",
        type=float,
        default=0.0,
        help="Seconds of padding to add before/after each clip boundary.",
    )
    p.add_argument(
        "--yes", action="store_true", help="Run Ego4D CLI without confirming download."
    )
    p.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    run(args)
