#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass
class SegmentRecord:
    sample_id: str
    video_id: str
    split: str
    start_sec: float
    end_sec: float
    center_sec: float
    duration_sec: float
    audio_path: str
    image_path: str
    source_video: str


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def check_ffmpeg_tools() -> None:
    for tool in ["ffmpeg", "ffprobe"]:
        if shutil.which(tool) is None:
            raise EnvironmentError(
                f"Required tool '{tool}' not found on PATH. "
                f"Install ffmpeg in Colab or your local environment."
            )


def list_video_files(raw_videos_dir: Path) -> List[Path]:
    files = [p for p in raw_videos_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    return sorted(files)


def get_video_duration(video_path: Path) -> Optional[float]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def stable_split(video_id: str, train_ratio: float, val_ratio: float, seed: int) -> str:
    # Stable split from video_id + seed, so reruns are deterministic.
    key = f"{video_id}|{seed}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF  # in [0,1]
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def extract_audio_segment(
    video_path: Path,
    out_wav: Path,
    start_sec: float,
    clip_len: float,
    sample_rate: int,
) -> bool:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(video_path),
        "-t", f"{clip_len:.3f}",
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        str(out_wav),
    ]
    result = run_cmd(cmd)
    return result.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 0


def extract_center_frame(
    video_path: Path,
    out_img: Path,
    center_sec: float,
) -> bool:
    out_img.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{center_sec:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(out_img),
    ]
    result = run_cmd(cmd)
    return result.returncode == 0 and out_img.exists() and out_img.stat().st_size > 0


def make_sample_id(video_id: str, start_sec: float, end_sec: float) -> str:
    token = f"{video_id}_{start_sec:.3f}_{end_sec:.3f}"
    return hashlib.md5(token.encode("utf-8")).hexdigest()[:16]


def write_metadata_csv(records: List[SegmentRecord], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "video_id",
        "split",
        "start_sec",
        "end_sec",
        "center_sec",
        "duration_sec",
        "audio_path",
        "image_path",
        "source_video",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "sample_id": r.sample_id,
                "video_id": r.video_id,
                "split": r.split,
                "start_sec": f"{r.start_sec:.3f}",
                "end_sec": f"{r.end_sec:.3f}",
                "center_sec": f"{r.center_sec:.3f}",
                "duration_sec": f"{r.duration_sec:.3f}",
                "audio_path": r.audio_path,
                "image_path": r.image_path,
                "source_video": r.source_video,
            })


def write_summary_json(summary: dict, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract paired audio segments and center frames from videos.")
    ap.add_argument("--project-root", type=str, required=True,
                    help="Root project directory, e.g. /content/drive/MyDrive/spectralbridge_landscape")
    ap.add_argument("--raw-videos-dir", type=str, default=None,
                    help="Optional override for raw videos directory")
    ap.add_argument("--audio-dir", type=str, default=None,
                    help="Optional override for extracted audio directory")
    ap.add_argument("--image-dir", type=str, default=None,
                    help="Optional override for extracted image directory")
    ap.add_argument("--metadata-dir", type=str, default=None,
                    help="Optional override for metadata directory")
    ap.add_argument("--segment-seconds", type=float, default=10.0)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--min-duration", type=float, default=10.0,
                    help="Minimum video duration to be considered")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--limit-videos", type=int, default=0,
                    help="If > 0, process only the first N videos (for debugging)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only compute planned segments and write metadata without extracting files")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    check_ffmpeg_tools()

    project_root = Path(args.project_root).expanduser().resolve()
    raw_videos_dir = Path(args.raw_videos_dir).resolve() if args.raw_videos_dir else project_root / "data" / "raw_videos"
    audio_dir = Path(args.audio_dir).resolve() if args.audio_dir else project_root / "data" / "audio"
    image_dir = Path(args.image_dir).resolve() if args.image_dir else project_root / "data" / "images"
    metadata_dir = Path(args.metadata_dir).resolve() if args.metadata_dir else project_root / "data" / "metadata"

    metadata_csv = metadata_dir / "segments.csv"
    summary_json = metadata_dir / "segments_summary.json"

    if not raw_videos_dir.exists():
        raise FileNotFoundError(f"Raw videos directory not found: {raw_videos_dir}")

    video_files = list_video_files(raw_videos_dir)
    if args.limit_videos > 0:
        video_files = video_files[:args.limit_videos]

    print(f"Found {len(video_files)} video files in {raw_videos_dir}")

    records: List[SegmentRecord] = []
    num_skipped_duration = 0
    num_failed_probe = 0
    num_failed_audio = 0
    num_failed_image = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    video_counts = {"train": 0, "val": 0, "test": 0}

    for i, video_path in enumerate(video_files, start=1):
        video_id = video_path.stem
        duration = get_video_duration(video_path)
        if duration is None:
            num_failed_probe += 1
            print(f"[{i}/{len(video_files)}] ffprobe failed: {video_path.name}")
            continue

        if duration < args.min_duration:
            num_skipped_duration += 1
            print(f"[{i}/{len(video_files)}] too short ({duration:.2f}s): {video_path.name}")
            continue

        split = stable_split(video_id, args.train_ratio, args.val_ratio, args.seed)
        video_counts[split] += 1

        num_segments = int(duration // args.segment_seconds)
        print(f"[{i}/{len(video_files)}] {video_path.name} | duration={duration:.2f}s | segments={num_segments} | split={split}")

        for seg_idx in range(num_segments):
            start_sec = seg_idx * args.segment_seconds
            end_sec = start_sec + args.segment_seconds
            center_sec = start_sec + args.segment_seconds / 2.0

            sample_id = make_sample_id(video_id, start_sec, end_sec)

            rel_audio = Path(split) / f"{sample_id}.wav"
            rel_image = Path(split) / f"{sample_id}.jpg"

            out_wav = audio_dir / rel_audio
            out_img = image_dir / rel_image

            ok_audio = True
            ok_image = True
            if not args.dry_run:
                ok_audio = extract_audio_segment(
                    video_path=video_path,
                    out_wav=out_wav,
                    start_sec=start_sec,
                    clip_len=args.segment_seconds,
                    sample_rate=args.sample_rate,
                )
                if not ok_audio:
                    num_failed_audio += 1
                    print(f"  audio extraction failed: {sample_id}")
                    continue

                ok_image = extract_center_frame(
                    video_path=video_path,
                    out_img=out_img,
                    center_sec=center_sec,
                )
                if not ok_image:
                    num_failed_image += 1
                    print(f"  image extraction failed: {sample_id}")
                    continue

            record = SegmentRecord(
                sample_id=sample_id,
                video_id=video_id,
                split=split,
                start_sec=start_sec,
                end_sec=end_sec,
                center_sec=center_sec,
                duration_sec=args.segment_seconds,
                audio_path=safe_relpath(out_wav, project_root),
                image_path=safe_relpath(out_img, project_root),
                source_video=safe_relpath(video_path, project_root),
            )
            records.append(record)
            split_counts[split] += 1

    write_metadata_csv(records, metadata_csv)

    summary = {
        "project_root": str(project_root),
        "raw_videos_dir": str(raw_videos_dir),
        "audio_dir": str(audio_dir),
        "image_dir": str(image_dir),
        "metadata_csv": str(metadata_csv),
        "segment_seconds": args.segment_seconds,
        "sample_rate": args.sample_rate,
        "num_videos_found": len(video_files),
        "num_records_written": len(records),
        "video_counts": video_counts,
        "split_counts": split_counts,
        "num_skipped_duration": num_skipped_duration,
        "num_failed_probe": num_failed_probe,
        "num_failed_audio": num_failed_audio,
        "num_failed_image": num_failed_image,
        "dry_run": args.dry_run,
    }
    write_summary_json(summary, summary_json)

    print("\nDone.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()