#!/usr/bin/env python3
"""
prepare_data.py - Convert RTime JSON files to CLIP4Clip CSV format

Usage:
    python scripts/prepare_data.py [--output_dir OUTPUT_DIR] [--no_rewrites]

This script converts the RTime dataset JSON files into CSV format compatible
with CLIP4Clip training and evaluation.

Output files:
    - data/rtime_train.csv: Training data with GPT rewrites expanded
    - data/rtime_valid.csv: Validation data (human captions only)
    - data/rtime_test.csv: Test data - Hard setting (original + reversed videos)
    - data/rtime_test_origin.csv: Test data - Origin setting (original videos only)
    - data/video_ids.txt: List of all video IDs for download verification
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_json(path: str) -> dict:
    """Load a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_train_csv(data: dict, output_path: str, include_rewrites: bool = True) -> int:
    """
    Create training CSV with expanded captions.

    For training, we include:
    - All forward_captions (human captions)
    - All forward_rewrites (GPT-4 paraphrases) if include_rewrites=True
    - All reverse_captions and reverse_rewrites for videos with reverse=True

    Args:
        data: RTime JSON data (video_id -> metadata dict)
        output_path: Path to save the CSV
        include_rewrites: Whether to include GPT rewrites

    Returns:
        Number of rows written
    """
    rows = []

    for video_id, meta in data.items():
        # Forward captions (human annotations)
        for caption in meta.get('forward_captions', []):
            if caption.strip():
                rows.append((video_id, caption.strip()))

        # Forward rewrites (GPT-4 paraphrases)
        if include_rewrites:
            for caption in meta.get('forward_rewrites', []):
                if caption.strip():
                    rows.append((video_id, caption.strip()))

        # If this video has a reversed version
        if meta.get('reverse', False):
            rev_video_id = f"{video_id}_rev"

            # Reverse captions (human annotations for reversed video)
            for caption in meta.get('reverse_captions', []):
                if caption.strip():
                    rows.append((rev_video_id, caption.strip()))

            # Reverse rewrites (GPT-4 paraphrases for reversed video)
            if include_rewrites:
                for caption in meta.get('reverse_rewrites', []):
                    if caption.strip():
                        rows.append((rev_video_id, caption.strip()))

    # Write CSV
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('video_id,sentence\n')
        for video_id, sentence in rows:
            # Escape quotes in sentence
            sentence_escaped = sentence.replace('"', '""')
            f.write(f'{video_id},"{sentence_escaped}"\n')

    return len(rows)


def create_eval_csv(data: dict, output_path: str, include_reversed: bool = False) -> int:
    """
    Create evaluation CSV.

    For evaluation, we use only the first human caption per video (no rewrites).

    Args:
        data: RTime JSON data (video_id -> metadata dict)
        output_path: Path to save the CSV
        include_reversed: If True, include reversed videos (Hard setting)

    Returns:
        Number of rows written
    """
    rows = []

    for video_id, meta in data.items():
        # Original video with first forward caption
        forward_captions = meta.get('forward_captions', [])
        if forward_captions:
            caption = forward_captions[0].strip()
            if caption:
                rows.append((video_id, caption))

        # Reversed video (only for Hard setting)
        if include_reversed and meta.get('reverse', False):
            rev_video_id = f"{video_id}_rev"
            reverse_captions = meta.get('reverse_captions', [])
            if reverse_captions:
                caption = reverse_captions[0].strip()
                if caption:
                    rows.append((rev_video_id, caption))

    # Write CSV
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('video_id,sentence\n')
        for video_id, sentence in rows:
            # Escape quotes in sentence
            sentence_escaped = sentence.replace('"', '""')
            f.write(f'{video_id},"{sentence_escaped}"\n')

    return len(rows)


def create_video_id_list(train_data: dict, valid_data: dict, test_data: dict, output_path: str) -> int:
    """
    Create a list of all video IDs needed for the benchmark.

    Args:
        train_data, valid_data, test_data: RTime JSON data
        output_path: Path to save the video ID list

    Returns:
        Number of unique video IDs
    """
    video_ids = set()

    for data in [train_data, valid_data, test_data]:
        for video_id, meta in data.items():
            video_ids.add(video_id)
            # Add reversed video ID if applicable
            if meta.get('reverse', False):
                video_ids.add(f"{video_id}_rev")

    # Write sorted list
    with open(output_path, 'w', encoding='utf-8') as f:
        for vid in sorted(video_ids):
            f.write(f'{vid}\n')

    return len(video_ids)


def get_dataset_stats(data: dict) -> dict:
    """Compute statistics for a dataset split."""
    total_videos = len(data)
    temporal_videos = sum(1 for m in data.values() if m.get('temporal', False))
    reversed_videos = sum(1 for m in data.values() if m.get('reverse', False))

    total_forward_captions = sum(len(m.get('forward_captions', [])) for m in data.values())
    total_forward_rewrites = sum(len(m.get('forward_rewrites', [])) for m in data.values())
    total_reverse_captions = sum(len(m.get('reverse_captions', [])) for m in data.values())
    total_reverse_rewrites = sum(len(m.get('reverse_rewrites', [])) for m in data.values())

    return {
        'total_videos': total_videos,
        'temporal_videos': temporal_videos,
        'reversed_videos': reversed_videos,
        'forward_captions': total_forward_captions,
        'forward_rewrites': total_forward_rewrites,
        'reverse_captions': total_reverse_captions,
        'reverse_rewrites': total_reverse_rewrites,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert RTime JSON files to CLIP4Clip CSV format'
    )
    parser.add_argument(
        '--rtime_dir', type=str, default='Reversed-in-Time',
        help='Directory containing RTime JSON files (default: Reversed-in-Time)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data',
        help='Output directory for CSV files (default: data)'
    )
    parser.add_argument(
        '--no_rewrites', action='store_true',
        help='Exclude GPT rewrites from training data'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Print detailed statistics'
    )

    args = parser.parse_args()

    # Determine project root (one level up from scripts/)
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    rtime_dir = project_dir / args.rtime_dir
    output_dir = project_dir / args.output_dir

    # Check input files exist
    train_json = rtime_dir / 'train.json'
    valid_json = rtime_dir / 'valid.json'
    test_json = rtime_dir / 'test.json'

    for json_file in [train_json, valid_json, test_json]:
        if not json_file.exists():
            print(f"ERROR: Required file not found: {json_file}")
            sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading RTime dataset...")
    train_data = load_json(train_json)
    valid_data = load_json(valid_json)
    test_data = load_json(test_json)

    if args.verbose:
        print("\nDataset statistics:")
        for name, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
            stats = get_dataset_stats(data)
            print(f"\n  {name}:")
            print(f"    Total videos: {stats['total_videos']}")
            print(f"    Temporal videos: {stats['temporal_videos']}")
            print(f"    Reversed videos: {stats['reversed_videos']}")
            print(f"    Forward captions: {stats['forward_captions']}")
            print(f"    Forward rewrites: {stats['forward_rewrites']}")
            print(f"    Reverse captions: {stats['reverse_captions']}")
            print(f"    Reverse rewrites: {stats['reverse_rewrites']}")

    # Create CSVs
    print("\nCreating CSV files...")

    # Training CSV (with rewrites expanded)
    train_csv = output_dir / 'rtime_train.csv'
    n_train = create_train_csv(train_data, train_csv, include_rewrites=not args.no_rewrites)
    print(f"  Created {train_csv} ({n_train} rows)")

    # Validation CSV (human captions only, no reversed videos)
    valid_csv = output_dir / 'rtime_valid.csv'
    n_valid = create_eval_csv(valid_data, valid_csv, include_reversed=False)
    print(f"  Created {valid_csv} ({n_valid} rows)")

    # Test CSV - Hard setting (original + reversed videos)
    test_hard_csv = output_dir / 'rtime_test.csv'
    n_test_hard = create_eval_csv(test_data, test_hard_csv, include_reversed=True)
    print(f"  Created {test_hard_csv} ({n_test_hard} rows) [Hard setting]")

    # Test CSV - Origin setting (original videos only)
    test_origin_csv = output_dir / 'rtime_test_origin.csv'
    n_test_origin = create_eval_csv(test_data, test_origin_csv, include_reversed=False)
    print(f"  Created {test_origin_csv} ({n_test_origin} rows) [Origin setting]")

    # Video ID list for download verification
    video_ids_file = output_dir / 'video_ids.txt'
    n_videos = create_video_id_list(train_data, valid_data, test_data, video_ids_file)
    print(f"  Created {video_ids_file} ({n_videos} unique video IDs)")

    print("\nData preparation complete!")
    print("\nSummary:")
    print(f"  Training samples: {n_train}")
    print(f"  Validation samples: {n_valid}")
    print(f"  Test samples (Hard): {n_test_hard}")
    print(f"  Test samples (Origin): {n_test_origin}")
    print(f"  Total unique videos: {n_videos}")

    print("\nNext steps:")
    print("  1. Place video files in data/videos/")
    print("     - Original videos: <video_id>.mp4")
    print("     - Reversed videos: <video_id>_rev.mp4")
    print("  2. Run the benchmark: bash scripts/run_benchmark.sh")


if __name__ == '__main__':
    main()
