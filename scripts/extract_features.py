#!/usr/bin/env python
"""
One-time feature extraction script for caching CLIP embeddings.

This script extracts and caches CLIP embeddings for all videos and captions
in the RTime dataset, enabling frozen CLIP training experiments.

Usage:
    python scripts/extract_features.py \
        --video_dir data/videos \
        --train_csv data/rtime_train.csv \
        --val_csv data/rtime_valid.csv \
        --test_csv data/rtime_test.csv \
        --output_dir rtime_cache \
        --clip_model ViT-B/32 \
        --max_frames 24 \
        --batch_size 32

For multi-GPU extraction:
    python -m torch.distributed.launch --nproc_per_node=4 \
        scripts/extract_features.py [args]
"""

import os
import sys
import json
import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2

# Add CLIP4Clip to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CLIP4Clip'))

from modules.module_clip import CLIP, convert_weights
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def get_args():
    parser = argparse.ArgumentParser(description='Extract CLIP features for RTime dataset')

    # Data paths
    parser.add_argument('--video_dir', type=str, default='data/videos',
                        help='Directory containing video files')
    parser.add_argument('--train_csv', type=str, default='data/rtime_train.csv',
                        help='Training CSV file')
    parser.add_argument('--val_csv', type=str, default='data/rtime_valid.csv',
                        help='Validation CSV file')
    parser.add_argument('--test_csv', type=str, default='data/rtime_test.csv',
                        help='Test CSV file')
    parser.add_argument('--test_origin_csv', type=str, default='data/rtime_test_origin.csv',
                        help='Test origin CSV file (original videos only)')
    parser.add_argument('--output_dir', type=str, default='rtime_cache',
                        help='Output directory for cached features')

    # Model config
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        choices=['ViT-B/32', 'ViT-B/16'],
                        help='CLIP model variant')

    # Video extraction config
    parser.add_argument('--max_frames', type=int, default=24,
                        help='Maximum number of frames to extract per video (T_max)')
    parser.add_argument('--fps', type=int, default=1,
                        help='Frames per second for sampling')
    parser.add_argument('--image_resolution', type=int, default=224,
                        help='Image resolution for CLIP')

    # Text config
    parser.add_argument('--max_words', type=int, default=32,
                        help='Maximum number of tokens for text')

    # Processing config
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--fp16', action='store_true',
                        help='Save features in fp16 format')

    # Distributed config
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--verify', action='store_true',
                        help='Verify extracted features (spot check)')

    args = parser.parse_args()
    return args


def setup_distributed(args):
    """Setup distributed training if available."""
    if args.local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.world_size = 1
        args.rank = 0
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

    args.device = device
    return args


def get_transform(image_resolution):
    """Get CLIP image preprocessing transform."""
    return Compose([
        Resize(image_resolution, interpolation=Image.BICUBIC),
        CenterCrop(image_resolution),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


def load_clip_model(clip_model_name, device):
    """Load CLIP model for feature extraction."""
    state_dict = CLIP.get_config(pretrained_clip_name=clip_model_name)

    # Build model from state dict
    vit = "visual.proj" in state_dict
    assert vit, "Only ViT models are supported"

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys()
                        if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict
                                 if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    ).float()

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, embed_dim


def extract_video_frames(video_path, transform, max_frames, fps):
    """Extract frames from a video file.

    Args:
        video_path: Path to video file
        transform: Image preprocessing transform
        max_frames: Maximum frames to extract
        fps: Frames per second for sampling

    Returns:
        frames: Tensor of shape [T, 3, H, W]
        num_frames: Actual number of valid frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0 or frame_count <= 0:
        cap.release()
        return None, 0

    # Calculate frame indices to sample
    duration = frame_count / video_fps
    total_sample_frames = int(duration * fps)

    if total_sample_frames <= 0:
        total_sample_frames = 1

    # Sample uniformly if more frames than max_frames
    if total_sample_frames > max_frames:
        sample_indices = np.linspace(0, total_sample_frames - 1, max_frames, dtype=int)
    else:
        sample_indices = np.arange(total_sample_frames)

    # Convert sample indices to actual frame indices
    frame_indices = (sample_indices / fps * video_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, frame_count - 1)

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = transform(frame_pil)
            frames.append(frame_tensor)

    cap.release()

    if len(frames) == 0:
        return None, 0

    frames_tensor = torch.stack(frames, dim=0)
    return frames_tensor, len(frames)


def extract_video_features(model, frames, device):
    """Extract CLIP visual features from video frames.

    Args:
        model: CLIP model
        frames: Tensor of shape [T, 3, H, W]
        device: Device to run on

    Returns:
        features: Tensor of shape [T, embed_dim]
    """
    with torch.no_grad():
        frames = frames.to(device)
        # Encode each frame
        features = model.encode_image(frames)  # [T, embed_dim]
    return features.cpu()


def extract_text_features(model, tokenizer, texts, device, max_words=32):
    """Extract CLIP text features.

    Args:
        model: CLIP model
        tokenizer: CLIP tokenizer
        texts: List of text strings
        device: Device to run on
        max_words: Maximum tokens

    Returns:
        pooled_features: Tensor of shape [N, embed_dim] (EoT token features)
        hidden_features: Tensor of shape [N, L, embed_dim] (all token features)
    """
    SPECIAL_TOKEN = {
        "CLS_TOKEN": "<|startoftext|>",
        "SEP_TOKEN": "<|endoftext|>"
    }

    all_input_ids = []
    for text in texts:
        words = tokenizer.tokenize(text)
        words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
        if len(words) > max_words - 1:
            words = words[:max_words - 1]
        words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = tokenizer.convert_tokens_to_ids(words)
        while len(input_ids) < max_words:
            input_ids.append(0)
        all_input_ids.append(input_ids)

    input_ids = torch.tensor(all_input_ids, dtype=torch.long).to(device)

    with torch.no_grad():
        pooled, hidden = model.encode_text(input_ids, return_hidden=True)

    return pooled.cpu(), hidden.cpu()


def get_unique_video_ids(csv_files):
    """Get unique video IDs from CSV files."""
    video_ids = set()
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            video_ids.update(df['video_id'].unique())
    return sorted(list(video_ids))


def get_all_captions(csv_files):
    """Get all unique (video_id, sentence) pairs from CSV files."""
    captions = {}
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                vid = str(row['video_id'])
                sent = row['sentence']
                # Use hash of sentence as unique ID
                sent_hash = hashlib.md5(sent.encode()).hexdigest()[:12]
                caption_id = f"{vid}_{sent_hash}"
                captions[caption_id] = {
                    'video_id': vid,
                    'sentence': sent
                }
    return captions


def compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    if not os.path.exists(filepath):
        return None
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_cache_metadata(args, output_dir, video_ids, caption_ids):
    """Save cache metadata for reproducibility."""
    metadata = {
        'clip_model': args.clip_model,
        'preprocessing': {
            'resize': args.image_resolution,
            'center_crop': args.image_resolution,
            'normalize_mean': [0.48145466, 0.4578275, 0.40821073],
            'normalize_std': [0.26862954, 0.26130258, 0.27577711]
        },
        'video_sampling': {
            'fps': args.fps,
            'T_max': args.max_frames,
            'sampling_rule': 'uniform'
        },
        'text_config': {
            'max_words': args.max_words
        },
        'embedding_dtype': 'float16' if args.fp16 else 'float32',
        'dataset_files': {
            'train_csv': args.train_csv,
            'val_csv': args.val_csv,
            'test_csv': args.test_csv,
            'test_origin_csv': args.test_origin_csv
        },
        'dataset_hashes': {
            'train_csv': compute_file_hash(args.train_csv),
            'val_csv': compute_file_hash(args.val_csv),
            'test_csv': compute_file_hash(args.test_csv),
            'test_origin_csv': compute_file_hash(args.test_origin_csv)
        },
        'num_videos': len(video_ids),
        'num_captions': len(caption_ids),
        'extraction_timestamp': datetime.now().isoformat(),
        'seed': args.seed
    }

    meta_path = os.path.join(output_dir, 'cache_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {meta_path}")


def main():
    args = get_args()
    args = setup_distributed(args)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directories
    output_dir = args.output_dir
    video_feat_dir = os.path.join(output_dir, 'video_feats')
    text_feat_dir = os.path.join(output_dir, 'text_feats')

    if args.rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(video_feat_dir, exist_ok=True)
        os.makedirs(text_feat_dir, exist_ok=True)

    if args.world_size > 1:
        torch.distributed.barrier()

    # Load model
    print(f"[Rank {args.rank}] Loading CLIP model: {args.clip_model}")
    model, embed_dim = load_clip_model(args.clip_model, args.device)
    tokenizer = ClipTokenizer()
    transform = get_transform(args.image_resolution)

    print(f"[Rank {args.rank}] Embedding dimension: {embed_dim}")

    # Get all video IDs and captions
    csv_files = [args.train_csv, args.val_csv, args.test_csv, args.test_origin_csv]
    video_ids = get_unique_video_ids(csv_files)
    captions = get_all_captions(csv_files)

    print(f"[Rank {args.rank}] Found {len(video_ids)} unique videos")
    print(f"[Rank {args.rank}] Found {len(captions)} unique captions")

    # Filter to base video IDs (not reversed)
    base_video_ids = [vid for vid in video_ids if not str(vid).endswith('_rev')]
    print(f"[Rank {args.rank}] Found {len(base_video_ids)} base videos (excluding _rev)")

    # Distribute videos across ranks
    videos_per_rank = len(base_video_ids) // args.world_size
    start_idx = args.rank * videos_per_rank
    end_idx = start_idx + videos_per_rank if args.rank < args.world_size - 1 else len(base_video_ids)
    rank_video_ids = base_video_ids[start_idx:end_idx]

    # Extract video features
    print(f"[Rank {args.rank}] Extracting features for {len(rank_video_ids)} videos...")

    video_errors = []
    for vid in tqdm(rank_video_ids, desc=f"[Rank {args.rank}] Videos", disable=args.rank != 0):
        vid_str = str(vid)
        output_path = os.path.join(video_feat_dir, f"{vid_str}.pt")

        # Skip if already extracted
        if os.path.exists(output_path):
            continue

        # Find video file
        video_path = None
        for ext in ['.mp4', '.webm', '.avi', '.mkv']:
            candidate = os.path.join(args.video_dir, f"{vid_str}{ext}")
            if os.path.exists(candidate):
                video_path = candidate
                break

        if video_path is None:
            video_errors.append(vid_str)
            continue

        # Extract frames
        frames, num_frames = extract_video_frames(video_path, transform, args.max_frames, args.fps)

        if frames is None:
            video_errors.append(vid_str)
            continue

        # Extract features
        features = extract_video_features(model, frames, args.device)

        # Pad to max_frames
        if features.shape[0] < args.max_frames:
            padding = torch.zeros(args.max_frames - features.shape[0], features.shape[1])
            features = torch.cat([features, padding], dim=0)

        # Convert to fp16 if requested
        if args.fp16:
            features = features.half()

        # Save with metadata
        save_dict = {
            'features': features,  # [T_max, embed_dim]
            'num_frames': num_frames,
            'video_id': vid_str
        }
        torch.save(save_dict, output_path)

    if video_errors:
        print(f"[Rank {args.rank}] Failed to extract {len(video_errors)} videos: {video_errors[:10]}...")

    # Synchronize before text extraction
    if args.world_size > 1:
        torch.distributed.barrier()

    # Extract text features (only on rank 0)
    if args.rank == 0:
        print("Extracting text features...")

        caption_items = list(captions.items())

        # Process in batches
        for i in tqdm(range(0, len(caption_items), args.batch_size), desc="Text batches"):
            batch_items = caption_items[i:i + args.batch_size]
            batch_ids = [item[0] for item in batch_items]
            batch_texts = [item[1]['sentence'] for item in batch_items]

            # Check if all already extracted
            all_exist = all(
                os.path.exists(os.path.join(text_feat_dir, f"{cid}_pool.pt"))
                for cid in batch_ids
            )
            if all_exist:
                continue

            # Extract features
            pooled, hidden = extract_text_features(
                model, tokenizer, batch_texts, args.device, args.max_words
            )

            # Convert to fp16 if requested
            if args.fp16:
                pooled = pooled.half()
                hidden = hidden.half()

            # Save each caption
            for j, (cid, info) in enumerate(batch_items):
                pool_path = os.path.join(text_feat_dir, f"{cid}_pool.pt")
                tok_path = os.path.join(text_feat_dir, f"{cid}_tok.pt")

                torch.save({
                    'features': pooled[j],  # [embed_dim]
                    'caption_id': cid,
                    'video_id': info['video_id']
                }, pool_path)

                torch.save({
                    'features': hidden[j],  # [L, embed_dim]
                    'caption_id': cid,
                    'video_id': info['video_id']
                }, tok_path)

        # Create caption index mapping
        caption_index = {}
        for cid, info in captions.items():
            caption_index[cid] = info

        index_path = os.path.join(text_feat_dir, 'caption_index.json')
        with open(index_path, 'w') as f:
            json.dump(caption_index, f, indent=2)

        # Save metadata
        save_cache_metadata(args, output_dir, video_ids, list(captions.keys()))

        # Create split files linking video_id, caption_id pairs
        for split_name, csv_path in [
            ('train', args.train_csv),
            ('valid', args.val_csv),
            ('test', args.test_csv),
            ('test_origin', args.test_origin_csv)
        ]:
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            split_data = []

            for _, row in df.iterrows():
                vid = str(row['video_id'])
                sent = row['sentence']
                sent_hash = hashlib.md5(sent.encode()).hexdigest()[:12]
                caption_id = f"{vid}_{sent_hash}"

                # Check if it's a reversed video
                is_reversed = vid.endswith('_rev')
                base_vid = vid[:-4] if is_reversed else vid

                split_data.append({
                    'video_id': vid,
                    'base_video_id': base_vid,
                    'caption_id': caption_id,
                    'is_reversed': is_reversed,
                    'sentence': sent
                })

            split_df = pd.DataFrame(split_data)
            split_path = os.path.join(output_dir, f'{split_name}_pairs.csv')
            split_df.to_csv(split_path, index=False)
            print(f"Saved {split_name} split with {len(split_df)} pairs to {split_path}")

    # Final synchronization
    if args.world_size > 1:
        torch.distributed.barrier()

    print(f"[Rank {args.rank}] Feature extraction complete!")

    # Verification (optional)
    if args.verify and args.rank == 0:
        print("\nVerifying extracted features...")
        verify_features(args, model, transform, tokenizer, video_feat_dir, text_feat_dir)


def verify_features(args, model, transform, tokenizer, video_feat_dir, text_feat_dir):
    """Spot-check that cached features match on-the-fly computation."""
    import random

    # Check a few random videos
    video_files = [f for f in os.listdir(video_feat_dir) if f.endswith('.pt')]
    sample_videos = random.sample(video_files, min(5, len(video_files)))

    print(f"Verifying {len(sample_videos)} videos...")

    for vf in sample_videos:
        vid = vf.replace('.pt', '')
        cached = torch.load(os.path.join(video_feat_dir, vf))

        # Re-extract
        video_path = None
        for ext in ['.mp4', '.webm', '.avi', '.mkv']:
            candidate = os.path.join(args.video_dir, f"{vid}{ext}")
            if os.path.exists(candidate):
                video_path = candidate
                break

        if video_path is None:
            print(f"  {vid}: SKIP (video not found)")
            continue

        frames, _ = extract_video_frames(video_path, transform, args.max_frames, args.fps)
        if frames is None:
            print(f"  {vid}: SKIP (frame extraction failed)")
            continue

        fresh_features = extract_video_features(model, frames, args.device)

        # Compare (allowing for fp16 tolerance)
        cached_feat = cached['features'][:fresh_features.shape[0]]
        if args.fp16:
            cached_feat = cached_feat.float()

        max_diff = (cached_feat - fresh_features).abs().max().item()
        print(f"  {vid}: max_diff = {max_diff:.6f} {'OK' if max_diff < 0.01 else 'WARN'}")

    # Check a few random texts
    text_files = [f for f in os.listdir(text_feat_dir) if f.endswith('_pool.pt')]
    sample_texts = random.sample(text_files, min(5, len(text_files)))

    print(f"\nVerifying {len(sample_texts)} captions...")

    # Load caption index
    index_path = os.path.join(text_feat_dir, 'caption_index.json')
    with open(index_path) as f:
        caption_index = json.load(f)

    for tf in sample_texts:
        cid = tf.replace('_pool.pt', '')
        cached = torch.load(os.path.join(text_feat_dir, tf))

        if cid not in caption_index:
            print(f"  {cid}: SKIP (not in index)")
            continue

        sentence = caption_index[cid]['sentence']
        fresh_pooled, _ = extract_text_features(model, tokenizer, [sentence], args.device, args.max_words)

        cached_feat = cached['features']
        if args.fp16:
            cached_feat = cached_feat.float()

        max_diff = (cached_feat - fresh_pooled[0]).abs().max().item()
        print(f"  {cid[:30]}...: max_diff = {max_diff:.6f} {'OK' if max_diff < 0.01 else 'WARN'}")


if __name__ == '__main__':
    main()
