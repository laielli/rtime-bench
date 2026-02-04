"""
RTime Cached Embedding Dataloader

This module provides dataloaders that use pre-computed CLIP embeddings
instead of raw videos, enabling efficient frozen CLIP training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import json
import hashlib
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class RTimeCachedDataset(Dataset):
    """Base dataset for cached RTime embeddings."""

    def __init__(
            self,
            csv_path,
            cache_path,
            max_frames=12,
            frame_order=0,
    ):
        """
        Args:
            csv_path: Path to CSV file with (video_id, sentence) pairs
            cache_path: Path to cache directory with pre-computed embeddings
            max_frames: Maximum frames to use (subsamples if cached > max_frames)
            frame_order: 0=normal, 1=reverse, 2=random
        """
        self.data = pd.read_csv(csv_path)
        self.cache_path = cache_path
        self.max_frames = max_frames
        self.frame_order = frame_order

        # Paths to feature directories
        self.video_feat_dir = os.path.join(cache_path, 'video_feats')
        self.text_feat_dir = os.path.join(cache_path, 'text_feats')

        # Load cache metadata
        meta_path = os.path.join(cache_path, 'cache_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.cache_meta = json.load(f)
            self.cached_max_frames = self.cache_meta['video_sampling']['T_max']
            self.embed_dim = 512  # Standard CLIP embed dim
        else:
            self.cached_max_frames = 24
            self.embed_dim = 512

        # Build caption ID mapping
        self._build_caption_index()

    def _build_caption_index(self):
        """Build mapping from (video_id, sentence) to caption_id."""
        self.caption_ids = []
        for idx in range(len(self.data)):
            vid = str(self.data['video_id'].values[idx])
            sent = self.data['sentence'].values[idx]
            sent_hash = hashlib.md5(sent.encode()).hexdigest()[:12]
            caption_id = f"{vid}_{sent_hash}"
            self.caption_ids.append(caption_id)

    def __len__(self):
        return len(self.data)

    def _get_base_video_id(self, video_id):
        """Get base video ID (strip _rev suffix if present)."""
        vid_str = str(video_id)
        if vid_str.endswith('_rev'):
            return vid_str[:-4], True
        return vid_str, False

    def _load_video_features(self, video_id):
        """Load cached video features.

        Args:
            video_id: Video ID (may include _rev suffix)

        Returns:
            video_emb: [max_frames, embed_dim] tensor
            video_mask: [max_frames] tensor
            num_frames: Actual number of valid frames
        """
        base_vid, is_reversed = self._get_base_video_id(video_id)

        feat_path = os.path.join(self.video_feat_dir, f"{base_vid}.pt")

        if not os.path.exists(feat_path):
            # Return zeros if not found
            video_emb = torch.zeros(self.max_frames, self.embed_dim)
            video_mask = torch.zeros(self.max_frames, dtype=torch.long)
            return video_emb, video_mask, 0

        cached = torch.load(feat_path, map_location='cpu')
        features = cached['features']  # [T_cached, embed_dim]
        num_frames = cached.get('num_frames', features.shape[0])

        # Handle reversed videos by flipping the frame order
        if is_reversed:
            # Only flip the valid frames
            features[:num_frames] = torch.flip(features[:num_frames], dims=[0])

        # Subsample or pad to max_frames
        if num_frames > self.max_frames:
            # Uniform subsampling
            indices = torch.linspace(0, num_frames - 1, self.max_frames).long()
            video_emb = features[indices]
            actual_frames = self.max_frames
        else:
            # Pad with zeros
            video_emb = torch.zeros(self.max_frames, features.shape[1])
            video_emb[:num_frames] = features[:num_frames]
            actual_frames = num_frames

        # Apply frame order transformation
        if self.frame_order == 1:
            # Reverse order (for non-reversed videos, this is additional reversal)
            video_emb[:actual_frames] = torch.flip(video_emb[:actual_frames], dims=[0])
        elif self.frame_order == 2:
            # Random order
            perm = torch.randperm(actual_frames)
            video_emb[:actual_frames] = video_emb[perm]

        # Create mask
        video_mask = torch.zeros(self.max_frames, dtype=torch.long)
        video_mask[:actual_frames] = 1

        # Convert to float32 if needed
        video_emb = video_emb.float()

        return video_emb, video_mask, actual_frames

    def _load_text_features(self, caption_id, pooled=True):
        """Load cached text features.

        Args:
            caption_id: Caption identifier
            pooled: If True, return pooled (EoT) features; else return all token features

        Returns:
            text_emb: [embed_dim] or [L, embed_dim] tensor
        """
        suffix = '_pool.pt' if pooled else '_tok.pt'
        feat_path = os.path.join(self.text_feat_dir, f"{caption_id}{suffix}")

        if not os.path.exists(feat_path):
            # Return zeros if not found
            if pooled:
                return torch.zeros(self.embed_dim)
            else:
                return torch.zeros(32, self.embed_dim)  # max_words default

        cached = torch.load(feat_path, map_location='cpu')
        features = cached['features'].float()
        return features


class RTimeCachedTrainDataset(RTimeCachedDataset):
    """Training dataset with cached embeddings."""

    def __getitem__(self, idx):
        video_id = str(self.data['video_id'].values[idx])
        caption_id = self.caption_ids[idx]

        # Load features
        text_emb = self._load_text_features(caption_id, pooled=True)  # [embed_dim]
        video_emb, video_mask, _ = self._load_video_features(video_id)  # [max_frames, embed_dim]

        return text_emb, video_emb, video_mask


class RTimeCachedTestDataset(RTimeCachedDataset):
    """Test/validation dataset with cached embeddings.

    Returns additional identifiers for evaluation.
    """

    def __init__(self, *args, return_ids=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_ids = return_ids

    def __getitem__(self, idx):
        video_id = str(self.data['video_id'].values[idx])
        caption_id = self.caption_ids[idx]

        # Load features
        text_emb = self._load_text_features(caption_id, pooled=True)
        video_emb, video_mask, _ = self._load_video_features(video_id)

        if self.return_ids:
            return text_emb, video_emb, video_mask, video_id, caption_id

        return text_emb, video_emb, video_mask


class RTimeCachedTestDatasetTightTransf(RTimeCachedDataset):
    """Test dataset for tightTransf that needs token-level text features."""

    def __init__(self, *args, return_ids=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_ids = return_ids

    def __getitem__(self, idx):
        video_id = str(self.data['video_id'].values[idx])
        caption_id = self.caption_ids[idx]

        # Load token-level text features for cross-attention
        text_emb_tok = self._load_text_features(caption_id, pooled=False)  # [L, embed_dim]
        text_emb_pool = self._load_text_features(caption_id, pooled=True)  # [embed_dim]
        video_emb, video_mask, _ = self._load_video_features(video_id)

        if self.return_ids:
            return text_emb_pool, text_emb_tok, video_emb, video_mask, video_id, caption_id

        return text_emb_pool, text_emb_tok, video_emb, video_mask
