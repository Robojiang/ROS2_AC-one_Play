#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Sparsify a Zarr dataset to roughly 1/ratio while keeping all keyframes.

Example:
  python sparsify_zarr_keep_keyframes.py \
    --input /mnt/afs/250010074/robot_manipulation/ROS2_AC-one_Play/datasets_zarr/pick_place_d405.zarr \
    --output /mnt/afs/250010074/robot_manipulation/ROS2_AC-one_Play/datasets_zarr/pick_place_d405_sparse.zarr \
    --keep_ratio 0.33
"""

import argparse
import os
import shutil

import numpy as np
import zarr


def build_episode_ranges(episode_ends):
    """Return list of (start, end) for each episode, end is exclusive."""
    ranges = []
    start = 0
    for end in episode_ends:
        end = int(end)
        ranges.append((start, end))
        start = end
    return ranges


def choose_indices_for_episode(keyframe_mask, keep_ratio):
    """
    Keep all keyframes, and subsample non-keyframes to reach keep_ratio.
    Returns sorted array of indices to keep.
    """
    n = len(keyframe_mask)
    all_idx = np.arange(n)
    key_idx = all_idx[keyframe_mask]
    non_key_idx = all_idx[~keyframe_mask]

    target_keep = int(np.round(n * keep_ratio))
    # Always keep at least 2 frames if possible to preserve dynamics.
    target_keep = max(target_keep, min(2, n))
    # Must keep all keyframes.
    remaining = max(target_keep - len(key_idx), 0)

    if remaining == 0 or len(non_key_idx) == 0:
        keep_idx = key_idx
    else:
        # Uniformly sample non-keyframes.
        if remaining >= len(non_key_idx):
            sampled_non_key = non_key_idx
        else:
            sample_positions = np.linspace(0, len(non_key_idx) - 1, remaining)
            sampled_non_key = non_key_idx[np.round(sample_positions).astype(int)]
        keep_idx = np.concatenate([key_idx, sampled_non_key])

    keep_idx = np.unique(np.sort(keep_idx))
    return keep_idx


def main():
    parser = argparse.ArgumentParser(description="Sparsify Zarr dataset while keeping keyframes")
    parser.add_argument("--input", default="/mnt/afs/250010074/robot_manipulation/ROS2_AC-one_Play/datasets_zarr/pick_place_d405.zarr", help="Path to input .zarr")
    parser.add_argument("--output", default="/mnt/afs/250010074/robot_manipulation/ROS2_AC-one_Play/datasets_zarr/pick_place_d405_sparse.zarr", help="Path to output .zarr")
    parser.add_argument("--keep_ratio", type=float, default=0.33, help="Fraction to keep, e.g. 0.33")
    args = parser.parse_args()

    if args.keep_ratio <= 0 or args.keep_ratio > 1:
        raise ValueError("keep_ratio must be in (0, 1]")

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    if os.path.exists(args.output):
        print(f"Removing existing output: {args.output}")
        shutil.rmtree(args.output)

    root_in = zarr.open_group(args.input, mode="r")
    data_in = root_in["data"]
    meta_in = root_in["meta"]

    # Create output structure
    root_out = zarr.open_group(args.output, mode="w")
    data_out = root_out.create_group("data")
    meta_out = root_out.create_group("meta")

    episode_ends_in = meta_in["episode_ends"][:]
    ranges = build_episode_ranges(episode_ends_in)

    # Pre-create datasets with same dtype and chunking where possible.
    datasets = {}
    for name, arr in data_in.items():
        chunks = arr.chunks
        datasets[name] = data_out.create_dataset(
            name,
            shape=(0,) + arr.shape[1:],
            maxshape=(None,) + arr.shape[1:],
            dtype=arr.dtype,
            chunks=chunks,
            compressor=arr.compressor,
        )

    episode_ends_out = meta_out.create_dataset(
        "episode_ends",
        shape=(0,),
        maxshape=(None,),
        dtype=episode_ends_in.dtype,
        chunks=meta_in["episode_ends"].chunks,
        compressor=meta_in["episode_ends"].compressor,
    )

    total_out = 0

    for ep_id, (start, end) in enumerate(ranges):
        key_mask = data_in["keyframe_mask"][start:end]
        keep_idx_local = choose_indices_for_episode(key_mask, args.keep_ratio)
        keep_idx_global = keep_idx_local + start

        for name, arr in data_in.items():
            datasets[name].append(arr.oindex[keep_idx_global])

        total_out += len(keep_idx_global)
        episode_ends_out.append([total_out])

        print(
            f"Episode {ep_id:03d}: in={end - start}, "
            f"keep={len(keep_idx_global)}, keyframes={int(key_mask.sum())}"
        )

    print(f"Done. Output frames: {total_out}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
