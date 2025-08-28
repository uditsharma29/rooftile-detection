#!/usr/bin/env python3
"""
Dataset analysis for roof classification dataset
"""

import os
import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image


def count_vertices(annotations):
    """Count vertices in annotations."""
    vertex_counts = []
    
    # Extract from numpy array format (parquet stores as array with single dict)
    if hasattr(annotations, 'shape') and annotations.shape == (1,):
        annotations = annotations[0]
    
    # Safety check for dictionary format
    if not isinstance(annotations, dict):
        return vertex_counts
        
    # Get objects list, default to empty if key doesn't exist
    objects = annotations.get('objects', [])
    
    # Count vertices in each polygon
    for obj in objects:
        keypoints = obj.get('keyPoints', [])
        if keypoints is None:
            continue
        for grp in keypoints:
            pts = grp.get('points', [])
            if pts is None:
                continue
            if len(pts) >= 3:
                vertex_counts.append(len(pts))
    
    return vertex_counts


def analyze_dataset():
    """Analyze and print dataset statistics."""
    print("Dataset Analysis")
    print("-" * 40)
    
    # Load and display basic dataset info
    df = pd.read_parquet('data/dataset.parquet')
    print(f"Total images: {len(df)}")
    
    # Count vertices in all polygon annotations
    all_vertex_counts = []
    for _, row in df.iterrows():
        vertex_counts = count_vertices(row['annotations'])
        all_vertex_counts.extend(vertex_counts)
    
    # Display polygon and vertex statistics
    if all_vertex_counts:
        print(f"Total polygons: {len(all_vertex_counts)}")
        print(f"Average polygons per image: {len(all_vertex_counts) / len(df):.1f}")
        print(f"Vertex count - Min: {min(all_vertex_counts)}, Max: {max(all_vertex_counts)}, Mean: {np.mean(all_vertex_counts):.1f}")
        
        # Show distribution of vertex counts
        vertex_dist = Counter(all_vertex_counts)
        print("\nVertex distribution:")
        for count in sorted(vertex_dist.keys()):
            percentage = (vertex_dist[count] / len(all_vertex_counts)) * 100
            print(f"  {count} vertices: {vertex_dist[count]} ({percentage:.1f}%)")
    
    # Show data split distribution (train/val/test)
    print(f"\nData splits:")
    split_counts = df['partition'].value_counts()
    for split, count in split_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {split}: {count} images ({percentage:.1f}%)")
    
    # Get basic image properties from sample
    print(f"\nImage info:")
    sample_img = df.iloc[0]["asset_url"]
    img_path = os.path.join("data", sample_img) if not sample_img.startswith("data/") else sample_img
    
    img = Image.open(img_path)
    print(f"  Dimensions: {img.size[0]}x{img.size[1]} pixels")
    print(f"  Format: {img.mode}")


if __name__ == "__main__":
    analyze_dataset()
