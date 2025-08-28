"""
Simple post-processing to merge overlapping polygons.
Just the essential functionality - easy to understand and explain.
"""

import numpy as np
import cv2
from typing import List, Dict, Any


def polygon_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Calculate IoU between two polygons using OpenCV."""
    if len(poly1) < 3 or len(poly2) < 3:
        return 0.0
    
    # Create masks
    h, w = max(poly1[:, 1].max(), poly2[:, 1].max()) + 10, max(poly1[:, 0].max(), poly2[:, 0].max()) + 10
    mask1 = np.zeros((int(h), int(w)), dtype=np.uint8)
    mask2 = np.zeros((int(h), int(w)), dtype=np.uint8)
    
    # Fill polygons
    cv2.fillPoly(mask1, [poly1.astype(np.int32)], 1)
    cv2.fillPoly(mask2, [poly2.astype(np.int32)], 1)
    
    # Calculate IoU
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection / union if union > 0 else 0.0


def merge_overlapping_polygons(polygons: List[np.ndarray], iou_threshold: float = 0.7) -> List[np.ndarray]:
    """
    Merge overlapping polygons based on IoU threshold.
    Simple approach: keep the first polygon, remove overlapping ones.
    """
    if len(polygons) <= 1:
        return polygons
    
    # Convert to numpy arrays if needed
    polygons = [np.array(p) if not isinstance(p, np.ndarray) else p for p in polygons]
    
    # Find and merge overlapping polygons
    merged = []
    used = [False] * len(polygons)
    
    for i in range(len(polygons)):
        if used[i]:
            continue
            
        current_poly = polygons[i]
        merged.append(current_poly)
        used[i] = True
        
        # Find overlapping polygons
        for j in range(i + 1, len(polygons)):
            if used[j]:
                continue
                
            iou = polygon_iou(current_poly, polygons[j])
            if iou > iou_threshold:
                used[j] = True  # Mark as merged
    
    return merged


def postprocess_predictions(results: List[Dict[str, Any]], iou_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Post-process prediction results by merging overlapping polygons.
    Simple and effective approach.
    """
    postprocessed_results = []
    
    for result in results:
        if "pred_polys" not in result or not result["pred_polys"]:
            postprocessed_results.append(result)
            continue
        
        # Get polygons
        polygons = result["pred_polys"]
        
        # Merge overlapping ones
        merged_polygons = merge_overlapping_polygons(polygons, iou_threshold)
        
        # Create result with merged polygons
        postprocessed_result = result.copy()
        postprocessed_result["pred_polys"] = [p.tolist() if isinstance(p, np.ndarray) else p for p in merged_polygons]
        postprocessed_result["original_count"] = len(polygons)
        postprocessed_result["merged_count"] = len(merged_polygons)
        
        postprocessed_results.append(postprocessed_result)
    
    return postprocessed_results