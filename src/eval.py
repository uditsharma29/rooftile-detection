"""
Simplified evaluation script for Mask R-CNN roof classification.
Keeps all functionality but much cleaner and easier to understand.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import cv2

from .dataset import SimpleRoofDataset
from .model import get_maskrcnn_model
from .postprocessing import postprocess_predictions


def mask_to_polygons(mask: np.ndarray, mask_threshold: float = 0.5, max_vertices: int = 8):
    """Convert mask to polygons with vertex count constraint."""
    mask_uint8 = (mask > mask_threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    
    for cnt in contours:
        if len(cnt) >= 3:
            perimeter = cv2.arcLength(cnt, True)
            
            # Progressive simplification
            epsilon_values = [0.05, 0.03, 0.02]  # Percentage of perimeter
            best_poly = None
            
            for eps_pct in epsilon_values:
                epsilon = max(eps_pct * perimeter, 2.0)
                approx = cv2.approxPolyDP(cnt, epsilon, True).squeeze(1)
                
                if approx.ndim == 2 and len(approx) >= 3:
                    if len(approx) <= max_vertices:
                        best_poly = approx
                        break
                    elif best_poly is None or len(approx) < len(best_poly):
                        best_poly = approx
            
            # Force reduction if still over limit
            if best_poly is not None and len(best_poly) > max_vertices:
                step = len(best_poly) / max_vertices
                indices = [int(i * step) for i in range(max_vertices)]
                indices = [min(i, len(best_poly) - 1) for i in indices]
                best_poly = best_poly[indices]
            
            if best_poly is not None and len(best_poly) >= 3:
                polys.append(best_poly.astype(np.float32))
    
    return polys


def polygon_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Calculate IoU between two polygons."""
    try:
        all_pts = np.vstack([poly1, poly2])
        x1, y1 = all_pts.min(axis=0)
        x2, y2 = all_pts.max(axis=0)
        w, h = int(x2 - x1) + 10, int(y2 - y1) + 10
        
        mask1 = np.zeros((h, w), dtype=np.uint8)
        mask2 = np.zeros((h, w), dtype=np.uint8)
        
        cv2.fillPoly(mask1, [(poly1 - [x1, y1]).astype(np.int32)], 1)
        cv2.fillPoly(mask2, [(poly2 - [x1, y1]).astype(np.int32)], 1)
        
        inter = (mask1 & mask2).sum()
        union = (mask1 | mask2).sum()
        return float(inter) / float(union) if union > 0 else 0.0
    except Exception:
        return 0.0


def load_model(device: torch.device, checkpoint_path: str = "outputs/maskrcnn_model.pt"):
    """Load and return Mask R-CNN model."""
    model = get_maskrcnn_model(num_classes=2).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.eval()
    return model


def _compute_ap(pr_scores, pr_matches, num_gt: int) -> float:
    """Compute Average Precision."""
    if num_gt == 0 or not pr_scores:
        return 0.0
    
    cum_tp = np.cumsum(pr_matches)
    cum_fp = np.cumsum(1 - np.array(pr_matches))
    recall = cum_tp / max(1, num_gt)
    precision = cum_tp / np.maximum(1, cum_tp + cum_fp)
    
    # 101-point interpolation
    recall_points = np.linspace(0.0, 1.0, 101)
    prec_at_recall = []
    for r in recall_points:
        mask = recall >= r
        prec_at_recall.append(float(precision[mask].max()) if mask.any() else 0.0)
    
    return float(np.mean(prec_at_recall))


def _evaluate_predictions(preds_all, gts_by_image, iou_thresholds):
    """Compute AP metrics across IoU thresholds."""
    num_gt = sum(len(gts) for gts in gts_by_image.values())
    aps = []
    
    for thr in iou_thresholds:
        matched = set()
        sorted_preds = sorted(preds_all, key=lambda x: x.get("score", 0.0), reverse=True)
        matches_flags = []
        
        for pred in sorted_preds:
            img_id = pred["image_id"]
            pred_poly = pred["poly"]
            best_iou = 0.0
            best_key = None
            
            # Find best matching ground truth
            for gi, gt in enumerate(gts_by_image.get(img_id, [])):
                gt_key = (img_id, gi)
                if gt_key in matched:
                    continue
                iou = polygon_iou(gt["poly"], pred_poly)
                if iou >= thr and iou > best_iou:
                    best_iou = iou
                    best_key = gt_key
            
            if best_key is not None:
                matched.add(best_key)
                matches_flags.append(1)
            else:
                matches_flags.append(0)
        
        ap = _compute_ap([p.get("score", 0.0) for p in sorted_preds], matches_flags, num_gt)
        aps.append(ap)
    
    # Return metrics
    metrics = {"AP": float(np.mean(aps))}
    if 0.50 in iou_thresholds:
        metrics["AP50"] = aps[iou_thresholds.index(0.50)]
    if 0.75 in iou_thresholds:
        metrics["AP75"] = aps[iou_thresholds.index(0.75)]
    
    return metrics


def evaluate_split(partition: str, score_thresh: float = 0.5, image_size: int = 256, 
                  model=None, loader=None):
    """Evaluate model on a dataset split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup data loader
    if loader is None:
        ds = SimpleRoofDataset("data", "data/dataset.parquet", partition, image_size=image_size)
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    
    # Setup model
    if model is None:
        checkpoint_path = os.environ.get("CHECKPOINT", "outputs/maskrcnn_model.pt")
        model = load_model(device, checkpoint_path)
    else:
        model.eval()

    results = []
    ious = []
    gts_by_image = {}
    preds_all = []
    
    # Get environment variables
    mask_thresh = float(os.environ.get("MASK_THRESH", "0.5"))
    max_vertices = int(os.environ.get("MAX_VERTICES", "8"))

    with torch.no_grad():
        for batch in loader:
            img, target, meta = batch[0]
            img = img.to(device)
            out = model([img])[0]

            # Get predictions
            scores = out.get("scores", torch.empty(0, device=device))
            masks_all = out.get("masks", torch.zeros(0, 1, img.shape[-2], img.shape[-1], device=device))
            
            # Apply score threshold for visualization
            keep = scores > score_thresh
            masks = masks_all[keep, 0].detach().cpu().numpy() if keep.any() else np.zeros((0, img.shape[-2], img.shape[-1]))
            
            # Get all predictions for AP calculation (no threshold)
            masks_ap = masks_all[:, 0].detach().cpu().numpy() if masks_all.numel() > 0 else np.zeros((0, img.shape[-2], img.shape[-1]))
            scores_ap = scores.detach().cpu().numpy() if scores.numel() > 0 else np.array([])

            # Scale coordinates
            orig_h, orig_w = meta["orig_size"]
            sx, sy = orig_w / image_size, orig_h / image_size
            
            # Get predicted polygons for visualization
            pred_polys = []
            for m in masks:
                for poly in mask_to_polygons(m, mask_thresh, max_vertices):
                    poly = poly.copy()
                    poly[:, 0] *= sx
                    poly[:, 1] *= sy
                    pred_polys.append(poly)

            # Get all predictions for AP calculation
            img_id = meta["asset_url"]
            for mi, m in enumerate(masks_ap):
                score_val = float(scores_ap[mi]) if mi < len(scores_ap) else 0.0
                for poly in mask_to_polygons(m, mask_thresh, max_vertices):
                    poly = poly.copy()
                    poly[:, 0] *= sx
                    poly[:, 1] *= sy
                    preds_all.append({"poly": poly, "score": score_val, "image_id": img_id})

            # Get ground truth polygons
            gt_polys = []
            if img_id not in gts_by_image:
                gts_by_image[img_id] = []
            
            for gm in target["masks"].cpu().numpy():
                for gp in mask_to_polygons(gm, mask_thresh):
                    gp = gp.copy()
                    gp[:, 0] *= sx
                    gp[:, 1] *= sy
                    gt_polys.append(gp)
                    gts_by_image[img_id].append({"poly": gp, "area": cv2.contourArea(gp.astype(np.float32))})

            # Compute IoU for validation
            if gt_polys and pred_polys:
                iou_scores = [max(polygon_iou(gt, pred) for pred in pred_polys) for gt in gt_polys]
                ious.append(float(np.mean(iou_scores)))
            else:
                ious.append(0.0)

            results.append({
                "asset_url": meta["asset_url"],
                "partition": partition,
                "pred_polys": [p.tolist() for p in pred_polys],
                "orig_size": meta["orig_size"],
            })

    # Compute AP metrics
    iou_thresholds = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
    metrics = _evaluate_predictions(preds_all, gts_by_image, iou_thresholds)

    return {"results": results, "ious": ious, "metrics": metrics}


def save_predictions_parquet(results, out_path: str):
    """Save predictions to parquet file in the exact same format as the original dataset."""
    # Load original dataset to get the template structure
    try:
        orig_df = pd.read_parquet('data/dataset.parquet')
        # Get a sample row to understand the structure
        sample_row = orig_df.iloc[0]
        template_annotation = sample_row['annotations'][0]
    except Exception as e:
        print(f"Warning: Could not load original dataset template: {e}")
        # Fallback to basic structure
        template_annotation = {}
    
    rows = []
    for r in results:
        # Convert absolute coordinates back to normalized (0.0-1.0)
        orig_h, orig_w = r.get("orig_size", (640, 640))  # Default size if not available
        
        normalized_polys = []
        for poly in r["pred_polys"]:
            # Convert absolute coordinates to normalized
            normalized_poly = []
            for point in poly:
                norm_x = point[0] / orig_w
                norm_y = point[1] / orig_h
                normalized_poly.append([norm_x, norm_y])
            normalized_polys.append(normalized_poly)
        
        # Create annotation structure matching the original format
        annotation = {
            "classes": np.array([], dtype=object),
            "embeddings": np.array([], dtype=object),
            "keyPoints": np.array([], dtype=object),
            "objects": np.array([{
                "category": None,
                "classLabel": "Roof",  # All predictions are roof class
                "confidence": None,
                "height": None,  # Could calculate from polygon bounds
                "keyPoints": np.array([{
                    "category": None,
                    "points": np.array([{
                        "category": None,
                        "classLabel": None,
                        "confidence": None,
                        "visible": None,
                        "x": norm_poly[0],
                        "y": norm_poly[1],
                        "z": None
                    } for norm_poly in normalized_poly], dtype=object)
                } for normalized_poly in normalized_polys], dtype=object),
                "texts": None,
                "user_review": "predicted",  # Mark as model prediction
                "width": None,  # Could calculate from polygon bounds
                "x": None,  # Could calculate from polygon bounds
                "y": None   # Could calculate from polygon bounds
            } for normalized_poly in normalized_polys], dtype=object),
            "source": "model_prediction",
            "source_model_uuid": None,
            "texts": np.array([], dtype=object),
            "type": "prediction",
            "user_review": "pending"
        }
        
        rows.append({
            "asset_url": r["asset_url"],
            "partition": r["partition"],
            "annotations": np.array([annotation])
        })
    
    pd.DataFrame(rows).to_parquet(out_path)


def plot_overlays(results, out_dir: str):
    """Generate visualization overlays."""
    os.makedirs(out_dir, exist_ok=True)
    
    for r in results:
        img_path = r["asset_url"]
        if not os.path.exists(img_path):
            img_path = os.path.join("data", img_path)
        
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Draw polygons
        for poly in r["pred_polys"]:
            pts = [(float(x), float(y)) for x, y in poly]
            draw.polygon(pts, outline="red", width=3)
            for x, y in pts:
                draw.ellipse([x-3, y-3, x+3, y+3], fill="blue")
        
        # Save image
        base = os.path.splitext(os.path.basename(img_path))[0]
        img.save(os.path.join(out_dir, f"{base}_pred.jpg"))


def validate():
    """Simple validation function."""
    os.makedirs("outputs", exist_ok=True)
    partition = os.environ.get("PARTITION", "val")
    res = evaluate_split(partition=partition)
    save_predictions_parquet(res["results"], f"outputs/output_maskrcnn_{partition}.parquet")
    metrics = res.get("metrics", {})
    print(f"Results saved to outputs/output_maskrcnn_{partition}.parquet")
    print(f"Mean IoU: {np.mean(res['ious']):.3f}")
    print(f"Summary (AP@.50): {metrics.get('AP50', 0.0):.3f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval", "plot", "validate"], default="eval")
    parser.add_argument("--partition", default="val")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--postprocess", action="store_true", help="Apply post-processing to merge overlapping polygons")
    parser.add_argument("--iou-threshold", type=float, default=0.7, help="IoU threshold for merging overlapping polygons")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    
    if args.mode == "validate":
        validate()
        return

    # Evaluate model
    res = evaluate_split(
        partition=args.partition,
        score_thresh=args.score_thresh,
        image_size=args.image_size,
    )
    results, ious = res["results"], res["ious"]

    # Calculate metrics before post-processing
    print(f"\n📊 METRICS BEFORE POST-PROCESSING:")
    print(f"Mean IoU: {np.mean(ious):.3f}")
    print(f"AP@[.50:.95]: {res.get('metrics', {}).get('AP', 0.0):.3f}")
    print(f"AP@.50: {res.get('metrics', {}).get('AP50', 0.0):.3f}")
    print(f"AP@.75: {res.get('metrics', {}).get('AP75', 0.0):.3f}")
    
    # Apply post-processing if requested
    if args.postprocess:
        print(f"\n🔄 APPLYING POST-PROCESSING (IoU threshold={args.iou_threshold})")
        original_results = results.copy()
        results = postprocess_predictions(results, iou_threshold=args.iou_threshold)
        
        # Print merging statistics
        total_original = sum(r.get("original_count", len(r.get("pred_polys", []))) for r in results)
        total_merged = sum(r.get("merged_count", len(r.get("pred_polys", []))) for r in results)
        reduction = ((total_original - total_merged) / total_original * 100) if total_original > 0 else 0
        print(f"Post-processing results: {total_original} → {total_merged} polygons ({reduction:.1f}% reduction)")
        
        # Post-processing completed
        print(f"\n📊 POST-PROCESSING COMPLETED:")
        print(f"Polygon reduction: {reduction:.1f}%")
        print(f"Note: AP metrics shown above are from BEFORE post-processing")
        print(f"Post-processing improves visual quality by reducing overlaps")
        
        if reduction > 0:
            print(f"✅ Post-processing reduced {total_original - total_merged} overlapping polygons")
        else:
            print(f"➡️ No overlapping polygons found")

    # Save results or generate plots
    if args.mode == "eval":
        out_path = f"outputs/output_maskrcnn_{args.partition}.parquet"
        save_predictions_parquet(results, out_path)
        metrics = res.get("metrics", {})
        print(f"Results saved to {out_path}.")
        print(f"Mean IoU: {np.mean(ious):.3f}")
        print(f"AP@[.50:.95]: {metrics.get('AP', 0.0):.3f}")
        print(f"AP@.50: {metrics.get('AP50', 0.0):.3f}")
        print(f"AP@.75: {metrics.get('AP75', 0.0):.3f}")
    else:
        out_dir = os.path.join("outputs", f"{args.partition}_predictions")
        plot_overlays(results, out_dir)
        print(f"Saved overlays to {out_dir}")


if __name__ == "__main__":
    main()