import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset


class SimpleRoofDataset(Dataset):
	def __init__(self, data_root: str, parquet_path: str, partition: str, image_size: int = 256):
		# Initialize dataset with data path, partition filter, and image size
		self.data_root = data_root
		self.image_size = image_size
		self.df = pd.read_parquet(parquet_path)
		self.df = self.df[self.df["partition"] == partition].reset_index(drop=True)

	def __len__(self) -> int:
		# Return total number of images in the partition
		return len(self.df)

	def _row_to_polygons(self, row: pd.Series):
		# Extract polygon coordinates from annotation row
		polys = []
		ann = row.get("annotations")
		
		# Handle numpy array format (parquet stores as array with single dict)
		if hasattr(ann, 'shape') and ann.shape == (1,):
			ann = ann[0]
		
		# Safety check for dictionary format
		if not isinstance(ann, dict):
			return polys
			
		# Get objects list, default to empty if key doesn't exist
		objects = ann.get("objects", [])
		if objects is None:
			return polys
			
		# Extract normalized polygon points from keypoints
		for obj in objects:
			keypoints = obj.get('keyPoints', [])
			if keypoints is None:
				continue
			for grp in keypoints:
				pts = grp.get('points', [])
				if pts is None:
					continue
				norm = [{"x": float(p.get("x") or 0.0), "y": float(p.get("y") or 0.0)} for p in pts]
				if len(norm) >= 3:
					polys.append(norm)
		return polys

	def __getitem__(self, idx: int):
		# Load image and resize to target size
		row = self.df.iloc[idx]
		img_path = row["asset_url"]
		if not img_path.startswith("data/"):
			img_path = os.path.join(self.data_root, img_path)
		
		img = Image.open(img_path).convert("RGB")
		orig_w, orig_h = img.size
		img = img.resize((self.image_size, self.image_size))

		# Create instance masks and bounding boxes from polygons
		instance_masks = []
		boxes = []
		for pts in self._row_to_polygons(row):
			# Scale coordinates to image size and create mask
			xy = [(p["x"] * self.image_size, p["y"] * self.image_size) for p in pts]
			mask_img = Image.new("L", (self.image_size, self.image_size), 0)
			ImageDraw.Draw(mask_img).polygon(xy, fill=1)
			mask = np.array(mask_img, dtype=np.uint8)
			if mask.sum() > 0:
				instance_masks.append(mask)
				# Calculate bounding box from mask
				yx = np.argwhere(mask > 0)
				ymin, xmin = yx.min(axis=0)
				ymax, xmax = yx.max(axis=0)
				boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])

		# Convert masks and boxes to PyTorch tensors
		n = len(instance_masks)
		if n > 0:
			masks_t = torch.from_numpy(np.stack(instance_masks, axis=0)).to(torch.uint8)
			boxes_t = torch.tensor(boxes, dtype=torch.float32)
		else:
			# Handle case with no instances
			masks_t = torch.empty((0, self.image_size, self.image_size), dtype=torch.uint8)
			boxes_t = torch.empty((0, 4), dtype=torch.float32)
		
		# Create labels tensor and normalize image
		labels_t = torch.ones((n,), dtype=torch.int64) if n > 0 else torch.empty((0,), dtype=torch.int64)
		image_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
		
		# Return image, target dict, and metadata
		target = {"boxes": boxes_t, "labels": labels_t, "masks": masks_t, "image_id": torch.tensor([idx])}
		meta = {"asset_url": row["asset_url"], "orig_size": (orig_h, orig_w)}
		return image_t, target, meta