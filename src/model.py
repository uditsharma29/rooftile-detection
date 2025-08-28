import os
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_maskrcnn_model(num_classes: int = 2):
	model = maskrcnn_resnet50_fpn_v2(weights=None)
	
	# Load pretrained COCO weights if available
	weights_path = "weights/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
	if os.path.exists(weights_path):
		state = torch.load(weights_path, map_location="cpu")
		model.load_state_dict(state, strict=False)
	
	# Replace heads for our classes (background + roof)
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	
	in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
	model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
	
	return model