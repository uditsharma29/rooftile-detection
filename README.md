# Instance Segmentation for Roof Tile Detection

## Overview
This project implements instance segmentation using Mask R-CNN to detect and segment roof sections from aerial imagery. The model outputs polygons that can be used for roof area analysis and classification.

## Project Structure
```
src/
├── dataset.py      # Dataset loading and preprocessing
├── model.py        # Mask R-CNN model definition
├── train.py        # Training script with improvements
├── eval.py         # Evaluation and inference
├── postprocessing.py # Post-processing utilities
└── smoke_tests.py  # Smoke tests for core functionality validation
```

## Setup

### Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Pretrained Weights
Download the COCO pretrained Mask R-CNN weights and save them to the `weights/` folder:
```bash
# Create weights directory
mkdir -p weights

# Download pretrained weights
wget https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth -O weights/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth
```

**Note**: The model will automatically load these weights if they exist in the `weights/` folder, otherwise it will start with random initialization.

## Usage

### Training
```bash
# Basic training with default parameters
python -m src.train

# Custom training parameters
IMAGE_SIZE=320 BATCH_SIZE=2 EPOCHS=25 python -m src.train
```

### Evaluation
```bash
# Evaluate on validation set
python -m src.eval --mode eval --partition val

# Evaluate on test set
python -m src.eval --mode eval --partition test

# With post-processing
python -m src.eval --mode eval --partition val --postprocess
```

### Inference
```bash
# Generate predictions and visualizations
python -m src.eval --mode eval --partition val --save-predictions
```

## Approach

### Model Architecture
- Mask R-CNN is ideal for this task as it naturally outputs instance segmentation masks that can be directly converted to polygons, eliminating the need for separate mask-to-polygon post-processing.
- **Backbone**: Mask R-CNN with ResNet-50 FPN (Feature Pyramid Network)
- **Components**: 
  - ResNet-50 backbone for feature extraction
  - FPN for multi-scale feature representation
  - Region Proposal Network (RPN) for object detection
  - Fast R-CNN head for classification and bounding box regression
  - Mask head for instance segmentation
- **Pretrained**: COCO weights for transfer learning
- **Output**: 2 classes (background + roof)
- **Polygonization**: Contour extraction with vertex constraints

### Loss Functions
Mask R-CNN uses a multi-task loss combining:
- **RPN Loss**: Objectness classification + bounding box regression
- **Fast R-CNN Loss**: Classification + bounding box regression  
- **Mask Loss**: Binary cross-entropy for pixel-wise segmentation
- **Total Loss**: L = L_rpn + L_fast_rcnn + L_mask

### Training Strategy
**Designed to solve training instability issues commonly encountered with Mask R-CNN:**
1. **Transfer Learning**: Start with COCO-pretrained weights to provide stable initialization
2. **Multi-task Loss**: Classification + bounding box + mask prediction with balanced loss weights
3. **Optimization**: AdamW with cosine learning rate scheduling to prevent learning rate oscillations and ensure smooth training
3. **Regularization**: Weight decay and early stopping to prevent overfitting and ensure stable convergence
5. **Reproducible Training**: Seeds ensure consistent results across training runs
6. **Checkpoint Saving**: Every 5 epochs preserves progress and enables recovery from training instability

### Hardware Used
- **Device**: CPU (Apple Silicon M4 Pro)
- **Memory**: Available system RAM
- **Training Time**: 25 epochs in ~100 mins
- **Note**: No GPU acceleration used during training

### Evaluation Metrics
- **IoU**: Intersection over Union for mask quality
- **AP50**: Average Precision at IoU threshold 0.5
- **AP@[.50:.95]**: Mean AP across IoU thresholds
- **AP75**: Average Precision at IoU threshold 0.75

### Post-processing
- **Polygon Simplification**: Douglas-Peucker algorithm with vertex limits
- **Overlap Removal**: IoU-based merging of duplicate predictions
- **Confidence Filtering**: Score threshold for quality control

## Results
- **IoU**: 0.622
- **Best AP50**: 0.669 (validation set)
- **AP@[0.50:0.95]**: 0.422 (mean AP across IoU thresholds)
- **AP75**: 0.454 (AP at IoU threshold 0.75)


## File Descriptions

- **`dataset.py`**: Handles data loading, polygon parsing, and tensor conversion
- **`model.py`**: Defines Mask R-CNN architecture and loads pretrained weights
- **`train.py`**: Training loop with scheduling, early stopping, validation and model checkpointing
- **`eval.py`**: Evaluation, metrics calculation, and visualization
- **`postprocessing.py`**: Polygon optimization and overlap removal utilities

### Output Structure
```
outputs/
├── maskrcnn_model_best.pt          # Best trained model (AP50=0.669)
├── output_maskrcnn_val.parquet     # Validation set predictions
├── output_maskrcnn_test.parquet    # Test set predictions  
├── val_predictions/                 # Validation image overlays with predicted polygons
├── test_predictions/                # Test image overlays with predicted polygons
└── train_improved_320x320_batch2.log # Training logs and metrics
```

## Challenges and Solutions

### Training Stability and Reproducibility
I observed unstable training patterns characterized by fluctuating loss curves and inconsistent AP scores across epochs. To address this, I implemented comprehensive seeding (numpy, torch, dataloader workers), learning rate scheduling with cosine annealing, and early stopping on AP50 to prevent overfitting and ensure reproducible results.

### Post-processing Integration
I identified a need to balance polygon quality improvement with maintaining model performance metrics. The challenge was to reduce visual clutter from overlapping predictions without degrading quantitative performance. I implemented IoU-based polygon merging that reduces overlapping predictions and achieving better visual quality. 

## Future Work

### Data Augmentation
The current implementation lacks data augmentation, which can reduce model robustness and generalization to unseen roof shapes. Implementing simple geometric transformations such as horizontal/vertical flips, rotations, and brightness adjustments during training would enhance the model's ability to handle diverse roof configurations.

### Boundary-Aware Loss Functions
The current model sometimes produces overly complex polygons requiring post-processing, indicating that boundary prediction could be improved. To address this, boundary-aware losses (like Boundary IoU) could be implemented during training to encourage cleaner polygon outputs and reduce the need for extensive post-processing.

### Multi-scale Training Strategy
I observed that AP75 (0.454) is significantly lower than AP50 (0.669), indicating the model struggles with stricter IoU thresholds and precise localization. This suggests the model could benefit from better multi-scale understanding to handle roofs of varying sizes and improve boundary accuracy. Implementing multi-scale training could improve detection across different roof sizes and enhance overall model robustness.

## Conclusion

This project achieves solid performance metrics (AP50: 0.669, mIoU: 0.622) and demonstrates best practices for training, reproducibility, and evaluation. It provides a strong foundation for a production-ready roof segmentation system. The next steps identified can further enhance model performance and deployment readiness.