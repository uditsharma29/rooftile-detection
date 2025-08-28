import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import SimpleRoofDataset
from src.model import get_maskrcnn_model
from src.eval import evaluate_split
import numpy as np
from tqdm import tqdm
import random

# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    # Initialize worker with different seed for reproducibility
    random.seed(42 + worker_id)
    np.random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)

def custom_collate(batch):
    # Custom collate function to handle variable numbers of objects
    # Dataset returns (image, target, meta) but we only need (images, targets)
    images = []
    targets = []
    
    for image, target, meta in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets

def main():
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get config from environment variables
    image_size = int(os.getenv('IMAGE_SIZE', 256))
    batch_size = int(os.getenv('BATCH_SIZE', 4))
    epochs = int(os.getenv('EPOCHS', 25))  # Reduced from 30 to prevent overfitting
    
    # Set evaluation threshold
    os.environ['SCORE_THRESH'] = '0.1'  # Lower threshold for evaluation
    
    print(f"Training with: image_size={image_size}, batch_size={batch_size}, epochs={epochs}")
    print(f"Evaluation score threshold: {os.getenv('SCORE_THRESH', '0.1')}")
    
    # Create datasets and dataloaders
    train_dataset = SimpleRoofDataset('data', 'data/dataset.parquet', 'train', image_size=image_size)
    val_dataset = SimpleRoofDataset('data', 'data/dataset.parquet', 'val', image_size=image_size)
    
    # Training dataloader with custom collate for variable object counts
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        worker_init_fn=worker_init_fn,
        collate_fn=custom_collate
    )
    
    # Validation dataloader without custom collate (for evaluation function compatibility)
    val_loader_eval = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,  # No multiprocessing for evaluation
        collate_fn=lambda x: x  # Original format for evaluation function
    )
    
    # Create model using existing model module
    model = get_maskrcnn_model(num_classes=2)
    model = model.to(device)
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.001, 
        weight_decay=1e-4,  # Added weight decay
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs,
        eta_min=1e-6
    )
    
    # Training loop with early stopping
    best_ap50 = 0.0
    patience = 5
    no_improvement = 0
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Early stopping patience: {patience} epochs")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        loss_components = {'loss_classifier': 0.0, 'loss_box_reg': 0.0, 'loss_mask': 0.0, 'loss_objectness': 0.0, 'loss_rpn_box_reg': 0.0}
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        for batch_idx, (images, targets) in enumerate(train_pbar):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            
            # Track individual loss components
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key].item()
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f'{total_loss.item():.3f}'})
        
        # Calculate average losses
        avg_total_loss = total_loss / len(train_loader)
        avg_losses = {k: v / len(train_loader) for k, v in loss_components.items()}
        
        # Validation phase
        model.eval()
        print(f"\nEpoch {epoch + 1}: loss={avg_total_loss:.3f}")
        print(f"Loss breakdown: {avg_losses}")
        
        # Evaluate on validation set
        val_results = evaluate_split('val', score_thresh=0.1, image_size=image_size, model=model, loader=val_loader_eval)
        ap50 = val_results.get('metrics', {}).get('AP50', 0.0)
        ap = val_results.get('metrics', {}).get('AP', 0.0)
        ap75 = val_results.get('metrics', {}).get('AP75', 0.0)
        mean_iou = np.mean(val_results.get('ious', [0.0]))
        
        print(f"Epoch {epoch + 1}: IoU={mean_iou:.3f}, AP50={ap50:.3f}, AP@[.50:.95]={ap:.3f}, AP75={ap75:.3f}")
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
        
        # Early stopping check
        if ap50 > best_ap50:
            best_ap50 = ap50
            no_improvement = 0
            print(f"New best model saved: AP50 = {ap50:.3f}")
            
            # Save best model
            torch.save(model.state_dict(), 'outputs/maskrcnn_model_best.pt')
        else:
            no_improvement += 1
            print(f"No improvement for {no_improvement} epochs")
            
            if no_improvement >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_ap50': best_ap50,
                'metrics': {'AP50': ap50, 'AP': ap, 'AP75': ap75, 'IoU': mean_iou}
            }, f'outputs/maskrcnn_checkpoint_epoch_{epoch + 1}.pt')
    
    print(f"\nTraining completed!")
    print(f"Best AP50 achieved: {best_ap50:.3f}")
    print(f"Best model saved to: outputs/maskrcnn_model_best.pt")

if __name__ == "__main__":
    main()