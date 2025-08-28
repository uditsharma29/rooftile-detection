#!/usr/bin/env python3
"""
Smoke tests for the simple architecture.
Tests only the most critical functionality to ensure everything works.
"""

import unittest
import torch
import numpy as np
import pandas as pd
import tempfile
import os

# Import the modules to test
from src.dataset import SimpleRoofDataset
from src.eval import mask_to_polygons, polygon_iou
from src.model import get_maskrcnn_model


class TestSmokeTests(unittest.TestCase):
    """Critical smoke tests for core functionality."""
    
    def setUp(self):
        """Set up minimal test data."""
        # Create minimal mock data
        self.mock_data = {
            'asset_url': ['test_image.jpg'],
            'partition': ['train'],
            'annotations': [
                np.array([{
                    'objects': [{
                        'keyPoints': [{
                            'points': [
                                {'x': 0.1, 'y': 0.1},
                                {'x': 0.9, 'y': 0.1},
                                {'x': 0.9, 'y': 0.9},
                                {'x': 0.1, 'y': 0.9}
                            ]
                        }]
                    }]
                }])
            ]
        }
        
        # Create temporary files
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_path = os.path.join(self.temp_dir, 'test_data.parquet')
        df = pd.DataFrame(self.mock_data)
        df.to_parquet(self.parquet_path)
        
        # Create dummy image
        self.image_dir = os.path.join(self.temp_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)
        img_path = os.path.join(self.image_dir, 'test_image.jpg')
        dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        from PIL import Image
        Image.fromarray(dummy_img).save(img_path)
    
    def test_dataset_returns_correct_format(self):
        """Smoke test: Dataset returns data in expected format."""
        dataset = SimpleRoofDataset(self.image_dir, self.parquet_path, 'train', image_size=64)
        
        # Test that dataset loads
        self.assertEqual(len(dataset), 1)
        
        # Test that __getitem__ returns expected format
        image, target, meta = dataset[0]
        
        # Check image tensor
        self.assertEqual(image.shape, (3, 64, 64))
        self.assertTrue(torch.is_tensor(image))
        
        # Check target structure
        self.assertIn('boxes', target)
        self.assertIn('labels', target)
        self.assertIn('masks', target)
        
        # Check metadata
        self.assertEqual(meta['asset_url'], 'test_image.jpg')
    
    def test_model_creates_and_runs(self):
        """Smoke test: Model can be created and run forward pass."""
        model = get_maskrcnn_model(num_classes=2)
        
        # Test model creation
        self.assertIsNotNone(model)
        
        # Test forward pass with dummy data
        model.eval()
        dummy_input = torch.randn(3, 256, 256)
        
        with torch.no_grad():
            output = model([dummy_input])
        
        # Check output structure
        self.assertEqual(len(output), 1)
        self.assertIn('boxes', output[0])
        self.assertIn('labels', output[0])
        self.assertIn('scores', output[0])
        self.assertIn('masks', output[0])
    
    def test_eval_functions_work(self):
        """Smoke test: Evaluation functions work with basic inputs."""
        # Test mask to polygons
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 1  # Simple rectangle
        
        polygons = mask_to_polygons(mask, mask_threshold=0.5, max_vertices=8)
        self.assertGreater(len(polygons), 0)
        
        # Test polygon IoU
        poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        poly2 = np.array([[5, 5], [15, 5], [15, 15], [5, 15]])
        
        iou = polygon_iou(poly1, poly2)
        self.assertGreaterEqual(iou, 0.0)
        self.assertLessEqual(iou, 1.0)
    
    def test_end_to_end_pipeline(self):
        """Smoke test: Complete pipeline from dataset to model works."""
        # Create dataset
        dataset = SimpleRoofDataset(self.image_dir, self.parquet_path, 'train', image_size=64)
        
        # Create model
        model = get_maskrcnn_model(num_classes=2)
        model.eval()
        
        # Get data from dataset
        image, target, meta = dataset[0]
        
        # Run model inference
        with torch.no_grad():
            output = model([image])
        
        # Verify everything worked
        self.assertIsNotNone(output)
        self.assertEqual(len(output), 1)
        self.assertIn('boxes', output[0])


def run_smoke_tests():
    """Run all smoke tests."""
    print("Running smoke tests for simple architecture...")
    
    # Create and run test suite
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSmokeTests))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*40}")
    print(f"Smoke Test Results")
    print(f"{'='*40}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_smoke_tests()
    
    if success:
        print(f"\n✅ All smoke tests passed! Core functionality works.")
    else:
        print(f"\n❌ Some smoke tests failed! Check core functionality.")
    
    exit(0 if success else 1)
