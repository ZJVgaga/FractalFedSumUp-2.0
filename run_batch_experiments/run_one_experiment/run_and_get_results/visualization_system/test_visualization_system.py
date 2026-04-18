#!/usr/bin/env python3
"""
Test the new visualization system
"""

import torch
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_tracker import ImageTracker
from client_visualizer import ClientVisualizer
from server_visualizer import ServerVisualizer

def test_image_tracker():
    """Test the ImageTracker class"""
    print("Testing ImageTracker...")
    
    # Create mock configuration
    config = {
        "visualization_save_dir": "./test_visualizations",
        "compressed_image_size": 24,
        "Dataset": "CIFAR10"
    }
    
    # Create mock dataset information
    dataset_info = {
        "im_size": [32, 32],
        "channels": 3,
        "num_classes": 10
    }
    
    # Create mock train_indices and train_labels
    train_indices = list(range(100))
    train_labels = [i % 10 for i in range(100)]  # 10 classes
    
    # Create ImageTracker
    image_tracker = ImageTracker(config, dataset_info, train_indices, train_labels)
    
    # Test getting tracked indices
    tracked_indices = image_tracker.get_tracked_indices()
    print(f"Tracked image indices: {tracked_indices}")
    
    # Test getting image information
    for idx, class_id in tracked_indices.items():
        info = image_tracker.get_image_info(idx)
        print(f"Image {idx} info: {info}")
    
    print("ImageTracker test completed")
    return image_tracker

def test_client_visualizer():
    """Test the ClientVisualizer class"""
    print("\nTesting ClientVisualizer...")
    
    # Create mock configuration
    config = {
        "visualization_save_dir": "./test_visualizations",
        "compressed_image_size": 24,
        "Dataset": "CIFAR10"
    }
    
    # Create mock dataset information
    dataset_info = {
        "im_size": [32, 32],
        "channels": 3,
        "num_classes": 10
    }
    
    # Create mock train_indices and train_labels
    train_indices = list(range(100))
    train_labels = [i % 10 for i in range(100)]
    
    # Create ImageTracker
    image_tracker = ImageTracker(config, dataset_info, train_indices, train_labels)
    
    # Create ClientVisualizer
    client_visualizer = ClientVisualizer(image_tracker, config)
    
    # Create mock data
    batch_size = 10
    original_images = torch.randn(batch_size, 3, 32, 32)
    compressed_images = torch.randn(batch_size, 3, 24, 24)
    labels = torch.randint(0, 10, (batch_size,))
    client_indices = list(range(batch_size))
    
    # Test saving client visualization
    try:
        client_visualizer.save_client_visualization(
            original_images=original_images,
            compressed_images=compressed_images,
            labels=labels,
            client_indices=client_indices,
            client_id=0,
            round_num=1,
            dataset=None
        )
        print("ClientVisualizer test completed")
    except Exception as e:
        print(f"ClientVisualizer test failed: {e}")

def test_server_visualizer():
    """Test the ServerVisualizer class"""
    print("\nTesting ServerVisualizer...")
    
    # Create mock configuration
    config = {
        "visualization_save_dir": "./test_visualizations",
        "compressed_image_size": 24,
        "Dataset": "CIFAR10"
    }
    
    # Create mock dataset information
    dataset_info = {
        "im_size": [32, 32],
        "channels": 3,
        "num_classes": 10
    }
    
    # Create mock train_indices and train_labels
    train_indices = list(range(100))
    train_labels = [i % 10 for i in range(100)]
    
    # Create ImageTracker
    image_tracker = ImageTracker(config, dataset_info, train_indices, train_labels)
    
    # Create ServerVisualizer
    server_visualizer = ServerVisualizer(image_tracker, config)
    
    # Create mock tracked image data
    tracked_images_data = {}
    for idx in range(5):
        tracked_images_data[idx] = {
            "original": torch.randn(1, 3, 32, 32),
            "compressed": torch.randn(1, 3, 24, 24),
            "class_id": idx % 10
        }
    
    # Test adding epoch visualization
    server_visualizer.add_epoch_visualization(0, tracked_images_data)
    
    # Test saving server visualization
    try:
        server_visualizer.save_server_visualization(round_num=1)
        print("ServerVisualizer test completed")
    except Exception as e:
        print(f"ServerVisualizer test failed: {e}")

def main():
    """Main test function"""
    print("Starting test of the new visualization system...")
    
    try:
        # Test ImageTracker
        image_tracker = test_image_tracker()
        
        # Test ClientVisualizer
        test_client_visualizer()
        
        # Test ServerVisualizer
        test_server_visualizer()
        
        print("\nAll tests completed!")
        print("Visualization results saved in: ./test_visualizations/")
        
    except Exception as e:
        print(f"Error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()