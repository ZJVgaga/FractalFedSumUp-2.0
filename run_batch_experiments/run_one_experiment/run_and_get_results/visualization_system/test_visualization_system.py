#!/usr/bin/env python3
"""
测试新的可视化系统
"""

import torch
import numpy as np
import os
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_tracker import ImageTracker
from client_visualizer import ClientVisualizer
from server_visualizer import ServerVisualizer

def test_image_tracker():
    """测试ImageTracker类"""
    print("测试ImageTracker...")
    
    # 创建模拟配置
    config = {
        "visualization_save_dir": "./test_visualizations",
        "compressed_image_size": 24,
        "Dataset": "CIFAR10"
    }
    
    # 创建模拟数据集信息
    dataset_info = {
        "im_size": [32, 32],
        "channels": 3,
        "num_classes": 10
    }
    
    # 创建模拟train_indices和train_labels
    train_indices = list(range(100))
    train_labels = [i % 10 for i in range(100)]  # 10个类别
    
    # 创建ImageTracker
    image_tracker = ImageTracker(config, dataset_info, train_indices, train_labels)
    
    # 测试获取追踪索引
    tracked_indices = image_tracker.get_tracked_indices()
    print(f"追踪的图片索引: {tracked_indices}")
    
    # 测试获取图片信息
    for idx, class_id in tracked_indices.items():
        info = image_tracker.get_image_info(idx)
        print(f"图片 {idx} 信息: {info}")
    
    print("ImageTracker测试完成")
    return image_tracker

def test_client_visualizer():
    """测试ClientVisualizer类"""
    print("\n测试ClientVisualizer...")
    
    # 创建模拟配置
    config = {
        "visualization_save_dir": "./test_visualizations",
        "compressed_image_size": 24,
        "Dataset": "CIFAR10"
    }
    
    # 创建模拟数据集信息
    dataset_info = {
        "im_size": [32, 32],
        "channels": 3,
        "num_classes": 10
    }
    
    # 创建模拟train_indices和train_labels
    train_indices = list(range(100))
    train_labels = [i % 10 for i in range(100)]
    
    # 创建ImageTracker
    image_tracker = ImageTracker(config, dataset_info, train_indices, train_labels)
    
    # 创建ClientVisualizer
    client_visualizer = ClientVisualizer(image_tracker, config)
    
    # 创建模拟数据
    batch_size = 10
    original_images = torch.randn(batch_size, 3, 32, 32)
    compressed_images = torch.randn(batch_size, 3, 24, 24)
    labels = torch.randint(0, 10, (batch_size,))
    client_indices = list(range(batch_size))
    
    # 测试保存客户端可视化
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
        print("ClientVisualizer测试完成")
    except Exception as e:
        print(f"ClientVisualizer测试失败: {e}")

def test_server_visualizer():
    """测试ServerVisualizer类"""
    print("\n测试ServerVisualizer...")
    
    # 创建模拟配置
    config = {
        "visualization_save_dir": "./test_visualizations",
        "compressed_image_size": 24,
        "Dataset": "CIFAR10"
    }
    
    # 创建模拟数据集信息
    dataset_info = {
        "im_size": [32, 32],
        "channels": 3,
        "num_classes": 10
    }
    
    # 创建模拟train_indices和train_labels
    train_indices = list(range(100))
    train_labels = [i % 10 for i in range(100)]
    
    # 创建ImageTracker
    image_tracker = ImageTracker(config, dataset_info, train_indices, train_labels)
    
    # 创建ServerVisualizer
    server_visualizer = ServerVisualizer(image_tracker, config)
    
    # 创建模拟追踪图片数据
    tracked_images_data = {}
    for idx in range(5):
        tracked_images_data[idx] = {
            "original": torch.randn(1, 3, 32, 32),
            "compressed": torch.randn(1, 3, 24, 24),
            "class_id": idx % 10
        }
    
    # 测试添加epoch可视化
    server_visualizer.add_epoch_visualization(0, tracked_images_data)
    
    # 测试保存服务器可视化
    try:
        server_visualizer.save_server_visualization(round_num=1)
        print("ServerVisualizer测试完成")
    except Exception as e:
        print(f"ServerVisualizer测试失败: {e}")

def main():
    """主测试函数"""
    print("开始测试新的可视化系统...")
    
    try:
        # 测试ImageTracker
        image_tracker = test_image_tracker()
        
        # 测试ClientVisualizer
        test_client_visualizer()
        
        # 测试ServerVisualizer
        test_server_visualizer()
        
        print("\n所有测试完成！")
        print("可视化结果保存在: ./test_visualizations/")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
