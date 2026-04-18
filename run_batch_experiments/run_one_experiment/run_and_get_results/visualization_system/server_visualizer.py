import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
from .image_tracker import ImageTracker


class ServerVisualizer:
    """
    服务器可视化器：负责服务器端的可视化
    """
    def __init__(self, image_tracker, config):
        """
        初始化服务器可视化器
        
        Parameters:
        - image_tracker: 图片追踪器实例
        - config: 配置字典
        """
        self.image_tracker = image_tracker
        self.config = config
        self.compressed_size = config.get("compressed_image_size", 24)
        
        # 存储每个epoch的可视化数据
        self.epoch_visualizations = {}
    
    def add_epoch_visualization(self, epoch, tracked_images_data):
        """
        添加一个epoch的可视化数据
        
        Parameters:
        - epoch: epoch编号
        - tracked_images_data: 字典，包含追踪图片的数据
            {index: {"original": original_tensor, "compressed": compressed_tensor, "class_id": class_id}}
        """
        self.epoch_visualizations[epoch] = tracked_images_data
    
    def save_server_visualization(self, round_num):
        """
        保存服务器端所有epoch的可视化结果
        
        Parameters:
        - round_num: 轮次
        """
        if not self.epoch_visualizations:
            print(f"服务器端第 {round_num} 轮没有可视化数据")
            return
        
        # 对每一张追踪的图片
        tracked_indices = self.image_tracker.get_tracked_indices()
        
        for index, class_id in tracked_indices.items():
            # 收集该图片在所有epoch的数据
            all_epochs_data = []
            
            for epoch in sorted(self.epoch_visualizations.keys()):
                epoch_data = self.epoch_visualizations[epoch]
                if index in epoch_data:
                    all_epochs_data.append((epoch, epoch_data[index]))
            
            if not all_epochs_data:
                continue
            
            # 保存该图片的可视化结果
            self._save_single_image_server_visualization(
                index, class_id, round_num, all_epochs_data
            )
        
        # 清空当前轮的缓存
        self.epoch_visualizations.clear()
    
    def _save_single_image_server_visualization(self, index, class_id, round_num, all_epochs_data):
        """
        保存单张图片在服务器端的可视化结果
        
        Parameters:
        - index: 图片索引
        - class_id: 类别ID
        - round_num: 轮次
        - all_epochs_data: 列表，包含(epoch, data)元组
        """
        # 获取原始图像尺寸（从第一个epoch的数据）
        first_epoch_data = all_epochs_data[0][1]
        orig_img = first_epoch_data["original"]
        orig_h, orig_w = orig_img.shape[2], orig_img.shape[3]
        
        # 准备所有行的图像数据
        all_rows_images = []
        all_rows_titles = []
        
        # 对每个epoch
        for epoch, data in all_epochs_data:
            orig_img_epoch = data["original"]
            comp_img_epoch = data["compressed"]
            
            # 获取压缩图像尺寸
            comp_h, comp_w = comp_img_epoch.shape[2], comp_img_epoch.shape[3]
            
            # 创建三个可视化图像
            # 1. 原始图像上采样到32x32（用于显示）- 使用最近点插值
            upsampled_bilinear = torch.nn.functional.interpolate(
                orig_img_epoch,
                size=(32, 32),
                mode='bilinear',
                align_corners=False  # 最近点插值通常不需要align_corners
            )
            
            # 2. 编码器压缩的图像（保持压缩尺寸）
            compressed = comp_img_epoch.clone()
            
            # 3. 编码器压缩的图像上采样到32x32 - 使用最近点插值
            compressed_upsample = torch.nn.functional.interpolate(
                comp_img_epoch,
                size=(32, 32),
                mode='bilinear',
                align_corners=False  # 最近点插值通常不需要align_corners
            )
            
            # 添加到行数据
            all_rows_images.extend([upsampled_bilinear, compressed, compressed_upsample])
            all_rows_titles.extend([
                f"Epoch {epoch}: Bilinear Upsample\nSize: 32x32",
                f"Epoch {epoch}: Encoder Compressed\nSize: {comp_h}x{comp_w}",
                f"Epoch {epoch}: Compressed Upsample\nSize: 32x32"
            ])
        
        # 计算总行数（每个epoch一行）
        num_epochs = len(all_epochs_data)
        
        # 创建图形（多行三列）
        fig, axes = plt.subplots(num_epochs, 3, figsize=(15, 5 * num_epochs))
        
        # 如果只有一个epoch，axes的形状需要调整
        if num_epochs == 1:
            axes = [axes]
        
        # 填充每个epoch的图像
        for row in range(num_epochs):
            for col in range(3):
                ax = axes[row][col] if num_epochs > 1 else axes[col]
                img_idx = row * 3 + col
                
                if img_idx < len(all_rows_images):
                    img = all_rows_images[img_idx]
                    title = all_rows_titles[img_idx]
                    
                    # 处理图像数据
                    if isinstance(img, torch.Tensor):
                        img = img.detach().cpu().numpy()
                    
                    # 调整图像维度顺序
                    if len(img.shape) == 4:  # [batch, channels, height, width]
                        img = img[0]  # 取第一张图像
                    
                    if len(img.shape) == 3:
                        if img.shape[0] == 1:  # 单通道图像
                            img = img[0]  # [height, width]
                        elif img.shape[0] == 3:  # RGB图像
                            img = img.transpose(1, 2, 0)  # [height, width, channels]
                    
                    # 显示图像
                    if len(img.shape) == 2:  # 灰度图像
                        ax.imshow(img, cmap='gray')
                    else:  # RGB图像
                        # 归一化到[0, 1]范围
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                        ax.imshow(img)
                    
                    ax.set_title(title, fontsize=10)
                    ax.axis('off')
        
        # 只保存总的汇总图像，不保存每个epoch的单独图像
        # 获取保存路径（不使用epoch参数）
        save_dir = self.image_tracker.get_save_path(index, class_id, round_num, is_server=True)
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存汇总图像
        save_path = os.path.join(save_dir, f"server_all_epochs.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # 保存汇总元数据
        metadata = {
            "index": int(index),
            "class_id": int(class_id),
            "round": int(round_num),
            "num_epochs": num_epochs,
            "original_size": f"{orig_h}x{orig_w}",
            "target_compressed_size": f"{self.compressed_size}x{self.compressed_size}",
            "epochs": [epoch for epoch, _ in all_epochs_data],
            "timestamp": str(np.datetime64('now'))
        }
        
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"服务器端保存了索引 {index} (类别 {class_id}) 的可视化结果到: {save_dir}")
