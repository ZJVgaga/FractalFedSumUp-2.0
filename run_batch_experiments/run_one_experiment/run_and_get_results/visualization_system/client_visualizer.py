import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
from PIL import Image as PILImage
from matplotlib.backends.backend_pdf import PdfPages
from .image_tracker import ImageTracker


class ClientVisualizer:
    """
    客户端可视化器：负责客户端端的可视化
    """
    def __init__(self, image_tracker, config):
        """
        初始化客户端可视化器
        
        Parameters:
        - image_tracker: 图片追踪器实例
        - config: 配置字典
        """
        self.image_tracker = image_tracker
        self.config = config
        self.compressed_size = config.get("compressed_image_size", 24)
    
    def save_client_visualization(self, original_images, compressed_images, labels,
                                 client_indices, client_id, round_num, dataset):
        """
        保存客户端可视化结果
        
        Parameters:
        - original_images: 原始图像 [batch, channels, height, width]
        - compressed_images: 压缩后的图像 [batch, channels, comp_h, comp_w]
        - labels: 标签
        - client_indices: 客户端的数据索引列表
        - client_id: 客户端ID
        - round_num: 轮次
        - dataset: 数据集对象（用于获取特定索引的图像）
        """
        # 获取客户端负责追踪的图片
        responsibility = self.image_tracker.get_client_responsibility(client_indices)
        
        if not responsibility:
            print(f"客户端 {client_id} 没有负责追踪的图片")
            return
        
        print(f"客户端 {client_id} 负责追踪 {len(responsibility)} 张图片: {list(responsibility.keys())}")
        
        # 对每一张负责的图片进行可视化
        for idx, class_id in responsibility.items():
            # 在客户端数据中找到该图片的位置
            if idx not in client_indices:
                continue
            
            position_in_client = client_indices.index(idx)
            
            # 获取该图片的数据
            if position_in_client < len(original_images):
                orig_img = original_images[position_in_client:position_in_client+1]
                comp_img = compressed_images[position_in_client:position_in_client+1]
                label = labels[position_in_client:position_in_client+1]
                
                # 保存可视化结果
                self._save_single_image_visualization(
                    orig_img, comp_img, label, idx, class_id, 
                    client_id, round_num
                )
    
    def _save_single_image_visualization(self, orig_img, comp_img, label, 
                                        index, class_id, client_id, round_num):
        """
        保存单张图片的可视化结果
        
        Parameters:
        - orig_img: 原始图像 [1, channels, height, width]
        - comp_img: 压缩图像 [1, channels, comp_h, comp_w]
        - label: 标签 [1]
        - index: 图片索引
        - class_id: 类别ID
        - client_id: 客户端ID
        - round_num: 轮次
        """
        # 获取保存路径
        save_dir = self.image_tracker.get_save_path(index, class_id, round_num, is_server=False)
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取图像尺寸
        orig_h, orig_w = orig_img.shape[2], orig_img.shape[3]
        comp_h, comp_w = comp_img.shape[2], comp_img.shape[3]
        
        # 创建四个可视化图像
        # 1. 原始图像
        orig_display = orig_img.clone()
        
        # 2. 直接缩放到编码器输出大小的图像（保持压缩尺寸）- 使用最近点插值
        orig_downsampled = torch.nn.functional.interpolate(
            orig_img,
            size=(comp_h, comp_w),
            mode='nearest'
            # 注意：最近点插值不支持align_corners参数
        )
        
        # 3. 编码器压缩的图像（保持压缩尺寸）
        comp_display = comp_img.clone()
        
        # 创建图像网格（单行四列）
        images_to_save = [orig_display, orig_downsampled, comp_display]
        titles = [
            f"Original\nSize: {orig_h}x{orig_w}",
            f"Direct Downsample\nSize: {comp_h}x{comp_w}",
            f"Encoder Compressed\nSize: {comp_h}x{comp_w}"
        ]
        
        # 保存图像
        save_path = os.path.join(save_dir, f"client_visualization.png")
        self._save_image_grid(images_to_save, titles, save_path, nrow=3, figsize=(15, 5))
        
        # 保存元数据
        metadata = {
            "index": int(index),
            "class_id": int(class_id),
            "client_id": int(client_id),
            "round": int(round_num),
            "original_size": f"{orig_h}x{orig_w}",
            "compressed_size": f"{comp_h}x{comp_w}",
            "target_compressed_size": f"{self.compressed_size}x{self.compressed_size}",
            "label": int(label[0].item()) if hasattr(label[0], 'item') else int(label[0]),
            "timestamp": str(np.datetime64('now'))
        }
        
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"客户端 {client_id} 保存了索引 {index} (类别 {class_id}) 的可视化结果到: {save_dir}")
        
        # 保存压缩图像用于后续PDF生成
        self._save_compressed_image_for_pdf(orig_img, comp_img, index, class_id, round_num)
    
    def _save_image_grid(self, images, titles, save_path, nrow=3, figsize=(15, 5)):
        """
        保存图像网格
        
        Parameters:
        - images: 图像列表
        - titles: 标题列表
        - save_path: 保存路径
        - nrow: 每行图像数量
        - figsize: 图像尺寸
        """
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 创建图形
        fig, axes = plt.subplots(1, len(images), figsize=figsize)
        if len(images) == 1:
            axes = [axes]
        
        for i, (img, title) in enumerate(zip(images, titles)):
            ax = axes[i]
            
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
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _save_compressed_image_for_pdf(self, orig_img, comp_img, index, class_id, round_num):
        """
        保存压缩图像为JPG文件用于后续PDF生成
        
        Parameters:
        - orig_img: 原始图像 [1, channels, height, width]
        - comp_img: 压缩图像 [1, channels, comp_h, comp_w]
        - index: 图片索引
        - class_id: 类别ID
        - round_num: 轮次
        """
        # 获取保存路径
        base_dir = self.config.get("visualization_save_dir", "./fedsumup_visualizations")
        dataset_name = self.config.get("Dataset", "CIFAR10")
        compressed_size = self.compressed_size
        
        # 创建JPG图像保存目录
        jpg_data_dir = os.path.join(
            base_dir,
            f"fedsumup_visualization_{dataset_name}",
            f"compressed_{compressed_size}",
            "jpg_data"
        )
        os.makedirs(jpg_data_dir, exist_ok=True)
        
        # 保存原始图像为JPG
        orig_img_np = orig_img.detach().cpu().numpy()
        if len(orig_img_np.shape) == 4:
            orig_img_np = orig_img_np[0]  # 取第一张图像
        
        # 调整图像维度顺序
        if len(orig_img_np.shape) == 3:
            if orig_img_np.shape[0] == 1:  # 单通道图像
                orig_img_np = orig_img_np[0]  # [height, width]
            elif orig_img_np.shape[0] == 3:  # RGB图像
                orig_img_np = orig_img_np.transpose(1, 2, 0)  # [height, width, channels]
        
        # 归一化到[0, 255]范围
        if orig_img_np.max() > orig_img_np.min():
            orig_img_np = (orig_img_np - orig_img_np.min()) / (orig_img_np.max() - orig_img_np.min() + 1e-8) * 255
        orig_img_np = orig_img_np.astype(np.uint8)
        
        # 保存原始图像
        orig_img_path = os.path.join(jpg_data_dir, f"{index}_{class_id}_round_{round_num}_original.jpg")
        if len(orig_img_np.shape) == 2:  # 灰度图像
            PILImage.fromarray(orig_img_np, mode='L').save(orig_img_path)
        else:  # RGB图像
            PILImage.fromarray(orig_img_np, mode='RGB').save(orig_img_path)
        
        # 保存压缩图像为JPG
        comp_img_np = comp_img.detach().cpu().numpy()
        if len(comp_img_np.shape) == 4:
            comp_img_np = comp_img_np[0]  # 取第一张图像
        
        # 调整图像维度顺序
        if len(comp_img_np.shape) == 3:
            if comp_img_np.shape[0] == 1:  # 单通道图像
                comp_img_np = comp_img_np[0]  # [height, width]
            elif comp_img_np.shape[0] == 3:  # RGB图像
                comp_img_np = comp_img_np.transpose(1, 2, 0)  # [height, width, channels]
        
        # 归一化到[0, 255]范围
        if comp_img_np.max() > comp_img_np.min():
            comp_img_np = (comp_img_np - comp_img_np.min()) / (comp_img_np.max() - comp_img_np.min() + 1e-8) * 255
        comp_img_np = comp_img_np.astype(np.uint8)
        
        # 保存压缩图像
        comp_img_path = os.path.join(jpg_data_dir, f"{index}_{class_id}_round_{round_num}_compressed.jpg")
        if len(comp_img_np.shape) == 2:  # 灰度图像
            PILImage.fromarray(comp_img_np, mode='L').save(comp_img_path)
        else:  # RGB图像
            PILImage.fromarray(comp_img_np, mode='RGB').save(comp_img_path)
        
        # 保存下采样图像（第二列）- 使用最近点插值
        # 获取压缩图像尺寸
        comp_h, comp_w = comp_img.shape[2], comp_img.shape[3]
        
        # 将原始图像转换为PIL图像
        if len(orig_img_np.shape) == 2:  # 灰度图像
            orig_pil = PILImage.fromarray(orig_img_np, mode='L')
            # 下采样到压缩尺寸
            downsampled_img = orig_pil.resize((comp_w, comp_h), PILImage.NEAREST)
            downsampled_np = np.array(downsampled_img)
        else:  # RGB图像
            orig_pil = PILImage.fromarray(orig_img_np, mode='RGB')
            # 下采样到压缩尺寸
            downsampled_img = orig_pil.resize((comp_w, comp_h), PILImage.NEAREST)
            downsampled_np = np.array(downsampled_img)
        
        # 保存下采样图像
        downsampled_img_path = os.path.join(jpg_data_dir, f"{index}_{class_id}_round_{round_num}_downsampled.jpg")
        if len(downsampled_np.shape) == 2:  # 灰度图像
            PILImage.fromarray(downsampled_np, mode='L').save(downsampled_img_path)
        else:  # RGB图像
            PILImage.fromarray(downsampled_np, mode='RGB').save(downsampled_img_path)
        
        # 保存元数据为JSON文件
        metadata = {
            "index": int(index),
            "class_id": int(class_id),
            "round": int(round_num),
            "original_image_path": orig_img_path,
            "downsampled_image_path": downsampled_img_path,
            "compressed_image_path": comp_img_path,
            "original_size": f"{orig_img.shape[2]}x{orig_img.shape[3]}",
            "downsampled_size": f"{comp_h}x{comp_w}",
            "compressed_size": f"{comp_img.shape[2]}x{comp_img.shape[3]}"
        }
        
        metadata_path = os.path.join(jpg_data_dir, f"{index}_{class_id}_round_{round_num}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def generate_summary_pdf(self, total_rounds, data_files=None):
        """
        生成汇总PDF，所有内容存成一页，每个class的每个图带有编号
        从JPG文件加载图像数据
        
        Parameters:
        - total_rounds: 总轮次数
        - data_files: 可选的数据文件路径列表，如果为None则从默认目录加载
        """
        # 获取保存路径
        base_dir = self.config.get("visualization_save_dir", "./fedsumup_visualizations")
        dataset_name = self.config.get("Dataset", "CIFAR10")
        compressed_size = self.compressed_size
        
        # PDF输出路径
        pdf_output_path = os.path.join(
            base_dir,
            f"fedsumup_visualization_{dataset_name}",
            f"compressed_{compressed_size}",
            "summary_visualization.pdf"
        )
        
        # JPG数据目录
        jpg_data_dir = os.path.join(
            base_dir,
            f"fedsumup_visualization_{dataset_name}",
            f"compressed_{compressed_size}",
            "jpg_data"
        )
        
        # 检查JPG数据目录是否存在
        if not os.path.exists(jpg_data_dir):
            print(f"JPG数据目录不存在: {jpg_data_dir}")
            return
        
        # 收集所有元数据JSON文件
        metadata_files = []
        for filename in os.listdir(jpg_data_dir):
            if filename.endswith("_metadata.json"):
                metadata_files.append(os.path.join(jpg_data_dir, filename))
        
        if not metadata_files:
            print("没有找到元数据文件")
            return
        
        # 按类别和索引组织数据
        class_data = {}
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            index = metadata["index"]
            class_id = metadata["class_id"]
            round_num = metadata["round"]
            original_image_path = metadata["original_image_path"]
            downsampled_image_path = metadata["downsampled_image_path"]
            compressed_image_path = metadata["compressed_image_path"]
            
            if class_id not in class_data:
                class_data[class_id] = {}
            
            if index not in class_data[class_id]:
                class_data[class_id][index] = {}
            
            # 存储图像路径和元数据
            class_data[class_id][index][round_num] = {
                "metadata": metadata,
                "original_image_path": original_image_path,
                "downsampled_image_path": downsampled_image_path,
                "compressed_image_path": compressed_image_path
            }
        
        # 创建PDF - 所有内容存成一页
        with PdfPages(pdf_output_path) as pdf:
            # 设置总宽度为90mm（转换为英寸：90mm / 25.4 = 3.543英寸）
            total_width_inch = 90 / 25.4
            
            # 获取所有类别
            all_classes = sorted(class_data.keys())
            if not all_classes:
                print("没有找到类别数据")
                return
            
            # 每个类别显示2个图片，每个图片4种类型
            # 总共列数：8列（2个图片 × 4种类型）
            n_cols = 8
            # 行数：类别数量
            n_rows = len(all_classes)
            
            # 计算图形尺寸
            # 总宽度固定为90mm（3.543英寸）
            # 高度根据行数计算
            fig_width = total_width_inch
            # 每行高度约为宽度的1/8，乘以行数
            fig_height = (fig_width / 8) * n_rows * 1.5  # 增加50%的空间给标题和间距
            
            # 创建图形（所有类别在一个页面上）
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
            
            # 如果只有一个类别，axes的形状需要调整
            if n_rows == 1:
                axes = [axes]
            else:
                axes = axes.reshape(n_rows, n_cols)
            
            # 设置列标题
            column_titles = [
                "Original", "Downsampled", "Round 0 Compressed", "Final Round Compressed",
                "Original", "Downsampled", "Round 0 Compressed", "Final Round Compressed"
            ]
            
            # 对于每个类别
            for row_idx, class_id in enumerate(all_classes):
                # 获取该类别的所有图片（最多2张）
                indices = sorted(class_data[class_id].keys())[:2]
                
                if not indices:
                    # 如果没有图片，跳过
                    continue
                
                col_idx = 0
                for img_idx, index in enumerate(indices):
                    # 获取该图片的所有轮次数据
                    round_data = class_data[class_id][index]
                    
                    # 获取第0轮的数据
                    round_0_data = round_data.get(0)
                    if not round_0_data:
                        # 如果没有第0轮数据，使用最早的一轮
                        available_rounds = sorted(round_data.keys())
                        if available_rounds:
                            round_0_data = round_data[available_rounds[0]]
                    
                    # 获取最后一轮的数据（使用所有轮次中最大的那一轮）
                    available_rounds = sorted(round_data.keys())
                    if available_rounds:
                        # 使用最大的轮次作为最后一轮
                        max_round = available_rounds[-1]
                        round_last_data = round_data[max_round]
                    else:
                        round_last_data = None
                    
                    # 准备4种类型的图像
                    image_paths = []
                    
                    if round_0_data:
                        # 1. 原始图像路径
                        orig_img_path = round_0_data["original_image_path"]
                        # 2. 下采样图像路径（从保存的文件中加载）
                        downsampled_img_path = round_0_data["downsampled_image_path"]
                        # 3. 第0轮压缩图像路径
                        comp_round_0_path = round_0_data["compressed_image_path"]
                        # 4. 最后一轮压缩图像路径
                        if round_last_data:
                            comp_round_last_path = round_last_data["compressed_image_path"]
                        else:
                            comp_round_last_path = comp_round_0_path
                        
                        image_paths = [orig_img_path, downsampled_img_path, comp_round_0_path, comp_round_last_path]
                    else:
                        # 如果没有数据，使用空图像
                        image_paths = [None, None, None, None]
                    
                    # 显示4种类型的图像
                    for i, img_path in enumerate(image_paths):
                        # 获取对应的坐标轴
                        if n_rows == 1:
                            ax = axes[col_idx]
                        else:
                            ax = axes[row_idx, col_idx]
                        
                        # 加载和显示图像
                        if img_path is not None:
                            # 从JPG文件加载图像
                            img = PILImage.open(img_path)
                            img = np.array(img)
                        else:
                            # 创建空图像
                            img = np.zeros((28, 28, 3), dtype=np.uint8) if i != 1 else np.zeros((28, 28), dtype=np.uint8)
                        
                        # 显示图像
                        if len(img.shape) == 2:  # 灰度图像
                            ax.imshow(img, cmap='gray')
                        else:  # RGB图像
                            # 归一化到[0, 1]范围
                            # 如果图像数据是uint8类型（0-255），则归一化到0-1
                            if img.dtype == np.uint8:
                                img = img.astype(np.float32) / 255.0
                            # 如果图像数据已经是浮点数，确保在0-1范围内
                            elif img.max() > 1.0 or img.min() < 0.0:
                                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                            ax.imshow(img)
                        
                        # 设置标题 - 添加编号
                        # 编号格式: 类别-图片序号-类型序号 (例如: 0-1-2 表示类别0的第1个图片的第2种类型)
                        type_names = ["Orig", "Down", "R0", "Final"]
                        type_idx = i  # 0: Original, 1: Downsampled, 2: Round 0 Compressed, 3: Final Round Compressed
                        
                        # 图片编号: 第一个图片为1，第二个图片为2
                        img_num = img_idx + 1
                        
                        # 创建标题
                        title = f"C{class_id}-I{img_num}-{type_names[type_idx]}"
                        
                        # 如果是第一个图片的第一个类型，添加类别标签
                        if img_idx == 0 and i == 0:
                            title = f"Class {class_id}\n{title}"
                        
                        ax.set_title(title, fontsize=6)
                        ax.axis('off')
                        
                        col_idx += 1
            
            # 设置图形标题
            plt.suptitle(f"FedSumUp Visualization - All Classes (Compressed Size: {compressed_size})", fontsize=10, y=1.02)
            
            plt.tight_layout()
            
            # 保存到PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        print(f"汇总PDF已生成: {pdf_output_path}")
        return pdf_output_path
