import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import sys
import numpy as np
import os
import torchvision
import matplotlib.pyplot as plt

# 导入漏斗编码器
try:
    from ..funnel_encoder import FunnelEncoder
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from funnel_encoder import FunnelEncoder

# 导入可视化系统
try:
    from ..visualization_system.image_tracker import ImageTracker
    from ..visualization_system.server_visualizer import ServerVisualizer
    VISUALIZATION_SYSTEM_AVAILABLE = True
except ImportError:
    try:
        # 如果相对导入失败，尝试绝对导入
        from visualization_system.image_tracker import ImageTracker
        from visualization_system.server_visualizer import ServerVisualizer
        VISUALIZATION_SYSTEM_AVAILABLE = True
    except ImportError:
        VISUALIZATION_SYSTEM_AVAILABLE = False
        print("警告: 可视化系统不可用")

class Server:
    def __init__(self, basic_modules, config, logger, clients):
        """
        初始化服务器（新流程版本）。
        
        参数:
        - basic_modules: 包含基本模块的字典，例如全局模型、设备等信息。
        - config: 配置字典，包含服务器运行所需的配置参数。
        - logger: 日志记录器，用于记录服务器运行过程中的日志信息。
        - clients: 客户端列表或相关信息。
        """
        self.clients = clients  # 存储所有客户端的信息
        self.logger = logger  # 设置日志记录器
        
        self.logger.info("正在初始化FedSumUp服务器（新流程版本）...")
        
        # 模型相关
        # 使用deepcopy确保全局模型的独立性，并将其移动到指定设备上
        
        self.model_strategy = config.get("Model")  # 获取模型相关的策略
        self.logger.info(f"服务器模型初始化完成: {self.model_strategy}")
        
        # 调度相关
        self.join_ratio = config.get("join_ratio")  # 获取参与率
        
        # 数据相关
        # 保存dataset_info用于denormalize
        self.dataset_info = basic_modules['dataset_info']
        # 加载测试集并使用SubsetRandomSampler来获取特定索引的数据样本
        self.test_set = basic_modules['dst_test']
        # 使用更大的batch_size提高效率，默认为32
        test_batch_size = 32
        self.test_loader = DataLoader(
            self.test_set,
            sampler=SubsetRandomSampler(basic_modules['target_test_indices']),
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        self.logger.info(f"服务器测试数据加载完成，batch_size={test_batch_size}，共{len(basic_modules['target_test_indices'])}个样本")
        
        # 设备相关
        self.device = basic_modules['device']  # 确定使用的设备（如CPU或GPU）
        
        # 使用从basic_modules传入的编码器和分类器
        self.encoder = basic_modules['vae']  # 注意：这里仍然使用'vae'键，但实际上是FunnelEncoder
        self.global_model = basic_modules['classifier']
        # 保存一个深拷贝作为"无辜的"分类器模板
        self.innocent_classifier_template = copy.deepcopy(basic_modules['classifier'])
        self.logger.info("编码器模型和分类器已从basic_modules加载")
        
        # 保存训练前的编码器状态（用于评估）
        self.encoder_before_training = None
        self._save_encoder_before_training()
        
        # 联邦学习策略
        self.fed_strategy = config.get("Federated_Learning_Config")  # 获取联邦学习策略
        self.logger.info(f"服务器使用联邦策略: {self.fed_strategy}")
        
        # 将config保存为实例变量，以便后续访问
        self.config = config
        
        # 保存basic_modules以便后续使用
        self.basic_modules = basic_modules
        
        # 初始化状态变量
        self.current_round = 0
        self.aggregated_latent_variables = None
        self.aggregated_labels = None
        
        # 初始化可视化系统
        self._init_visualization_system(config)
        
        self.logger.info("FedSumUp服务器（新流程版本）初始化完成")

    def _init_visualization_system(self, config):
        """
        初始化可视化系统
        """
        if not VISUALIZATION_SYSTEM_AVAILABLE:
            self.logger.warning("可视化系统不可用，跳过初始化")
            self.image_tracker = None
            self.server_visualizer = None
            return
        
        try:
            # 获取配置参数
            dataset_name = config.get("dataset_name", "CIFAR10")
            compressed_size = config.get("compressed_image_size", 8)
            num_tracked_images = config.get("num_tracked_images", 10)
            
            # 从basic_modules获取必要的数据
            dataset_info = self.basic_modules.get('dataset_info', {})
            train_indices = self.basic_modules.get('train_indices', [])
            train_labels = self.basic_modules.get('train_labels', [])
            
            # 创建图片追踪器（使用正确的参数）
            self.image_tracker = ImageTracker(
                config=config,
                dataset_info=dataset_info,
                train_indices=train_indices,
                train_labels=train_labels
            )
            
            # 创建服务器可视化器
            self.server_visualizer = ServerVisualizer(
                image_tracker=self.image_tracker,
                config=config
            )
            
            self.logger.info(f"可视化系统初始化完成，数据集: {dataset_name}, 压缩尺寸: {compressed_size}, 追踪图片数: {num_tracked_images}")
        except Exception as e:
            self.logger.error(f"可视化系统初始化失败: {e}")
            self.image_tracker = None
            self.server_visualizer = None

    def arrange_server_data_to_client(self):
        """
        准备要发送给客户端的数据
        新需求：发送加权平均后的编码器、当前轮次和总轮次信息
        """
        server_data = {}
        server_data["current_round"] = self.current_round
        server_data["total_rounds"] = self.config.get("num_rounds", 10)
        
        # 发送加权平均后的编码器
        server_data["encoder"] = copy.deepcopy(self.encoder)
        self.logger.info(f"服务器准备发送加权平均后的编码器到客户端（第{self.current_round}轮，总轮次{server_data['total_rounds']}）")
        
        return server_data
    
    def merge_data(self, received_data_list):
        """
        合并从客户端接收到的数据（实验演示版本）
        新需求：
        1. 每轮都加权平均各客户端上传的 encoder 参数
        2. 每轮都合并压缩图片和标签
        """
        self.logger.info(f"开始合并第{self.current_round}轮客户端数据，共{len(received_data_list)}个客户端")
        
        # 每轮都进行 encoder 参数加权平均
        self.logger.info(f"第{self.current_round}轮，进行 encoder 参数加权平均")
        encoder_merged = self._merge_encoder_parameters(received_data_list)
        
        # 每轮都合并压缩图片和标签
        self.logger.info(f"第{self.current_round}轮，合并压缩图片和标签")
        images_merged = self._merge_compressed_images(received_data_list)
        
        # 合并结果
        merged_data = {}
        if encoder_merged is not None:
            merged_data.update(encoder_merged)
        if images_merged is not None:
            merged_data.update(images_merged)
        
        return merged_data
    
    def _merge_encoder_parameters(self, received_data_list):
        """
        加权平均各客户端上传的 encoder 参数
        """
        self.logger.info(f"开始加权平均 encoder 参数，共{len(received_data_list)}个客户端")
        
        # 收集所有客户端的 encoder 状态和样本数量
        encoder_states = []
        sample_counts = []
        client_ids = []
        
        for data in received_data_list:
            if data is None:
                self.logger.warning(f"客户端数据为None，跳过")
                continue
            
            if "encoder_state" not in data:
                self.logger.warning(f"客户端数据中缺少encoder_state字段，跳过")
                continue
            
            if "num_samples" not in data:
                self.logger.warning(f"客户端数据中缺少num_samples字段，跳过")
                continue
            
            encoder_states.append(data["encoder_state"])
            sample_counts.append(data["num_samples"])
            client_ids.append(data.get("client_id", "unknown"))
        
        if not encoder_states:
            self.logger.warning("没有收到有效的 encoder 参数数据")
            return {}
        
        # 计算总样本数
        total_samples = sum(sample_counts)
        self.logger.info(f"总样本数: {total_samples}, 各客户端样本数: {sample_counts}")
        
        # 初始化加权平均后的 encoder 状态
        avg_encoder_state = {}
        
        # 遍历所有参数键
        for key in encoder_states[0].keys():
            # 初始化加权和
            weighted_sum = None
            
            for i, state in enumerate(encoder_states):
                param = state[key]
                weight = sample_counts[i] / total_samples
                
                if weighted_sum is None:
                    weighted_sum = param * weight
                else:
                    weighted_sum += param * weight
            
            avg_encoder_state[key] = weighted_sum
        
        # 更新服务器的 encoder
        self.encoder.load_state_dict(avg_encoder_state)
        
        self.logger.info(f"encoder 参数加权平均完成，共{len(encoder_states)}个客户端参与")
        
        # 返回包含encoder状态的信息
        return {"encoder_updated": True}
    
    def _merge_compressed_images(self, received_data_list):
        """
        合并每轮客户端上传的压缩图片和标签，并保存可视化结果
        """
        self.logger.info(f"开始合并第{self.current_round}轮的压缩图片和标签，共{len(received_data_list)}个客户端")
        
        compressed_images_list = []
        labels_list = []
        indices_list = []
        
        # 收集所有客户端的原始图片数据（用于可视化）
        original_images_list = []
        original_indices_list = []
        
        for data in received_data_list:
            if data is None:
                self.logger.warning(f"客户端数据为None，跳过")
                continue
            
            if "compressed_images" not in data:
                self.logger.warning(f"客户端数据中缺少compressed_images字段，跳过")
                continue
            
            if "labels" not in data:
                self.logger.warning(f"客户端数据中缺少labels字段，跳过")
                continue
            
            if "client_indices" not in data:
                self.logger.warning(f"客户端数据中缺少client_indices字段，跳过")
                continue
            
            compressed_images_list.append(data["compressed_images"])
            labels_list.append(data["labels"])
            indices_list.append(data["client_indices"])
            
            # 收集原始图片数据（如果存在）
            if "original_images" in data and data["original_images"] is not None:
                original_images_list.append(data["original_images"])
                original_indices_list.append(data["client_indices"])
        
        if not compressed_images_list or not labels_list:
            self.logger.warning("没有收到有效的压缩图片或标签数据")
            return {}
        
        # 合并所有客户端的压缩图片和标签
        self.aggregated_compressed_images = torch.cat(compressed_images_list, dim=0)
        self.aggregated_labels = torch.cat(labels_list, dim=0)
        # 合并所有客户端的索引
        self.aggregated_indices = []
        for indices in indices_list:
            self.aggregated_indices.extend(indices)
        
        self.logger.info(f"合并压缩图片和标签完成，压缩图片形状: {self.aggregated_compressed_images.shape}, 标签形状: {self.aggregated_labels.shape}, 索引数量: {len(self.aggregated_indices)}")
        
        # 保存可视化结果（如果可视化系统可用）
        if (self.image_tracker is not None and self.server_visualizer is not None and 
            original_images_list and original_indices_list):
            try:
                self._save_visualization_results(
                    original_images_list, original_indices_list,
                    compressed_images_list, labels_list, indices_list
                )
            except Exception as e:
                self.logger.error(f"保存可视化结果时出错: {e}")
        
        merged_data = {
            "compressed_images": self.aggregated_compressed_images,
            "labels": self.aggregated_labels,
            "client_indices": self.aggregated_indices
        }
        
        return merged_data
    
    def _save_visualization_results(self, original_images_list, original_indices_list,
                                   compressed_images_list, labels_list, indices_list):
        """
        保存可视化结果
        """
        self.logger.info(f"开始保存第{self.current_round}轮的可视化结果")
        
        # 合并所有原始图片
        all_original_images = torch.cat(original_images_list, dim=0)
        all_original_indices = []
        for indices in original_indices_list:
            all_original_indices.extend(indices)
        
        # 合并所有压缩图片
        all_compressed_images = torch.cat(compressed_images_list, dim=0)
        
        # 合并所有标签
        all_labels = torch.cat(labels_list, dim=0)
        
        # 获取追踪的图片索引
        tracked_indices = self.image_tracker.get_tracked_indices()
        
        # 创建索引到位置的映射，提高查找效率
        index_to_position = {}
        for i, idx in enumerate(all_original_indices):
            index_to_position[idx] = i
        
        # 对每个追踪的图片，找到对应的数据
        for index, class_id in tracked_indices.items():
            # 在原始索引中查找该图片
            if index not in index_to_position:
                self.logger.debug(f"追踪的图片索引 {index} 不在客户端上传的数据中")
                continue
            
            # 找到该图片在所有列表中的位置
            idx = index_to_position[index]
            
            # 获取原始图片和压缩图片
            original_img = all_original_images[idx:idx+1]  # 保持batch维度
            compressed_img = all_compressed_images[idx:idx+1]
            label = all_labels[idx:idx+1]
            
            # 准备可视化数据
            tracked_image_data = {
                index: {
                    "original": original_img,
                    "compressed": compressed_img,
                    "class_id": class_id
                }
            }
            
            # 添加到服务器可视化器
            self.server_visualizer.add_epoch_visualization(
                epoch=self.current_round,
                tracked_images_data=tracked_image_data
            )
        
        # 保存服务器可视化结果
        self.server_visualizer.save_server_visualization(round_num=self.current_round)
        
        self.logger.info(f"第{self.current_round}轮的可视化结果保存完成")
    
    def process(self, merged_data):
        """
        处理合并后的数据（实验演示版本）
        
        新需求（实验演示版本）：
        1. 每轮都训练 classifier
        2. 每轮都计算 acc 和 mia acc
        3. 报告每一轮的 acc 和 mia acc
        """
        self.logger.info(f"开始处理第{self.current_round}轮合并数据（实验演示版本）")
        
        # 检查必要字段是否存在
        if "compressed_images" not in merged_data:
            self.logger.warning("合并数据中缺少compressed_images字段，跳过classifier训练")
            # 只更新轮次
            self.current_round += 1
            self.logger.info(f"第{self.current_round-1}轮处理完成（跳过训练），进入第{self.current_round}轮")
            return
        
        if "labels" not in merged_data:
            self.logger.warning("合并数据中缺少labels字段，跳过classifier训练")
            # 只更新轮次
            self.current_round += 1
            self.logger.info(f"第{self.current_round-1}轮处理完成（跳过训练），进入第{self.current_round}轮")
            return
        
        if "client_indices" not in merged_data:
            self.logger.warning("合并数据中缺少client_indices字段，跳过classifier训练")
            # 只更新轮次
            self.current_round += 1
            self.logger.info(f"第{self.current_round-1}轮处理完成（跳过训练），进入第{self.current_round}轮")
            return
        
        if merged_data["compressed_images"] is None:
            self.logger.warning("compressed_images为None，跳过classifier训练")
            # 只更新轮次
            self.current_round += 1
            self.logger.info(f"第{self.current_round-1}轮处理完成（跳过训练），进入第{self.current_round}轮")
            return
        
        if merged_data["labels"] is None:
            self.logger.warning("labels为None，跳过classifier训练")
            # 只更新轮次
            self.current_round += 1
            self.logger.info(f"第{self.current_round-1}轮处理完成（跳过训练），进入第{self.current_round}轮")
            return
        
        if merged_data["client_indices"] is None:
            self.logger.warning("client_indices为None，跳过classifier训练")
            # 只更新轮次
            self.current_round += 1
            self.logger.info(f"第{self.current_round-1}轮处理完成（跳过训练），进入第{self.current_round}轮")
            return
        
        # 准备数据（压缩图片）
        compressed_images = merged_data["compressed_images"].to(self.device)
        labels = merged_data["labels"].to(self.device)
        client_indices = merged_data["client_indices"]
        
        self.logger.info(f"收到压缩图片数据: {compressed_images.shape}, 标签: {labels.shape}, 索引数量: {len(client_indices)}")
        
        # 每轮都训练 classifier
        self.logger.info(f"第{self.current_round}轮：开始训练 classifier...")
        self._train_classifier_every_round(compressed_images, labels, client_indices)
        
        # 评估模型性能
        self.logger.info(f"评估第{self.current_round}轮 encoder 和 classifier 的整体性能...")

        

        
        # 更新轮次
        self.current_round += 1
        self.logger.info(f"第{self.current_round-1}轮处理完成，进入第{self.current_round}轮")
        
        return
    
    def _train_classifier_every_round(self, compressed_images, labels, client_indices):
        """
        每轮都从头开始训练 classifier
        """
        self.logger.info(f"第{self.current_round}轮：从头开始训练 classifier...")
        
        # 准备数据
        compressed_images = compressed_images.to(self.device)
        labels = labels.to(self.device)
        
        # 训练多个epoch
        num_epochs = self.config.get("train_model_epochs", 10)
        
        # 冻结编码器参数
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 每轮都从头开始：重新初始化分类器
        # 使用无辜分类器模板的深拷贝作为新的分类器
        self.global_model = copy.deepcopy(self.innocent_classifier_template)
        self.global_model.to(self.device)
        
        # 解冻分类器参数
        for param in self.global_model.parameters():
            param.requires_grad = True
        
        # 定义优化器 - 只优化分类器（编码器参数冻结）
        classifier_optimizer = torch.optim.Adam(
            self.global_model.parameters(),
            lr=0.001, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        
        self.encoder.eval()  # 编码器设为评估模式，因为参数冻结
        self.global_model.train()
        
        # 创建数据集和数据加载器
        # 将client_indices转换为tensor
        indices_tensor = torch.tensor(client_indices, device=self.device)
        dataset = TensorDataset(compressed_images, labels, indices_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (batch, target, indices) in enumerate(dataloader):
                batch, target = batch.to(self.device), target.to(self.device)
                
                classifier_optimizer.zero_grad()
                
                # 将压缩图片放大到分类器期望的输入大小（32x32）
                batch_upsampled = F.interpolate(
                    batch,
                    size=(32, 32),
                    mode='bilinear',
                    align_corners=True
                )
                
                # 分类器处理放大后的图
                # 新版分类器要求4D输入 (B, C, H, W)，不需要展平
                output = self.global_model(batch_upsampled)
                
                # 计算分类损失
                loss = criterion(output, target)
                
                # 检查损失是否为NaN或Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"损失为NaN或Inf，跳过此batch")
                    continue
                
                loss.backward()
                
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=0.1)
                
                classifier_optimizer.step()
                
                epoch_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
            accuracy = 100 * correct / total if total > 0 else 0
            self.logger.info(f"第{self.current_round}轮分类器训练 Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        self.logger.info(f"第{self.current_round}轮：分类器训练完成（从头开始）")
        
        # 训练完成后，分类器参数保持可训练状态
        self.logger.info("分类器参数保持可训练状态")
        
        return
    

    
    def _train_classifier_final_round(self, compressed_images, labels, client_indices):
        """
        最后一轮：使用压缩图片训练 classifier
        """
        self.logger.info("最后一轮：训练 classifier...")
        
        # 准备数据
        compressed_images = compressed_images.to(self.device)
        labels = labels.to(self.device)
        
        # 训练多个epoch
        num_epochs = self.config.get("train_model_epochs", 10)
        
        # 冻结编码器参数
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 解冻分类器参数
        for param in self.global_model.parameters():
            param.requires_grad = True
        
        # 定义优化器 - 只优化分类器（编码器参数冻结）
        classifier_optimizer = torch.optim.Adam(
            self.global_model.parameters(),
            lr=0.001, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        
        self.encoder.eval()  # 编码器设为评估模式，因为参数冻结
        self.global_model.train()
        
        # 创建数据集和数据加载器
        # 将client_indices转换为tensor
        indices_tensor = torch.tensor(client_indices, device=self.device)
        dataset = TensorDataset(compressed_images, labels, indices_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (batch, target, indices) in enumerate(dataloader):
                batch, target = batch.to(self.device), target.to(self.device)
                
                classifier_optimizer.zero_grad()
                
                # 将压缩图片放大到分类器期望的输入大小（32x32）
                batch_upsampled = F.interpolate(
                    batch,
                    size=(32, 32),
                    mode='bilinear',
                    align_corners=True
                )
                
                # 分类器处理放大后的图
                # 新版分类器要求4D输入 (B, C, H, W)，不需要展平
                output = self.global_model(batch_upsampled)
                
                # 计算分类损失
                loss = criterion(output, target)
                
                # 检查损失是否为NaN或Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"损失为NaN或Inf，跳过此batch")
                    continue
                
                loss.backward()
                
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=0.1)
                
                classifier_optimizer.step()
                
                epoch_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
            accuracy = 100 * correct / total if total > 0 else 0
            self.logger.info(f"最后一轮分类器训练 Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        self.logger.info("最后一轮：分类器训练完成")
        
        # 训练完成后，分类器参数保持可训练状态
        self.logger.info("分类器参数保持可训练状态")
 
    def _save_encoder_before_training(self):
        """保存训练前的编码器状态（深拷贝）"""
        self.encoder_before_training = copy.deepcopy(self.encoder)
        self.logger.info("已保存训练前的编码器状态（用于评估）")
    
    def select_clients(self):
        return (
            self.clients if self.join_ratio == 1.0
            else random.sample(self.clients, int(round(len(self.clients) * self.join_ratio)))
        )
    
    def evaluate(self):
        """
        评估模型性能
        修改：使用当前轮的VAE进行评估
        """
        self.logger.info("开始评估模型性能（使用当前轮的VAE和分类器）...")
        
        # 使用当前轮的VAE进行评估
        encoder_for_eval = self.encoder
        self.logger.info(f"使用当前轮VAE进行评估（第{self.current_round}轮）")
        
        # 使用最新的分类器
        classifier_for_eval = self.global_model
        
        encoder_for_eval.eval()
        classifier_for_eval.eval()
        
        with torch.no_grad():
            correct, total = 0, 0
            for x, target in self.test_loader:
                x, target = x.to(self.device), target.to(self.device, dtype=torch.int64)
                
                # 使用当前轮编码器将数据编码到潜空间
                # 新版FunnelEncoder不再需要class_ids参数
                z = encoder_for_eval.encode(x)
                
                # 将编码器输出上采样到分类器期望的输入大小（32x32）
                # 分类器是在原始图像大小（如32x32）上训练的
                z_upsampled = F.interpolate(
                    z,
                    size=(32, 32),
                    mode='bilinear',
                    align_corners=True
                )
                
                # 新版分类器要求4D输入 (B, C, H, W)，不需要展平
                # 直接传递z_upsampled，形状为 [B, C, 32, 32]
                pred = classifier_for_eval(z_upsampled)
                _, pred_label = torch.max(pred.data, 1)
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()
        
        accuracy = correct / float(total)
        self.logger.info(f"模型评估准确率（使用当前轮VAE）: {accuracy:.4f}")
        
        return accuracy
