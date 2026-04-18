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
        
        self.logger.info("FedSumUp服务器（新流程版本）初始化完成")
    
    def _compute_logits_consistency_loss(self, logits, labels):
        """
        计算logits一致性损失（向量化优化版本）
        
        参数:
        - logits: 形状为 [batch_size, num_classes] 的logits
        - labels: 形状为 [batch_size] 的标签

        返回:
        - total_loss: 总的一致性损失
        - same_loss: 同一标签内相似性损失
        - diff_loss: 不同标签间差异性损失
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        
        # 获取所有唯一的标签
        unique_labels, counts = torch.unique(labels, return_counts=True)
        num_unique_labels = len(unique_labels)
        
        # ========== 向量化计算同一标签内相似性损失 ==========
        # 初始化同一标签损失
        same_loss = torch.tensor(0.0, device=logits.device)
        same_count = 0
        
        # 使用向量化方法计算每个标签的方差
        for i, label in enumerate(unique_labels):
            if counts[i] > 1:  # 至少需要两个样本才能计算相似性
                # 获取同一标签的所有样本
                mask = (labels == label)
                same_logits = logits[mask]  # [count_i, num_classes]
                
                # 计算方差并取平均（向量化操作）
                variance = torch.var(same_logits, dim=0, unbiased=True).mean()
                same_loss += variance
                same_count += 1
        
        if same_count > 0:
            same_loss = same_loss / same_count
        
        # ========== 向量化计算不同标签间差异性损失 ==========
        diff_loss = torch.tensor(0.0, device=logits.device)
        
        if num_unique_labels > 1:
            # 计算每个标签的logits均值
            label_means = []
            valid_labels = []
            
            for i, label in enumerate(unique_labels):
                if counts[i] > 0:  # 确保标签有样本
                    mask = (labels == label)
                    label_mean = logits[mask].mean(dim=0, keepdim=True)  # [1, num_classes]
                    label_means.append(label_mean)
                    valid_labels.append(label)
            
            if len(label_means) > 1:
                # 将所有标签均值堆叠成矩阵 [num_valid_labels, num_classes]
                means_matrix = torch.cat(label_means, dim=0)  # [M, num_classes]
                M = means_matrix.size(0)
                
                # 计算所有标签对之间的余弦相似度矩阵
                # 归一化均值向量
                means_norm = F.normalize(means_matrix, p=2, dim=1)  # [M, num_classes]
                
                # 计算余弦相似度矩阵 [M, M]
                cos_sim_matrix = torch.mm(means_norm, means_norm.t())  # [M, M]
                
                # 获取上三角矩阵（不包括对角线）
                triu_mask = torch.triu(torch.ones(M, M, device=logits.device), diagonal=1).bool()
                cos_sim_values = cos_sim_matrix[triu_mask]  # 获取上三角元素
                
                if len(cos_sim_values) > 0:
                    # 计算差异性损失：1 - 余弦相似度（我们希望相似度小）
                    diff_loss = (1.0 - cos_sim_values).mean()
        
        # 总损失 = 同一标签内相似性损失 + 不同标签间差异性损失
        total_loss = same_loss +  diff_loss
        
        
        return total_loss, same_loss, diff_loss

    def arrange_server_data_to_client(self):
        """
        准备要发送给客户端的数据
        新版本：每个round都发送Encoder到客户端
        """
        server_data = {}
        server_data["current_round"] = self.current_round
        
        # 新版本：每个round都发送编码器
        server_data["encoder"] = copy.deepcopy(self.encoder)
        self.logger.info(f"服务器准备发送编码器到客户端（第{self.current_round}轮）")
        
        return server_data
    
    def merge_data(self, received_data_list):
        """
        合并从客户端接收到的数据
        新版本：每个round都合并潜空间编码、labels和对应的索引
        """
        self.logger.info(f"开始合并第{self.current_round}轮客户端数据，共{len(received_data_list)}个客户端")
        
        latent_list = []
        label_list = []
        indices_list = []
        
        for data in received_data_list:
            if data is None:
                raise ValueError(f"客户端数据为None")
            if "latent_variables" not in data:
                raise ValueError(f"客户端数据中缺少latent_variables字段")
            if "labels" not in data:
                raise ValueError(f"客户端数据中缺少labels字段")
            if "client_indices" not in data:
                raise ValueError(f"客户端数据中缺少client_indices字段")
            
            latent_list.append(data["latent_variables"])
            label_list.append(data["labels"])
            indices_list.append(data["client_indices"])
        
        if not latent_list or not label_list:
            raise ValueError("没有收到有效的潜变量或标签数据")
        
        self.aggregated_latent_variables = torch.cat(latent_list, dim=0)
        self.aggregated_labels = torch.cat(label_list, dim=0)
        # 合并所有客户端的索引
        self.aggregated_indices = []
        for indices in indices_list:
            self.aggregated_indices.extend(indices)
        
        self.logger.info(f"合并潜变量、标签和索引完成，潜变量形状: {self.aggregated_latent_variables.shape}, 标签形状: {self.aggregated_labels.shape}, 索引数量: {len(self.aggregated_indices)}")
        
        merged_data = {
            "latent_variables": self.aggregated_latent_variables,
            "labels": self.aggregated_labels,
            "client_indices": self.aggregated_indices
        }
        
        return merged_data
    
    def process(self, merged_data):
        """
        处理合并后的数据（按照用户提出的两阶段流程）
        
        用户流程：
        第1轮服务器：
        第一阶段：分类器学习
          1. 小长宽数据 I₀ 训练分类器 C₀ 得到 C₁
        第二阶段：有损压缩器学习
          1. 冻结 C₁
          2. 小长宽数据 I₀ 直接缩放成 D₀ 大小得到 I₀'
          3. 把 V₀ 和 C₁ 当成一个整体，用 I₀' 和 L₀ 的数据对训练 V₀ 得到 V₁
          4. 用 V₀ 和 C₁ 的整体去评估测试集上的准确率
          5. 把 V₁ 传回客户端
        """
        self.logger.info(f"开始处理第{self.current_round}轮合并数据（两阶段流程）")
        
        # 检查必要字段是否存在
        if "latent_variables" not in merged_data:
            raise ValueError("合并数据中缺少latent_variables字段")
        if "labels" not in merged_data:
            raise ValueError("合并数据中缺少labels字段")
        if "client_indices" not in merged_data:
            raise ValueError("合并数据中缺少client_indices字段")
        
        if merged_data["latent_variables"] is None:
            raise ValueError("latent_variables为None")
        if merged_data["labels"] is None:
            raise ValueError("labels为None")
        if merged_data["client_indices"] is None:
            raise ValueError("client_indices为None")
        
        # 保存训练前的编码器状态
        self._save_encoder_before_training()
        
        # 准备数据（小图I₀）
        small_images = merged_data["latent_variables"].to(self.device)
        labels = merged_data["labels"].to(self.device)
        client_indices = merged_data["client_indices"]
        
        self.logger.info(f"收到小图数据: {small_images.shape}, 标签: {labels.shape}, 索引数量: {len(client_indices)}")
        
        # ========== 第一阶段：分类器学习 ==========
        self.logger.info(f"第一阶段：训练分类器 C_{self.current_round}...")
        self._train_classifier_stage(small_images, labels, client_indices)
        
        # ========== 第二阶段：有损压缩器学习 ==========
        self.logger.info(f"第二阶段：训练编码器 V_{self.current_round}（冻结分类器）...")
        self._train_encoder_stage(small_images, labels, client_indices)
        
        # ========== 评估 ==========
        self.logger.info(f"评估 V_{self.current_round+1} 和 C_{self.current_round+1} 的整体性能...")
        accuracy = self.evaluate()
        
        self.logger.info(f"生成新的编码器V_{self.current_round+1}和分类器C_{self.current_round+1}完成，测试准确率: {accuracy:.4f}")
        
        # 更新轮次
        self.current_round += 1
        self.logger.info(f"第{self.current_round-1}轮处理完成，进入第{self.current_round}轮")
        
        return
    

    
    def _train_classifier_stage(self, small_images, labels, client_indices):
        """
        第一阶段：分类器学习
        
        根据用户流程：
        1. 小长宽数据 I₀ 训练分类器 C₀ 得到 C₁
        
        步骤：
        1. 冻结编码器参数
        2. 解冻分类器参数
        3. 将所有小图 I₀ 放大到32x32
        4. 用放大后的图训练分类器
        5. 添加logits一致性损失：减小同一标签内logits差异，维持不同标签间logits差异
        """
        self.logger.info("第一阶段：训练分类器（冻结编码器，添加logits一致性损失）...")
        
        # 准备数据
        small_images = small_images.to(self.device)
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
        
        # 创建数据集和数据加载器（使用原始小图和标签，在训练时动态放大）
        # 将client_indices转换为tensor
        indices_tensor = torch.tensor(client_indices, device=self.device)
        dataset = TensorDataset(small_images, labels, indices_tensor)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_same_loss = 0.0
            epoch_diff_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (batch, target, indices) in enumerate(dataloader):
                batch, target = batch.to(self.device), target.to(self.device)
                
                classifier_optimizer.zero_grad()
                
                # 将小图放大到32x32
                batch_upsampled = F.interpolate(
                    batch,
                    size=(32, 32),
                    mode='bilinear',
                    align_corners=True
                )
                
                # 分类器处理放大后的图
                # 新版分类器要求4D输入 (B, C, H, W)，不需要展平
                output = self.global_model(batch_upsampled)
                
                # 计算分类损失（交叉熵）
                ce_loss = criterion(output, target)
                
                # 计算logits一致性损失
                consistency_loss, same_loss, diff_loss = self._compute_logits_consistency_loss(
                    output, target
                )
                utility_ratio = self.config.get("sumup_utility_ratio")
                # 总损失 = 交叉熵损失 + logits一致性损失
                total_loss =utility_ratio * ce_loss + consistency_loss
                
                # 检查损失是否为NaN或Inf
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    self.logger.warning(f"损失为NaN或Inf，跳过此batch")
                    continue
                
                total_loss.backward()
                
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=0.1)
                
                classifier_optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_same_loss += same_loss.item()
                epoch_diff_loss += diff_loss.item()
                
                # 计算准确率
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
            avg_ce_loss = epoch_ce_loss / len(dataloader) if len(dataloader) > 0 else 0
            avg_same_loss = epoch_same_loss / len(dataloader) if len(dataloader) > 0 else 0
            avg_diff_loss = epoch_diff_loss / len(dataloader) if len(dataloader) > 0 else 0
            accuracy = 100 * correct / total if total > 0 else 0
            
            self.logger.info(
                f"分类器训练 Epoch {epoch+1}/{num_epochs}, "
                f"总损失: {avg_loss:.4f}, "
                f"交叉熵损失: {avg_ce_loss:.4f}, "
                f"同一标签损失: {avg_same_loss:.4f}, "
                f"不同标签损失: {avg_diff_loss:.4f}, "
                f"准确率: {accuracy:.2f}%"
            )
        
        self.logger.info("第一阶段：分类器训练完成（编码器参数冻结，已添加logits一致性损失）")
        
        # 训练完成后，分类器参数保持可训练状态
        self.logger.info("分类器参数保持可训练状态")
    
    def _train_encoder_stage(self, small_images, labels, client_indices):
        """
        第二阶段：有损压缩器学习
        
        根据用户流程：
        1. 冻结 C₁
        2. 小长宽数据 I₀ 直接缩放成 D₀ 大小得到 I₀'
        3. 把 V₀ 和 C₁ 当成一个整体，用 I₀' 和 L₀ 的数据对训练 V₀ 得到 V₁
        
        关键修改：
        1. 对 L₀^转化的硬标签中与 L₀ 一致的部分的交叉熵损失最小，而不一致的部分交叉熵损失尽可能大
        2. 添加logits一致性损失：减小同一标签内logits差异，维持不同标签间logits差异
        
        步骤：
        1. 冻结分类器参数
        2. 解冻编码器参数
        3. 将小图 I₀ 放大到大图 I₀'（原始尺寸）
        4. 将 I₀' 输入编码器 V，得到压缩后的小图
        5. 将压缩后的小图输入分类器 C（冻结），计算分类损失
        6. 添加logits一致性损失
        7. 用总损失更新编码器参数
        
        参数：
        - small_images: 小图数据
        - labels: 标签数据
        - client_indices: 客户端发送的索引，用于正确匹配追踪的图像
        """
        self.logger.info("第二阶段：训练编码器（冻结分类器，使用改进的损失函数，添加logits一致性损失）...")
        
        # 准备数据
        small_images = small_images.to(self.device)
        labels = labels.to(self.device)  # 原始标签 L₀
        
        # 检查client_indices是否有效
        if client_indices is None:
            raise ValueError("client_indices不能为None")
        if len(client_indices) != len(small_images):
            raise ValueError(f"client_indices数量({len(client_indices)})与small_images数量({len(small_images)})不匹配")
        
        self.logger.info(f"使用客户端索引进行可视化，索引数量: {len(client_indices)}")
        
        # 训练多个epoch
        num_epochs = self.config.get("train_model_epochs", 10)
        
        # 解冻编码器参数
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        # 冻结分类器参数
        for param in self.global_model.parameters():
            param.requires_grad = False
        
        # 定义优化器 - 只优化编码器（分类器参数冻结）
        # 增加学习率以加速编码器训练
        encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=0.01, weight_decay=1e-4  # 从0.001增加到0.01
        )
        criterion = nn.CrossEntropyLoss(reduction='none')  # 使用'reduction=none'获取每个样本的损失
        
        self.encoder.train()
        self.global_model.eval()  # 分类器设为评估模式，因为参数冻结
        
        # ========== 关键修改：先用冻结的分类器计算硬标签 L₀^ ==========
        with torch.no_grad():
            # 分批处理以避免内存溢出
            batch_size = 64  # 使用较小的batch size
            hard_target_labels_list = []
            
            # 分批处理小图
            for i in range(0, len(small_images), batch_size):
                end_idx = min(i + batch_size, len(small_images))
                small_batch = small_images[i:end_idx]
                
                # 将小图 I₀ 上采样到分类器期望的输入大小（32x32）
                small_upsampled_batch = F.interpolate(
                    small_batch,
                    size=(32, 32),
                    mode='bilinear',
                    align_corners=True
                )
                
                # 用冻结的分类器 C₁ 预测硬标签 L₀^
                logits_batch = self.global_model(small_upsampled_batch)
                hard_target_labels_batch = logits_batch.argmax(dim=1)  # 硬标签 L₀^
                hard_target_labels_list.append(hard_target_labels_batch)
                
                # 释放内存
                del small_upsampled_batch, logits_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 合并所有batch的结果
            hard_target_labels = torch.cat(hard_target_labels_list, dim=0)
            
            # 比较原始标签 L₀ 和硬标签 L₀^，找出哪些样本的预测与原始标签一致
            consistent_mask = (labels == hard_target_labels)  # 一致的部分
            inconsistent_mask = (labels != hard_target_labels)  # 不一致的部分
            
            self.logger.info(f"原始标签 L₀ 形状: {labels.shape}")
            self.logger.info(f"硬标签 L₀^ 形状: {hard_target_labels.shape}")
            self.logger.info(f"一致样本比例: {consistent_mask.float().mean().item():.4f} ({consistent_mask.sum().item()}/{len(labels)})")
            self.logger.info(f"不一致样本比例: {inconsistent_mask.float().mean().item():.4f} ({inconsistent_mask.sum().item()}/{len(labels)})")
            self.logger.info(f"原始标签示例: {labels[:10].tolist()}")
            self.logger.info(f"硬标签示例: {hard_target_labels[:10].tolist()}")
            self.logger.info(f"一致掩码示例: {consistent_mask[:10].tolist()}")
        
        # 创建数据集和数据加载器（需要原始标签用于计算一致性）
        dataset = TensorDataset(small_images, hard_target_labels, labels, consistent_mask, inconsistent_mask)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # 获取输入图像大小（M）
        input_size = self.encoder.input_size
        
        # 保存第一个batch用于可视化
        first_batch_data = None
        
        # 初始化可视化系统（如果可用）
        server_visualizer = None
        if VISUALIZATION_SYSTEM_AVAILABLE:
            try:
                # 创建ImageTracker（需要train_indices和train_labels）
                # 从basic_modules中获取target_train_indices和target_train_labels
                from ..visualization_system.image_tracker import ImageTracker
                from ..visualization_system.server_visualizer import ServerVisualizer
                
                # 从basic_modules中获取train_indices和train_labels
                train_indices = self.basic_modules.get('target_train_indices', [])
                train_labels = self.basic_modules.get('target_train_labels', [])
                
                if len(train_indices) == 0 or len(train_labels) == 0:
                    self.logger.warning("basic_modules中没有找到target_train_indices或target_train_labels，使用虚拟数据")
                    # 创建虚拟的train_indices和train_labels
                    train_indices = list(range(small_images.shape[0]))
                    train_labels = labels.cpu().numpy().tolist()
                
                # 创建ImageTracker
                image_tracker = ImageTracker(
                    self.config, 
                    self.dataset_info, 
                    train_indices, 
                    train_labels
                )
                
                # 创建ServerVisualizer
                server_visualizer = ServerVisualizer(image_tracker, self.config)
                self.logger.info("服务器可视化系统初始化完成")
                self.logger.info(f"从basic_modules中获取到train_indices数量: {len(train_indices)}, train_labels数量: {len(train_labels)}")
            except Exception as e:
                self.logger.warning(f"服务器可视化系统初始化失败: {str(e)}")
                server_visualizer = None
        
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            consistent_loss_sum = 0.0
            consistent_count = 0
            inconsistent_count = 0
            correct = 0
            total = 0
            
            for batch_idx, (small_batch, target_labels, orig_labels, cons_mask, incons_mask) in enumerate(dataloader):
                small_batch = small_batch.to(self.device)
                target_labels = target_labels.to(self.device)
                orig_labels = orig_labels.to(self.device)
                cons_mask = cons_mask.to(self.device)
                incons_mask = incons_mask.to(self.device)

                encoder_optimizer.zero_grad()
                
                # 1. 小长宽数据 I₀ 直接缩放成 32x32 大小得到 I₀'
                # 使用双线性插值进行平滑放大
                large_batch = F.interpolate(
                    small_batch,
                    size=(32, 32),
                    mode='bilinear',
                    align_corners=True
                )
                
                # 2. 把 I₀' 输入 V，得到压缩后的小图
                # 新版FunnelEncoder不再需要class_ids参数
                compressed_batch = self.encoder.encode(large_batch)
                
                # 保存第一个batch的数据用于可视化
                if epoch == 0 and batch_idx == 0 and first_batch_data is None:
                    first_batch_data = {
                        'small_images': small_batch.detach().clone(),
                        'large_images': large_batch.detach().clone(),
                        'compressed_images': compressed_batch.detach().clone()
                    }
                
                # 3. 把压缩后的小图输入分类器 C（冻结）
                # 先将压缩后的小图上采样到分类器期望的输入大小（32x32）
                compressed_upsampled = F.interpolate(
                    compressed_batch,
                    size=(32, 32),
                    mode='bilinear',
                    align_corners=True
                )
                
                # 新版分类器要求4D输入 (B, C, H, W)，不需要展平
                output = self.global_model(compressed_upsampled)
                
                # ========== 关键修改：改进的损失函数 ==========
                # 计算每个样本的交叉熵损失
                per_sample_loss = criterion(output, target_labels)  # 形状: [batch_size]
                
                # 对一致的部分：损失最小化（正常权重）
                # 对不一致的部分：损失最大化（负权重）
                # 实现方式：一致的部分使用正常损失，不一致的部分使用负损失
                
                # 计算一致样本的损失（正常）
                consistent_loss = (per_sample_loss * cons_mask).sum()
                # 计算不一致样本的损失（负值，使其最大化）
                # 注意：这里我们只最小化一致部分的损失，不一致部分的损失不参与优化
                
                # 总损失 = 一致损失 - 不一致损失 * α
                # 其中α是权重系数，控制不一致部分的影响程度
                # 使用较小的α值，因为我们要最小化一致部分的损失，最大化不一致部分的损失
                total_ce_loss = consistent_loss
                
                # ========== 添加logits一致性损失 ==========
                # 从配置获取logits一致性损失的权重
       
                # 计算logits一致性损失
                consistency_loss, same_loss, diff_loss = self._compute_logits_consistency_loss(
                    output, target_labels)
                utility_ratio = self.config.get("sumup_utility_ratio")

                # 总损失 = 交叉熵损失 + logits一致性损失
                total_loss = utility_ratio*total_ce_loss + consistency_loss
                
                # 总损失 = 总损失 / batch_size
                loss = total_loss / small_batch.size(0)
                
                # 检查损失是否为NaN或Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"损失为NaN或Inf，跳过此batch")
                    continue
                
                loss.backward()
                
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=0.1)
                
                encoder_optimizer.step()
                
                epoch_loss += loss.item() * small_batch.size(0)
                consistent_loss_sum += consistent_loss.item()

                consistent_count += cons_mask.sum().item()
                inconsistent_count += incons_mask.sum().item()
                
                # 计算准确率（相对于硬标签 L₀^）
                _, predicted = torch.max(output.data, 1)
                total += target_labels.size(0)
                correct += (predicted == target_labels).sum().item()
            
            avg_loss = epoch_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
            avg_consistent_loss = consistent_loss_sum / consistent_count if consistent_count > 0 else 0
            accuracy = 100 * correct / total if total > 0 else 0

            self.logger.info(f"编码器训练 Epoch {epoch+1}/{num_epochs}, "
                           f"总损失: {avg_loss:.4f}, "
                           f"一致损失: {avg_consistent_loss:.4f}, "
                           f"准确率: {accuracy:.2f}%, "
                           f"一致样本数: {consistent_count}, "
                           f"不一致样本数: {inconsistent_count}, ")
                         
            

            # 如果可视化系统可用，收集当前epoch的追踪图片数据
            if server_visualizer is not None:
                try:
                    # 获取追踪的图片索引
                    tracked_indices = server_visualizer.image_tracker.get_tracked_indices()
                    
                    # 收集追踪图片的数据
                    tracked_images_data = {}
                    
                    # 遍历所有追踪的图片
                    for idx, class_id in tracked_indices.items():
                        # 注意：idx是相对于train_indices的索引
                        # 我们需要检查这个索引是否在客户端发送的client_indices中
                        # 如果在，就使用该索引在small_images中的位置
                        
                        # 首先，在client_indices中找到idx的位置
                        if idx in client_indices:
                            position = client_indices.index(idx)
                            
                            # 获取该图片的数据
                            small_img = small_images[position:position+1]
                            # 放大到32x32
                            large_img = F.interpolate(
                                small_img,
                                size=(32, 32),
                                mode='bilinear',
                                align_corners=True
                            )
                            # 编码器压缩
                            compressed_img = self.encoder.encode(large_img)
                            
                            # 保存数据
                            tracked_images_data[idx] = {
                                "original": small_img,
                                "compressed": compressed_img,
                                "class_id": class_id,
                                "position_in_small_images": position
                            }
                            self.logger.debug(f"找到追踪图片 {idx} (类别 {class_id}) 在small_images中的位置: {position} (使用客户端索引)")
                        else:
                            self.logger.warning(f"追踪图片 {idx} (类别 {class_id}) 不在客户端发送的索引中，跳过可视化")
                    
                    # 添加到ServerVisualizer
                    if tracked_images_data:
                        server_visualizer.add_epoch_visualization(epoch, tracked_images_data)
                        self.logger.info(f"已收集第{epoch}个epoch的追踪图片数据，共{len(tracked_images_data)}张图片")
                    else:
                        self.logger.warning(f"第{epoch}个epoch没有收集到任何追踪图片数据")
                    
                except Exception as e:
                    self.logger.warning(f"收集追踪图片数据失败: {str(e)}")
                    import traceback
                    self.logger.debug(f"详细错误信息: {traceback.format_exc()}")
        
        self.logger.info("第二阶段：编码器训练完成（分类器参数冻结）")
        
        # 训练完成后，重新冻结编码器参数
        self.logger.info("编码器训练完成，重新冻结编码器参数...")
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 分类器参数保持冻结状态
        for param in self.global_model.parameters():
            param.requires_grad = False
        
        self.logger.info("编码器参数已重新冻结，分类器参数保持冻结")
        
        # 如果可视化系统可用，保存服务器可视化结果
        if server_visualizer is not None:
            try:
                server_visualizer.save_server_visualization(self.current_round)
                self.logger.info(f"服务器第{self.current_round}轮可视化结果保存完成")
            except Exception as e:
                self.logger.warning(f"保存服务器可视化结果失败: {str(e)}")
    
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
        修改：使用上一轮的VAE进行评估（而不是最新的编码器）
        """
        self.logger.info("开始评估模型性能（使用上一轮的VAE和最新的分类器）...")
        
        # 修改：使用上一轮的VAE（训练前的编码器）进行评估
        if self.encoder_before_training is not None:
            encoder_for_eval = self.encoder_before_training
            self.logger.info(f"使用上一轮的VAE进行评估（第{self.current_round}轮训练前）")
        else:
            # 如果没有保存训练前的编码器，使用当前编码器
        
            encoder_for_eval = self.encoder
            self.logger.info(f"使用当前编码器进行评估（第{self.current_round}轮）")
        
        # 使用最新的分类器
        classifier_for_eval = self.global_model
        
        encoder_for_eval.eval()
        classifier_for_eval.eval()
        
        with torch.no_grad():
            correct, total = 0, 0
            for x, target in self.test_loader:
                x, target = x.to(self.device), target.to(self.device, dtype=torch.int64)
                
                # 使用上一轮编码器将数据编码到潜空间
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
        self.logger.info(f"模型评估准确率（使用上一轮VAE）: {accuracy:.4f}")
        
        # 同时记录使用最新编码器的准确率（用于对比）
        if self.encoder is not None and self.encoder is not encoder_for_eval:
            encoder_latest_eval = self.encoder
            encoder_latest_eval.eval()
            correct_latest, total_latest = 0, 0
            with torch.no_grad():
                for x, target in self.test_loader:
                    x, target = x.to(self.device), target.to(self.device, dtype=torch.int64)
                    # 新版FunnelEncoder不再需要class_ids参数
                    z = encoder_latest_eval.encode(x)
                    # 将编码器输出上采样到分类器期望的输入大小（32x32）
                    z_upsampled = F.interpolate(
                        z,
                        size=(32, 32),
                        mode='bilinear',
                        align_corners=True
                    )
                    # 新版分类器要求4D输入 (B, C, H, W)，不需要展平
                    pred = classifier_for_eval(z_upsampled)
                    _, pred_label = torch.max(pred.data, 1)
                    total_latest += x.data.size()[0]
                    correct_latest += (pred_label == target.data).sum().item()
            
            accuracy_latest = correct_latest / float(total_latest)
            self.logger.info(f"模型评估准确率（使用最新编码器）: {accuracy_latest:.4f}")
            self.logger.info(f"准确率差异（最新编码器 - 上一轮编码器）: {accuracy_latest - accuracy:.4f}")
        
        return accuracy
