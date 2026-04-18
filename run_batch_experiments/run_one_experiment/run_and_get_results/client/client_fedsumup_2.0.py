import copy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import sys
import os

# 导入可视化系统
try:
    from ..visualization_system.image_tracker import ImageTracker
    from ..visualization_system.client_visualizer import ClientVisualizer
    VISUALIZATION_SYSTEM_AVAILABLE = True
except ImportError:
    VISUALIZATION_SYSTEM_AVAILABLE = False
    print("警告: 可视化系统不可用")

class Client:
    def __init__(self, client_modules, config, logger, i):
        """
        客户端初始化函数（新流程版本）
        参数解释：
        - client_modules: 包含客户端所需的各种模块的字典
        - config: 配置参数的字典
        - logger: 日志记录器
        - i: 客户端ID
        """
        # 初始化配置和日志记录器
        self.config = config
        self.logger = logger
        
        # 记录开始初始化客户端的日志
        self.logger.info(f"正在初始化客户端 {i}（新流程版本）...")
        
        # 从配置中获取模型训练周期数
        self.model_epochs = config.get("train_model_epochs")
        
        # 设置客户端ID
        self.cid = i
        
        # 数据相关初始化
        self.dst_train = client_modules['dst_train']  # 训练数据集
        self.client_indices = client_modules['client_indices'][i]  # 当前客户端的数据索引
        self.classes = client_modules['client_classes'][i]  # 当前客户端的类别信息
        self.dataset_info = client_modules['dataset_info']  # 数据集信息
        self.test_set = client_modules['dst_test']  # 测试数据集
        
        # 创建测试数据加载器
        self.test_loader = DataLoader(
            self.test_set,
            sampler=SubsetRandomSampler(client_modules['target_test_indices']),
            batch_size=1, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )
        # 记录数据加载完成的日志
        self.logger.info(f"客户端 {i} 数据加载完成，包含类别: {self.classes}")
        
        # 模型相关初始化
        self.model_strategy = config.get("Model")  # 获取模型策略
        self.local_model = copy.deepcopy(client_modules["classifier"])  # 全局模型
        
        # 联邦学习策略相关初始化
        self.fed_strategy = config.get("Federated_Learning_Config")  # 获取联邦学习策略
        self.logger.info(f"客户端 {i} 使用联邦策略: {self.fed_strategy}")
        
        # 如果采用的是FedSumUp策略，则进行特定参数的初始化
        if self.fed_strategy == "FedSumUp":
            self.num_crop = config.get("sumup_num_crop", 5)
            self.input_size = self.dataset_info["im_size"][0]
        
            self.sumup_iterations = config.get("sumup_iterations", 100)
            self.foulier_alpha = config.get("sumup_foulier_alpha", 0.5)
            self.logger.info(f"客户端 {i} FedSumUp参数初始化完成")
        
        # 设备相关初始化
        self.device = client_modules['device']
        
        # 保存client_modules以便后续使用
        self.client_modules = client_modules
        
        # 初始化状态变量
        self.current_round = 0
        self.latent_variables = None
        self.labels = None
        
        # 记录客户端初始化完成的日志
        self.logger.info(f"客户端 {i} 初始化完成")
        
    def receive_data_from_server(self, server_data):
        """
        接收来自服务器的数据
        新流程：接收编码器、当前轮次和总轮次信息
        """
        self.current_round = server_data.get("current_round", 0)
        self.total_rounds = server_data.get("total_rounds", 1)
        self.logger.info(f"客户端 {self.cid} 接收到第 {self.current_round}/{self.total_rounds} 轮服务器数据")
        
        # 判断是否是最后一轮
        self.is_final_round = (self.current_round == self.total_rounds - 1)
        if self.is_final_round:
            self.logger.info(f"客户端 {self.cid} 这是最后一轮，将上传压缩图片")
        
        # 接收编码器
        if "encoder" in server_data:
            self.encoder = copy.deepcopy(server_data["encoder"]).to(self.device)
            self.logger.info(f"客户端 {self.cid} 接收到编码器")
        else:
            self.logger.warning(f"客户端 {self.cid} 没有接收到编码器")
        
        return

    def process(self):
        """
        客户端处理流程（新需求版本）
        
        新需求（实验演示版本）：
        1. 每轮都训练 encoder：首先将自己所拥有的图片按标签归类，对每一个epoch,对每一个类，
           要求同类图片所encode出来的图片的差距尽可能小。
        2. 每轮都选择并压缩图片（按照配置中的比例 r）
        3. 每轮都上传训练好的 encoder 参数和压缩图片
        """
        self.logger.info(f"客户端 {self.cid} 开始第 {self.current_round} 轮处理（实验演示版本）")
        
        # 检查是否有编码器
        if not hasattr(self, 'encoder'):
            raise ValueError(f"客户端 {self.cid} 缺少编码器，无法进行训练或压缩")
        
        # 每轮都训练 encoder
        self.logger.info(f"客户端 {self.cid} 开始训练 encoder（第{self.current_round}轮）...")
        self.train_encoder()
        
        # 每轮都选择并压缩图片
        self.logger.info(f"客户端 {self.cid} 选择并压缩图片（第{self.current_round}轮）...")
        self.select_and_compress_images()
        
        self.logger.info(f"客户端 {self.cid} 处理流程完成")
        return
    
    def train_encoder(self):
        """
        训练 encoder 使同类图片编码相似
        对每一个epoch，对每一个类，要求同类图片所encode出来的图片的差距尽可能小
        修改：遍历每个类别的所有图片，而不是只取前几个
        """
        self.logger.info(f"客户端 {self.cid} 开始训练 encoder...")
        
        # 设置 encoder 为训练模式
        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        # 获取配置参数
        encoder_epochs = self.config.get("encoder_epochs", 5)
        learning_rate = self.config.get("encoder_learning_rate", 0.001)
        
        # 优化器
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        
        # 按类别分组图片索引
        class_indices = {}
        for idx in self.client_indices:
            _, label = self.dst_train[idx]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        self.logger.info(f"客户端 {self.cid} 有 {len(class_indices)} 个类别，开始训练 encoder...")
        
        for epoch in range(encoder_epochs):
            epoch_loss = 0.0
            total_samples = 0
            
            # 对每个类别进行训练
            for class_label, indices in class_indices.items():
                if len(indices) < 2:
                    continue  # 至少需要2张图片才能计算相似度
                
                # 使用所有图片，而不是只取前几个
                # 如果图片太多，可以分批处理
                batch_size = 32  # 设置合适的batch size
                num_batches = (len(indices) + batch_size - 1) // batch_size
                
                for batch_idx in range(num_batches):
                    # 获取当前批次的索引
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(indices))
                    batch_indices = indices[start_idx:end_idx]
                    
                    # 准备图片数据
                    images = []
                    for idx in batch_indices:
                        image_tensor, _ = self.dst_train[idx]
                        images.append(image_tensor)
                    
                    if len(images) < 2:
                        continue  # 如果批次中图片少于2张，跳过
                    
                    # 将图片堆叠成批次
                    images_batch = torch.stack(images).to(self.device)
                    
                    # 编码图片
                    encoded_images = self.encoder.encode(images_batch)
                    
                    # 计算同类图片编码的相似度损失
                    # 使用均方误差（MSE）作为相似度度量，使同类图片编码尽可能相似
                    # 计算所有编码之间的两两距离
                    if encoded_images.shape[0] > 1:
                        # 展平编码
                        encoded_flat = encoded_images.view(encoded_images.shape[0], -1)
                        
                        # 计算两两之间的欧氏距离
                        distances = torch.cdist(encoded_flat, encoded_flat, p=2)
                        
                        # 对角线设为0（自己与自己的距离）
                        mask = torch.eye(distances.shape[0], device=self.device).bool()
                        distances = distances[~mask].view(distances.shape[0], distances.shape[0] - 1)
                        
                        # 计算平均距离作为损失（使同类图片编码尽可能接近）
                        loss = distances.mean()
                        
                        # 反向传播
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item() * len(batch_indices)
                        total_samples += len(batch_indices)
            
            if total_samples > 0:
                avg_loss = epoch_loss / total_samples
                self.logger.info(f"客户端 {self.cid} Encoder 训练 Epoch {epoch+1}/{encoder_epochs}, 平均损失: {avg_loss:.4f}")
        
        # 训练完成后，设置 encoder 为评估模式
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.logger.info(f"客户端 {self.cid} encoder 训练完成")
        
        # 保存训练好的 encoder 参数
        self.trained_encoder_state = copy.deepcopy(self.encoder.state_dict())
        
        return
    
    def select_and_compress_images(self):
        """
        每轮都按照配置中的比例 r，从每个类别中选择图片并压缩
        """
        self.logger.info(f"客户端 {self.cid} 开始选择并压缩图片...")
        
        # 获取配置中的比例 r
        data_representation_ratio = self.config.get("data_representation_ratio", 0.1)
        self.logger.info(f"客户端 {self.cid} 数据代表比例 r: {data_representation_ratio}")
        
        # 按类别分组图片索引
        class_indices = {}
        for idx in self.client_indices:
            _, label = self.dst_train[idx]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # 选择并压缩图片
        compressed_images = []
        labels = []
        selected_indices = []
        
        for class_label, indices in class_indices.items():
            # 计算需要选择的图片数量
            num_to_select = max(1, int(len(indices) * data_representation_ratio))
            num_to_select = min(num_to_select, len(indices))
            
            # 随机选择图片
            import numpy as np
            selected_class_indices = np.random.choice(indices, num_to_select, replace=False)
            
            for idx in selected_class_indices:
                # 获取图片
                image_tensor, label = self.dst_train[idx]
                
                # 压缩图片
                compressed_image = self.compress_image(image_tensor)
                
                compressed_images.append(compressed_image)
                labels.append(label)
                selected_indices.append(idx)
        
        # 保存压缩结果
        self.compressed_images = torch.stack(compressed_images) if compressed_images else None
        self.compressed_labels = torch.tensor(labels) if labels else None
        self.compressed_indices = selected_indices
        
        self.logger.info(f"客户端 {self.cid} 选择并压缩了 {len(compressed_images)} 张图片")
        return
    
    def compress_image(self, image_tensor):
        """
        压缩单张图片
        """
        # 将图片移动到设备并添加批次维度
        image_batch = image_tensor.unsqueeze(0).to(self.device)
        
        # 使用 encoder 压缩图片
        with torch.no_grad():
            compressed = self.encoder.encode(image_batch)
        
        # 移除批次维度
        compressed = compressed.squeeze(0)
        
        return compressed
    
    def send_data_to_server(self):
        """
        将客户端数据发送到服务器
        新需求（实验演示版本）：
        1. 每轮都发送训练好的 encoder 参数
        2. 每轮都发送压缩图片
        3. 发送原始图片数据用于可视化
        """
        data_sent = {}
        data_sent["client_id"] = self.cid
        data_sent["current_round"] = self.current_round
        
        # 每轮都发送训练好的 encoder 参数
        if hasattr(self, 'trained_encoder_state'):
            data_sent["encoder_state"] = self.trained_encoder_state
            data_sent["num_samples"] = len(self.client_indices)
            self.logger.info(f"客户端 {self.cid} 准备发送训练好的 encoder 参数")
        else:
            self.logger.warning(f"客户端 {self.cid} 没有训练好的 encoder 参数可发送")
        
        # 每轮都发送压缩图片
        if hasattr(self, 'compressed_images') and self.compressed_images is not None:
            data_sent["compressed_images"] = self.compressed_images.cpu()
            data_sent["labels"] = self.compressed_labels.cpu()
            data_sent["client_indices"] = self.compressed_indices
            self.logger.info(f"客户端 {self.cid} 准备发送 {len(self.compressed_images)} 张压缩图片")
            
            # 发送原始图片数据用于可视化
            if hasattr(self, 'compressed_indices') and self.compressed_indices:
                original_images = []
                for idx in self.compressed_indices:
                    image_tensor, _ = self.dst_train[idx]
                    original_images.append(image_tensor)
                
                if original_images:
                    data_sent["original_images"] = torch.stack(original_images).cpu()
                    self.logger.info(f"客户端 {self.cid} 同时发送 {len(original_images)} 张原始图片用于可视化")
        else:
            self.logger.warning(f"客户端 {self.cid} 没有压缩图片可发送")
        
        return data_sent
