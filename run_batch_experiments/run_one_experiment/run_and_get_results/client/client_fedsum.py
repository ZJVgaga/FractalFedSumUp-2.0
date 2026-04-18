import copy
import io
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import softmax



# pyRAPL测量函数
try:
    from pyRAPL import Measurement
    PY_RAPL_AVAILABLE = True
except ImportError:
    PY_RAPL_AVAILABLE = False
    print("警告: pyRAPL库不可用，将使用备用测量方法")
    
def measure_with_pyrapl(func, *args, **kwargs):
        """
        使用Intel RAPL测量功耗和估算FLOPs（如果可用），否则使用备用方法
        
        参数：
        - func: 要测量的函数
        - *args, **kwargs: 函数的参数
        
        返回：
        - tuple: (函数结果, 估算的FLOPs)
        """
        # 如果pyRAPL不可用，使用备用方法
        if not PY_RAPL_AVAILABLE:
            import time
            # 使用时间作为代理测量
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # 基于执行时间估算FLOPs（假设每秒10^9 FLOPs）
            execution_time = end_time - start_time
            estimated_flops = execution_time * 1e9  # 假设每秒10^9 FLOPs
            
            return result, estimated_flops
        
        try:
            # 创建测量对象
            measurement = Measurement("function_measurement")
            measurement.begin()
            
            result = func(*args, **kwargs)
            
            measurement.end()
            
            # 获取能量消耗（焦耳）
            energy_joules = measurement.result.pkg[0] if hasattr(measurement.result, 'pkg') and measurement.result.pkg else 0
            
            # RAPL提供能量消耗，可以估算FLOPs（每焦耳约10^10 FLOPs）
            estimated_flops = energy_joules * 1e10
            
            return result, estimated_flops
        except Exception as e:
            print(f"pyRAPL测量失败: {e}，使用备用方法")
            import time
            # 使用时间作为代理测量
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # 基于执行时间估算FLOPs（假设每秒10^9 FLOPs）
            execution_time = end_time - start_time
            estimated_flops = execution_time * 1e9  # 假设每秒10^9 FLOPs
            
            return result, estimated_flops


class Client:
    def __init__(self, client_modules, config, logger, i):
        """
        客户端初始化函数
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
        self.logger.info(f"正在初始化客户端 {i}...")
        
        # 从配置中获取模型训练周期数
        self.model_epochs = config.get("train_model_epochs")
        
        # 设置客户端ID
        self.cid = i
        self.first_round = True
        
        # 数据相关初始化
        self.dst_train = client_modules['dst_train']  # 训练数据集
        self.client_indices = client_modules['client_indices'][i]  # 当前客户端的数据索引
        self.classes = client_modules['client_classes'][i]  # 当前客户端的类别信息
        self.dataset_info = client_modules['dataset_info']  # 数据集信息
        self.test_set = client_modules['dst_test']  # 测试数据集
        self.vae = client_modules['vae']
        
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
        
        # 联邦学习策略相关初始化
        self.fed_strategy = config.get("Federated_Learning_Config")  # 获取联邦学习策略
        self.logger.info(f"客户端 {i} 使用联邦策略: {self.fed_strategy}")
        
        # 设备相关初始化
        self.device = client_modules['device']
        self.local_model = copy.deepcopy(client_modules["global_model"]).to(self.device)  # 全局模型
        
        # 1. 获取允许的总预算
        self.total_budget = config.get("compute_budget_per_client_per_round")
        self.logger.info(f"客户端 {i} 允许的总预算: {self.total_budget:.2e} FLOPs")
        
        # 2. 测量单图片处理函数的实际开销
        self.logger.info(f"客户端 {i} 开始测量单图片处理函数的实际开销...")
        self.single_image_processing_cost = self.measure_single_image_processing_cost()
        self.logger.info(f"客户端 {i} 单图片处理成本测量完成: {self.single_image_processing_cost:.2e} FLOPs")
        
        # 3. 计算process最开始的函数的固定开销（prepare_and_select_coreset的成本）
        self.logger.info(f"客户端 {i} 开始计算process最开始的函数的固定开销...")
        self.fixed_preparation_cost = self.calculate_fixed_preparation_cost()
        self.logger.info(f"客户端 {i} 固定准备成本计算完成: {self.fixed_preparation_cost:.2e} FLOPs")
        
        # 4. 根据总预算和固定开销反推出每类图片数量（核心集大小）
        self.logger.info(f"客户端 {i} 开始根据总预算和固定开销反推出每类图片数量...")
        self.images_per_class = self.calculate_images_per_class_from_budget_and_fixed_cost()
        
        self.input_size = self.dataset_info["im_size"][0]
        # 为sumup_num_crop提供默认值，如果配置中没有则使用1
        self.num_crop = config.get("sumup_num_crop", 1)
        
        self.logger.info(f"客户端 {i} FedSumUp参数初始化完成: "
                        f"总预算={self.total_budget:.2e} FLOPs, "
                        f"固定准备成本={self.fixed_preparation_cost:.2e} FLOPs, "
                        f"单图片处理成本={self.single_image_processing_cost:.2e} FLOPs, "
                        f"每类图片数量={self.images_per_class}")
        
        # 记录客户端初始化完成的日志
        self.logger.info(f"客户端 {i} 初始化完成")
        


    def receive_data_from_server(self, server_data):
        self.logger.info(f"客户端 {self.cid} 正在接收服务器数据...")
        
        # 接收服务器发送的 encoder
        if 'encoder' in server_data:
            self.encoder = copy.deepcopy(server_data['encoder']).to(self.device)
            self.logger.info(f"客户端 {self.cid} 已接收服务器发送的 encoder")
        
        # 接收当前轮次信息
        if 'current_round' in server_data:
            self.current_round = server_data['current_round']
            self.logger.info(f"客户端 {self.cid} 当前轮次: {self.current_round}")
        
        # 接收总轮次信息
        if 'total_rounds' in server_data:
            self.total_rounds = server_data['total_rounds']
            self.logger.info(f"客户端 {self.cid} 总轮次: {self.total_rounds}")
        
        # 判断是否是最后一轮
        self.is_final_round = (self.current_round == self.total_rounds - 1) if hasattr(self, 'total_rounds') else False
        if self.is_final_round:
            self.logger.info(f"客户端 {self.cid} 这是最后一轮，将上传压缩图片")
        
        # 重置单图片处理结果列表
        self.single_image_results = []
        self.current_image_index = 0
        
        self.logger.info(f"客户端 {self.cid} 服务器数据接收完成")
        return 

    def process(self):
        """
        客户端处理流程：训练 encoder 使同类图片编码相似
        """
        self.logger.info(f"客户端 {self.cid} 开始处理流程...")
        
        # 检查是否有 encoder
        if not hasattr(self, 'encoder'):
            self.logger.error(f"客户端 {self.cid} 没有 encoder，无法进行训练")
            return
        
        # 如果不是最后一轮，训练 encoder
        if not self.is_final_round:
            self.logger.info(f"客户端 {self.cid} 开始训练 encoder（第{self.current_round}轮）...")
            self.train_encoder()
        else:
            # 最后一轮：选择图片并压缩上传
            self.logger.info(f"客户端 {self.cid} 最后一轮，选择图片并压缩上传...")
            self.select_and_compress_images()
        
        self.logger.info(f"客户端 {self.cid} 处理流程完成")
        return
    
    def train_encoder(self):
        """
        训练 encoder 使同类图片编码相似
        对每一个epoch，对每一个类，要求同类图片所encode出来的图片的差距尽可能小
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
                
                # 随机选择该类别的图片（批量处理）
                batch_size = min(16, len(indices))  # 小批量处理
                selected_indices = np.random.choice(indices, batch_size, replace=False)
                
                # 准备图片数据
                images = []
                for idx in selected_indices:
                    image_tensor, _ = self.dst_train[idx]
                    images.append(image_tensor)
                
                if len(images) < 2:
                    continue
                
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
                    
                    epoch_loss += loss.item() * len(indices)
                    total_samples += len(indices)
            
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
    
    def calculate_preparation_cost(self):
        """
        计算准备阶段的固定计算成本。
        
        返回：
        - float: 准备阶段的固定计算成本（FLOPs）
        """
        self.logger.info(f"客户端 {self.cid} 开始测量准备阶段的固定计算成本...")
        
        # 准备测试数据
        test_local_model = copy.deepcopy(self.local_model)
        test_vae = self.vae
        
        # 使用pyRAPL测量准备阶段开销
        def preparation_cost_wrapper():
            # 模拟准备阶段流程
            # 1. 模型准备
            test_local_model.eval()
            for param in test_local_model.parameters():    
                param.requires_grad = False
            
            # 2. VAE准备
            for p in test_vae.parameters():
                p.requires_grad = False
            test_vae.eval()
            test_vae.to(self.device)
            
            return test_vae
        
        # 测量计算开销
        _, estimated_flops = measure_with_pyrapl(preparation_cost_wrapper)
        
        self.logger.info(f"客户端 {self.cid} 准备阶段固定成本测量完成: {estimated_flops:.2e} FLOPs")
        
        return estimated_flops
    
    def process_single_core_image(self, image_index, vae):
        """
        处理核心集中的单张图片。
        
        参数：
        - image_index (int): 图片索引
        - vae: VAE模型
        
        返回：
        - dict: 处理结果，包含潜在变量、软标签和特征
        """
        self.logger.info(f"客户端 {self.cid} 开始处理核心集图片 {image_index}...")
        
        # 1. 获取图片
        image_tensor, label = self.dst_train[image_index]
        self.logger.info(f"客户端 {self.cid} 处理索引 {image_index} 的图片，标签: {label}")
        
        # 2. 处理单张图片
        # 2.1 扰动图片（如果需要）
        perturbed_image = self.process_single_image(image_tensor, vae)
        
        # 2.2 编码图片为潜在变量
        latent_variable = self.encode_single_image(perturbed_image, vae)
        
        # 2.3 生成软标签
        soft_label = self.generate_single_soft_label(image_tensor)
        
        # 2.4 提取特征
        feature = self.extract_single_feature(perturbed_image)
        
        # 3. 估算处理成本
        processing_cost = self.estimate_single_image_processing_cost()
        
        # 4. 准备返回结果
        result = {
            "latent_variable": latent_variable,
            "soft_label": soft_label,
            "feature": feature,
            "image_index": image_index,
            "processing_cost": processing_cost
        }
        
        self.logger.info(f"客户端 {self.cid} 核心集图片 {image_index} 处理完成，计算成本: {processing_cost:.2e} FLOPs")
        
        return result
    
    def process_single_image(self, image_tensor, vae):
        """
        处理单张图片，包括扰动和预处理。
        
        参数：
        - image_tensor (torch.Tensor): 输入图片，形状为 [C, H, W]
        - vae: VAE模型
        
        返回：
        - torch.Tensor: 处理后的图片
        """
        self.logger.info(f"客户端 {self.cid} 开始处理单张图片...")
        
        # 如果需要扰动，可以在这里添加扰动逻辑
        # 当前版本不进行扰动，直接返回原始图片
        perturbed_image = image_tensor
        
        # 确保图片在设备上
        perturbed_image = perturbed_image.to(self.device)
        
        self.logger.info(f"客户端 {self.cid} 单张图片处理完成，形状: {perturbed_image.shape}")
        
        return perturbed_image
    
    def encode_single_image(self, image_tensor, vae):
        """
        将单张图片编码为潜在变量。
        
        参数：
        - image_tensor (torch.Tensor): 输入图片，形状为 [C, H, W]
        - vae: VAE模型
        
        返回：
        - torch.Tensor: 潜在变量
        """
        self.logger.info(f"客户端 {self.cid} 开始编码单张图片...")
        
        # 反归一化图片
        x = denormalize(image_tensor.unsqueeze(0), self.dataset_info).to(self.device)
        x = x.to(torch.float32)  # 确保输入是 float32
        
        # 使用VAE编码
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mode().clone().detach()
        
        # 移除批次维度
        z = z.squeeze(0)
        
        self.logger.info(f"客户端 {self.cid} 单张图片编码完成，潜在变量形状: {z.shape}")
        
        return z
    
    def generate_single_soft_label(self, image_tensor):
        """
        为单张图片生成软标签。
        
        参数：
        - image_tensor (torch.Tensor): 输入图片，形状为 [C, H, W]
        
        返回：
        - torch.Tensor: 软标签
        """
        self.logger.info(f"客户端 {self.cid} 开始生成单张图片软标签...")
        
        # 获取图片标签（这里需要根据实际情况获取）
        # 假设我们有一个方法可以获取图片的标签
        # 这里使用简化版本，创建一个随机的软标签
        num_classes = self.dataset_info["num_classes"]
        
        # 创建随机的软标签（在实际应用中应该根据图片内容生成）
        soft_label = torch.randn(num_classes)
        soft_label = torch.softmax(soft_label, dim=0)
        
        self.logger.info(f"客户端 {self.cid} 单张图片软标签生成完成，形状: {soft_label.shape}")
        
        return soft_label
    
    def extract_single_feature(self, image_tensor):
        """
        从单张图片中提取特征。
        
        参数：
        - image_tensor (torch.Tensor): 输入图片，形状为 [C, H, W]
        
        返回：
        - torch.Tensor: 特征向量
        """
        self.logger.info(f"客户端 {self.cid} 开始提取单张图片特征...")
        
        # 使用本地模型提取特征
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).float().to(self.device)
            feature = self.local_model.embed(image_tensor).detach()
        
        # 移除批次维度
        feature = feature.squeeze(0)
        
        self.logger.info(f"客户端 {self.cid} 单张图片特征提取完成，特征形状: {feature.shape}")
        
        return feature
    
    def estimate_single_image_processing_cost(self):
        """
        估算单张图片处理的计算成本。
        
        返回：
        - float: 单张图片处理的估算计算成本（FLOPs）
        """
        # 直接返回之前测量的单图片处理成本
        return self.single_image_processing_cost
    
    def measure_single_image_processing_cost(self):
        """
        测量单图片处理函数的实际开销。
        
        返回：
        - float: 单张图片处理的实际计算成本（FLOPs）
        """
        self.logger.info(f"客户端 {self.cid} 开始测量单图片处理函数的实际开销...")
        
        # 准备测试图片
        test_image_index = self.client_indices[0]
        test_image_tensor, _ = self.dst_train[test_image_index]
        
        # 准备VAE
        vae = self.vae
        for p in vae.parameters():
            p.requires_grad = False
        vae.eval()
        vae.to(self.device)
        
        # 使用pyRAPL测量单图片处理开销
        def process_single_image_wrapper():
            # 模拟单图片处理流程
            perturbed_image = self.process_single_image(test_image_tensor, vae)
            latent_variable = self.encode_single_image(perturbed_image, vae)
            soft_label = self.generate_single_soft_label(test_image_tensor)
            feature = self.extract_single_feature(perturbed_image)
            return perturbed_image, latent_variable, soft_label, feature
        
        # 测量计算开销
        _, estimated_flops = measure_with_pyrapl(process_single_image_wrapper)
        
        self.logger.info(f"客户端 {self.cid} 单图片处理函数实际开销测量完成: {estimated_flops:.2e} FLOPs")
        
        return estimated_flops
    
    def calculate_fixed_preparation_cost(self):
        """
        计算process最开始的函数的固定开销（prepare_and_select_coreset的成本）。
        
        返回：
        - float: 固定准备成本（FLOPs）
        """
        # 使用pyRAPL测量prepare_and_select_coreset的实际开销
        self.logger.info(f"客户端 {self.cid} 开始测量prepare_and_select_coreset的实际开销...")
        
        # 保存原始状态
        original_local_model = self.local_model
        original_vae = self.vae
        
        # 使用pyRAPL测量实际prepare_and_select_coreset函数的开销
        def prepare_and_select_coreset_wrapper():
            # 调用实际的prepare_and_select_coreset函数
            return self.prepare_and_select_coreset()
        
        # 测量计算开销
        _, estimated_flops = measure_with_pyrapl(prepare_and_select_coreset_wrapper)
        
        self.logger.info(f"客户端 {self.cid} prepare_and_select_coreset实际开销测量完成: {estimated_flops:.2e} FLOPs")
        
        return estimated_flops
    
    def calculate_images_per_class_from_budget_and_fixed_cost(self):
        """
        根据总预算和固定开销计算每类可处理的图片数量。
        
        改进算法：
        1. 计算每个客户端的类别比例 r_c
        2. 根据预算计算可处理的总图片数量 N
        3. 按比例分配每类图片数量 N_c = round(N × r_c)
        
        返回：
        - dict: 每类可处理的图片数量 {class_label: num_images}
        """
        # 计算可用于处理图片的预算
        available_budget = self.total_budget - self.fixed_preparation_cost
        
        if available_budget <= 0:
            self.logger.warning(f"客户端 {self.cid} 可用预算不足: "
                              f"总预算={self.total_budget:.2e}, "
                              f"固定准备成本={self.fixed_preparation_cost:.2e}, "
                              f"可用预算={available_budget:.2e}")
            # 返回最小分配：每类至少1张图片
            return {label: 1 for label in self.classes}
        
        # 计算最大可处理总图片数量
        max_total_images = int(available_budget / self.single_image_processing_cost)
        
        # 确保至少可以处理1张图片
        max_total_images = max(1, max_total_images)
        
        # 获取配置中的原始每类图片数量
        original_images_per_class = self.config.get("images_per_class")
        
        # 计算理论上可处理的最大总图片数（基于原始每类图片数量）
        num_classes = len(self.classes)
        theoretical_max_images = original_images_per_class * num_classes
        
        # 实际可处理的总图片数量
        total_images_to_process = min(max_total_images, theoretical_max_images)
        
        # 计算每个类别的比例
        # 首先统计每个类别的样本数量
        class_counts = {}
        for idx in self.client_indices:
            _, label = self.dst_train[idx]
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # 计算每个类别的比例
        total_samples = len(self.client_indices)
        class_ratios = {}
        for label, count in class_counts.items():
            class_ratios[label] = count / total_samples
        
        # 按比例分配每类图片数量
        class_images = {}
        remaining_images = total_images_to_process
        
        # 第一轮分配：按比例分配，向下取整
        for label, ratio in class_ratios.items():
            allocated = int(total_images_to_process * ratio)
            class_images[label] = max(1, allocated)  # 每类至少1张
            remaining_images -= class_images[label]
        
        # 第二轮分配：将剩余的图片分配给比例最高的类别
        if remaining_images > 0:
            # 按比例排序，从高到低
            sorted_labels = sorted(class_ratios.items(), key=lambda x: x[1], reverse=True)
            for label, ratio in sorted_labels:
                if remaining_images <= 0:
                    break
                class_images[label] += 1
                remaining_images -= 1
        
        # 验证总图片数量
        total_allocated = sum(class_images.values())
        if total_allocated != total_images_to_process:
            self.logger.warning(f"客户端 {self.cid} 图片分配不一致: "
                              f"分配总数={total_allocated}, 目标总数={total_images_to_process}")
            # 调整以确保总数正确
            diff = total_images_to_process - total_allocated
            if diff > 0:
                # 增加比例最高的类别
                max_label = max(class_ratios.items(), key=lambda x: x[1])[0]
                class_images[max_label] += diff
            else:
                # 减少比例最低的类别
                min_label = min(class_ratios.items(), key=lambda x: x[1])[0]
                class_images[min_label] = max(1, class_images[min_label] + diff)
        
        self.logger.info(f"客户端 {self.cid} 预算分配计算完成: "
                        f"总预算={self.total_budget:.2e}, "
                        f"固定准备成本={self.fixed_preparation_cost:.2e}, "
                        f"可用预算={available_budget:.2e}, "
                        f"单图片成本={self.single_image_processing_cost:.2e}, "
                        f"最大可处理图片={max_total_images}, "
                        f"原始每类图片数量={original_images_per_class}, "
                        f"理论最大图片数={theoretical_max_images}, "
                        f"实际处理图片数={total_images_to_process}, "
                        f"类别分配={class_images}")
        
        # 保存分配结果供后续使用
        self.class_images_allocation = class_images
        self.total_images_to_process = total_images_to_process
        
        return class_images
    


    def select_and_compress_images(self):
        """
        最后一轮：按照配置中的比例 r，从每个类别中选择图片并压缩
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
        """
        data_sent = {}
        
        # 如果不是最后一轮，发送训练好的 encoder 参数
        if not self.is_final_round:
            if hasattr(self, 'trained_encoder_state'):
                data_sent["encoder_state"] = self.trained_encoder_state
                data_sent["client_id"] = self.cid
                data_sent["num_samples"] = len(self.client_indices)
                self.logger.info(f"客户端 {self.cid} 准备发送训练好的 encoder 参数")
            else:
                self.logger.warning(f"客户端 {self.cid} 没有训练好的 encoder 参数可发送")
        else:
            # 最后一轮：发送压缩图片
            if hasattr(self, 'compressed_images') and self.compressed_images is not None:
                data_sent["compressed_images"] = self.compressed_images
                data_sent["labels"] = self.compressed_labels
                data_sent["client_indices"] = self.compressed_indices
                data_sent["client_id"] = self.cid
                self.logger.info(f"客户端 {self.cid} 准备发送 {len(self.compressed_images)} 张压缩图片")
            else:
                self.logger.warning(f"客户端 {self.cid} 没有压缩图片可发送")
        
        return data_sent
    
    def select_random_indices_based_on_trained_local_model(self, local_model, client_indices):
        """
        按类别随机选择核心集索引，根据预算分配每类图片数量。
        
        改进算法：
        1. 使用calculate_images_per_class_from_budget_and_fixed_cost计算的每类图片数量
        2. 按比例从每个类别中随机选择图片
        3. 确保总图片数量不超过预算限制
        """
        self.logger.info(f"客户端 {self.cid} 开始按类别和预算分配选择核心集索引...")
        
        # 按类别对client_indices进行分组
        class_indices = {}
        for idx in client_indices:
            _, label = self.dst_train[idx]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # 获取每类图片数量分配
        if hasattr(self, 'class_images_allocation'):
            class_images_allocation = self.class_images_allocation
        else:
            # 如果没有分配，使用原始每类图片数量
            original_images_per_class = self.config.get("images_per_class")
            class_images_allocation = {label: original_images_per_class for label in class_indices.keys()}
        
        core_set_original_indices = []
        non_core_set_original_indices = []
        
        # 从每个类别中按分配数量随机选择图像作为核心集
        for label, indices in class_indices.items():
            # 获取该类别的分配数量
            num_to_select = class_images_allocation.get(label, 0)
            
            # 确保不超过该类别的可用样本数
            num_to_select = min(num_to_select, len(indices))
            
            if num_to_select > 0:
                selected_indices = np.random.choice(indices, num_to_select, replace=False).tolist()
                core_set_original_indices.extend(selected_indices)
                
                # 剩余样本作为非核心集
                non_selected_indices = [idx for idx in indices if idx not in selected_indices]
                non_core_set_original_indices.extend(non_selected_indices)
            else:
                # 如果分配数量为0，所有样本都作为非核心集
                non_core_set_original_indices.extend(indices)
        
        # 初始化核心集和非核心集数据字典
        self.core_set_data = {
            "core_set_original_indices": core_set_original_indices,
            "core_set_loss": [],
            "core_set_preds": [],
        }
        self.non_core_set_data = {
            "non_core_set_original_indices": non_core_set_original_indices,
            "non_core_set_loss": [],
            "non_core_set_preds": [],
        }
        
        # 记录选择结果
        total_selected = len(core_set_original_indices)
        if total_selected > 0:
            first_10_indices = core_set_original_indices[:min(10, total_selected)]
            self.logger.info(f"核心集的前{len(first_10_indices)}个索引为{first_10_indices}")
        
        # 验证选择是否正确
        all_indices = set(core_set_original_indices + non_core_set_original_indices)
        if all_indices != set(client_indices):
            self.logger.warning(f"客户端 {self.cid} 核心集与非核心集的并集与原始索引不匹配")
            self.logger.warning(f"原始索引数量: {len(client_indices)}, 并集数量: {len(all_indices)}")
        
        # 验证每类选择数量是否符合分配
        selected_by_class = {}
        for idx in core_set_original_indices:
            _, label = self.dst_train[idx]
            selected_by_class[label] = selected_by_class.get(label, 0) + 1
        
        for label, allocated in class_images_allocation.items():
            selected = selected_by_class.get(label, 0)
            if selected > allocated:
                self.logger.warning(f"客户端 {self.cid} 类别{label}选择了{selected}张图片，超过分配数量{allocated}")
        
        self.logger.info(f"客户端 {self.cid} 核心集选择完成: "
                        f"总选择图片数={total_selected}, "
                        f"类别分配={class_images_allocation}, "
                        f"实际选择={selected_by_class}")
        
        return


def denormalize(image_tensor, dataset_info, use_fp16=False, inplace=False):
    """
    将归一化的图像张量反归一化回原始输入。
    
    参数:
        image_tensor: 输入的归一化图像张量
        dataset_info: 包含均值和标准差的信息字典
        use_fp16: 是否使用半精度浮点数
        inplace: 是否在原地修改输入张量
        
    返回:
        反归一化后的图像张量
    """
    mean_list = dataset_info["mean"]
    std_list = dataset_info["std"]
    if use_fp16:
        mean = np.array(mean_list, dtype=np.float16)
        std = np.array(std_list, dtype=np.float16)
    else:
        mean = np.array(mean_list)
        std = np.array(std_list)

    if not inplace:
        image_tensor = image_tensor.clone()

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)
    return image_tensor
