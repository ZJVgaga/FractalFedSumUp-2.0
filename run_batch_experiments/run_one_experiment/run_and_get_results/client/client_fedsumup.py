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
        
        新需求（根据用户描述）：
        1. 客户端不再训练 encoder，只调用 encoder 进行压缩
        2. 效用提取器V0（本质为有损压缩器）读取原始数据对(D0, L0)
        3. V0将原始数据压缩为融合效用与隐私的混合信息表示I0及其对应标签L0
        4. 客户端将混合信息对(I0, L0)上传至服务器
        """
        self.logger.info(f"客户端 {self.cid} 开始第 {self.current_round} 轮处理（新流程版本）")
        
        # 检查是否有编码器
        if not hasattr(self, 'encoder'):
            raise ValueError(f"客户端 {self.cid} 缺少编码器，无法进行压缩")
        
        # 不再训练 encoder，只调用 encoder 进行压缩
        self.logger.info(f"客户端 {self.cid} 开始压缩图片（第{self.current_round}轮）...")
        self.compress_images()
        
        self.logger.info(f"客户端 {self.cid} 处理流程完成")
        return
    
    def compress_images(self):
        """
        压缩图片（新需求版本）
        
        根据用户描述：
        1. 效用提取器V0（本质为有损压缩器）读取原始数据对(D0, L0)
        2. V0将原始数据压缩为融合效用与隐私的混合信息表示I0及其对应标签L0
        3. 其中I0中效用信息占比为u%，隐私信息占比为(1-u)%
        
        实现：
        1. 使用接收到的编码器V0压缩所有图片
        2. 保存压缩后的图片I0和对应的标签L0
        """
        self.logger.info(f"客户端 {self.cid} 开始压缩图片...")
        
        # 设置编码器为评估模式
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 获取配置中的效用信息占比u%（默认为50%）
        utility_ratio = self.config.get("utility_ratio", 0.5)
        self.logger.info(f"客户端 {self.cid} 效用信息占比: {utility_ratio*100:.1f}%，隐私信息占比: {(1-utility_ratio)*100:.1f}%")
        
        # 按类别分组图片索引
        class_indices = {}
        for idx in self.client_indices:
            _, label = self.dst_train[idx]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # 压缩所有图片
        compressed_images = []
        labels = []
        selected_indices = []
        
        for class_label, indices in class_indices.items():
            for idx in indices:
                # 获取原始图片数据对(D0, L0)
                image_tensor, label = self.dst_train[idx]
                
                # 使用编码器V0压缩图片，得到混合信息表示I0
                compressed_image = self.compress_image(image_tensor)
                
                compressed_images.append(compressed_image)
                labels.append(label)
                selected_indices.append(idx)
        
        # 保存压缩结果
        self.compressed_images = torch.stack(compressed_images) if compressed_images else None
        self.compressed_labels = torch.tensor(labels) if labels else None
        self.compressed_indices = selected_indices
        
        self.logger.info(f"客户端 {self.cid} 压缩了 {len(compressed_images)} 张图片")
        
        # 如果可视化系统可用，进行可视化
        if VISUALIZATION_SYSTEM_AVAILABLE:
            try:
                self._visualize_compressed_images(compressed_images, labels, selected_indices)
            except Exception as e:
                self.logger.warning(f"客户端可视化失败: {str(e)}")
        
        return
    
    def _visualize_compressed_images(self, compressed_images, labels, selected_indices):
        """
        可视化压缩图片
        
        参数:
        - compressed_images: 压缩后的图片列表
        - labels: 标签列表
        - selected_indices: 选择的索引列表
        """
        try:
            from ..visualization_system.image_tracker import ImageTracker
            from ..visualization_system.client_visualizer import ClientVisualizer
            
            # 创建ImageTracker
            # 从client_modules中获取train_indices和train_labels
            train_indices = self.client_modules.get('target_train_indices', [])
            train_labels = self.client_modules.get('target_train_labels', [])
            
            if len(train_indices) == 0 or len(train_labels) == 0:
                self.logger.warning("client_modules中没有找到target_train_indices或target_train_labels，使用虚拟数据")
                # 创建虚拟的train_indices和train_labels
                train_indices = list(range(len(self.client_indices)))
                train_labels = [self.dst_train[idx][1] for idx in self.client_indices]
            
            image_tracker = ImageTracker(
                self.config, 
                self.dataset_info, 
                train_indices, 
                train_labels
            )
            
            # 创建ClientVisualizer
            client_visualizer = ClientVisualizer(image_tracker, self.config)
            
            # 将压缩图片转换为tensor
            compressed_tensor = torch.stack(compressed_images) if isinstance(compressed_images, list) else compressed_images
            
            # 获取原始图片
            original_images = []
            for idx in selected_indices:
                image_tensor, _ = self.dst_train[idx]
                original_images.append(image_tensor)
            original_tensor = torch.stack(original_images)
            
            # 保存可视化结果（只保存图片，不生成PDF）
            client_visualizer.save_client_visualization(
                original_images=original_tensor,
                compressed_images=compressed_tensor,
                labels=torch.tensor(labels),
                client_indices=selected_indices,
                client_id=self.cid,
                round_num=self.current_round,
                dataset=self.dst_train
            )
            
            self.logger.info(f"客户端 {self.cid} 可视化完成，保存了 {len(selected_indices)} 张图片的可视化结果")
            
            # 注意：PDF生成现在在run_and_get_results.py中处理，不在客户端中处理
            
        except Exception as e:
            self.logger.warning(f"客户端可视化系统初始化失败: {str(e)}")
            import traceback
            self.logger.debug(f"详细错误信息: {traceback.format_exc()}")
    
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
        新需求（根据用户描述）：
        1. 发送压缩后的混合信息表示I0及其对应标签L0
        2. 不再发送训练好的encoder参数（因为客户端不再训练encoder）
        """
        data_sent = {}
        data_sent["client_id"] = self.cid
        data_sent["current_round"] = self.current_round
        
        # 发送压缩图片（混合信息表示I0）和标签L0
        if hasattr(self, 'compressed_images') and self.compressed_images is not None:
            data_sent["latent_variables"] = self.compressed_images.cpu()
            data_sent["labels"] = self.compressed_labels.cpu()
            data_sent["client_indices"] = self.compressed_indices
            self.logger.info(f"客户端 {self.cid} 准备发送 {len(self.compressed_images)} 张压缩图片（混合信息表示I0）和标签L0")
            
            # 发送原始图片数据用于可视化（可选）
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
    
    def get_visualization_data_paths(self):
        """
        获取客户端保存的可视化数据路径
        返回客户端保存的所有图片数据的路径列表
        """
        if not VISUALIZATION_SYSTEM_AVAILABLE:
            return []
        
        try:
            # 获取保存路径
            base_dir = self.config.get("visualization_save_dir", "./fedsumup_visualizations")
            dataset_name = self.config.get("Dataset", "CIFAR10")
            compressed_size = self.config.get("compressed_image_size", 24)
            
            # PDF数据目录
            pdf_data_dir = os.path.join(
                base_dir,
                f"fedsumup_visualization_{dataset_name}",
                f"compressed_{compressed_size}",
                "pdf_data"
            )
            
            # 检查目录是否存在
            if not os.path.exists(pdf_data_dir):
                return []
            
            # 收集该客户端的所有数据文件
            data_files = []
            for filename in os.listdir(pdf_data_dir):
                if filename.endswith(".pkl"):
                    data_files.append(os.path.join(pdf_data_dir, filename))
            
            return data_files
            
        except Exception as e:
            self.logger.warning(f"客户端 {self.cid} 获取可视化数据路径失败: {str(e)}")
            return []
