import copy
import os
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


from torch.utils.data.sampler import SubsetRandomSampler
#事实上

#用于测量通讯开销的
class Server:
    #初始化函数
    def __init__(self, basic_modules, config, logger,clients):
        random.seed(config.get("seed"))
        """
        服务器初始化
        参数：
        - basic_modules: 基本模块字典
        - config: 配置字典
        """
        self.clients=clients
        self.logger = logger
        self.logger.info("正在初始化服务器...")
        self.config=config
        
        # 模型相关
        self.global_model = copy.deepcopy(basic_modules['global_model']).to(basic_modules['device'])
        self.model_strategy = config.get("Model")
        self.logger.info(f"服务器模型初始化完成: {self.model_strategy}")
        
        # 调度相关
        
        self.join_ratio = config.get("join_ratio")
        
           
        # 数据相关
        self.test_set = basic_modules['dst_test']
        self.test_loader = DataLoader(
            self.test_set,
            sampler=SubsetRandomSampler(basic_modules['target_test_indices']),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        self.logger.info("服务器测试数据加载完成")
        
        # 设备相关
        self.device = basic_modules['device']
        
        # 联邦学习策略
        self.fed_strategy = config.get("Federated_Learning_Config")
        self.logger.info(f"服务器使用联邦策略: {self.fed_strategy}")
        
        self.logger.info("服务器初始化完成")

    def arrange_server_data_to_client(self):
        """
        准备发送给客户端的数据
        返回：
        - server_data: 要发送的数据字典
        """
        self.logger.info("服务器准备发送数据给客户端...")
        
        server_data = {}
        
        # 发送全局模型
        server_data['global_model'] = self.global_model
        self.logger.info("已准备全局模型数据")
        
        
        
        self.logger.info("服务器数据准备完成")
        return server_data

    def merge_data(self, received_data_list):
        """
        合并从客户端接收的数据
        参数：
        - received_data_list: 从客户端接收的数据列表
        返回：
        - merged_data: 合并后的数据字典
        """
        self.logger.info("服务器开始合并客户端数据...")
        
        merged_data = {}
        
        # 如果是FedDM策略，合并合成数据
        if self.fed_strategy == "FedDM":
            synthetic_images = []
            synthetic_labels = []
            
            for data in received_data_list:
                if 'synthetic_images' in data and 'synthetic_labels' in data:
                    synthetic_images.append(data['synthetic_images'])
                    synthetic_labels.append(data['synthetic_labels'])
            
            if synthetic_images:
                merged_data['synthetic_images'] = torch.cat(synthetic_images, dim=0).cpu()
                merged_data['synthetic_labels'] = torch.cat(synthetic_labels, dim=0)
                self.logger.info(f"合并后合成数据: 图像形状 {merged_data['synthetic_images'].shape}, 标签形状 {merged_data['synthetic_labels'].shape}")
            else:
                self.logger.info("警告: 没有接收到有效的合成数据")
        
        
        self.logger.info("服务器数据合并完成")
        return merged_data

    def process(self, merged_data):
        config=self.config
        random.seed(self.config.get("seed"))
        """
        处理合并后的数据
        参数：
        - merged_data: 合并后的数据字典
        """
        self.logger.info("服务器开始处理合并数据...")
        
        # 如果是FedDM策略，使用合成数据训练全局模型
        if self.fed_strategy == "FedDM" and 'synthetic_images' in merged_data and 'synthetic_labels' in merged_data:
            self.logger.info("使用合成数据训练全局模型...")
            
            # 创建数据集和数据加载器
            synthetic_dataset = TensorDataset(merged_data['synthetic_images'], merged_data['synthetic_labels'])
            batch_size = min(config.get("train_batch_size"), len(synthetic_dataset))
            synthetic_dataloader = DataLoader(
                synthetic_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
            
            # 训练配置
            self.global_model.train()
            model_optimizer = torch.optim.SGD(
                self.global_model.parameters(),
                lr=config.get("learning_rate"),
                weight_decay=config.get("weight_decay"),
                momentum=config.get("momentum")
            )
            loss_function = torch.nn.CrossEntropyLoss()
            total_loss = 0
            
            # 训练过程
            for epoch in range(config.get("train_model_epochs")):
                for x, target in synthetic_dataloader:
                    x, target = x.to(self.device), target.to(self.device)
                    model_optimizer.zero_grad()
                    pred = self.global_model(x)
                    loss = loss_function(pred, target)
                    loss.backward()
                    model_optimizer.step()
                    total_loss += loss.item()
                
                self.logger.info(f"全局模型训练 epoch {epoch+1}/{config.get('train_model_epochs')}, 损失: {loss.item():.4f}")
            
            self.logger.info(f"全局模型训练完成，平均损失: {total_loss/config.get('train_model_epochs'):.4f}")
        
        self.logger.info("服务器数据处理完成")
    

    def select_clients(self):
        return (
            self.clients if self.join_ratio == 1.0
            else random.sample(self.clients, int(round(len(self.clients) * self.join_ratio)))
        )

    def evaluate(self):
        self.global_model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for x, target in self.test_loader:
                x, target = x.to(self.device), target.to(self.device, dtype=torch.int64)
                pred = self.global_model(x)
                _, pred_label = torch.max(pred.data, 1)
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()
        return correct / float(total)

