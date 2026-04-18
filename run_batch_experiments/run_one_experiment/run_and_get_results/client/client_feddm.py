import copy

import io
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import softmax
from run_one_experiment.run_and_get_results.network_tools import random_pertube_model
from .DatasetNonIIDClass.DatasetNonIIDClass import PerLabelDatasetNonIID
import time

class Client:
    def __init__(self, client_modules, config,logger, i):
        """
        客户端初始化
        参数：
        - client_modules: 客户端需要的模块字典
        - config: 配置字典
        - i: 客户端ID
        """
        self.logger=logger
        self.logger.info(f"正在初始化客户端 {i}...")
        
        # 客户端ID
        self.cid = i
        
        self.config=config
        # 数据相关
        self.train_set = PerLabelDatasetNonIID(
            client_modules['dst_train'],
            client_modules['client_indices'][i],
            client_modules['client_classes'][i],
            client_modules['dataset_info']['channel'],
            client_modules['device'],
            config.get("seed")
        )
        self.classes = client_modules['client_classes'][i]
        self.dataset_info = client_modules['dataset_info']
        self.test_set = client_modules['dst_test']
        self.test_loader = DataLoader(
            self.test_set,
            sampler=SubsetRandomSampler(client_modules['target_test_indices']),
            batch_size=1, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )
        self.logger.info(f"客户端 {i} 数据加载完成，包含类别: {self.classes}")
        
        # 模型相关
        self.model_strategy = config.get("Model")
        self.global_model = None
        
        # 联邦学习策略
        self.fed_strategy = config.get("Federated_Learning_Config")
        self.logger.info(f"客户端 {i} 使用联邦策略: {self.fed_strategy}")
        
        # 如果是FedDM策略，初始化相关参数
        if self.fed_strategy == "FedDM":
            self.num_per_class = config.get("images_per_class")
            self.model_noise = config.get("dm_model_noise")
            self.client_sumup_epochs = config.get("dm_client_sumup_epochs")
            self.real_batch_size = config.get("dm_client_sumup_batch_size")
            self.image_learning_rate = config.get("dm_image_learning_rate")
            self.batch_size = config.get("train_batch_size")
            self.model_epochs = config.get("train_model_epochs")
            self.logger.info(f"客户端 {i} FedDM参数初始化完成")
        
        # 设备相关
        self.device = client_modules['device']
        
        self.logger.info(f"客户端 {i} 初始化完成")
        return

    def receive_data_from_server(self, server_data):
        """
        接收来自服务器的数据
        参数：
        - server_data: 服务器发送的数据字典
        """
        self.logger.info(f"客户端 {self.cid} 正在接收服务器数据...")
        
        # 接收全局模型
        if 'global_model' in server_data:
            self.global_model = copy.deepcopy(server_data['global_model'])
            self.global_model.eval()
            self.logger.info(f"客户端 {self.cid} 已接收全局模型")
        
        
        self.logger.info(f"客户端 {self.cid} 服务器数据接收完成")

    def process(self):
        """
        客户端处理过程
        """
        self.logger.info(f"客户端 {self.cid} 开始处理...")
        
        # 如果是FedDM策略，执行数据蒸馏
        if self.fed_strategy == "FedDM":
            self.logger.info(f"客户端 {self.cid} 执行FedDM数据蒸馏...")
            self.final_synthetic_images, self.final_synthetic_labels = self.train()
            self.logger.info(f"客户端 {self.cid} 数据蒸馏完成")
        
        
        self.logger.info(f"客户端 {self.cid} 处理完成")

    def send_data_to_server(self):
        """
        发送数据到服务器
        返回：
        - data_sent: 要发送的数据字典
        """
        self.logger.info(f"客户端 {self.cid} 准备发送数据到服务器...")
        
        data_sent = {}
        
        # 如果是FedDM策略，发送合成数据
        if self.fed_strategy == "FedDM":
            data_sent['synthetic_images'] = self.final_synthetic_images
            data_sent['synthetic_labels'] = self.final_synthetic_labels
            self.logger.info(f"客户端 {self.cid} 发送合成数据: 图像形状 {self.final_synthetic_images.shape}, 标签形状 {self.final_synthetic_labels.shape}")
        
        
        self.logger.info(f"客户端 {self.cid} 数据准备完成")
        return data_sent
    
    def train(self):
        config=self.config

        # 初始化最终返回的合成图像和标签
        final_synthetic_images = None
        final_synthetic_labels = None
         
        #专有部分
        if self.fed_strategy=="FedDM":
            self.logger.info("以IPC模式训练")
            all_synthetic_images = []
            all_synthetic_labels = []
             # 对每个类别 c 计算损失
            for i, c in enumerate(self.classes):
                # 获取类别 c 的 real_batch_size 个真实图像
                real_images= self.train_set.get_all_class_c_images(c)
                if real_images is None or len(real_images) == 0:
                    self.logger.info(f"类别 {c} 没有找到真实图像，跳过...")
                    continue
                #初始化合成张量
                # 确保所有参数都是整数
                # 确保num_per_class是整数
                num_per_class_int = int(self.num_per_class)
                size_tuple = (
                    num_per_class_int,
                    int(self.dataset_info['channel']),
                    int(self.dataset_info['im_size'][0]),
                    int(self.dataset_info['im_size'][1]),
                )
                
                synthetic_images = torch.randn(
                    size_tuple,
                    device=self.device
                )
                synthetic_images = synthetic_images.to(dtype=torch.float)
                synthetic_images.requires_grad_(True)
                
                if config.get("dm_data_template_mode") == 'real':
                    self.logger.info('initialize synthetic data from random real images')
                    synthetic_images.data= self.train_set.get_images(c, num_per_class_int).detach().data
                
                labels = torch.full((num_per_class_int,), c, dtype=torch.long, device=self.device)
                synthetic_images.to(self.device)
                synthetic_images=synthetic_images.reshape(
                    (num_per_class_int, self.dataset_info['channel'], self.dataset_info['im_size'][0], self.dataset_info['im_size'][1]))
                synthetic_images = synthetic_images.clone().detach().requires_grad_(True).to(self.device)  # 确保仍然是叶子节点
                # 使用 SGD 优化器对 synthetic_images 进行优化，这个地方实际上是合成数据的模型而不是训练模型，合成数据的模型是多出来的模型，因此不用和其他算法的训练模型的参数保持一致                                                                   
                optimizer_image = torch.optim.SGD([synthetic_images], lr=self.image_learning_rate, momentum=0.5, weight_decay=0)
                # 清零优化器的梯度
                optimizer_image.zero_grad()
                # 对每个客户端进行 client_sumup_epochs 轮的数据凝练
                for epoch in range(self.client_sumup_epochs):
                # 从全局模型 global_model 中随机扰动生成一个样本模型 sample_model
                    sample_model=random_pertube_model(self.global_model, self.model_noise)
                # 将 sample_model 设置为评估模式
                    sample_model.eval()
                    real_images=self.train_set.get_images(c, self.real_batch_size)
                # 初始化损失为 0
                    loss = torch.tensor(0.0).to(self.device)
                   # 提取特征
                    with torch.no_grad():
                        real_feature = sample_model.embed(real_images.to(self.device)).detach()
                    synthetic_feature = sample_model.embed(synthetic_images)
                    # 获取真实图像和合成图像的预测 logits
                    real_logits = sample_model(real_images).detach()
                    synthetic_logits = sample_model(synthetic_images)

                    # 计算特征和 logits 的均值差异，并累加到总损失中
                    loss += torch.sum((torch.mean(real_feature, dim=0) - torch.mean(synthetic_feature, dim=0)) ** 2)
                    loss += torch.sum((torch.mean(real_logits, dim=0) - torch.mean(synthetic_logits, dim=0)) ** 2)

                # 更新合成图像 S_k
                    optimizer_image.zero_grad()# 清零优化器的梯度
                    loss.backward()# 反向传播计算梯度
                
                # 更新合成图像
                    optimizer_image.step()
                
                self.logger.info(f'client {self.cid}, data condensation {epoch}, total loss = {loss.item()}, avg loss = {loss.item() / len(self.classes)}')

                # 将合成图像保存到列表中
                all_synthetic_images.append(synthetic_images.detach())  
                all_synthetic_labels.append(labels)

            # 将所有类别的合成数据整合为一个完整张量
            if all_synthetic_images:
                final_synthetic_images = torch.cat(all_synthetic_images, dim=0).detach()  # 添加 .detach()
                final_synthetic_labels = torch.cat(all_synthetic_labels, dim=0)
            else:
                final_synthetic_images = None  # 如果没有合成数据，则返回 None
                final_synthetic_labels = None
                self.logger.info("client的合成数据为空！！")
            self.logger.info("所有类别的合成数据已成功整合！")

            # 打印最终整合后的张量形状
            if final_synthetic_images is not None:
                self.logger.info(f"final_synthetic_images 的形状: {final_synthetic_images.shape}")
            else:
                self.logger.info("final_synthetic_images 为空")

            if final_synthetic_labels is not None:
                self.logger.info(f"final_synthetic_labels 的形状: {final_synthetic_labels.shape}")
            else:
                self.logger.info("final_synthetic_labels 为空")

        
        

        return final_synthetic_images,final_synthetic_labels
