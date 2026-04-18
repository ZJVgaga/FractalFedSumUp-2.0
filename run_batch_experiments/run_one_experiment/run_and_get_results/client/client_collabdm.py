import copy

import io
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import softmax

from .DatasetNonIIDClass.DatasetNonIIDClass import PerLabelDatasetNonIID
import time
from torchvision.models import ResNet

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
        self.first_round=True
        self.images_per_class=self.config.get("images_per_class")
        # 数据相关初始化
        self.dst_train = client_modules['dst_train']  # 训练数据集
        self.client_indices = client_modules['client_indices'][i]  # 当前客户端的数据索引
        self.classes = client_modules['client_classes'][i]  # 当前客户端的类别信息
        self.dataset_info = client_modules['dataset_info']  # 数据集信息
        self.test_set = client_modules['dst_test']  # 测试数据集
        # 构建每个类别的索引字典（只包含该客户端拥有的类）
        self.class_indices = {}

# 遍历当前客户端的所有样本索引，按类别组织成字典
        for idx in self.client_indices:
            _, label = self.dst_train[idx]  # 获取标签（假设返回格式是 (image, label)）
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
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
        
        # 如果采用的是Fedsumup策略，则进行特定参数的初始化

        # 设备相关初始化
        self.device = client_modules['device']
        self.local_model = copy.deepcopy(client_modules["global_model"]).to(self.device)  # 全局模型
        # 记录客户端初始化完成的日志
        self.logger.info(f"客户端 {i} 初始化完成")

    def receive_data_from_server(self,server_data):
        self.logger.info(f"客户端 {self.cid} 正在接收服务器数据...")
        if not self.first_round==True:
            if 'global_model' in server_data:
                self.local_model = copy.deepcopy(server_data['global_model'])
                self.local_model.eval()
                self.logger.info(f"客户端 {self.cid} 已接收全局模型")
        self.logger.info(f"客户端 {self.cid} 服务器数据接收完成")
        #在这里填写接收到的server_data怎么处理
         #可以用到self.config中的超参数
        return 

    def process(self):
       # 使用 distribution matching 方法生成本地合成数据
        self.client_syn = self.distribution_matching_idm(self.images_per_class, self.config)
          # 构建合成数据的标签，注意这里只包含客户端拥有的类别
        local_classes = list(self.class_indices.keys())  # 替换原来的 self.classes
       
        
        # 设置迭代次数（可以来自 config）
        iterations = self.config.get("collabdm_iterations")  # 默认 5 次

    # 在不计算梯度的情况下计算嵌入向量（多次迭代）
        with torch.no_grad():
            client_embedding = self.compute_embeddings(local_classes, iterations)

     # 将嵌入和合成数据发送到服务器
        self.client_embedding = client_embedding  # 现在是一个类 -> 多次迭代结果的字典
        return    
    
    def distribution_matching_idm(self, images_per_class, config):
        lr_img = 1.0
        batch_real = 256 
        iteration = self.config.get("collabdm_iterations")  # 默认 5 次

        # 获取本地拥有的类别列表
        local_classes = list(self.class_indices.keys())  # 替换原来的 self.classes
        
        # 按本地类别初始化合成图像和标签
        # 确保所有参数都是整数
        # 确保images_per_class是整数
        images_per_class_int = int(images_per_class)
        size_tuple = (
            int(len(local_classes) * images_per_class_int),
            int(self.dataset_info["channel"]),
            int(self.dataset_info["im_size"][0]),
            int(self.dataset_info["im_size"][1])
        )
        
        image_syn = torch.randn(
            size_tuple,
            device=self.device
        )
        image_syn = image_syn.to(dtype=torch.float)
        image_syn.requires_grad_(True)
        label_syn = torch.tensor(
            [np.ones(images_per_class_int) * cls for cls in local_classes],
            dtype=torch.long,
            requires_grad=False,
            device=self.device
        ).view(-1)

       # 初始化方式：real / noise
        if self.config.get("collabdm_mode") == 'real':
            print('initialize synthetic data from random real images')
            for idx, cls in enumerate(local_classes):
                img_real = self.get_images(cls, images_per_class_int)
                image_syn.data[idx * images_per_class_int:(idx + 1) * images_per_class_int] = img_real.detach().data
        else:
            print('initialize synthetic data from random noise')
        ''' training '''
       
        optimizer_img = torch.optim.SGD([image_syn],lr_img , momentum=0.5)
    
        optimizer_img.zero_grad()
        print(' training begins')

        net = self.local_model.to(self.device)


        # train syntheitc data
        for it in range(iteration + 1):

            loss_avg = 0

            for param in net.parameters():
                param.requires_grad = False

            for idx, cls in enumerate(local_classes):
                img_real = self.get_images(cls, batch_real).to(self.device)
                img_syn = image_syn[idx * images_per_class_int: (idx + 1) * images_per_class_int].reshape(
                    images_per_class_int, self.dataset_info["channel"], self.dataset_info["im_size"][0], self.dataset_info["im_size"][1]
                )

                with torch.no_grad():
                    output_real = net.embed(img_real).detach()

                output_syn = net.embed(img_syn)

                loss_c = torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)

                optimizer_img.zero_grad()
                loss_c.backward()
                optimizer_img.step()

                loss_avg += loss_c.item()

            loss_avg /= len(local_classes)

            if it % 100 == 0:
                print(f'iter = {it:04d}, loss = {loss_avg:.4f}')

    # 将合成图像按类组织成字典
        image_by_class = {
            cls: image_syn[i * images_per_class_int: (i + 1) * images_per_class_int].detach().cpu()
            for i, cls in enumerate(local_classes)
        }

        return image_by_class  # 返回的是一个按类组织的字典
    
    def get_images(self, cls, n):
        indices = np.random.choice(self.class_indices[cls], size=n, replace=len(self.class_indices[cls]) < n)
        images = torch.stack([self.dst_train[i][0] for i in indices]).to(self.device)
        return images
    
    def compute_embeddings(self, local_classes, iterations):
        """
        对每个类别运行多次网络初始化 + 嵌入提取，保存每次的 mean feature。
        返回格式：{ class: [embedding_0, embedding_1, ..., embedding_T] }
        """
        embeddings = {cls: [] for cls in local_classes}

        for it in range(iterations):
            print(f'Embedding iteration [{it+1}/{iterations}]')
    
        
            net = self.local_model
            net.train()
            for param in net.parameters():
                param.requires_grad = False

            embed_layer = net.module.embed if isinstance(net, torch.nn.DataParallel) else net.embed

        # 遍历本地类别，提取均值嵌入
            for cls in local_classes:
                img_real = self.get_images(cls, 256).to(self.device)  # 从 class_indices 取样
                with torch.no_grad():
                    feat = embed_layer(img_real).detach()

                mean_feat = torch.mean(feat, dim=0).cpu()  # 移动到 CPU
                embeddings[cls].append(mean_feat)

           
        torch.cuda.empty_cache()
    
        return embeddings
    

    def send_data_to_server(self):
        """
        将 process() 中生成的关键数据打包发送给服务器。
        包括：
        - 合成图像（按类组织）
        - 每个类的多轮 embedding 均值
        """
    # 创建一个字典用于存储要传输的数据
        data_sent = {}

    # 添加合成图像数据
        self.logger.info(f"客户端 {self.cid} 开始准备合成图像以发送到服务器...")
        data_sent["synthetic_images"] = self.client_syn  # {class: tensor}

    # 添加多轮嵌入数据
        self.logger.info(f"客户端 {self.cid} 开始准备多轮嵌入特征以发送到服务器...")
        data_sent["client_embeddings"] = self.client_embedding  # {class: [tensor, tensor, ...]}


    # 记录数据准备完成的信息
        self.logger.info(f"客户端 {self.cid} 数据准备完成，即将发送到服务器。")

        return data_sent
