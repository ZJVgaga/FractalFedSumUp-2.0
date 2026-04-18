import copy
import os
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler


#用于测量通讯开销的
class Server:
    def __init__(self, basic_modules, config, logger, clients):
        """
        初始化服务器。
        
        参数:
        - basic_modules: 包含基本模块的字典，例如全局模型、设备等信息。
        - config: 配置字典，包含服务器运行所需的配置参数。
        - logger: 日志记录器，用于记录服务器运行过程中的日志信息。
        - clients: 客户端列表或相关信息。
        """
        self.clients = clients  # 存储所有客户端的信息
        self.logger = logger  # 设置日志记录器
        
        self.logger.info("正在初始化服务器...")
        import torch.nn as nn

        self.criterion = nn.CrossEntropyLoss()
        # 模型相关
        # 使用deepcopy确保全局模型的独立性，并将其移动到指定设备上
        self.global_model = copy.deepcopy(basic_modules['global_model']).to(basic_modules['device'])
        self.model_strategy = config.get("Model")  # 获取模型相关的策略
        self.logger.info(f"服务器模型初始化完成: {self.model_strategy}")
       
        # 调度相关
        self.join_ratio = config.get("join_ratio")  # 获取参与率
        from diffusers import AutoencoderKL
        # 数据相关
        # 加载测试集并使用SubsetRandomSampler来获取特定索引的数据样本
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
        self.first_round=True
        # 设备相关
        self.device = basic_modules['device']  # 确定使用的设备（如CPU或GPU）
        
      
        # 联邦学习策略
        self.fed_strategy = config.get("Federated_Learning_Config")  # 获取联邦学习策略
        self.logger.info(f"服务器使用联邦策略: {self.fed_strategy}")
        
        # 将config保存为实例变量，以便后续访问
        self.config = config
        
        self.logger.info("服务器初始化完成")

    def arrange_server_data_to_client(self):
        if self.first_round==True:
            self.first_round=False
            server_data={}
            return server_data
        else:
            server_data = {
            "global_model": self.global_model
        }#server_data是一个字典
        #可以用到self.config中的超参数

        #在这里填写你要准备给每个客户端的数据

            return server_data
    
    def merge_data(self, received_data_list):
        
        merged_data = {
            'synthetic_images': {},  # 每个类别的合成图像列表（来自不同客户端）
            'embeddings': {}         # 每个类别的嵌入特征列表（包含 client_id）
        }

        
        for client_idx, client_data in enumerate(received_data_list):
            if not client_data:
                continue

            # 处理合成图像
            synthetic_images = client_data["synthetic_images"]
            for cls, img_tensor in synthetic_images.items():
                if cls not in merged_data['synthetic_images']:
                    merged_data['synthetic_images'][cls] = []
                merged_data['synthetic_images'][cls].append(img_tensor)

            # 处理嵌入特征
            client_embeddings = client_data["client_embeddings"]
            for cls, feat_list in client_embeddings.items():
                if cls not in merged_data['embeddings']:
                    merged_data['embeddings'][cls] = []
                
                merged_data['embeddings'][cls].append({
                        'mean': feat_list,
                        'client_id': client_idx
                })
        # 将每个类别的图像 list 合并为一个大 tensor
        for cls in merged_data['synthetic_images']:
            merged_data['synthetic_images'][cls] = torch.cat(merged_data['synthetic_images'][cls], dim=0)
        self.logger.info("Data merging completed.")
        return merged_data
      
    
    def process(self, merged_data):
        #对merged_data
        #获取iterations=self.config.get("collabdm_iterations")
        #对每一轮iterations中的t
            #对每一个类c
            #从merged_data['synthetic_images'][c]中随机选256张X
            #feature=self.global_model.embed(x)
            #然后计算L_original_data=merged_data['embeddings'][cls]中所有客户端feat的第t个元素的加权平均
            #然后计算loss=torch.sum((L_original_data - torch.mean(feature, dim=0)) ** 2)
            #用这个loss优化merged_data['synthetic_images'][c]

        iterations = self.config.get("collabdm_iterations")  # 默认 5 次
        batch_size = 256 # 每次采样 256 张图用于计算目标 embed
        device = self.device

        # 提取模型 embed 层
        model = self.global_model
      

        # 只保留需要优化的类
        all_classes = list(merged_data['synthetic_images'].keys())
        self.logger.info(f"开始对类别 {all_classes} 进行合成图像优化")

        # 设置优化器
        syn_images_dict = {}
        optimizer_dict = {}

        for cls in all_classes:
            # 复制原始合成图像作为可学习参数
            syn_images = merged_data['synthetic_images'][cls].clone().detach().to(device).requires_grad_(True)
            syn_images_dict[cls] = syn_images
            optimizer_dict[cls] = torch.optim.Adam([syn_images], lr=1)

        # 开始多轮优化
        for t in range(iterations):
            self.logger.info(f"优化轮次 [{t+1}/{iterations}]")

            for cls in all_classes:
                # Step 1: 从合成图像中随机选取 batch_size 张图像
                indices = torch.randperm(syn_images_dict[cls].shape[0])[:batch_size]
                x = syn_images_dict[cls][indices].to(device)

                # Step 2: 前向传播得到特征
                
                feature = model.embed(x)

                # Step 3: 获取真实数据在第 t 轮的平均 embed（所有客户端的平均）
                real_feats_t = []
                for item in merged_data['embeddings'][cls]:
                    real_feats_t.append(item['mean'][t])  # 假设 item['mean'] 是 Tensor 或者可以转换为 Tensor 的形式

                real_feat_avg = torch.mean(torch.stack(real_feats_t), dim=0).to(device)

                # Step 4: 计算 loss``
                loss = torch.sum((real_feat_avg - torch.mean(feature, dim=0)) ** 2)

                # Step 5: 反向传播 & 优化
                optimizer = optimizer_dict[cls]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.logger.info(f"类别 {cls}: loss = {loss.item():.4f}")

        # 最终更新 merged_data 中的合成图像
        for cls in all_classes:
            merged_data['synthetic_images'][cls] = syn_images_dict[cls].cpu().detach()

        self.logger.info("合成图像优化完成")
        """
        使用 collabDM 合成图像训练全局模型，使用硬标签（cls）而不是 soft label
        """
        self.logger.info("开始使用 collabDM 合成图像进行全局模型训练")

        # Step 1: 构造训练数据集
        synthetic_data_list = []
        target_labels_list = []
  
        all_classes = list(merged_data['synthetic_images'].keys())

        for cls in all_classes:
            images = merged_data['synthetic_images'][cls].to(self.device)
            labels = torch.tensor([cls] * images.shape[0], dtype=torch.long, device=self.device)

            synthetic_data_list.append(images)
            target_labels_list.append(labels)

        synthetic_data = torch.cat(synthetic_data_list, dim=0)
        target_labels = torch.cat(target_labels_list, dim=0)

        # Step 2: 创建 DataLoader
        dataset = TensorDataset(synthetic_data, target_labels)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        # Step 3: 获取训练轮数
        num_epochs = self.config.get("train_model_epochs")
        for param in self.global_model.parameters():
            param.requires_grad = True
        self.global_model.to(self.device).train()
        # ✅ 添加优化器
        self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=self.config.get("learning_rate"))
        # Step 4: 训练循环
        total_loss = 0.0


        for epoch in range(num_epochs):
            self.logger.info(f"Epoch [{epoch + 1}/{num_epochs}] 开始")
            epoch_loss = 0.0

            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)

                logits = self.global_model(data)
                loss = self.criterion(logits, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                if batch_idx % 10 == 0:
                    self.logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / len(dataloader)
            total_loss += avg_epoch_loss
            self.logger.info(f"Epoch [{epoch + 1}/{num_epochs}] 完成，平均 Loss: {avg_epoch_loss:.4f}")

        avg_total_loss = total_loss / num_epochs
        self.logger.info(f"collabDM 数据训练完成，总平均 Loss: {avg_total_loss:.4f}")

        return avg_total_loss
    

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

