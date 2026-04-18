import copy
import os
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
      
        # 模型相关
        # 使用deepcopy确保全局模型的独立性，并将其移动到指定设备上
        self.global_model = copy.deepcopy(basic_modules['global_model']).to(basic_modules['device'])
        self.model_strategy = config.get("Model")  # 获取模型相关的策略
        self.logger.info(f"服务器模型初始化完成: {self.model_strategy}")
        self.sumup_iterations_vae=config.get("sumup_iterations_vae")
        self.sumup_iterations_img=config.get("sumup_iterations_img")
        # 调度相关
        self.join_ratio = config.get("join_ratio")  # 获取参与率
        self.vae = basic_modules['vae']
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
        
        with torch.no_grad():
            self.vae.to(self.device)  # 将VAE移动到指定设备
        self.vae.eval()
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
        random.seed(self.config.get("seed"))
        """
        合并从客户端接收到的数据。
        
        参数:
            received_data_list: 客户端发送的数据列表，每个元素都是一个字典，
                                包含"latent_variables_tensor"和"soft_labels"键。
                                
        返回:
            merged_data: 包含合并后的潜在变量张量和软标签的字典。
        """
        # 初始化存储列表
        self.latent_variable_tensor_list = []
        self.soft_labels_list = []
        self.decoded_image_list= []
        # 遍历接收到的所有客户端数据
        def compute_synthetic_loss(latent_zi, batch_mean_feature,vae):
            decoded_data_zi = self.vae.decode(latent_zi.float()).sample.to(self.device)
            zi_feature = self.temp_model.embed(decoded_data_zi.float().to(self.device))
            batch_loss = torch.sum((torch.mean(zi_feature, dim=0) - batch_mean_feature.to(self.device)) ** 2)
            self.logger.info(f"计算合成损失...，损失为{batch_loss}")
            return batch_loss
        # 遍历接收到的所有客户端数据
        for j,data_sent_i in enumerate(received_data_list):
            # 提取潜在变量张量和软标签，并添加到列表中
            self.soft_labels_list.append(data_sent_i["soft_labels"])
            latent_variables_tensor=data_sent_i["latent_variables_tensor"].to(self.device)
            batch_mean_feature_list = data_sent_i["batch_mean_feature_list"]  # List of tensor (dim: feature_dim)
            # 用于记录当前处理的位置
            if "local_model" in data_sent_i:
                self.temp_model = None
                self.temp_model=copy.deepcopy(data_sent_i["local_model"]).to(self.device)
                for param in self.temp_model.parameters():    
                    param.requires_grad = False
                self.temp_model.eval()
                self.logger.info(f"客户端{j}传输过来了模型，使用该模型作为优化合成数据的观察者")
            else:
                self.temp_model=copy.deepcopy(self.global_model).to(self.device)
                self.temp_model.eval()
                for param in self.temp_model.parameters():    
                    param.requires_grad = False
                self.logger.info(f"客户端{j}没有传输过来模型，使用全局模型作为优化合成数据的观察者")
            # 用于记录当前处理的位置
            dataloader = DataLoader(latent_variables_tensor, batch_size=256, shuffle=False)
            for i, zi in enumerate(dataloader):
                zi=zi.to(self.device)
                if not zi.requires_grad:
                    zi.requires_grad_(True)
                
                optimizer = torch.optim.Adam([zi], lr=0.1)
                self.logger.info(f"计算第{j}个客户端潜在空间的第{i}批次数据的合成损失...")
                for t in range(self.sumup_iterations_vae):  # 对于t = 1到T_syn
                    optimizer.zero_grad()
                    L_syn = compute_synthetic_loss(zi, batch_mean_feature_list[i].to(self.device),self.vae)
                    L_syn.backward()
                    optimizer.step()
                    # 将优化后的 zi 写回到原始 latent_tensor 中 
                with torch.no_grad():            
                    start_index = i * dataloader.batch_size 
                    end_index = start_index + zi.size(0)            
                    latent_variables_tensor[start_index:end_index] = zi.cpu()
            

            # 对优化后的潜在变量进行解码            
            with torch.no_grad():
                decoded_image = self.vae.decode(latent_variables_tensor.to(self.device).float()).sample
            
            # 对复原后的图像张量进行优化
            decoded_image.requires_grad = True
            optimizer_decoded = torch.optim.Adam([decoded_image], lr=0.001)
            num_optimization_steps =self.sumup_iterations_img
            batch_size=256
            image_dataloader = DataLoader(decoded_image, batch_size=batch_size, shuffle=False)
            for step in range(num_optimization_steps):
                for batch_idx, batch_image in enumerate(image_dataloader):
                    optimizer_decoded.zero_grad()
                    zi_feature = self.temp_model.embed(batch_image.float().to(self.device))
                    start_index = batch_idx * batch_size
                    end_index = start_index + batch_image.size(0)
                    target_mean = batch_mean_feature_list[batch_idx].to(self.device).detach()
                    loss_decoded = torch.sum((torch.mean(zi_feature, dim=0) - target_mean) ** 2)
                    loss_decoded.backward()
                    optimizer_decoded.step()
                    self.logger.info(f"Client {j}, Decoded image optimization step {step + 1}/{num_optimization_steps}, "
                                     f"Batch {batch_idx}, Loss: {loss_decoded.item()}")
            
            self.decoded_image_list.append(decoded_image)
        # 将列表中的张量拼接成一个大的张量

        merged_soft_labels = torch.cat(self.soft_labels_list, dim=0)
        merged_decoded_images = torch.cat(self.decoded_image_list, dim=0)
        # 构造最终的合并数据字典
        merged_data = {
            
            "soft_labels": merged_soft_labels,                   # 形状: [total_samples, num_classes]
            "decoded_images": merged_decoded_images,
        }
        self.logger.info("Data merging completed.")
        return merged_data
    
    def process(self, merged_data):
        random.seed(self.config.get("seed"))
        """
        使用合并的数据对全局模型进行训练。
        
        参数:
            merged_data: 包含合并后的潜在变量张量和软标签的字典。
            
        返回:
            avg_epoch_loss: 每个epoch的平均损失值，用于监控训练过程。
        """
        from torch.optim import Adam
        self.global_model.to(self.device)  # 确保全局模型位于正确的设备上
        
        # 初始化优化器，针对全局模型的所有可训练参数
        self.optimizer = Adam(self.global_model.parameters(), lr=0.001)  # 根据需要调整学习率
        # 加载并固定VAE（SDXL-VAE）
       
        vae = self.vae
    
        # 冻结VAE所有参数
        for p in vae.parameters():
            p.requires_grad = False
        vae.eval()  # 设置为评估模式
        vae.to(self.device)  # 移动到指定设备

        # 从merged_data提取数据
        target_soft_labels = merged_data["soft_labels"].to(self.device)
        decoded_images = merged_data["decoded_images"].to(self.device)

        # 打印合成数据及其目标软标签的形状
        self.logger.info(f"Decoded images shape: {decoded_images.shape}")
        self.logger.info(f"Target soft labels shape: {target_soft_labels.shape}")
        
        # 创建TensorDataset和DataLoader来分批处理数据
        dataset = TensorDataset(decoded_images, target_soft_labels)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        

        # 多个epoch训练循环
        num_epochs = self.config.get("train_model_epochs")
        for epoch in range(num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            self.global_model.train()  # 确保全局模型在训练模式

            epoch_loss = 0.0  # 记录每轮epoch的总损失

            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)

                logits = self.global_model(data)  # 注意这里应该使用'data'而不是'synthetic_data'
                
                                # 计算交叉熵损失
                hard_label = torch.argmax(target, dim=1)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(logits, hard_label)
                
                '''
                pred_soft_labels = torch.softmax(logits, dim=1)  # 模型预测的概率分布
                # 计算KL散度损失
                loss = torch.nn.functional.kl_div(
                    input=torch.log(pred_soft_labels + 1e-10),  # 避免log(0)
                    target=target,
                    reduction='batchmean',  # 按batch平均
                    log_target=False  # target是概率分布（不是log概率）
                )
                '''
                # 反向传播优化全局模型
                self.global_model.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数

                epoch_loss += loss.item()

                if batch_idx % 10 == 0:
                    self.logger.info(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}")

            # 打印每个epoch结束时的平均损失
            avg_epoch_loss = epoch_loss / len(dataloader)
            self.logger.info(f"Epoch {epoch + 1} completed, Average Loss: {avg_epoch_loss}")

        return avg_epoch_loss  # 返回最终的平均损失值用于监控训练过程
    

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

