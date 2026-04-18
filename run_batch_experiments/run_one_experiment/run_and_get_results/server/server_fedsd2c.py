import copy
import os
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torch.utils.data.sampler import SubsetRandomSampler
from ..client.client_fedsd2c import denormalize


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
        
        # 调度相关
        self.join_ratio = config.get("join_ratio")  # 获取参与率
        
        # 数据相关
        # 保存dataset_info用于denormalize
        self.dataset_info = basic_modules['dataset_info']
        # 加载测试集并使用SubsetRandomSampler来获取特定索引的数据样本
        self.test_set = basic_modules['dst_test']
        # 使用更大的batch_size提高效率，默认为32
        test_batch_size =32
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
        
        # 使用从basic_modules传入的VAE
        self.vae = basic_modules['vae']
        self.logger.info("VAE模型已从basic_modules加载")
        
        # 联邦学习策略
        self.fed_strategy = config.get("Federated_Learning_Config")  # 获取联邦学习策略
        self.logger.info(f"服务器使用联邦策略: {self.fed_strategy}")
        
        # 将config保存为实例变量，以便后续访问
        self.config = config
        
        self.logger.info("服务器初始化完成")

    def arrange_server_data_to_client(self):
        server_data={}#server_data是一个字典
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

        # 遍历接收到的所有客户端数据
        for data_sent_i in received_data_list:
            # 提取潜在变量张量和软标签，并添加到列表中
            self.latent_variable_tensor_list.append(data_sent_i["latent_variables_tensor"])
            self.soft_labels_list.append(data_sent_i["soft_labels"])

        # 将列表中的张量拼接成一个大的张量
        merged_latent_variables = torch.cat(self.latent_variable_tensor_list, dim=0)
        merged_soft_labels = torch.cat(self.soft_labels_list, dim=0)

        # 构造最终的合并数据字典
        merged_data = {
            "latent_variables_tensor": merged_latent_variables,  # 形状: [total_samples, latent_dim]
            "soft_labels": merged_soft_labels                   # 形状: [total_samples, num_classes]
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
        # 使用从basic_modules传入的VAE
        vae = self.vae
    
        # 冻结VAE所有参数
        for p in vae.parameters():
            p.requires_grad = False
        vae.eval()  # 设置为评估模式
        vae.to(self.device)  # 移动到指定设备

        # 从merged_data提取数据
        latent_variables_tensor = merged_data["latent_variables_tensor"].to(self.device)
        target_soft_labels = merged_data["soft_labels"].to(self.device)
        
        # 用VAE解码潜在变量（无梯度计算）
        with torch.no_grad():
            synthetic_data_list = []
            batch_size = 256  # 和后面的 DataLoader 批次大小一致

            for i in range(0, latent_variables_tensor.size(0), batch_size):
                batch = latent_variables_tensor[i:i+batch_size]
                decoded_batch = vae.decode(batch).sample
                synthetic_data_list.append(decoded_batch)

            synthetic_data = torch.cat(synthetic_data_list, dim=0)
            
        # 打印合成数据及其目标软标签的形状
        self.logger.info(f"Synthetic_data shape: {synthetic_data.shape}")
        self.logger.info(f"Target soft labels shape: {target_soft_labels.shape}")

        # 创建TensorDataset和DataLoader来分批处理数据
        dataset = TensorDataset(synthetic_data, target_soft_labels)
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
                pred_soft_labels = torch.softmax(logits, dim=1)  # 模型预测的概率分布

                # 计算KL散度损失
                loss = torch.nn.functional.kl_div(
                    input=torch.log(pred_soft_labels + 1e-10),  # 避免log(0)
                    target=target,
                    reduction='batchmean',  # 按batch平均
                    log_target=False  # target是概率分布（不是log概率）
                )

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

