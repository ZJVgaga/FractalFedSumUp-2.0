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
        self.local_model =  copy.deepcopy(client_modules["global_model"])  # 全局模型
        
        # 联邦学习策略相关初始化
        self.fed_strategy = config.get("Federated_Learning_Config")  # 获取联邦学习策略
        self.logger.info(f"客户端 {i} 使用联邦策略: {self.fed_strategy}")
        
        # 如果采用的是FedSD2C策略，则进行特定参数的初始化
        if self.fed_strategy == "FedSD2C":
            self.num_crop = config.get("sd2c_num_crop")
            self.input_size = self.dataset_info["im_size"][0]
            self.images_per_class = int(config.get("images_per_class"))
            self.sd2c_iterations = config.get("sd2c_iterations")
            self.logger.info(f"客户端 {i} FedSD2C参数初始化完成")
        
        # 设备相关初始化
        self.device = client_modules['device']
        
        # 记录客户端初始化完成的日志
        self.logger.info(f"客户端 {i} 初始化完成")
        
    def receive_data_from_server(self,server_data):
        
        #在这里填写接收到的server_data怎么处理
         #可以用到self.config中的超参数
        return 

    def process(self):
        """
        客户端处理流程，包括模型预训练、核心集选择、数据扰动、VAE创建与使用、数据编码以及合成数据软标签生成。
        """
        
        # 预训练客户端模型，使用客户端原始数据
        self.logger.info(f"客户端 {self.cid} 开始预训练本地模型...")
        self.pretrain_client_model_use_client_original_data()
        self.logger.info(f"客户端 {self.cid} 本地模型预训练完成。")
        # 冻结模型的所有参数
        self.local_model.eval()
        for param in self.local_model.parameters():    
            param.requires_grad = False
        # 根据训练好的本地模型选择核心集索引
        self.logger.info(f"客户端 {self.cid} 开始基于训练好的本地模型选择核心集索引...")
        self.select_coreset_indices_based_on_trained_local_model(self.local_model, self.client_indices)
        self.logger.info(f"客户端 {self.cid} 核心集索引选择完成。")
        
        # 扰动核心集图像索引
        self.logger.info(f"客户端 {self.cid} 开始扰动核心集图像索引...")
        self.pertubed_core_set_images_dict = self.perturb_coreset_indices_images()
        self.logger.info(f"客户端 {self.cid} 核心集图像索引扰动完成。")
        
        # 创建一个VAE用于后续处理
        self.logger.info(f"客户端 {self.cid} 开始创建VAE...")
        from diffusers import AutoencoderKL
        # 使用本地模型路径而不是从Hugging Face下载
        local_vae_path = "FractalFedSumUp/run_batch_experiments/run_one_experiment/run_and_get_results/model/sdxl-vae"
        vae = AutoencoderKL.from_pretrained(local_vae_path, use_safetensors=True)
        for p in vae.parameters():
            p.requires_grad = False  # 冻结参数
        with torch.no_grad():
            vae.to(self.device)  # 将VAE移动到指定设备
        self.logger.info(f"客户端 {self.cid} VAE创建并配置完成。")
        
        # 使用本地模型和VAE对合成数据进行编码
        self.logger.info(f"客户端 {self.cid} 开始使用本地模型和VAE对合成数据进行编码...")
        self.encode_synthesis_data(self.local_model, vae)
        self.logger.info(f"客户端 {self.cid} 合成数据编码完成。")
        
        # 为合成潜在变量生成软标签
        self.logger.info(f"客户端 {self.cid} 开始为合成潜在变量生成软标签...")
        self.pertubed_core_set_soft_labels_dict = self.generate_soft_label_for_synthetic_latents(self.local_model, vae)
        self.logger.info(f"客户端 {self.cid} 合成潜在变量软标签生成完成。")
        
        return
    
    def send_data_to_server(self):
        """
        将客户端处理后的数据（如潜在变量张量和软标签）打包并准备发送到服务器。
        """
        
        # 创建一个字典用于存储要传输的数据
        data_sent = {}
        
        # 添加潜在变量张量到数据字典中
        self.logger.info(f"客户端 {self.cid} 开始准备潜在变量张量以发送到服务器...")
        data_sent["latent_variables_tensor"] = self.latent_variables_tensor
        
        # 添加软标签到数据字典中
        self.logger.info(f"客户端 {self.cid} 开始准备软标签以发送到服务器...")
        data_sent["soft_labels"] = self.soft_labels
        
        # 记录数据准备完成的信息
        self.logger.info(f"客户端 {self.cid} 数据准备完成，即将发送到服务器。")
        
        return data_sent
    
    def select_coreset_indices_based_on_trained_local_model(self, local_model, client_indices):
        """
        根据训练好的本地模型从客户端的数据中选择核心集索引。
        """
        from copy import deepcopy
        from collections import defaultdict
        import torch
        # 记录开始选择核心集索引的日志
        self.logger.info(f"客户端 {self.cid} 开始基于训练好的本地模型选择核心集索引...")
        
        # 复制传入的模型，避免修改原始模型的状态
        tool_model = deepcopy(local_model)
        self.logger.info(f"客户端 {self.cid} 已复制本地模型。")
        
        # 设置模型为评估模式，关闭dropout和batch normalization等训练时使用的层
        tool_model.eval()
        self.logger.info(f"客户端 {self.cid} 模型设置为评估模式。")
        
        # 禁用所有参数的梯度计算，因为此阶段不涉及训练
        for p in tool_model.parameters():
            p.requires_grad = False
        self.logger.info(f"客户端 {self.cid} 已禁用所有参数的梯度计算。")
        
        # 将模型置入device里面
        tool_model.to(self.device)
        self.logger.info(f"客户端 {self.cid} 模型已移动到指定设备。")
        
        # 深拷贝训练数据集，避免修改原始数据集
        tool_dataset = deepcopy(self.dst_train)
        self.logger.info(f"客户端 {self.cid} 已深拷贝训练数据集。")
        
        # 对client_indices按照类进行分组
        grouped_indices = defaultdict(list)
        for index in client_indices:
            target = tool_dataset.targets[index]
            if isinstance(target, int):    
                label = target
            else:    
                label = target.item()
            grouped_indices[label].append(index)
        
        # 对每组的indices排序
        for label in grouped_indices:
            grouped_indices[label].sort()
        
        # 初始化核心集和非核心集数据字典
        self.core_set_data = {
            "core_set_original_indices": [],
            "core_set_loss": [],
            "core_set_preds": [],
        }
        self.non_core_set_data = {
            "non_core_set_original_indices": [],
            "non_core_set_loss": [],
            "non_core_set_preds": [],
        }
        
        #
        # 对每一类进行核心集选择
        for label in grouped_indices:
            with torch.no_grad():  # 关闭梯度计算，节省内存
                mrc = MultiRandomCrop(self.num_crop, self.input_size, 1, 1)
                
                core_set_original_indices, core_set_loss, core_set_preds, \
                non_core_set_original_indices, non_core_set_loss, non_core_set_preds = select_coreset(
                    self.images_per_class,
                    tool_model,
                    tool_dataset,
                    grouped_indices[label],
                    self.input_size,
                    device=self.device,
                    mrc=mrc,
                    m=self.num_crop,
                    descending=False,
                    ret_all=True
                )
                
                # 扩展核心集数据
                self.core_set_data["core_set_original_indices"].extend(core_set_original_indices)
                self.core_set_data["core_set_loss"].extend([data.squeeze() for data in torch.split(core_set_loss.cpu(), 1)])
                self.core_set_data["core_set_preds"].extend([data.squeeze() for data in torch.split(core_set_preds.cpu(), 1)])

                # 扩展非核心集数据
                self.non_core_set_data["non_core_set_original_indices"].extend(non_core_set_original_indices)
                self.non_core_set_data["non_core_set_loss"].extend([data.squeeze() for data in torch.split(non_core_set_loss.cpu(), 1)])
                self.non_core_set_data["non_core_set_preds"].extend([data.squeeze() for data in torch.split(non_core_set_preds.cpu(), 1)])
        
        # 测试核心集与非核心集的选择是否正确
        all_indices = set(self.core_set_data["core_set_original_indices"] + self.non_core_set_data["non_core_set_original_indices"])
        assert all_indices == set(client_indices), "Test 1 Failed: The union of core and non-core indices does not match the given indices range."
        
        combined_indices = sorted(self.core_set_data["core_set_original_indices"] + self.non_core_set_data["non_core_set_original_indices"])
        given_indices_sorted = sorted(client_indices)
        assert combined_indices == given_indices_sorted, "Test 2 Failed: The combination of core and non-core original indices does not match the given indices."
        
        # 记录选择完成的日志
        self.logger.info(f"客户端 {self.cid} 核心集索引选择完成。")
        
        return

    def pretrain_client_model_use_client_original_data(self):
        """
        使用客户端原始数据预训练全局模型。
        """
        
        # 记录开始预训练本地模型的日志
        self.logger.info(f"客户端 {self.cid} 开始使用原始数据预训练本地模型...")
        
        # 创建DataLoader用于加载训练数据
        dataloader = DataLoader(
            self.dst_train,
            sampler=SubsetRandomSampler(self.client_indices),
            batch_size=self.config.get("train_batch_size"),
            shuffle=False,
            num_workers=0
        )
        
        # 设置模型为训练模式
        self.local_model.train()
        self.logger.info(f"客户端 {self.cid} 模型设置为训练模式。")
        
        # 定义优化器
        model_optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=float(self.config.get("learning_rate")),
            weight_decay=float(self.config.get("weight_decay")),
            momentum=float(self.config.get("momentum"))
        )
        self.logger.info(f"客户端 {self.cid} 优化器配置完成。")
        
        # 定义损失函数
        loss_function = torch.nn.CrossEntropyLoss()
        total_loss = 0
        
        # 开始训练循环
        for epoch in tqdm(range(self.model_epochs), desc='global model training', leave=True):
            for x, target in dataloader:
                x, target = x.to(self.device), target.to(self.device)  # 将数据移动到指定设备(cpu/gpu)
                target = target.long()  # 确保目标类型为long
                
                model_optimizer.zero_grad()  # 清除之前的梯度
                pred = self.local_model(x)  # 前向传播
                loss = loss_function(pred, target)  # 计算损失
                loss.backward()  # 反向传播
                model_optimizer.step()  # 更新权重
                
                total_loss += loss.item()  # 累加损失值
            
            # 在每个epoch结束时打印当前平均损失
            avg_loss = total_loss / (epoch + 1)
            self.logger.info(f"客户端 {self.cid} Epoch {epoch + 1}/{self.model_epochs}, 平均损失: {avg_loss}")
        
        # 记录每一轮的平均损失并打印
        final_avg_loss = total_loss / self.model_epochs
        print(f'每一轮的平均损失 = {final_avg_loss}')
        self.logger.info(f"客户端 {self.cid} 预训练完成，最终平均损失: {final_avg_loss}")
        
        return

    def perturb_coreset_indices_images(self):
        """
        对核心集中的每张图像进行扰动处理，并返回一个字典，其中包含核心集索引到扰动后图像的映射。
        """
        
        # 记录开始扰动核心集图像的日志
        self.logger.info(f"客户端 {self.cid} 开始对核心集图像进行扰动处理...")
        
        # 根据类别对non_core_set_data["non_core_set_original_indices"]进行分组
        non_core_set_by_class = {}
        
        # 创建一个字典，键是类别，值是该类别的非核心集图像索引列表
        for idx in self.non_core_set_data["non_core_set_original_indices"]:
            _, label = self.dst_train[idx]  # 假设第二个元素是标签
            if label not in non_core_set_by_class:
                non_core_set_by_class[label] = []
            non_core_set_by_class[label].append(idx)
        
        # 初始化一个字典用于存储扰动后的图像
        perturbed_images = {}
        
        # 遍历核心集中的每一张图像
        for i in self.core_set_data["core_set_original_indices"]:
            core_image, core_label = self.dst_train[i]
            
            # 从non-core-set中随机选择一张相同类别的图像
            if core_label in non_core_set_by_class and len(non_core_set_by_class[core_label]) > 0:
                selected_non_core_idx = np.random.choice(non_core_set_by_class[core_label])
                non_core_image, _ = self.dst_train[selected_non_core_idx]
            else:
                # 如果没有找到相同类别的图像，则使用均匀分布噪声生成一个替代图像
                non_core_image = torch.rand_like(core_image)  # 均匀分布噪声
            
            # 使用colorful_spectrum_mix进行扰动
            perturbed_img, _ = self.colorful_spectrum_mix(core_image, non_core_image, self.config.get("sd2c_foulier_alpha"))
            
            # 裁剪图像以确保其在合理范围内
            perturbed_img = self.clip(perturbed_img)
            
            # 将扰动后的图像存储到字典中
            perturbed_images[i] = perturbed_img
        
        # 记录扰动处理完成的日志
        self.logger.info(f"客户端 {self.cid} 核心集图像扰动处理完成。")
        
        return perturbed_images
        
        #对于每一张core_set_indices的图像(在self.dst_train取 self.core_set_data["core_set_original_indices"])
        #随机选择一张相同类别的non_core_set_original_indices的图像(在self.dst_train取 self.core_set_data["core_set_original_indices"])
        #用colorful_spectrum_mix扰动core_set_indices图像
        #clip图像
        #返回一个字典，coreset_indices对应的扰动后的图像

    def colorful_spectrum_mix(self, img1, img2, alpha, ratio=1.0):
        """
        对两张图像进行颜色频谱混合。
        输入图像大小: tensor of [C, H, W]
        """
        self.logger.info(f"开始颜色频谱混合操作...")
        
        from math import sqrt
        
        lam = alpha  # 混合系数
        
        # 调整图像维度顺序以适应numpy fft操作
        img1 = img1.permute(1, 2, 0)
        img2 = img2.permute(1, 2, 0)
        assert img1.shape == img2.shape, "输入图像尺寸不匹配"
        
        h, w, c = img1.shape  # 获取图像的高度、宽度和通道数
        h_crop = int(h * sqrt(ratio))  # 计算裁剪高度
        w_crop = int(w * sqrt(ratio))  # 计算裁剪宽度
        h_start = h // 2 - h_crop // 2  # 计算中心裁剪起点高度
        w_start = w // 2 - w_crop // 2  # 计算中心裁剪起点宽度
        
        # 执行快速傅里叶变换并分离幅度和相位信息
        img1_fft = np.fft.fft2(img1.numpy(), axes=(0, 1))
        img2_fft = np.fft.fft2(img2.numpy(), axes=(0, 1))
        img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
        img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

        # 将频谱移至中心
        img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
        img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

        # 创建副本用于后续处理
        img1_abs_ = np.copy(img1_abs)
        img2_abs_ = np.copy(img2_abs)
        
        # 应用混合系数lam进行频谱混合
        img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
        img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

        # 将频谱恢复原状
        img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
        img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

        # 重新组合幅度和相位信息，并执行逆FFT
        img21 = img1_abs * (np.e ** (1j * img1_pha))
        img12 = img2_abs * (np.e ** (1j * img2_pha))
        img21 = np.real(np.fft.ifft2(img21, axes=(0, 1))).transpose(2, 0, 1)
        img12 = np.real(np.fft.ifft2(img12, axes=(0, 1))).transpose(2, 0, 1)
        
        self.logger.info(f"颜色频谱混合操作完成。")
        
        return torch.from_numpy(img21), torch.from_numpy(img12)
        
    def clip(self, image_tensor, use_fp16=False, inplace=False):
        """
        根据给定的均值和标准差调整输入图像张量，并确保其值在合理范围内。
        
        参数:
        - image_tensor: 输入的图像张量，形状为 [C, H, W]。
        - use_fp16: 是否使用半精度浮点数进行计算，默认为False。
        - inplace: 是否直接在原张量上进行操作，默认为False。
        """
        
        self.logger.info(f"开始对图像张量进行裁剪调整...")
        
        # 获取数据集的均值和标准差信息
        mean, std = self.dataset_info["mean"], self.dataset_info["std"]
        
        # 根据use_fp16参数选择合适的数值类型
        if use_fp16:
            mean = np.array(mean, dtype=np.float16)
            std = np.array(std, dtype=np.float16)
            self.logger.info("使用半精度浮点数(fp16)进行计算。")
        else:
            mean = np.array(mean)
            std = np.array(std)
            self.logger.info("使用全精度浮点数(fp32)进行计算。")
        
        # 确保输入是PyTorch张量
        if isinstance(image_tensor, np.ndarray):
            image_tensor = torch.from_numpy(image_tensor)
            self.logger.info("将输入从NumPy数组转换为PyTorch张量。")
        
        # 如果不进行就地操作，则复制张量
        if not inplace:
            image_tensor = image_tensor.clone()
            self.logger.info("创建了输入张量的副本以避免就地修改。")
        
        # 对每个通道分别进行裁剪
        for c in range(3):  # 假设输入图像是三通道(RGB)
            m, s = mean[c], std[c]
            # 使用clamp函数限制每个通道的值范围
            image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
            self.logger.info(f"通道 {c} 已根据均值 {m} 和标准差 {s} 进行裁剪调整。")
        
        self.logger.info(f"图像张量裁剪调整完成。")
        
        return image_tensor
       

    def encode_synthesis_data(self, local_model, vae):
        """
        使用预训练的VAE对扰动后的核心集图像进行编码，并转换为张量形式。
        """
        self.logger.info(f"开始编码合成数据...")
        
        from copy import deepcopy
        
        # 初始化潜变量集 Z，其中 Z 包含通过预训练的 VAE 编码器 E 对每个 Xi_s 集合中的元素 xij 进行编码得到的结果
        self.initialize_latent_through_pretrained_vae(vae)
        self.transform_image_dict_to_tensor(vae)
        
        self.logger.info(f"合成数据编码完成。")

    def initialize_latent_through_pretrained_vae(self, vae):
        """
        通过预训练的VAE编码器对每个扰动后的核心集图像进行编码。
        """
        self.logger.info(f"初始化通过预训练VAE获得的潜变量...")
        
        self.latent_variable_set_Z = {}
        for i in self.core_set_data["core_set_original_indices"]:
            x = denormalize(self.pertubed_core_set_images_dict[i].unsqueeze(0), self.dataset_info).to(self.device)
            x = x.to(torch.float32)  # 确保输入是 float32
            z = vae.encode(x).latent_dist.mode().clone().detach()
            self.latent_variable_set_Z[i] = z
            
        self.logger.info(f"潜变量初始化完成。")
                
                    
    def transform_image_dict_to_tensor(self, vae):
        """
        将扰动后的图像字典和对应的潜变量转换为张量格式。
        """
        self.logger.info(f"将图像字典转换为张量...")
        
        # 初始化列表以保存提取的数据
        perturbed_images_list = []
        latent_variables_list = []

        # 遍历每个索引 i 并从字典中提取对应的数据
        for i in self.core_set_data["core_set_original_indices"]:
            perturbed_image = self.pertubed_core_set_images_dict[i]
            latent_variable = self.latent_variable_set_Z[i]

            perturbed_images_list.append(perturbed_image)
            latent_variables_list.append(latent_variable)

        self.perturbed_images_tensor = torch.stack(perturbed_images_list)  # 形状: (num_indices, ...)
        self.latent_variables_tensor = torch.stack(latent_variables_list)  # 形状: (num_indices, ...)
        self.latent_variables_tensor = self.latent_variables_tensor.squeeze(1)  # 移除第2维
        
        self.logger.info(f"图像字典转换为张量完成。Perturbed Images Tensor Shape: {self.perturbed_images_tensor.shape}, Latent Variables Tensor Shape: {self.latent_variables_tensor.shape}")
        
        # 计算合成损失并优化潜在变量
        self.optimize_latent_variables(vae)
                
    def optimize_latent_variables(self, vae):
        """
        计算合成损失并对潜在变量进行优化。
        """
        self.logger.info(f"开始优化潜在变量...")
        
        def compute_synthetic_loss(latent_zi, original_data_xi, vae):
            #self.logger.info(f"计算合成损失...")
            decoded_data_zi = vae.decode(latent_zi.float()).sample.to(self.device)
            zi_feature = self.local_model.embed(decoded_data_zi.float().to(self.device))
            xi_feature = self.local_model.embed(original_data_xi.float().to(self.device)).detach()
            batch_loss = torch.sum((torch.mean(zi_feature, dim=0) - torch.mean(xi_feature, dim=0)) ** 2)
            return batch_loss
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(list(zip(self.latent_variables_tensor, self.perturbed_images_tensor)), batch_size=256, shuffle=False)
        current_index = 0
        for zi, xi_s in dataloader:
            if not zi.requires_grad:
                zi.requires_grad_(True)
            
            optimizer = torch.optim.Adam([zi], lr=0.1)
            for t in range(self.sd2c_iterations):  # 对于t = 1到T_syn
                optimizer.zero_grad()
                L_syn = compute_synthetic_loss(zi, xi_s, vae)
                L_syn.backward()
                optimizer.step()
            # 将优化后的 zi 写回到原始 latent_tensor 中
            with torch.no_grad():
                self.latent_variables_tensor[current_index:current_index + zi.size(0)] = zi.cpu()
                current_index += zi.size(0)

        self.logger.info(f"潜在变量优化完成。")
                

    def generate_soft_label_for_synthetic_latents(self, local_model, vae):
        """
        使用预训练的VAE解码潜在变量，并生成软标签。
        """
        self.logger.info(f"开始为合成潜变量生成软标签...")
        
        # 确保local_model和vae模型都在评估模式下
        local_model.eval()
        
        # 先把vae的latent张量解码成图像张量
        latent_decode_images_tensor = vae.decode(self.latent_variables_tensor.to(self.device)).sample.float().to(self.device)
        
        batch_size = 256
        soft_labels = []
        for i in range(0, len(latent_decode_images_tensor), batch_size):
            batch = latent_decode_images_tensor[i:i + batch_size]
            with torch.no_grad():
                logits = local_model(batch)
                soft_labels.append(torch.softmax(logits, dim=1))
                
        self.soft_labels = torch.cat(soft_labels, dim=0)
        self.logger.info(f"软标签生成完成。Soft Labels Shape: {self.soft_labels.shape}")

import torch
#下面的都是需要用到的工具函数
import torch.nn.functional as F

def select_coreset(n, model, tool_dataset, given_indices, size, device, mrc, m=5, descending=False, ret_all=False):
    """
    从给定的数据集中选择一个大小为n的核心集，并计算相应的损失和预测结果。
    
    参数:
    - n: 核心集的大小
    - model: 用于前向传播的模型
    - tool_dataset: 数据集对象
    - given_indices: 从中选择核心集的初始索引列表
    - size: 图像调整后的大小
    - device: 设备类型（如'cuda'或'cpu'）
    - mrc: 一个预处理函数，对图像进行预处理
    - m: 每个样本重复的次数，默认为5
    - descending: 布尔值，指示是否按降序排列损失，默认为False
    - ret_all: 布尔值，指示是否返回所有信息，默认为False
    
    返回:
    - 包含核心集原始索引、核心集损失、核心集预测、非核心集原始索引、非核心集损失、非核心集预测的元组
    """
    # 禁用梯度计算以节省内存和加速推理过程
    with torch.no_grad():
        # 根据给定的indices从tool_dataset中提取图像和标签
        images = torch.stack([tool_dataset[i][0] for i in given_indices])
        labels = torch.tensor([tool_dataset[i][1] for i in given_indices])
        images = mrc(images)  # 对图像应用预处理
        
        # 将图像张量移动到指定设备（如GPU）
        images = images.to(device)
        s = images.shape

        # 调整图像张量的维度顺序为 [m, mipc, 3, 224, 224] 并展平为 [mipc * m, 3, 224, 224]
        images = images.permute(1, 0, 2, 3, 4)
        images = images.reshape(s[0] * s[1], s[2], s[3], s[4])
        
        # 将标签重复 m 次以匹配调整后的图像张量的大小，并转换为长整型
        labels = labels.repeat(m).to(device).to(torch.int64)

        # 设置批量大小，用于分批次前向传播以适应显存限制
        batch_size = 64  # 根据显存调整此值
        
        # 使用模型对填充后的图像进行批量前向传播，获取预测结果
        preds = batched_forward(model, pad(images, size).to(device), batch_size)

        # 计算预测结果与真实标签之间的交叉熵损失，并获取每张图像的最大概率对应的类别
        dist = cross_entropy(preds, labels)
        preds = torch.argmax(preds, dim=1)

        # 将损失和预测结果重塑为 [m, mipc] 形状
        dist = dist.reshape(m, s[0])
        preds = preds.reshape(m, s[0])

        # 对每个类别选择损失最小的样本索引
        index = torch.argmin(dist, 0)
        dist = dist[index, torch.arange(s[0])]
        preds = preds[index, torch.arange(s[0])]

        # [mipc, 3, 224, 224]
        sa = images.shape
        images = images.reshape(m, s[0], sa[1], sa[2], sa[3])
        images = images[index, torch.arange(s[0])]

        # 根据损失排序，选择前n个作为核心集
        core_set_idx_tensor = torch.argsort(dist, descending=descending)[:n]
        core_set_idx = core_set_idx_tensor.tolist()  # 如果需要，将其转换为列表来比较
        non_core_set_idx = [i for i in range(len(given_indices)) if i not in core_set_idx]

        # 将索引列表转换为 long 类型的张量
        core_set_idx = torch.tensor(core_set_idx, dtype=torch.long)
        non_core_set_idx = torch.tensor(non_core_set_idx, dtype=torch.long)

        # 维护原始数据集的索引映射关系
        core_set_original_indices = [given_indices[idx] for idx in core_set_idx]
        non_core_set_original_indices = [given_indices[idx] for idx in non_core_set_idx]

        # 断言检查，确保数据完整性
        

        # 清理缓存以释放显存
        torch.cuda.empty_cache()

    return (core_set_original_indices,
            dist[core_set_idx].detach(),
            preds[core_set_idx].detach(),
            non_core_set_original_indices,
            dist[non_core_set_idx].detach(),
            preds[non_core_set_idx].detach())


def batched_forward(model, tensor, batch_size):
    """
    使用指定的批量大小对输入张量进行模型前向传播。
    
    参数:
        model: 要使用的PyTorch模型
        tensor: 输入的图像张量
        batch_size: 每批数据的大小
        
    返回:
        所有批次输出的拼接结果
    """
    total_samples = tensor.size(0)
    all_outputs = []
    model.eval()  # 设置模型为评估模式

    with torch.no_grad():  # 禁用梯度计算以节省内存
        for i in range(0, total_samples, batch_size):
            batch_data = tensor[i: min(i + batch_size, total_samples)]
            output = model(batch_data)
            all_outputs.append(output)

    final_output = torch.cat(all_outputs, dim=0)  # 将所有批次的输出拼接在一起
    return final_output

import torch.nn.functional as F

def cross_entropy(y_pre, y):
    """
    计算预测值与真实标签之间的交叉熵损失。
    
    参数:
        y_pre: 预测的概率分布
        y: 真实标签
        
    返回:
        每个样本的交叉熵损失
    """
    y_pre = F.softmax(y_pre, dim=1)  # 应用Softmax函数得到概率分布
    return (-torch.log(y_pre.gather(1, y.view(-1, 1))))[:, 0]  # 计算负对数似然

import torch.nn.functional as F

def pad(input_tensor, target_height, target_width=None):
    """
    对输入张量进行填充，使其达到目标的高度和宽度。
    
    参数:
        input_tensor: 输入的图像张量
        target_height: 目标高度
        target_width: 目标宽度（如果未提供，则使用target_height）
        
    返回:
        填充后的图像张量
    """
    if target_width is None:
        target_width = target_height
    vertical_padding = target_height - input_tensor.size(2)
    horizontal_padding = target_width - input_tensor.size(3)

    top_padding = vertical_padding // 2
    bottom_padding = vertical_padding - top_padding
    left_padding = horizontal_padding // 2
    right_padding = horizontal_padding - left_padding

    padded_tensor = F.pad(
        input_tensor, (left_padding, right_padding, top_padding, bottom_padding)
    )
    return padded_tensor


import torchvision.transforms as transforms

import torchvision.transforms as transforms

class MultiRandomCrop(torch.nn.Module):
    def __init__(self, num_crop=5, size=224, factor=2, stack_dim=0):
        """
        初始化多随机裁剪模块。
        
        参数:
            num_crop: 裁剪次数
            size: 输出尺寸
            factor: 缩放因子
            stack_dim: 在哪个维度上堆叠裁剪结果
        """
        super().__init__()
        self.num_crop = num_crop
        self.size = size
        self.factor = factor
        self.stack_dim = stack_dim

    def forward(self, image):
        """
        对输入图像进行多次随机裁剪。
        
        参数:
            image: 输入图像
            
        返回:
            多次裁剪结果的堆叠张量
        """
        cropper = transforms.RandomResizedCrop(
            self.size // self.factor,
            ratio=(1, 1),
            antialias=True,
        )
        patches = [cropper(image) for _ in range(self.num_crop)]
        return torch.stack(patches, self.stack_dim)

    def __repr__(self):
        detail = f"(num_crop={self.num_crop}, size={self.size})"
        return f"{self.__class__.__name__}{detail}"


import numpy as np

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

from typing import Tuple, Iterator

def batch(latent_vars: torch.Tensor, perturbed_images: torch.Tensor,
          batch_size: int = 32, shuffle: bool = True) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    生成批数据的生成器函数
    
    参数:
        latent_vars: 潜在变量张量 (N, latent_dim)
        perturbed_images: 扰动图像张量 (N, C, H, W)
        batch_size: 每批的大小
        shuffle: 是否打乱数据顺序
        
    返回:
        生成器，每次产生一个批次的 (latent_batch, images_batch)
    """
    num_samples = perturbed_images.size(0)
    
    if shuffle:
        indices = torch.randperm(num_samples)
        perturbed_images = perturbed_images[indices]
        latent_vars = latent_vars[indices]
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        yield (
            latent_vars[start_idx:end_idx], perturbed_images[start_idx:end_idx]
        )
