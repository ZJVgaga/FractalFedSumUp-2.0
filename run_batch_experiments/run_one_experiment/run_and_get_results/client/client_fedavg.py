import copy
import io
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, SubsetRandomSampler

# 自定义模块
from .DatasetNonIIDClass.DatasetNonIIDClass import PerLabelDatasetNonIID

# Opacus 差分隐私相关
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import copy
from opacus import PrivacyEngine

# 自定义DP工具
from ..dp_tools import DPClipper


class Client:
    def __init__(self, client_modules, config, logger, i):
        self.config = config
        self.logger = logger
        self.cid = i
        self.device = th.device(config.get("device"))
        self.times = 0

        # 训练参数
        self.real_batch_size = config.get("train_batch_size")
        self.model_epochs = config.get("train_model_epochs")

        # 数据相关
        self.train_set = client_modules["dst_train"]
        self.test_set = client_modules["dst_test"]
        self.test_indices = client_modules["target_test_indices"]
        self.classes = client_modules["client_classes"][i]
        self.dataset_info = client_modules["dataset_info"]
        self.client_indices = client_modules['client_indices'][i]
        self.test_loader = DataLoader(
            self.test_set,
            sampler=SubsetRandomSampler(self.test_indices),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        # 模型相关（初始为空）
        self.global_model = copy.deepcopy(client_modules["global_model"])
        self.global_model.to(self.device)
        self.global_model.train()
        self.optimizer = None
        self.dataloader = None
        self.privacy_engine = None
        
        # DP相关
        self.dp_mechanism = config.get("dp_mechanism", "no_dp")
        self.use_dp = self.dp_mechanism != "no_dp"
        self.dp_clipper = None
        
        if self.use_dp:
            # 使用自定义DP工具
            self.dp_clipper = DPClipper(config, logger, self.device)
            self.logger.info(f"客户端 {self.cid} 使用自定义DP机制: {self.dp_mechanism}")
        else:
            self.logger.info(f"客户端 {self.cid} 不使用DP机制")

    def receive_data_from_server(self, server_data):

        if 'global_model' in server_data:
            new_state_dict = server_data['global_model']  # OrderedDict，不含 Opacus 包装

            # 直接加载到整个模型
            self.global_model.load_state_dict(new_state_dict)
            self.global_model.train()

            self.logger.info(f"客户端 {self.cid} 已接收并更新全局模型参数")

        self.logger.info(f"客户端 {self.cid} 服务器数据接收完成")
        
    def _setup_dataloader_and_optimizer(self):
        """
        设置数据加载器和优化器
        """
        if self.dataloader is None or self.optimizer is None:
            self.dataloader = DataLoader(
                self.train_set,
                sampler=SubsetRandomSampler(self.client_indices),
                batch_size=self.real_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )

            self.optimizer = th.optim.SGD(
                self.global_model.parameters(),
                lr=self.config.get("learning_rate"),
                weight_decay=self.config.get("weight_decay"),
                momentum=self.config.get("momentum")
            )

    def process(self):
        self.times += 1

        # 设置数据加载器和优化器
        self._setup_dataloader_and_optimizer()

        self.global_model.train()

        total_loss = 0
        loss_function = th.nn.CrossEntropyLoss()

        for epoch in tqdm(range(self.model_epochs), desc='global model training', leave=True, position=0):
            for x, target in self.dataloader:
                x, target = x.to(self.device), target.to(self.device)
                target = target.long()

                self.optimizer.zero_grad()
                pred = self.global_model(x)
                loss = loss_function(pred, target)
                loss.backward()
                
                # 应用DP机制（如果启用）
                if self.dp_clipper is not None:
                    self.dp_clipper.apply_dp_to_gradients(
                        self.global_model, 
                        lr=self.config.get("learning_rate")
                    )
                
                self.optimizer.step()

                total_loss += loss.item()

                # 🔥 及时删除中间变量并清空缓存
                del x, target, pred, loss
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        avg_loss = total_loss / (len(self.dataloader) * self.model_epochs)
        self.logger.info(f'Client {self.cid} epoch avg loss = {avg_loss}')
        
        # 🔥 在整个 epoch 结束后也清空一次缓存
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 打印隐私预算信息
        if self.dp_clipper is not None:
            privacy_info = self.dp_clipper.get_privacy_budget()
            self.logger.info(f"Client {self.cid} | {privacy_info}")

        return
   
    def send_data_to_server(self):
        """
        发送给 server 的模型参数
        """
        # 使用原始模型的state_dict
        original_state_dict = self.global_model.state_dict()

        data_for_server = {
            'client_model': original_state_dict,
        }
        return data_for_server
