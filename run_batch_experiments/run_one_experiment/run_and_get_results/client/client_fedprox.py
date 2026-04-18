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

class Client:
    def __init__(
            self,client_modules,config,logger,i,
        ):
         #client_modules是一个字典,config中存储了超参数，可以用config.get("param")提取
       #在这里填写你要初始化的server模块,把server_modules["param"]存为self.变量，也把config存为self.config
        self.config = config
        self.logger = logger
        self.model_strategy = config.get("Model")  # 获取模型策略
        self.cid = i
        self.device = torch.device(config.get("device"))
        
        # 从配置中获取训练相关参数
        self.real_batch_size = config.get("train_batch_size")
        self.model_epochs = config.get("train_model_epochs")
 
        self.train_set = client_modules["dst_train"]
        self.test_set = client_modules["dst_test"]
        self.test_indices = client_modules["target_test_indices"]
        self.classes = client_modules["client_classes"][i]
        self.dataset_info = client_modules["dataset_info"]
        self.client_indices = client_modules['client_indices'][i]  # 当前客户端的数据索引
        self.test_loader = DataLoader(self.test_set, sampler=SubsetRandomSampler(self.test_indices), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        self.global_model = None
        self.fedprox_mu = self.config.get(f"fedprox_mu")
        self.old_global_model=None
        return
        
    def receive_data_from_server(self,server_data):
        
        #在这里填写接收到的server_data怎么处理
         #可以用到self.config中的超参数
        self.logger.info(f"客户端 {self.cid} 正在接收服务器数据...")
        #FedProx专有部分
      
            #保存上一步的模型以进行对比
        if not self.global_model is None:
            self.old_global_model = copy.deepcopy(self.global_model)  # 保存旧模型
        else:
            self.old_global_model=None
        # 接收全局模型
        if 'global_model' in server_data:
            self.global_model = copy.deepcopy(server_data['global_model'])
            self.global_model.eval()
            self.logger.info(f"客户端 {self.cid} 已接收全局模型")
        
        
        self.logger.info(f"客户端 {self.cid} 服务器数据接收完成")

    def process(self):

        dataloader = DataLoader(self.train_set, sampler=SubsetRandomSampler(self.client_indices), batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
        self.global_model.train()
 
        
        model_optimizer = torch.optim.SGD(
                self.global_model.parameters(),
                lr=self.config.get("learning_rate"),
                weight_decay=self.config.get("weight_decay"),
                momentum=self.config.get("momentum")
            )
 
        loss_function = torch.nn.CrossEntropyLoss()
        total_loss = 0
 
        for epoch in tqdm(range(self.model_epochs), desc='global model training', leave=True):
            for x, target in dataloader:
                x, target = x.to(self.device), target.to(self.device)
                target = target.long()
 
                model_optimizer.zero_grad()
                pred = self.global_model(x)
                loss = loss_function(pred, target)
 
                if self.fedprox_mu > 0:
                    proximal_term = 0.0
                    if not self.old_global_model is None:
                        for w, w_t in zip(self.global_model.parameters(), self.old_global_model.parameters()):
                            proximal_term += (w - w_t).norm(2)**2
                        loss += (self.fedprox_mu / 2) * proximal_term


                loss.backward()
                model_optimizer.step()
                total_loss += loss.item()
 
        avg_loss = total_loss / (len(dataloader) * self.model_epochs)
        self.logger.info(f'Client {self.cid} epoch avg loss = {avg_loss}')
        
        return 

    def send_data_to_server(self):
        
        #要传输的数据用字典表示
        
        data_for_server = {
                
                'client_model': self.global_model,
            }
         #可以用到self.config中的超参数
        
        return  data_for_server