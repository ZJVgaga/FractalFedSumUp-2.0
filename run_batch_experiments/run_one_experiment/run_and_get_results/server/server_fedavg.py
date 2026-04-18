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
    #初始化函数
    def __init__(
        self,
        server_modules,config,logger,clients
    ):
       #server_modules是一个字典,config中存储了超参数，可以用config.get("param")提取
       #在这里填写你要初始化的server模块,把server_modules["param"]存为self.变量，也把config存为self.config
        self.config = config
        self.logger = logger
        self.device= server_modules["device"]
        self.global_model = server_modules["global_model"].to(self.config.get("device"))
        self.model_strategy = config.get("Model")
        self.fed_strategy = config.get("Federated_Learning_Config")
        #self.safety_strategy = config.get("Safety-Config")
        self.communication_rounds = config.get("communication_rounds")
        self.join_ratio = config.get("join_ratio")
        self.eval_gap = config.get("eval_gap")
        self.clients = clients
        self.test_set = server_modules["dst_test"]
        self.test_indices = server_modules["target_test_indices"]
        self.test_loader = DataLoader(self.test_set, sampler=SubsetRandomSampler(self.test_indices), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
      
      
        return
     
    #选择若干个客户端进行更新


    def arrange_server_data_to_client(self):
        server_data = {
            "global_model": self.global_model.state_dict()
        }#server_data是一个字典
        #可以用到self.config中的超参数

        #在这里填写你要准备给每个客户端的数据

        return server_data
    

    def merge_data(self,received_data_list):
        merged_data={}

        #在这里填写你合并的所有client的data的方式
        #可以用到self.config中的超参数
        client_models = [data["client_model"] for data in received_data_list]        
        client_counts = [1] * len(client_models)
        total_num = sum(client_counts)
        state_dict = self.global_model.state_dict()
 
        for key in state_dict.keys():
            state_dict[key] = sum([
                client_models[i][key] * (client_counts[i] / total_num)
                for i in range(len(client_models))
            ])
        self.global_model.load_state_dict(state_dict)
        
        return merged_data
    


    def process(self,merged_data):
        
        #在这里填写你对合并后的data的处理方式
        #可以用到self.config中的超参数

        return
    

    def select_clients(self):
        random.seed(self.config.get("seed"))
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

