import copy
import os
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


from torch.utils.data.sampler import SubsetRandomSampler


# For measuring communication overhead
class Server:
    # Initialization function
    def __init__(
        self,
        server_modules,config,logger,clients
    ):
       # server_modules is a dictionary, config stores hyperparameters, can be extracted using config.get("param")
       # Here, fill in the server modules you want to initialize, store server_modules["param"] as self.variable, also store config as self.config
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
        self.momentum = None
        self.momentum_beta1 = float(self.config.get(f"adam_momentum_beta1"))
        self.momentum_beta2 = float(self.config.get(f"adam_momentum_beta2"))
        self.momentum_epsilon = float(self.config.get(f"adam_momentum_epsilon"))
        self.momentum_learning_rate = float(self.config.get(f"adam_momentum_learning_rate"))
        self.round=0
      
        return
     
    # Select several clients for updating


    def arrange_server_data_to_client(self):
        server_data = {
            "global_model": self.global_model
        }# server_data is a dictionary
        # Can use hyperparameters from self.config

        # Here, fill in the data you want to prepare for each client

        return server_data
    

    def merge_data(self,received_data_list):
        merged_data={}

        # Here, fill in how you merge all client data
        # Can use hyperparameters from self.config
        client_models = [data["client_model"].state_dict() for data in received_data_list]        
        client_counts = [1] * len(client_models)
        total_num = sum(client_counts)
        state_dict = self.global_model.state_dict()
        
        if self.momentum is None:
                self.momentum,self.vector_momentum={},{}
                for key in state_dict.keys():
                    self.momentum[key]=torch.zeros_like(state_dict[key])
                    self.vector_momentum[key]=torch.zeros_like(state_dict[key])


        for key in state_dict.keys():
            updates = torch.stack([model[key] - state_dict[key] for model in client_models], 0).mean(0)
            self.momentum[key] = self.momentum_beta1 * self.momentum[key] + (1 - self.momentum_beta1) * updates
            self.vector_momentum[key] = self.momentum_beta2 * self.vector_momentum[key] + (1 - self.momentum_beta2) * updates.pow(2)
            momentum_hat = self.momentum[key] / (1 - self.momentum_beta1 ** (self.round+1))
            vector_momentum_hat = self.vector_momentum[key] / (1 - self.momentum_beta2 ** (self.round+1))

            state_dict[key] += self.momentum_learning_rate * momentum_hat / (vector_momentum_hat.sqrt() + self.momentum_epsilon)
        # Load the weighted average model parameters into the global model
        
        self.global_model.load_state_dict(state_dict)
        self.round+=1
        return merged_data
    


    def process(self,merged_data):
        
        # Here, fill in how you process the merged data
        # Can use hyperparameters from self.config

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