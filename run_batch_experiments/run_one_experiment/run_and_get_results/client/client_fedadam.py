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
         # client_modules is a dictionary, config stores hyperparameters, can be extracted using config.get("param")
       # Initialize your server modules here, store server_modules["param"] as self.variable, also store config as self.config
        self.config = config
        self.logger = logger
        self.model_strategy = config.get("Model")  # Get model strategy
        self.cid = i
        self.device = torch.device(config.get("device"))
 
        # Get training-related parameters from config
        self.real_batch_size = config.get("train_batch_size")
        self.model_epochs = config.get("train_model_epochs")
 
        self.train_set = client_modules["dst_train"]
        self.test_set = client_modules["dst_test"]
        self.test_indices = client_modules["target_test_indices"]
        self.classes = client_modules["client_classes"][i]
        self.dataset_info = client_modules["dataset_info"]
        self.client_indices = client_modules['client_indices'][i]  # Data indices for the current client
        self.test_loader = DataLoader(self.test_set, sampler=SubsetRandomSampler(self.test_indices), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        self.global_model = None
        return
        
    def receive_data_from_server(self,server_data):
        
        # Specify here how to process the received server_data
         # Can use hyperparameters from self.config
        self.logger.info(f"Client {self.cid} is receiving server data...")
        
        # Receive global model
        if 'global_model' in server_data:
            self.global_model = copy.deepcopy(server_data['global_model'])
            self.global_model.eval()
            self.logger.info(f"Client {self.cid} has received the global model")
        
        
        self.logger.info(f"Client {self.cid} server data reception completed")

    def process(self):

        dataloader = DataLoader(self.train_set, sampler=SubsetRandomSampler(self.client_indices), batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
        self.global_model.train()
 
        
        model_optimizer = torch.optim.Adam(
                self.global_model.parameters(),
                lr=self.config.get("learning_rate"),
               
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
 
 
                loss.backward()
                model_optimizer.step()
                total_loss += loss.item()
 
        avg_loss = total_loss / (len(dataloader) * self.model_epochs)
        self.logger.info(f'Client {self.cid} epoch avg loss = {avg_loss}')
        
        return 

    def send_data_to_server(self):
        
        # Data to be transmitted is represented by a dictionary
        
        data_for_server = {
                
                'client_model': self.global_model,
            }
         # Can use hyperparameters from self.config
        
        return  data_for_server