import copy
import io
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, SubsetRandomSampler

# Custom modules
from .DatasetNonIIDClass.DatasetNonIIDClass import PerLabelDatasetNonIID

# Opacus differential privacy related
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import copy
from opacus import PrivacyEngine

# Custom DP tools
from ..dp_tools import DPClipper


class Client:
    def __init__(self, client_modules, config, logger, i):
        self.config = config
        self.logger = logger
        self.cid = i
        self.device = th.device(config.get("device"))
        self.times = 0

        # Training parameters
        self.real_batch_size = config.get("train_batch_size")
        self.model_epochs = config.get("train_model_epochs")

        # Data related
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

        # Model related (initially empty)
        self.global_model = copy.deepcopy(client_modules["global_model"])
        self.global_model.to(self.device)
        self.global_model.train()
        self.optimizer = None
        self.dataloader = None
        self.privacy_engine = None
        
        # DP related
        self.dp_mechanism = config.get("dp_mechanism", "no_dp")
        self.use_dp = self.dp_mechanism != "no_dp"
        self.dp_clipper = None
        
        if self.use_dp:
            # Use custom DP tools
            self.dp_clipper = DPClipper(config, logger, self.device)
            self.logger.info(f"Client {self.cid} uses custom DP mechanism: {self.dp_mechanism}")
        else:
            self.logger.info(f"Client {self.cid} does not use DP mechanism")

    def receive_data_from_server(self, server_data):

        if 'global_model' in server_data:
            new_state_dict = server_data['global_model']  # OrderedDict, without Opacus wrapper

            # Load directly into the entire model
            self.global_model.load_state_dict(new_state_dict)
            self.global_model.train()

            self.logger.info(f"Client {self.cid} has received and updated global model parameters")

        self.logger.info(f"Client {self.cid} server data reception completed")
        
    def _setup_dataloader_and_optimizer(self):
        """
        Set up data loader and optimizer
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

        # Set up data loader and optimizer
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
                
                # Apply DP mechanism (if enabled)
                if self.dp_clipper is not None:
                    self.dp_clipper.apply_dp_to_gradients(
                        self.global_model, 
                        lr=self.config.get("learning_rate")
                    )
                
                self.optimizer.step()

                total_loss += loss.item()

                # 🔥 Promptly delete intermediate variables and clear cache
                del x, target, pred, loss
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        avg_loss = total_loss / (len(self.dataloader) * self.model_epochs)
        self.logger.info(f'Client {self.cid} epoch avg loss = {avg_loss}')
        
        # 🔥 Also clear cache once after the entire epoch ends
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Print privacy budget information
        if self.dp_clipper is not None:
            privacy_info = self.dp_clipper.get_privacy_budget()
            self.logger.info(f"Client {self.cid} | {privacy_info}")

        return
   
    def send_data_to_server(self):
        """
        Model parameters sent to the server
        """
        # Use the original model's state_dict
        original_state_dict = self.global_model.state_dict()

        data_for_server = {
            'client_model': original_state_dict,
        }
        return data_for_server