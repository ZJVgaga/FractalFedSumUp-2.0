import copy
import os
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


from torch.utils.data.sampler import SubsetRandomSampler
# In fact

# Used to measure communication overhead
class Server:
    # Initialization function
    def __init__(self, basic_modules, config, logger, clients):
        random.seed(config.get("seed"))
        """
        Server initialization
        Parameters:
        - basic_modules: Basic modules dictionary
        - config: Configuration dictionary
        """
        self.clients = clients
        self.logger = logger
        self.logger.info("Initializing server...")
        self.config = config
        
        # Model related
        self.global_model = copy.deepcopy(basic_modules['global_model']).to(basic_modules['device'])
        self.model_strategy = config.get("Model")
        self.logger.info(f"Server model initialization completed: {self.model_strategy}")
        
        # Scheduling related
        
        self.join_ratio = config.get("join_ratio")
        
           
        # Data related
        self.test_set = basic_modules['dst_test']
        self.test_loader = DataLoader(
            self.test_set,
            sampler=SubsetRandomSampler(basic_modules['target_test_indices']),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        self.logger.info("Server test data loading completed")
        
        # Device related
        self.device = basic_modules['device']
        
        # Federated learning strategy
        self.fed_strategy = config.get("Federated_Learning_Config")
        self.logger.info(f"Server using federated strategy: {self.fed_strategy}")
        
        self.logger.info("Server initialization completed")

    def arrange_server_data_to_client(self):
        """
        Prepare data to send to clients
        Returns:
        - server_data: Dictionary of data to send
        """
        self.logger.info("Server preparing to send data to clients...")
        
        server_data = {}
        
        # Send global model
        server_data['global_model'] = self.global_model
        self.logger.info("Global model data prepared")
        
        
        
        self.logger.info("Server data preparation completed")
        return server_data

    def merge_data(self, received_data_list):
        """
        Merge data received from clients
        Parameters:
        - received_data_list: List of data received from clients
        Returns:
        - merged_data: Dictionary of merged data
        """
        self.logger.info("Server starting to merge client data...")
        
        merged_data = {}
        
        # If using FedDM strategy, merge synthetic data
        if self.fed_strategy == "FedDM":
            synthetic_images = []
            synthetic_labels = []
            
            for data in received_data_list:
                if 'synthetic_images' in data and 'synthetic_labels' in data:
                    synthetic_images.append(data['synthetic_images'])
                    synthetic_labels.append(data['synthetic_labels'])
            
            if synthetic_images:
                merged_data['synthetic_images'] = torch.cat(synthetic_images, dim=0).cpu()
                merged_data['synthetic_labels'] = torch.cat(synthetic_labels, dim=0)
                self.logger.info(f"Merged synthetic data: Image shape {merged_data['synthetic_images'].shape}, Label shape {merged_data['synthetic_labels'].shape}")
            else:
                self.logger.info("Warning: No valid synthetic data received")
        
        
        self.logger.info("Server data merging completed")
        return merged_data

    def process(self, merged_data):
        config = self.config
        random.seed(self.config.get("seed"))
        """
        Process merged data
        Parameters:
        - merged_data: Dictionary of merged data
        """
        self.logger.info("Server starting to process merged data...")
        
        # If using FedDM strategy, train global model with synthetic data
        if self.fed_strategy == "FedDM" and 'synthetic_images' in merged_data and 'synthetic_labels' in merged_data:
            self.logger.info("Training global model with synthetic data...")
            
            # Create dataset and data loader
            synthetic_dataset = TensorDataset(merged_data['synthetic_images'], merged_data['synthetic_labels'])
            batch_size = min(config.get("train_batch_size"), len(synthetic_dataset))
            synthetic_dataloader = DataLoader(
                synthetic_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
            
            # Training configuration
            self.global_model.train()
            model_optimizer = torch.optim.SGD(
                self.global_model.parameters(),
                lr=config.get("learning_rate"),
                weight_decay=config.get("weight_decay"),
                momentum=config.get("momentum")
            )
            loss_function = torch.nn.CrossEntropyLoss()
            total_loss = 0
            
            # Training process
            for epoch in range(config.get("train_model_epochs")):
                for x, target in synthetic_dataloader:
                    x, target = x.to(self.device), target.to(self.device)
                    model_optimizer.zero_grad()
                    pred = self.global_model(x)
                    loss = loss_function(pred, target)
                    loss.backward()
                    model_optimizer.step()
                    total_loss += loss.item()
                
                self.logger.info(f"Global model training epoch {epoch+1}/{config.get('train_model_epochs')}, Loss: {loss.item():.4f}")
            
            self.logger.info(f"Global model training completed, Average loss: {total_loss/config.get('train_model_epochs'):.4f}")
        
        self.logger.info("Server data processing completed")
    

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