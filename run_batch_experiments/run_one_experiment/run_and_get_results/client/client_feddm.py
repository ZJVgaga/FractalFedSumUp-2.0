import copy

import io
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import softmax
from run_one_experiment.run_and_get_results.network_tools import random_pertube_model
from .DatasetNonIIDClass.DatasetNonIIDClass import PerLabelDatasetNonIID
import time

class Client:
    def __init__(self, client_modules, config,logger, i):
        """
        Client initialization
        Parameters:
        - client_modules: Dictionary of modules required by the client
        - config: Configuration dictionary
        - i: Client ID
        """
        self.logger=logger
        self.logger.info(f"Initializing client {i}...")
        
        # Client ID
        self.cid = i
        
        self.config=config
        # Data related
        self.train_set = PerLabelDatasetNonIID(
            client_modules['dst_train'],
            client_modules['client_indices'][i],
            client_modules['client_classes'][i],
            client_modules['dataset_info']['channel'],
            client_modules['device'],
            config.get("seed")
        )
        self.classes = client_modules['client_classes'][i]
        self.dataset_info = client_modules['dataset_info']
        self.test_set = client_modules['dst_test']
        self.test_loader = DataLoader(
            self.test_set,
            sampler=SubsetRandomSampler(client_modules['target_test_indices']),
            batch_size=1, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )
        self.logger.info(f"Client {i} data loading completed, contains classes: {self.classes}")
        
        # Model related
        self.model_strategy = config.get("Model")
        self.global_model = None
        
        # Federated learning strategy
        self.fed_strategy = config.get("Federated_Learning_Config")
        self.logger.info(f"Client {i} uses federated strategy: {self.fed_strategy}")
        
        # If it's the FedDM strategy, initialize related parameters
        if self.fed_strategy == "FedDM":
            self.num_per_class = config.get("images_per_class")
            self.model_noise = config.get("dm_model_noise")
            self.client_sumup_epochs = config.get("dm_client_sumup_epochs")
            self.real_batch_size = config.get("dm_client_sumup_batch_size")
            self.image_learning_rate = config.get("dm_image_learning_rate")
            self.batch_size = config.get("train_batch_size")
            self.model_epochs = config.get("train_model_epochs")
            self.logger.info(f"Client {i} FedDM parameter initialization completed")
        
        # Device related
        self.device = client_modules['device']
        
        self.logger.info(f"Client {i} initialization completed")
        return

    def receive_data_from_server(self, server_data):
        """
        Receive data from the server
        Parameters:
        - server_data: Data dictionary sent by the server
        """
        self.logger.info(f"Client {self.cid} is receiving server data...")
        
        # Receive the global model
        if 'global_model' in server_data:
            self.global_model = copy.deepcopy(server_data['global_model'])
            self.global_model.eval()
            self.logger.info(f"Client {self.cid} has received the global model")
        
        
        self.logger.info(f"Client {self.cid} server data reception completed")

    def process(self):
        """
        Client processing procedure
        """
        self.logger.info(f"Client {self.cid} starting processing...")
        
        # If it's the FedDM strategy, perform data distillation
        if self.fed_strategy == "FedDM":
            self.logger.info(f"Client {self.cid} executing FedDM data distillation...")
            self.final_synthetic_images, self.final_synthetic_labels = self.train()
            self.logger.info(f"Client {self.cid} data distillation completed")
        
        
        self.logger.info(f"Client {self.cid} processing completed")

    def send_data_to_server(self):
        """
        Send data to the server
        Returns:
        - data_sent: Dictionary of data to be sent
        """
        self.logger.info(f"Client {self.cid} preparing to send data to the server...")
        
        data_sent = {}
        
        # If it's the FedDM strategy, send synthetic data
        if self.fed_strategy == "FedDM":
            data_sent['synthetic_images'] = self.final_synthetic_images
            data_sent['synthetic_labels'] = self.final_synthetic_labels
            self.logger.info(f"Client {self.cid} sending synthetic data: image shape {self.final_synthetic_images.shape}, label shape {self.final_synthetic_labels.shape}")
        
        
        self.logger.info(f"Client {self.cid} data preparation completed")
        return data_sent
    
    def train(self):
        config=self.config

        # Initialize the final synthetic images and labels to return
        final_synthetic_images = None
        final_synthetic_labels = None
         
        # Exclusive part
        if self.fed_strategy=="FedDM":
            self.logger.info("Training in IPC mode")
            all_synthetic_images = []
            all_synthetic_labels = []
             # Calculate loss for each class c
            for i, c in enumerate(self.classes):
                # Get real_batch_size real images for class c
                real_images= self.train_set.get_all_class_c_images(c)
                if real_images is None or len(real_images) == 0:
                    self.logger.info(f"No real images found for class {c}, skipping...")
                    continue
                # Initialize synthetic tensor
                # Ensure all parameters are integers
                # Ensure num_per_class is an integer
                num_per_class_int = int(self.num_per_class)
                size_tuple = (
                    num_per_class_int,
                    int(self.dataset_info['channel']),
                    int(self.dataset_info['im_size'][0]),
                    int(self.dataset_info['im_size'][1]),
                )
                
                synthetic_images = torch.randn(
                    size_tuple,
                    device=self.device
                )
                synthetic_images = synthetic_images.to(dtype=torch.float)
                synthetic_images.requires_grad_(True)
                
                if config.get("dm_data_template_mode") == 'real':
                    self.logger.info('initialize synthetic data from random real images')
                    synthetic_images.data= self.train_set.get_images(c, num_per_class_int).detach().data
                
                labels = torch.full((num_per_class_int,), c, dtype=torch.long, device=self.device)
                synthetic_images.to(self.device)
                synthetic_images=synthetic_images.reshape(
                    (num_per_class_int, self.dataset_info['channel'], self.dataset_info['im_size'][0], self.dataset_info['im_size'][1]))
                synthetic_images = synthetic_images.clone().detach().requires_grad_(True).to(self.device)  # Ensure it remains a leaf node
                # Use SGD optimizer to optimize synthetic_images, this is actually the model for synthetic data rather than the training model. The synthetic data model is an extra model, so its parameters do not need to be consistent with the training model parameters of other algorithms.                                                                   
                optimizer_image = torch.optim.SGD([synthetic_images], lr=self.image_learning_rate, momentum=0.5, weight_decay=0)
                # Zero the optimizer gradients
                optimizer_image.zero_grad()
                # Perform client_sumup_epochs rounds of data condensation for each client
                for epoch in range(self.client_sumup_epochs):
                # Randomly perturb the global model to generate a sample model sample_model
                    sample_model=random_pertube_model(self.global_model, self.model_noise)
                # Set sample_model to evaluation mode
                    sample_model.eval()
                    real_images=self.train_set.get_images(c, self.real_batch_size)
                # Initialize loss to 0
                    loss = torch.tensor(0.0).to(self.device)
                   # Extract features
                    with torch.no_grad():
                        real_feature = sample_model.embed(real_images.to(self.device)).detach()
                    synthetic_feature = sample_model.embed(synthetic_images)
                    # Get prediction logits for real and synthetic images
                    real_logits = sample_model(real_images).detach()
                    synthetic_logits = sample_model(synthetic_images)

                    # Calculate the mean difference of features and logits, and accumulate to total loss
                    loss += torch.sum((torch.mean(real_feature, dim=0) - torch.mean(synthetic_feature, dim=0)) ** 2)
                    loss += torch.sum((torch.mean(real_logits, dim=0) - torch.mean(synthetic_logits, dim=0)) ** 2)

                # Update synthetic images S_k
                    optimizer_image.zero_grad()# Zero the optimizer gradients
                    loss.backward()# Backward propagation to calculate gradients
                
                # Update synthetic images
                    optimizer_image.step()
                
                self.logger.info(f'client {self.cid}, data condensation {epoch}, total loss = {loss.item()}, avg loss = {loss.item() / len(self.classes)}')

                # Save synthetic images to the list
                all_synthetic_images.append(synthetic_images.detach())  
                all_synthetic_labels.append(labels)

            # Consolidate synthetic data from all classes into a complete tensor
            if all_synthetic_images:
                final_synthetic_images = torch.cat(all_synthetic_images, dim=0).detach()  # Add .detach()
                final_synthetic_labels = torch.cat(all_synthetic_labels, dim=0)
            else:
                final_synthetic_images = None  # Return None if there is no synthetic data
                final_synthetic_labels = None
                self.logger.info("Client's synthetic data is empty!!")
            self.logger.info("Synthetic data for all classes successfully consolidated!")

            # Print the shape of the final consolidated tensor
            if final_synthetic_images is not None:
                self.logger.info(f"final_synthetic_images shape: {final_synthetic_images.shape}")
            else:
                self.logger.info("final_synthetic_images is empty")

            if final_synthetic_labels is not None:
                self.logger.info(f"final_synthetic_labels shape: {final_synthetic_labels.shape}")
            else:
                self.logger.info("final_synthetic_labels is empty")

        
        

        return final_synthetic_images,final_synthetic_labels