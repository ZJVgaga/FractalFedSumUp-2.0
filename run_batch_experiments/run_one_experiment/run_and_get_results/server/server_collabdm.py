import copy
import os
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler


# For measuring communication overhead
class Server:
    def __init__(self, basic_modules, config, logger, clients):
        """
        Initialize the server.
        
        Parameters:
        - basic_modules: Dictionary containing basic modules, such as global model, device information, etc.
        - config: Configuration dictionary containing parameters required for server operation.
        - logger: Logger for recording log information during server operation.
        - clients: Client list or related information.
        """
        self.clients = clients  # Store information of all clients
        self.logger = logger  # Set the logger
        
        self.logger.info("Initializing server...")
        import torch.nn as nn

        self.criterion = nn.CrossEntropyLoss()
        # Model related
        # Use deepcopy to ensure independence of the global model and move it to the specified device
        self.global_model = copy.deepcopy(basic_modules['global_model']).to(basic_modules['device'])
        self.model_strategy = config.get("Model")  # Get model-related strategy
        self.logger.info(f"Server model initialization completed: {self.model_strategy}")
       
        # Scheduling related
        self.join_ratio = config.get("join_ratio")  # Get participation rate
        from diffusers import AutoencoderKL
        # Data related
        # Load test set and use SubsetRandomSampler to get data samples for specific indices
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
        self.first_round=True
        # Device related
        self.device = basic_modules['device']  # Determine the device to use (e.g., CPU or GPU)
        
      
        # Federated learning strategy
        self.fed_strategy = config.get("Federated_Learning_Config")  # Get federated learning strategy
        self.logger.info(f"Server using federated strategy: {self.fed_strategy}")
        
        # Save config as an instance variable for later access
        self.config = config
        
        self.logger.info("Server initialization completed")

    def arrange_server_data_to_client(self):
        if self.first_round==True:
            self.first_round=False
            server_data={}
            return server_data
        else:
            server_data = {
            "global_model": self.global_model
        }#server_data is a dictionary
        #Can use hyperparameters from self.config

        #Fill in the data you want to prepare for each client here

            return server_data
    
    def merge_data(self, received_data_list):
        
        merged_data = {
            'synthetic_images': {},  # List of synthetic images per category (from different clients)
            'embeddings': {}         # List of embedding features per category (including client_id)
        }

        
        for client_idx, client_data in enumerate(received_data_list):
            if not client_data:
                continue

            # Process synthetic images
            synthetic_images = client_data["synthetic_images"]
            for cls, img_tensor in synthetic_images.items():
                if cls not in merged_data['synthetic_images']:
                    merged_data['synthetic_images'][cls] = []
                merged_data['synthetic_images'][cls].append(img_tensor)

            # Process embedding features
            client_embeddings = client_data["client_embeddings"]
            for cls, feat_list in client_embeddings.items():
                if cls not in merged_data['embeddings']:
                    merged_data['embeddings'][cls] = []
                
                merged_data['embeddings'][cls].append({
                        'mean': feat_list,
                        'client_id': client_idx
                })
        # Merge the image list for each category into one large tensor
        for cls in merged_data['synthetic_images']:
            merged_data['synthetic_images'][cls] = torch.cat(merged_data['synthetic_images'][cls], dim=0)
        self.logger.info("Data merging completed.")
        return merged_data
      
    
    def process(self, merged_data):
        #Process merged_data
        #Get iterations=self.config.get("collabdm_iterations")
        #For each t in iterations
            #For each class c
            #Randomly select 256 images X from merged_data['synthetic_images'][c]
            #feature=self.global_model.embed(x)
            #Then calculate L_original_data=weighted average of the t-th element of all client feats in merged_data['embeddings'][cls]
            #Then calculate loss=torch.sum((L_original_data - torch.mean(feature, dim=0)) ** 2)
            #Use this loss to optimize merged_data['synthetic_images'][c]

        iterations = self.config.get("collabdm_iterations")  # Default 5 times
        batch_size = 256 # Sample 256 images each time for calculating target embed
        device = self.device

        # Extract model embed layer
        model = self.global_model
      

        # Keep only classes that need optimization
        all_classes = list(merged_data['synthetic_images'].keys())
        self.logger.info(f"Starting synthetic image optimization for classes {all_classes}")

        # Set up optimizers
        syn_images_dict = {}
        optimizer_dict = {}

        for cls in all_classes:
            # Copy original synthetic images as learnable parameters
            syn_images = merged_data['synthetic_images'][cls].clone().detach().to(device).requires_grad_(True)
            syn_images_dict[cls] = syn_images
            optimizer_dict[cls] = torch.optim.Adam([syn_images], lr=1)

        # Start multi-round optimization
        for t in range(iterations):
            self.logger.info(f"Optimization round [{t+1}/{iterations}]")

            for cls in all_classes:
                # Step 1: Randomly select batch_size images from synthetic images
                indices = torch.randperm(syn_images_dict[cls].shape[0])[:batch_size]
                x = syn_images_dict[cls][indices].to(device)

                # Step 2: Forward propagation to get features
                
                feature = model.embed(x)

                # Step 3: Get the average embed of real data at round t (average across all clients)
                real_feats_t = []
                for item in merged_data['embeddings'][cls]:
                    real_feats_t.append(item['mean'][t])  # Assume item['mean'] is Tensor or can be converted to Tensor

                real_feat_avg = torch.mean(torch.stack(real_feats_t), dim=0).to(device)

                # Step 4: Calculate loss
                loss = torch.sum((real_feat_avg - torch.mean(feature, dim=0)) ** 2)

                # Step 5: Backward propagation & optimization
                optimizer = optimizer_dict[cls]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.logger.info(f"Class {cls}: loss = {loss.item():.4f}")

        # Finally update synthetic images in merged_data
        for cls in all_classes:
            merged_data['synthetic_images'][cls] = syn_images_dict[cls].cpu().detach()

        self.logger.info("Synthetic image optimization completed")
        """
        Train global model using collabDM synthetic images, using hard labels (cls) instead of soft labels
        """
        self.logger.info("Starting global model training using collabDM synthetic images")

        # Step 1: Construct training dataset
        synthetic_data_list = []
        target_labels_list = []
  
        all_classes = list(merged_data['synthetic_images'].keys())

        for cls in all_classes:
            images = merged_data['synthetic_images'][cls].to(self.device)
            labels = torch.tensor([cls] * images.shape[0], dtype=torch.long, device=self.device)

            synthetic_data_list.append(images)
            target_labels_list.append(labels)

        synthetic_data = torch.cat(synthetic_data_list, dim=0)
        target_labels = torch.cat(target_labels_list, dim=0)

        # Step 2: Create DataLoader
        dataset = TensorDataset(synthetic_data, target_labels)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        # Step 3: Get number of training epochs
        num_epochs = self.config.get("train_model_epochs")
        for param in self.global_model.parameters():
            param.requires_grad = True
        self.global_model.to(self.device).train()
        # ✅ Add optimizer
        self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=self.config.get("learning_rate"))
        # Step 4: Training loop
        total_loss = 0.0


        for epoch in range(num_epochs):
            self.logger.info(f"Epoch [{epoch + 1}/{num_epochs}] started")
            epoch_loss = 0.0

            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)

                logits = self.global_model(data)
                loss = self.criterion(logits, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                if batch_idx % 10 == 0:
                    self.logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / len(dataloader)
            total_loss += avg_epoch_loss
            self.logger.info(f"Epoch [{epoch + 1}/{num_epochs}] completed, average Loss: {avg_epoch_loss:.4f}")

        avg_total_loss = total_loss / num_epochs
        self.logger.info(f"collabDM data training completed, total average Loss: {avg_total_loss:.4f}")

        return avg_total_loss
    

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