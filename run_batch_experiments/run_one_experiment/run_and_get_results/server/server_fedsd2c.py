import copy
import os
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torch.utils.data.sampler import SubsetRandomSampler
from ..client.client_fedsd2c import denormalize


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
        
        # Model related
        # Use deepcopy to ensure independence of the global model and move it to the specified device
        self.global_model = copy.deepcopy(basic_modules['global_model']).to(basic_modules['device'])
        self.model_strategy = config.get("Model")  # Get model-related strategy
        self.logger.info(f"Server model initialization completed: {self.model_strategy}")
        
        # Scheduling related
        self.join_ratio = config.get("join_ratio")  # Get participation rate
        
        # Data related
        # Save dataset_info for denormalize
        self.dataset_info = basic_modules['dataset_info']
        # Load test set and use SubsetRandomSampler to get data samples with specific indices
        self.test_set = basic_modules['dst_test']
        # Use a larger batch_size for efficiency, default is 32
        test_batch_size =32
        self.test_loader = DataLoader(
            self.test_set,
            sampler=SubsetRandomSampler(basic_modules['target_test_indices']),
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        self.logger.info(f"Server test data loading completed, batch_size={test_batch_size}, total {len(basic_modules['target_test_indices'])} samples")
        
        # Device related
        self.device = basic_modules['device']  # Determine the device to use (e.g., CPU or GPU)
        
        # Use VAE passed from basic_modules
        self.vae = basic_modules['vae']
        self.logger.info("VAE model loaded from basic_modules")
        
        # Federated learning strategy
        self.fed_strategy = config.get("Federated_Learning_Config")  # Get federated learning strategy
        self.logger.info(f"Server using federated strategy: {self.fed_strategy}")
        
        # Save config as an instance variable for future access
        self.config = config
        
        self.logger.info("Server initialization completed")

    def arrange_server_data_to_client(self):
        server_data={}#server_data is a dictionary
        # Can use hyperparameters from self.config

        # Fill in the data you want to prepare for each client here

        return server_data
    
    def merge_data(self, received_data_list):
        random.seed(self.config.get("seed"))
        """
        Merge data received from clients.
        
        Parameters:
            received_data_list: List of data sent by clients, each element is a dictionary,
                                containing "latent_variables_tensor" and "soft_labels" keys.
                                
        Returns:
            merged_data: Dictionary containing merged latent variable tensor and soft labels.
        """
        # Initialize storage lists
        self.latent_variable_tensor_list = []
        self.soft_labels_list = []

        # Traverse all received client data
        for data_sent_i in received_data_list:
            # Extract latent variable tensor and soft labels, and add to lists
            self.latent_variable_tensor_list.append(data_sent_i["latent_variables_tensor"])
            self.soft_labels_list.append(data_sent_i["soft_labels"])

        # Concatenate tensors in the list into one large tensor
        merged_latent_variables = torch.cat(self.latent_variable_tensor_list, dim=0)
        merged_soft_labels = torch.cat(self.soft_labels_list, dim=0)

        # Construct final merged data dictionary
        merged_data = {
            "latent_variables_tensor": merged_latent_variables,  # Shape: [total_samples, latent_dim]
            "soft_labels": merged_soft_labels                   # Shape: [total_samples, num_classes]
        }
        self.logger.info("Data merging completed.")
        return merged_data
    
    def process(self, merged_data):
        random.seed(self.config.get("seed"))
        """
        Train the global model using merged data.
        
        Parameters:
            merged_data: Dictionary containing merged latent variable tensor and soft labels.
            
        Returns:
            avg_epoch_loss: Average loss per epoch, used for monitoring the training process.
        """
        from torch.optim import Adam
        self.global_model.to(self.device)  # Ensure global model is on the correct device
        
        # Initialize optimizer for all trainable parameters of the global model
        self.optimizer = Adam(self.global_model.parameters(), lr=0.001)  # Adjust learning rate as needed
        # Use VAE passed from basic_modules
        vae = self.vae
    
        # Freeze all VAE parameters
        for p in vae.parameters():
            p.requires_grad = False
        vae.eval()  # Set to evaluation mode
        vae.to(self.device)  # Move to specified device

        # Extract data from merged_data
        latent_variables_tensor = merged_data["latent_variables_tensor"].to(self.device)
        target_soft_labels = merged_data["soft_labels"].to(self.device)
        
        # Decode latent variables with VAE (no gradient computation)
        with torch.no_grad():
            synthetic_data_list = []
            batch_size = 256  # Consistent with later DataLoader batch size

            for i in range(0, latent_variables_tensor.size(0), batch_size):
                batch = latent_variables_tensor[i:i+batch_size]
                decoded_batch = vae.decode(batch).sample
                synthetic_data_list.append(decoded_batch)

            synthetic_data = torch.cat(synthetic_data_list, dim=0)
            
        # Print shapes of synthetic data and target soft labels
        self.logger.info(f"Synthetic_data shape: {synthetic_data.shape}")
        self.logger.info(f"Target soft labels shape: {target_soft_labels.shape}")

        # Create TensorDataset and DataLoader to process data in batches
        dataset = TensorDataset(synthetic_data, target_soft_labels)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        # Multiple epoch training loop
        num_epochs = self.config.get("train_model_epochs")
        for epoch in range(num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            self.global_model.train()  # Ensure global model is in training mode

            epoch_loss = 0.0  # Record total loss per epoch

            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)

                logits = self.global_model(data)  # Note: should use 'data' here, not 'synthetic_data'
                pred_soft_labels = torch.softmax(logits, dim=1)  # Probability distribution predicted by model

                # Calculate KL divergence loss
                loss = torch.nn.functional.kl_div(
                    input=torch.log(pred_soft_labels + 1e-10),  # Avoid log(0)
                    target=target,
                    reduction='batchmean',  # Average per batch
                    log_target=False  # target is probability distribution (not log probability)
                )

                # Backpropagation to optimize global model
                self.global_model.zero_grad()  # Clear gradients
                loss.backward()  # Backward propagation
                self.optimizer.step()  # Update parameters

                epoch_loss += loss.item()

                if batch_idx % 10 == 0:
                    self.logger.info(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}")

            # Print average loss at the end of each epoch
            avg_epoch_loss = epoch_loss / len(dataloader)
            self.logger.info(f"Epoch {epoch + 1} completed, Average Loss: {avg_epoch_loss}")

        return avg_epoch_loss  # Return final average loss value for monitoring training process
    

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