import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import sys
import numpy as np
import os
import torchvision
import matplotlib.pyplot as plt

# Import funnel encoder
try:
    from ..funnel_encoder import FunnelEncoder
except ImportError:
    # If relative import fails, try absolute import
    from funnel_encoder import FunnelEncoder

# Import visualization system
try:
    from ..visualization_system.image_tracker import ImageTracker
    from ..visualization_system.server_visualizer import ServerVisualizer
    VISUALIZATION_SYSTEM_AVAILABLE = True
except ImportError:
    try:
        # If relative import fails, try absolute import
        from visualization_system.image_tracker import ImageTracker
        from visualization_system.server_visualizer import ServerVisualizer
        VISUALIZATION_SYSTEM_AVAILABLE = True
    except ImportError:
        VISUALIZATION_SYSTEM_AVAILABLE = False
        print("Warning: Visualization system unavailable")

class Server:
    def __init__(self, basic_modules, config, logger, clients):
        """
        Initialize the server (new process version).
        
        Parameters:
        - basic_modules: Dictionary containing basic modules, such as global model, device, etc.
        - config: Configuration dictionary containing parameters required for server operation.
        - logger: Logger for recording log information during server operation.
        - clients: Client list or related information.
        """
        self.clients = clients  # Store information of all clients
        self.logger = logger  # Set logger
        
        self.logger.info("Initializing FedSumUp server (new process version)...")
        
        # Model related
        # Use deepcopy to ensure independence of the global model and move it to the specified device
        
        self.model_strategy = config.get("Model")  # Get model-related strategy
        self.logger.info(f"Server model initialization completed: {self.model_strategy}")
        
        # Scheduling related
        self.join_ratio = config.get("join_ratio")  # Get participation rate
        
        # Data related
        # Save dataset_info for denormalize
        self.dataset_info = basic_modules['dataset_info']
        # Load test set and use SubsetRandomSampler to get data samples with specific indices
        self.test_set = basic_modules['dst_test']
        # Use larger batch_size for efficiency, default is 32
        test_batch_size = 32
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
        
        # Use encoder and classifier passed from basic_modules
        self.encoder = basic_modules['vae']  # Note: Still using 'vae' key, but actually FunnelEncoder
        self.global_model = basic_modules['classifier']
        # Save a deep copy as an "innocent" classifier template
        self.innocent_classifier_template = copy.deepcopy(basic_modules['classifier'])
        self.logger.info("Encoder model and classifier loaded from basic_modules")
        
        # Save encoder state before training (for evaluation)
        self.encoder_before_training = None
        self._save_encoder_before_training()
        
        # Federated learning strategy
        self.fed_strategy = config.get("Federated_Learning_Config")  # Get federated learning strategy
        self.logger.info(f"Server using federated strategy: {self.fed_strategy}")
        
        # Save config as instance variable for future access
        self.config = config
        
        # Save basic_modules for future use
        self.basic_modules = basic_modules
        
        # Initialize state variables
        self.current_round = 0
        self.aggregated_latent_variables = None
        self.aggregated_labels = None
        
        self.logger.info("FedSumUp server (new process version) initialization completed")
    
    def _compute_logits_consistency_loss(self, logits, labels):
        """
        Compute logits consistency loss (vectorized optimized version)
        
        Parameters:
        - logits: Logits with shape [batch_size, num_classes]
        - labels: Labels with shape [batch_size]

        Returns:
        - total_loss: Total consistency loss
        - same_loss: Similarity loss within same labels
        - diff_loss: Dissimilarity loss between different labels
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        
        # Get all unique labels
        unique_labels, counts = torch.unique(labels, return_counts=True)
        num_unique_labels = len(unique_labels)
        
        # ========== Vectorized calculation of similarity loss within same labels ==========
        # Initialize same label loss
        same_loss = torch.tensor(0.0, device=logits.device)
        same_count = 0
        
        # Use vectorized method to calculate variance for each label
        for i, label in enumerate(unique_labels):
            if counts[i] > 1:  # At least two samples needed to calculate similarity
                # Get all samples with the same label
                mask = (labels == label)
                same_logits = logits[mask]  # [count_i, num_classes]
                
                # Calculate variance and take average (vectorized operation)
                variance = torch.var(same_logits, dim=0, unbiased=True).mean()
                same_loss += variance
                same_count += 1
        
        if same_count > 0:
            same_loss = same_loss / same_count
        
        # ========== Vectorized calculation of dissimilarity loss between different labels ==========
        diff_loss = torch.tensor(0.0, device=logits.device)
        
        if num_unique_labels > 1:
            # Calculate mean logits for each label
            label_means = []
            valid_labels = []
            
            for i, label in enumerate(unique_labels):
                if counts[i] > 0:  # Ensure label has samples
                    mask = (labels == label)
                    label_mean = logits[mask].mean(dim=0, keepdim=True)  # [1, num_classes]
                    label_means.append(label_mean)
                    valid_labels.append(label)
            
            if len(label_means) > 1:
                # Stack all label means into a matrix [num_valid_labels, num_classes]
                means_matrix = torch.cat(label_means, dim=0)  # [M, num_classes]
                M = means_matrix.size(0)
                
                # Calculate cosine similarity matrix between all label pairs
                # Normalize mean vectors
                means_norm = F.normalize(means_matrix, p=2, dim=1)  # [M, num_classes]
                
                # Calculate cosine similarity matrix [M, M]
                cos_sim_matrix = torch.mm(means_norm, means_norm.t())  # [M, M]
                
                # Get upper triangular matrix (excluding diagonal)
                triu_mask = torch.triu(torch.ones(M, M, device=logits.device), diagonal=1).bool()
                cos_sim_values = cos_sim_matrix[triu_mask]  # Get upper triangular elements
                
                if len(cos_sim_values) > 0:
                    # Calculate dissimilarity loss: 1 - cosine similarity (we want similarity to be small)
                    diff_loss = (1.0 - cos_sim_values).mean()
        
        # Total loss = similarity loss within same labels + dissimilarity loss between different labels
        total_loss = same_loss +  diff_loss
        
        
        return total_loss, same_loss, diff_loss

    def arrange_server_data_to_client(self):
        """
        Prepare data to send to clients
        New version: Send Encoder to clients every round
        """
        server_data = {}
        server_data["current_round"] = self.current_round
        
        # New version: Send encoder every round
        server_data["encoder"] = copy.deepcopy(self.encoder)
        self.logger.info(f"Server preparing to send encoder to clients (Round {self.current_round})")
        
        return server_data
    
    def merge_data(self, received_data_list):
        """
        Merge data received from clients
        New version: Merge latent space encodings, labels, and corresponding indices every round
        """
        self.logger.info(f"Starting to merge client data for round {self.current_round}, total {len(received_data_list)} clients")
        
        latent_list = []
        label_list = []
        indices_list = []
        
        for data in received_data_list:
            if data is None:
                raise ValueError(f"Client data is None")
            if "latent_variables" not in data:
                raise ValueError(f"Client data missing latent_variables field")
            if "labels" not in data:
                raise ValueError(f"Client data missing labels field")
            if "client_indices" not in data:
                raise ValueError(f"Client data missing client_indices field")
            
            latent_list.append(data["latent_variables"])
            label_list.append(data["labels"])
            indices_list.append(data["client_indices"])
        
        if not latent_list or not label_list:
            raise ValueError("No valid latent variables or label data received")
        
        self.aggregated_latent_variables = torch.cat(latent_list, dim=0)
        self.aggregated_labels = torch.cat(label_list, dim=0)
        # Merge indices from all clients
        self.aggregated_indices = []
        for indices in indices_list:
            self.aggregated_indices.extend(indices)
        
        self.logger.info(f"Merging latent variables, labels, and indices completed, latent variable shape: {self.aggregated_latent_variables.shape}, label shape: {self.aggregated_labels.shape}, index count: {len(self.aggregated_indices)}")
        
        merged_data = {
            "latent_variables": self.aggregated_latent_variables,
            "labels": self.aggregated_labels,
            "client_indices": self.aggregated_indices
        }
        
        return merged_data
    
    def process(self, merged_data):
        """
        Process merged data (following the two-stage process proposed by the user)
        
        User process:
        Round 1 server:
        Stage 1: Classifier learning
          1. Small aspect ratio data I₀ trains classifier C₀ to get C₁
        Stage 2: Lossy compressor learning
          1. Freeze C₁
          2. Small aspect ratio data I₀ directly scaled to D₀ size to get I₀'
          3. Treat V₀ and C₁ as a whole, train V₀ using I₀' and L₀ data pairs to get V₁
          4. Use V₀ and C₁ as a whole to evaluate accuracy on test set
          5. Send V₁ back to clients
        """
        self.logger.info(f"Starting to process merged data for round {self.current_round} (two-stage process)")
        
        # Check if necessary fields exist
        if "latent_variables" not in merged_data:
            raise ValueError("Merged data missing latent_variables field")
        if "labels" not in merged_data:
            raise ValueError("Merged data missing labels field")
        if "client_indices" not in merged_data:
            raise ValueError("Merged data missing client_indices field")
        
        if merged_data["latent_variables"] is None:
            raise ValueError("latent_variables is None")
        if merged_data["labels"] is None:
            raise ValueError("labels is None")
        if merged_data["client_indices"] is None:
            raise ValueError("client_indices is None")
        
        # Save encoder state before training
        self._save_encoder_before_training()
        
        # Prepare data (small images I₀)
        small_images = merged_data["latent_variables"].to(self.device)
        labels = merged_data["labels"].to(self.device)
        client_indices = merged_data["client_indices"]
        
        self.logger.info(f"Received small image data: {small_images.shape}, labels: {labels.shape}, index count: {len(client_indices)}")
        
        # ========== Stage 1: Classifier learning ==========
        self.logger.info(f"Stage 1: Training classifier C_{self.current_round}...")
        self._train_classifier_stage(small_images, labels, client_indices)
        
        # ========== Stage 2: Lossy compressor learning ==========
        self.logger.info(f"Stage 2: Training encoder V_{self.current_round} (freeze classifier)...")
        self._train_encoder_stage(small_images, labels, client_indices)
        
        # ========== Evaluation ==========
        self.logger.info(f"Evaluating overall performance of V_{self.current_round+1} and C_{self.current_round+1}...")
        accuracy = self.evaluate()
        
        self.logger.info(f"Generation of new encoder V_{self.current_round+1} and classifier C_{self.current_round+1} completed, test accuracy: {accuracy:.4f}")
        
        # Update round
        self.current_round += 1
        self.logger.info(f"Round {self.current_round-1} processing completed, entering round {self.current_round}")
        
        return
    

    
    def _train_classifier_stage(self, small_images, labels, client_indices):
        """
        Stage 1: Classifier learning
        
        According to user process:
        1. Small aspect ratio data I₀ trains classifier C₀ to get C₁
        
        Steps:
        1. Freeze encoder parameters
        2. Unfreeze classifier parameters
        3. Upscale all small images I₀ to 32x32
        4. Train classifier with upscaled images
        5. Add logits consistency loss: reduce logits differences within same labels, maintain logits differences between different labels
        """
        self.logger.info("Stage 1: Training classifier (freeze encoder, add logits consistency loss)...")
        
        # Prepare data
        small_images = small_images.to(self.device)
        labels = labels.to(self.device)
        
        # Train for multiple epochs
        num_epochs = self.config.get("train_model_epochs", 10)
        

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier parameters
        for param in self.global_model.parameters():
            param.requires_grad = True
        
        # Define optimizer - only optimize classifier (encoder parameters frozen)
        classifier_optimizer = torch.optim.Adam(
            self.global_model.parameters(),
            lr=0.001, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        
        self.encoder.eval()  # Set encoder to evaluation mode since parameters are frozen
        self.global_model.train()
        
        # Create dataset and data loader (use original small images and labels, dynamically upscale during training)
        # Convert client_indices to tensor
        indices_tensor = torch.tensor(client_indices, device=self.device)
        dataset = TensorDataset(small_images, labels, indices_tensor)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_same_loss = 0.0
            epoch_diff_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (batch, target, indices) in enumerate(dataloader):
                batch, target = batch.to(self.device), target.to(self.device)
                
                classifier_optimizer.zero_grad()
                
                # Upscale small images to 32x32
                batch_upsampled = F.interpolate(
                    batch,
                    size=(32, 32),
                    mode='bilinear',
                    align_corners=True
                )
                
                # Classifier processes upscaled images
                # New version classifier expects 4D input (B, C, H, W), no need to flatten
                output = self.global_model(batch_upsampled)
                
                # Calculate classification loss (cross entropy)
                ce_loss = criterion(output, target)
                
                # Calculate logits consistency loss
                consistency_loss, same_loss, diff_loss = self._compute_logits_consistency_loss(
                    output, target
                )
                utility_ratio = self.config.get("sumup_utility_ratio")
                # Total loss = cross entropy loss + logits consistency loss
                total_loss =utility_ratio * ce_loss + consistency_loss
                
                # Check if loss is NaN or Inf
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    self.logger.warning(f"Loss is NaN or Inf, skipping this batch")
                    continue
                
                total_loss.backward()
                
                # Add gradient clipping to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=0.1)
                
                classifier_optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_same_loss += same_loss.item()
                epoch_diff_loss += diff_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
            avg_ce_loss = epoch_ce_loss / len(dataloader) if len(dataloader) > 0 else 0
            avg_same_loss = epoch_same_loss / len(dataloader) if len(dataloader