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
from torchvision.models import ResNet

class Client:
    def __init__(self, client_modules, config, logger, i):
        """
        Client initialization function
        Parameter explanation:
        - client_modules: Dictionary containing various modules required by the client
        - config: Dictionary of configuration parameters
        - logger: Logger
        - i: Client ID
        """
        # Initialize configuration and logger
        self.config = config
        self.logger = logger
        
        # Log the start of client initialization
        self.logger.info(f"Initializing client {i}...")
        
        # Get the number of model training epochs from the configuration
        self.model_epochs = config.get("train_model_epochs")
        
        # Set client ID
        self.cid = i
        
        # Data-related initialization
        self.dst_train = client_modules['dst_train']  # Training dataset
        self.client_indices = client_modules['client_indices'][i]  # Current client's data indices
        self.classes = client_modules['client_classes'][i]  # Current client's class information
        self.dataset_info = client_modules['dataset_info']  # Dataset information
        self.test_set = client_modules['dst_test']  # Test dataset
        
        # Create test data loader
        self.test_loader = DataLoader(
            self.test_set,
            sampler=SubsetRandomSampler(client_modules['target_test_indices']),
            batch_size=1, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )
        # Log completion of data loading
        self.logger.info(f"Client {i} data loading completed, contains classes: {self.classes}")
        
        # Model-related initialization
        self.model_strategy = config.get("Model")  # Get model strategy
        self.local_model =  copy.deepcopy(client_modules["global_model"])  # Global model
        
        # Federated learning strategy-related initialization
        self.fed_strategy = config.get("Federated_Learning_Config")  # Get federated learning strategy
        self.logger.info(f"Client {i} using federated strategy: {self.fed_strategy}")
        
        # If using the FedSD2C strategy, perform specific parameter initialization
        if self.fed_strategy == "FedSD2C":
            self.num_crop = config.get("sd2c_num_crop")
            self.input_size = self.dataset_info["im_size"][0]
            self.images_per_class = int(config.get("images_per_class"))
            self.sd2c_iterations = config.get("sd2c_iterations")
            self.logger.info(f"Client {i} FedSD2C parameter initialization completed")
        
        # Device-related initialization
        self.device = client_modules['device']
        
        # Log completion of client initialization
        self.logger.info(f"Client {i} initialization completed")
        
    def receive_data_from_server(self,server_data):
        
        # Fill in here how to process the received server_data
         # Can use hyperparameters from self.config
        return 

    def process(self):
        """
        Client processing flow, including model pre-training, core set selection, data perturbation, VAE creation and usage, data encoding, and synthetic data soft label generation.
        """
        
        # Pre-train the client model using the client's original data
        self.logger.info(f"Client {self.cid} starting to pre-train local model...")
        self.pretrain_client_model_use_client_original_data()
        self.logger.info(f"Client {self.cid} local model pre-training completed.")
        # Freeze all parameters of the model
        self.local_model.eval()
        for param in self.local_model.parameters():    
            param.requires_grad = False
        # Select core set indices based on the trained local model
        self.logger.info(f"Client {self.cid} starting to select core set indices based on the trained local model...")
        self.select_coreset_indices_based_on_trained_local_model(self.local_model, self.client_indices)
        self.logger.info(f"Client {self.cid} core set index selection completed.")
        
        # Perturb core set image indices
        self.logger.info(f"Client {self.cid} starting to perturb core set image indices...")
        self.pertubed_core_set_images_dict = self.perturb_coreset_indices_images()
        self.logger.info(f"Client {self.cid} core set image index perturbation completed.")
        
        # Create a VAE for subsequent processing
        self.logger.info(f"Client {self.cid} starting to create VAE...")
        from diffusers import AutoencoderKL
        # Use local model path instead of downloading from Hugging Face
        local_vae_path = ""
        vae = AutoencoderKL.from_pretrained(local_vae_path, use_safetensors=True)
        for p in vae.parameters():
            p.requires_grad = False  # Freeze parameters
        with torch.no_grad():
            vae.to(self.device)  # Move VAE to the specified device
        self.logger.info(f"Client {self.cid} VAE creation and configuration completed.")
        
        # Encode synthetic data using the local model and VAE
        self.logger.info(f"Client {self.cid} starting to encode synthetic data using local model and VAE...")
        self.encode_synthesis_data(self.local_model, vae)
        self.logger.info(f"Client {self.cid} synthetic data encoding completed.")
        
        # Generate soft labels for synthetic latent variables
        self.logger.info(f"Client {self.cid} starting to generate soft labels for synthetic latent variables...")
        self.pertubed_core_set_soft_labels_dict = self.generate_soft_label_for_synthetic_latents(self.local_model, vae)
        self.logger.info(f"Client {self.cid} synthetic latent variable soft label generation completed.")
        
        return
    
    def send_data_to_server(self):
        """
        Package and prepare the client's processed data (such as latent variable tensors and soft labels) for sending to the server.
        """
        
        # Create a dictionary to store the data to be transmitted
        data_sent = {}
        
        # Add latent variable tensor to the data dictionary
        self.logger.info(f"Client {self.cid} starting to prepare latent variable tensor to send to server...")
        data_sent["latent_variables_tensor"] = self.latent_variables_tensor
        
        # Add soft labels to the data dictionary
        self.logger.info(f"Client {self.cid} starting to prepare soft labels to send to server...")
        data_sent["soft_labels"] = self.soft_labels
        
        # Log completion of data preparation
        self.logger.info(f"Client {self.cid} data preparation completed, ready to send to server.")
        
        return data_sent
    
    def select_coreset_indices_based_on_trained_local_model(self, local_model, client_indices):
        """
        Select core set indices from the client's data based on the trained local model.
        """
        from copy import deepcopy
        from collections import defaultdict
        import torch
        # Log the start of core set index selection
        self.logger.info(f"Client {self.cid} starting to select core set indices based on the trained local model...")
        
        # Copy the incoming model to avoid modifying the original model's state
        tool_model = deepcopy(local_model)
        self.logger.info(f"Client {self.cid} has copied the local model.")
        
        # Set the model to evaluation mode, turning off layers used during training like dropout and batch normalization
        tool_model.eval()
        self.logger.info(f"Client {self.cid} model set to evaluation mode.")
        
        # Disable gradient computation for all parameters as this stage does not involve training
        for p in tool_model.parameters():
            p.requires_grad = False
        self.logger.info(f"Client {self.cid} has disabled gradient computation for all parameters.")
        
        # Move the model to the device
        tool_model.to(self.device)
        self.logger.info(f"Client {self.cid} model moved to the specified device.")
        
        # Deep copy the training dataset to avoid modifying the original dataset
        tool_dataset = deepcopy(self.dst_train)
        self.logger.info(f"Client {self.cid} has deep copied the training dataset.")
        
        # Group client_indices by class
        grouped_indices = defaultdict(list)
        for index in client_indices:
            target = tool_dataset.targets[index]
            if isinstance(target, int):    
                label = target
            else:    
                label = target.item()
            grouped_indices[label].append(index)
        
        # Sort indices within each group
        for label in grouped_indices:
            grouped_indices[label].sort()
        
        # Initialize core set and non-core set data dictionaries
        self.core_set_data = {
            "core_set_original_indices": [],
            "core_set_loss": [],
            "core_set_preds": [],
        }
        self.non_core_set_data = {
            "non_core_set_original_indices": [],
            "non_core_set_loss": [],
            "non_core_set_preds": [],
        }
        
        #
        # Perform core set selection for each class
        for label in grouped_indices:
            with torch.no_grad():  # Turn off gradient computation to save memory
                mrc = MultiRandomCrop(self.num_crop, self.input_size, 1, 1)
                
                core_set_original_indices, core_set_loss, core_set_preds, \
                non_core_set_original_indices, non_core_set_loss, non_core_set_preds = select_coreset(
                    self.images_per_class,
                    tool_model,
                    tool_dataset,
                    grouped_indices[label],
                    self.input_size,
                    device=self.device,
                    mrc=mrc,
                    m=self.num_crop,
                    descending=False,
                    ret_all=True
                )
                
                # Extend core set data
                self.core_set_data["core_set_original_indices"].extend(core_set_original_indices)
                self.core_set_data["core_set_loss"].extend([data.squeeze() for data in torch.split(core_set_loss.cpu(), 1)])
                self.core_set_data["core_set_preds"].extend([data.squeeze() for data in torch.split(core_set_preds.cpu(), 1)])

                # Extend non-core set data
                self.non_core_set_data["non_core_set_original_indices"].extend(non_core_set_original_indices)
                self.non_core_set_data["non_core_set_loss"].extend([data.squeeze() for data in torch.split(non_core_set_loss.cpu(), 1)])
                self.non_core_set_data["non_core_set_preds"].extend([data.squeeze() for data in torch.split(non_core_set_preds.cpu(), 1)])
        
        # Test if core set and non-core set selection is correct
        all_indices = set(self.core_set_data["core_set_original_indices"] + self.non_core_set_data["non_core_set_original_indices"])
        assert all_indices == set(client_indices), "Test 1 Failed: The union of core and non-core indices does not match the given indices range."
        
        combined_indices = sorted(self.core_set_data["core_set_original_indices"] + self.non_core_set_data["non_core_set_original_indices"])
        given_indices_sorted = sorted(client_indices)
        assert combined_indices == given_indices_sorted, "Test 2 Failed: The combination of core and non-core original indices does not match the given indices."
        
        # Log completion of selection
        self.logger.info(f"Client {self.cid} core set index selection completed.")
        
        return

    def pretrain_client_model_use_client_original_data(self):
        """
        Pre-train the global model using the client's original data.
        """
        
        # Log the start of local model pre-training
        self.logger.info(f"Client {self.cid} starting to pre-train local model using original data...")
        
        # Create DataLoader for loading training data
        dataloader = DataLoader(
            self.dst_train,
            sampler=SubsetRandomSampler(self.client_indices),
            batch_size=self.config.get("train_batch_size"),
            shuffle=False,
            num_workers=0
        )
        
        # Set model to training mode
        self.local_model.train()
        self.logger.info(f"Client {self.cid} model set to training mode.")
        
        # Define optimizer
        model_optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=float(self.config.get("learning_rate")),
            weight_decay=float(self.config.get("weight_decay")),
            momentum=float(self.config.get("momentum"))
        )
        self.logger.info(f"Client {self.cid} optimizer configuration completed.")
        
        # Define loss function
        loss_function = torch.nn.CrossEntropyLoss()
        total_loss = 0
        
        # Start training loop
        for epoch in tqdm(range(self.model_epochs), desc='global model training', leave=True):
            for x, target in dataloader:
                x, target = x.to(self.device), target.to(self.device)  # Move data to the specified device (cpu/gpu)
                target = target.long()  # Ensure target type is long
                
                model_optimizer.zero_grad()  # Clear previous gradients
                pred = self.local_model(x)  # Forward pass
                loss = loss_function(pred, target)  # Compute loss
                loss.backward()  # Backward pass
                model_optimizer.step()  # Update weights
                
                total_loss += loss.item()  # Accumulate loss value
            
            # Print current average loss at the end of each epoch
            avg_loss = total_loss / (epoch + 1)
            self.logger.info(f"Client {self.cid} Epoch {epoch + 1}/{self.model_epochs}, Average Loss: {avg_loss}")
        
        # Log and print the average loss per round
        final_avg_loss = total_loss / self.model_epochs
        print(f'Average loss per round = {final_avg_loss}')
        self.logger.info(f"Client {self.cid} pre-training completed, final average loss: {final_avg_loss}")
        
        return

    def perturb_coreset_indices_images(self):
        """
        Perform perturbation processing on each image in the core set, and return a dictionary containing the mapping from core set indices to perturbed images.
        """
        
        # Log the start of core set image perturbation
        self.logger.info(f"Client {self.cid} starting to perturb core set images...")
        
        # Group non_core_set_data["non_core_set_original_indices"] by class
        non_core_set_by_class = {}
        
        # Create a dictionary where keys are classes and values are lists of non-core set image indices for that class
        for idx in self.non_core_set_data["non_core_set_original_indices"]:
            _, label = self.dst_train[idx]  # Assuming the second element is the label
            if label not in non_core_set_by_class:
                non_core_set_by_class[label] = []
            non_core_set_by_class[label].append(idx)
        
        # Initialize a dictionary to store perturbed images
        perturbed_images = {}
        
        # Iterate over each image in the core set
        for i in self.core_set_data["core_set_original_indices"]:
            core_image, core_label = self.dst_train[i]
            
            # Randomly select an image of the same class from the non-core-set
            if core_label in non_core_set_by_class and len(non_core_set_by_class[core_label]) > 0:
                selected_non_core_idx = np.random.choice(non_core_set_by_class[core_label])
                non_core_image, _ = self.dst_train[selected_non_core_idx]
            else:
                # If no image of the same class is found, generate a substitute image using uniform distribution noise
                non_core_image = torch.rand_like(core_image)  # Uniform distribution noise
            
            # Use colorful_spectrum_mix for perturbation
            perturbed_img, _ = self.colorful_spectrum_mix(core_image, non_core_image, self.config.get("sd2c_foulier_alpha"))
            
            # Clip the image to ensure it is within a reasonable range
            perturbed_img = self.clip(perturbed_img)
            
            # Store the perturbed image in the dictionary
            perturbed_images[i] = perturbed_img
        
        # Log completion of perturbation processing
        self.logger.info(f"Client {self.cid} core set image perturbation processing completed.")
        
        return perturbed_images
        
        # For each image in core_set_indices (retrieve from self.dst_train using self.core_set_data["core_set_original_indices"])
        # Randomly select an image of the same class from non_core_set_original_indices (retrieve from self.dst_train using self.core_set_data["core_set_original_indices"])
        # Perturb the core_set_indices image using colorful_spectrum_mix
        # Clip the image
        # Return a dictionary mapping coreset_indices to perturbed images

    def colorful_spectrum_mix(self, img1, img2, alpha, ratio=1.0):
        """
        Perform color spectrum mixing on two images.
        Input image size: tensor of [C, H, W]
        """
        self.logger.info(f"Starting color spectrum mixing operation...")
        
        from math import sqrt
        
        lam = alpha  # Mixing coefficient
        
        # Adjust image dimension order to fit numpy fft operation
        img1 = img1.permute(