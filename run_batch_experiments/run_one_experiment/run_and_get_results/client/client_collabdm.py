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
        - logger: Logger instance
        - i: Client ID
        """
        # Initialize configuration and logger
        self.config = config
        self.logger = logger
        
        # Log the start of client initialization
        self.logger.info(f"Initializing client {i}...")
        
        # Get the number of model training epochs from configuration
        self.model_epochs = config.get("train_model_epochs")
        
        # Set client ID
        self.cid = i
        self.first_round=True
        self.images_per_class=self.config.get("images_per_class")
        # Data-related initialization
        self.dst_train = client_modules['dst_train']  # Training dataset
        self.client_indices = client_modules['client_indices'][i]  # Current client's data indices
        self.classes = client_modules['client_classes'][i]  # Current client's class information
        self.dataset_info = client_modules['dataset_info']  # Dataset information
        self.test_set = client_modules['dst_test']  # Test dataset
        # Build a dictionary of indices per class (only containing classes owned by this client)
        self.class_indices = {}

        # Iterate through all sample indices of the current client, organize them into a dictionary by class
        for idx in self.client_indices:
            _, label = self.dst_train[idx]  # Get label (assuming return format is (image, label))
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
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
        
        
        # Federated learning strategy-related initialization
        self.fed_strategy = config.get("Federated_Learning_Config")  # Get federated learning strategy
        self.logger.info(f"Client {i} uses federated strategy: {self.fed_strategy}")
        
        # If using the Fedsumup strategy, perform specific parameter initialization

        # Device-related initialization
        self.device = client_modules['device']
        self.local_model = copy.deepcopy(client_modules["global_model"]).to(self.device)  # Global model
        # Log completion of client initialization
        self.logger.info(f"Client {i} initialization completed")

    def receive_data_from_server(self,server_data):
        self.logger.info(f"Client {self.cid} is receiving data from server...")
        if not self.first_round==True:
            if 'global_model' in server_data:
                self.local_model = copy.deepcopy(server_data['global_model'])
                self.local_model.eval()
                self.logger.info(f"Client {self.cid} has received the global model")
        self.logger.info(f"Client {self.cid} server data reception completed")
        # Fill in here how to handle the received server_data
         # Can use hyperparameters from self.config
        return 

    def process(self):
       # Generate local synthetic data using the distribution matching method
        self.client_syn = self.distribution_matching_idm(self.images_per_class, self.config)
          # Build labels for synthetic data, note that only classes owned by the client are included here
        local_classes = list(self.class_indices.keys())  # Replace the original self.classes
       
        
        # Set number of iterations (can come from config)
        iterations = self.config.get("collabdm_iterations")  # Default 5 times

    # Compute embedding vectors without calculating gradients (multiple iterations)
        with torch.no_grad():
            client_embedding = self.compute_embeddings(local_classes, iterations)

     # Send embeddings and synthetic data to server
        self.client_embedding = client_embedding  # Now a dictionary of class -> multiple iteration results
        return    
    
    def distribution_matching_idm(self, images_per_class, config):
        lr_img = 1.0
        batch_real = 256 
        iteration = self.config.get("collabdm_iterations")  # Default 5 times

        # Get list of classes owned locally
        local_classes = list(self.class_indices.keys())  # Replace the original self.classes
        
        # Initialize synthetic images and labels according to local classes
        # Ensure all parameters are integers
        # Ensure images_per_class is an integer
        images_per_class_int = int(images_per_class)
        size_tuple = (
            int(len(local_classes) * images_per_class_int),
            int(self.dataset_info["channel"]),
            int(self.dataset_info["im_size"][0]),
            int(self.dataset_info["im_size"][1])
        )
        
        image_syn = torch.randn(
            size_tuple,
            device=self.device
        )
        image_syn = image_syn.to(dtype=torch.float)
        image_syn.requires_grad_(True)
        label_syn = torch.tensor(
            [np.ones(images_per_class_int) * cls for cls in local_classes],
            dtype=torch.long,
            requires_grad=False,
            device=self.device
        ).view(-1)

       # Initialization method: real / noise
        if self.config.get("collabdm_mode") == 'real':
            print('initialize synthetic data from random real images')
            for idx, cls in enumerate(local_classes):
                img_real = self.get_images(cls, images_per_class_int)
                image_syn.data[idx * images_per_class_int:(idx + 1) * images_per_class_int] = img_real.detach().data
        else:
            print('initialize synthetic data from random noise')
        ''' training '''
       
        optimizer_img = torch.optim.SGD([image_syn],lr_img , momentum=0.5)
    
        optimizer_img.zero_grad()
        print(' training begins')

        net = self.local_model.to(self.device)


        # train syntheitc data
        for it in range(iteration + 1):

            loss_avg = 0

            for param in net.parameters():
                param.requires_grad = False

            for idx, cls in enumerate(local_classes):
                img_real = self.get_images(cls, batch_real).to(self.device)
                img_syn = image_syn[idx * images_per_class_int: (idx + 1) * images_per_class_int].reshape(
                    images_per_class_int, self.dataset_info["channel"], self.dataset_info["im_size"][0], self.dataset_info["im_size"][1]
                )

                with torch.no_grad():
                    output_real = net.embed(img_real).detach()

                output_syn = net.embed(img_syn)

                loss_c = torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)

                optimizer_img.zero_grad()
                loss_c.backward()
                optimizer_img.step()

                loss_avg += loss_c.item()

            loss_avg /= len(local_classes)

            if it % 100 == 0:
                print(f'iter = {it:04d}, loss = {loss_avg:.4f}')

    # Organize synthetic images into a dictionary by class
        image_by_class = {
            cls: image_syn[i * images_per_class_int: (i + 1) * images_per_class_int].detach().cpu()
            for i, cls in enumerate(local_classes)
        }

        return image_by_class  # Returns a dictionary organized by class
    
    def get_images(self, cls, n):
        indices = np.random.choice(self.class_indices[cls], size=n, replace=len(self.class_indices[cls]) < n)
        images = torch.stack([self.dst_train[i][0] for i in indices]).to(self.device)
        return images
    
    def compute_embeddings(self, local_classes, iterations):
        """
        Run multiple network initializations + embedding extraction for each class, saving the mean feature each time.
        Return format: { class: [embedding_0, embedding_1, ..., embedding_T] }
        """
        embeddings = {cls: [] for cls in local_classes}

        for it in range(iterations):
            print(f'Embedding iteration [{it+1}/{iterations}]')
    
        
            net = self.local_model
            net.train()
            for param in net.parameters():
                param.requires_grad = False

            embed_layer = net.module.embed if isinstance(net, torch.nn.DataParallel) else net.embed

        # Iterate through local classes, extract mean embeddings
            for cls in local_classes:
                img_real = self.get_images(cls, 256).to(self.device)  # Sample from class_indices
                with torch.no_grad():
                    feat = embed_layer(img_real).detach()

                mean_feat = torch.mean(feat, dim=0).cpu()  # Move to CPU
                embeddings[cls].append(mean_feat)

           
        torch.cuda.empty_cache()
    
        return embeddings
    

    def send_data_to_server(self):
        """
        Package and send key data generated in process() to the server.
        Includes:
        - Synthetic images (organized by class)
        - Multi-round embedding means for each class
        """
    # Create a dictionary to store data to be transmitted
        data_sent = {}

    # Add synthetic image data
        self.logger.info(f"Client {self.cid} starting to prepare synthetic images to send to server...")
        data_sent["synthetic_images"] = self.client_syn  # {class: tensor}

    # Add multi-round embedding data
        self.logger.info(f"Client {self.cid} starting to prepare multi-round embedding features to send to server...")
        data_sent["client_embeddings"] = self.client_embedding  # {class: [tensor, tensor, ...]}


    # Log completion of data preparation
        self.logger.info(f"Client {self.cid} data preparation completed, ready to send to server.")

        return data_sent