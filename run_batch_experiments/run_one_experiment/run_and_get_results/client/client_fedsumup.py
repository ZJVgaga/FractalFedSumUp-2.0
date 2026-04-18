import copy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import sys
import os

# Import visualization system
try:
    from ..visualization_system.image_tracker import ImageTracker
    from ..visualization_system.client_visualizer import ClientVisualizer
    VISUALIZATION_SYSTEM_AVAILABLE = True
except ImportError:
    VISUALIZATION_SYSTEM_AVAILABLE = False
    print("Warning: Visualization system unavailable")

class Client:
    def __init__(self, client_modules, config, logger, i):
        """
        Client initialization function (new process version)
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
        self.logger.info(f"Initializing client {i} (new process version)...")
        
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
        self.local_model = copy.deepcopy(client_modules["classifier"])  # Global model
        
        # Federated learning strategy-related initialization
        self.fed_strategy = config.get("Federated_Learning_Config")  # Get federated learning strategy
        self.logger.info(f"Client {i} uses federated strategy: {self.fed_strategy}")
        
        # If using FedSumUp strategy, initialize specific parameters
        if self.fed_strategy == "FedSumUp":
            self.num_crop = config.get("sumup_num_crop", 5)
            self.input_size = self.dataset_info["im_size"][0]
        
            self.sumup_iterations = config.get("sumup_iterations", 100)
            self.foulier_alpha = config.get("sumup_foulier_alpha", 0.5)
            self.logger.info(f"Client {i} FedSumUp parameter initialization completed")
        
        # Device-related initialization
        self.device = client_modules['device']
        
        # Save client_modules for future use
        self.client_modules = client_modules
        
        # Initialize state variables
        self.current_round = 0
        self.latent_variables = None
        self.labels = None
        
        # Log completion of client initialization
        self.logger.info(f"Client {i} initialization completed")
        
    def receive_data_from_server(self, server_data):
        """
        Receive data from the server
        New process: Receive encoder, current round, and total rounds information
        """
        self.current_round = server_data.get("current_round", 0)
        self.total_rounds = server_data.get("total_rounds", 1)
        self.logger.info(f"Client {self.cid} received server data for round {self.current_round}/{self.total_rounds}")
        
        # Determine if it's the final round
        self.is_final_round = (self.current_round == self.total_rounds - 1)
        if self.is_final_round:
            self.logger.info(f"Client {self.cid} This is the final round, will upload compressed images")
        
        # Receive encoder
        if "encoder" in server_data:
            self.encoder = copy.deepcopy(server_data["encoder"]).to(self.device)
            self.logger.info(f"Client {self.cid} received encoder")
        else:
            self.logger.warning(f"Client {self.cid} did not receive encoder")
        
        return

    def process(self):
        """
        Client processing flow (new requirement version)
        
        New requirements (based on user description):
        1. Client no longer trains encoder, only calls encoder for compression
        2. Utility extractor V0 (essentially a lossy compressor) reads original data pairs (D0, L0)
        3. V0 compresses original data into mixed information representation I0 and its corresponding label L0
        4. Client uploads mixed information pairs (I0, L0) to the server
        """
        self.logger.info(f"Client {self.cid} starting round {self.current_round} processing (new process version)")
        
        # Check if encoder exists
        if not hasattr(self, 'encoder'):
            raise ValueError(f"Client {self.cid} lacks encoder, cannot perform compression")
        
        # No longer train encoder, only call encoder for compression
        self.logger.info(f"Client {self.cid} starting image compression (round {self.current_round})...")
        self.compress_images()
        
        self.logger.info(f"Client {self.cid} processing flow completed")
        return
    
    def compress_images(self):
        """
        Compress images (new requirement version)
        
        Based on user description:
        1. Utility extractor V0 (essentially a lossy compressor) reads original data pairs (D0, L0)
        2. V0 compresses original data into mixed information representation I0 and its corresponding label L0
        3. In I0, utility information accounts for u%, privacy information accounts for (1-u)%
        
        Implementation:
        1. Use the received encoder V0 to compress all images
        2. Save compressed images I0 and corresponding labels L0
        """
        self.logger.info(f"Client {self.cid} starting image compression...")
        
        # Set encoder to evaluation mode
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Get utility information ratio u% from configuration (default 50%)
        utility_ratio = self.config.get("utility_ratio", 0.5)
        self.logger.info(f"Client {self.cid} utility information ratio: {utility_ratio*100:.1f}%, privacy information ratio: {(1-utility_ratio)*100:.1f}%")
        
        # Group image indices by class
        class_indices = {}
        for idx in self.client_indices:
            _, label = self.dst_train[idx]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # Compress all images
        compressed_images = []
        labels = []
        selected_indices = []
        
        for class_label, indices in class_indices.items():
            for idx in indices:
                # Get original image data pair (D0, L0)
                image_tensor, label = self.dst_train[idx]
                
                # Use encoder V0 to compress image, obtaining mixed information representation I0
                compressed_image = self.compress_image(image_tensor)
                
                compressed_images.append(compressed_image)
                labels.append(label)
                selected_indices.append(idx)
        
        # Save compression results
        self.compressed_images = torch.stack(compressed_images) if compressed_images else None
        self.compressed_labels = torch.tensor(labels) if labels else None
        self.compressed_indices = selected_indices
        
        self.logger.info(f"Client {self.cid} compressed {len(compressed_images)} images")
        
        # If visualization system is available, perform visualization
        if VISUALIZATION_SYSTEM_AVAILABLE:
            try:
                self._visualize_compressed_images(compressed_images, labels, selected_indices)
            except Exception as e:
                self.logger.warning(f"Client visualization failed: {str(e)}")
        
        return
    
    def _visualize_compressed_images(self, compressed_images, labels, selected_indices):
        """
        Visualize compressed images
        
        Parameters:
        - compressed_images: List of compressed images
        - labels: List of labels
        - selected_indices: List of selected indices
        """
        try:
            from ..visualization_system.image_tracker import ImageTracker
            from ..visualization_system.client_visualizer import ClientVisualizer
            
            # Create ImageTracker
            # Get train_indices and train_labels from client_modules
            train_indices = self.client_modules.get('target_train_indices', [])
            train_labels = self.client_modules.get('target_train_labels', [])
            
            if len(train_indices) == 0 or len(train_labels) == 0:
                self.logger.warning("target_train_indices or target_train_labels not found in client_modules, using dummy data")
                # Create dummy train_indices and train_labels
                train_indices = list(range(len(self.client_indices)))
                train_labels = [self.dst_train[idx][1] for idx in self.client_indices]
            
            image_tracker = ImageTracker(
                self.config, 
                self.dataset_info, 
                train_indices, 
                train_labels
            )
            
            # Create ClientVisualizer
            client_visualizer = ClientVisualizer(image_tracker, self.config)
            
            # Convert compressed images to tensor
            compressed_tensor = torch.stack(compressed_images) if isinstance(compressed_images, list) else compressed_images
            
            # Get original images
            original_images = []
            for idx in selected_indices:
                image_tensor, _ = self.dst_train[idx]
                original_images.append(image_tensor)
            original_tensor = torch.stack(original_images)
            
            # Save visualization results (only save images, do not generate PDF)
            client_visualizer.save_client_visualization(
                original_images=original_tensor,
                compressed_images=compressed_tensor,
                labels=torch.tensor(labels),
                client_indices=selected_indices,
                client_id=self.cid,
                round_num=self.current_round,
                dataset=self.dst_train
            )
            
            self.logger.info(f"Client {self.cid} visualization completed, saved visualization results for {len(selected_indices)} images")
            
            # Note: PDF generation is now handled in run_and_get_results.py, not in the client
            
        except Exception as e:
            self.logger.warning(f"Client visualization system initialization failed: {str(e)}")
            import traceback
            self.logger.debug(f"Detailed error information: {traceback.format_exc()}")
    
    def compress_image(self, image_tensor):
        """
        Compress a single image
        """
        # Move image to device and add batch dimension
        image_batch = image_tensor.unsqueeze(0).to(self.device)
        
        # Use encoder to compress image
        with torch.no_grad():
            compressed = self.encoder.encode(image_batch)
        
        # Remove batch dimension
        compressed = compressed.squeeze(0)
        
        return compressed
    
    def send_data_to_server(self):
        """
        Send client data to the server
        New requirements (based on user description):
        1. Send compressed mixed information representation I0 and its corresponding label L0
        2. No longer send trained encoder parameters (because client no longer trains encoder)
        """
        data_sent = {}
        data_sent["client_id"] = self.cid
        data_sent["current_round"] = self.current_round
        
        # Send compressed images (mixed information representation I0) and labels L0
        if hasattr(self, 'compressed_images') and self.compressed_images is not None:
            data_sent["latent_variables"] = self.compressed_images.cpu()
            data_sent["labels"] = self.compressed_labels.cpu()
            data_sent["client_indices"] = self.compressed_indices
            self.logger.info(f"Client {self.cid} preparing to send {len(self.compressed_images)} compressed images (mixed information representation I0) and labels L0")
            
            # Send original image data for visualization (optional)
            if hasattr(self, 'compressed_indices') and self.compressed_indices:
                original_images = []
                for idx in self.compressed_indices:
                    image_tensor, _ = self.dst_train[idx]
                    original_images.append(image_tensor)
                
                if original_images:
                    data_sent["original_images"] = torch.stack(original_images).cpu()
                    self.logger.info(f"Client {self.cid} simultaneously sending {len(original_images)} original images for visualization")
        else:
            self.logger.warning(f"Client {self.cid} has no compressed images to send")
        
        return data_sent
    
    def get_visualization_data_paths(self):
        """
        Get paths to client-saved visualization data
        Returns a list of paths to all image data saved by the client
        """
        if not VISUALIZATION_SYSTEM_AVAILABLE:
            return []
        
        try:
            # Get save path
            base_dir = self.config.get("visualization_save_dir", "./fedsumup_visualizations")
            dataset_name = self.config.get("Dataset", "CIFAR10")
            compressed_size = self.config.get("compressed_image_size", 24)
            
            # PDF data directory
            pdf_data_dir = os.path.join(
                base_dir,
                f"fedsumup_visualization_{dataset_name}",
                f"compressed_{compressed_size}",
                "pdf_data"
            )
            
            # Check if directory exists
            if not os.path.exists(pdf_data_dir):
                return []
            
            # Collect all data files for this client
            data_files = []
            for filename in os.listdir(pdf_data_dir):
                if filename.endswith(".pkl"):
                    data_files.append(os.path.join(pdf_data_dir, filename))
            
            return data_files
            
        except Exception as e:
            self.logger.warning(f"Client {self.cid} failed to get visualization data paths: {str(e)}")
            return []