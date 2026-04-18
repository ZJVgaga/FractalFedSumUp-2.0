import os
import json


class ImageTracker:
    """
    Image Tracker: Responsible for selecting and managing images to be tracked
    """
    def __init__(self, config, dataset_info, train_indices, train_labels):
        """
        Initialize the image tracker
        
        Parameters:
        - config: Configuration dictionary
        - dataset_info: Dataset information
        - train_indices: List of training data indices
        - train_labels: List of training data labels
        """
        self.config = config
        self.dataset_info = dataset_info
        self.train_indices = train_indices
        self.train_labels = train_labels
        
        # Get dataset information
        self.dataset_name = config.get("Dataset", "CIFAR10")
        self.num_classes = dataset_info.get("num_classes", 10)
        self.compressed_size = config.get("compressed_image_size", 24)
        
        # Select images to track
        self.tracked_indices = self._select_tracked_images()
        
        # Print tracking information
        print(f"ImageTracker: Selected {len(self.tracked_indices)} images for tracking")
        for idx, class_id in self.tracked_indices.items():
            print(f"  - Index {idx}: Class {class_id}")
    
    def _select_tracked_images(self):
        """
        Select two images from train_indices for each class and obtain their indices
        Returns dictionary: {index: class_id}
        """
        tracked_indices = {}
        
        # Group indices by class
        class_to_indices = {}
        for idx, label in zip(self.train_indices, self.train_labels):
            class_to_indices.setdefault(label, []).append(idx)
        
        # Take the first two images for each class
        for class_id in range(self.num_classes):
            if class_id in class_to_indices and class_to_indices[class_id]:
                # Take the first two images of this class
                for i in range(min(2, len(class_to_indices[class_id]))):
                    selected_idx = class_to_indices[class_id][i]
                    tracked_indices[selected_idx] = class_id
        
        return tracked_indices
    
    def get_tracked_indices(self):
        """Get the tracked image indices"""
        return self.tracked_indices
    
    def get_client_responsibility(self, client_indices):
        """
        Check if a client is responsible for tracking certain images
        
        Parameters:
        - client_indices: List of client's data indices
        
        Returns:
        - Dictionary: {index: class_id}, images this client is responsible for tracking
        """
        responsibility = {}
        for idx, class_id in self.tracked_indices.items():
            if idx in client_indices:
                responsibility[idx] = class_id
        
        return responsibility
    
    def get_save_path(self, index, class_id, round_num, is_server=False, epoch=None):
        """
        Get the save path
        
        Parameters:
        - index: Image index
        - class_id: Class ID
        - round_num: Round number
        - is_server: Whether it's the server side
        - epoch: Server epoch (only needed for server side)
        
        Returns:
        - Save path
        """
        # Base path
        base_dir = self.config.get("visualization_save_dir", "./fedsumup_visualizations")
        
        # According to the required folder structure
        # fedsumup_visualization_{datasetname}/compressed_{size}/{index}_{class}/round/client or server_epochs
        save_dir = os.path.join(
            base_dir,
            f"fedsumup_visualization_{self.dataset_name}",
            f"compressed_{self.compressed_size}",
            f"{index}_{class_id}",
            f"round_{round_num}"
        )
        
        if is_server and epoch is not None:
            save_dir = os.path.join(save_dir, f"server_epoch_{epoch}")
        else:
            save_dir = os.path.join(save_dir, "client")
        
        return save_dir