import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
from .image_tracker import ImageTracker


class ServerVisualizer:
    """
    Server Visualizer: Responsible for server-side visualization
    """
    def __init__(self, image_tracker, config):
        """
        Initialize the server visualizer
        
        Parameters:
        - image_tracker: Image tracker instance
        - config: Configuration dictionary
        """
        self.image_tracker = image_tracker
        self.config = config
        self.compressed_size = config.get("compressed_image_size", 24)
        
        # Store visualization data for each epoch
        self.epoch_visualizations = {}
    
    def add_epoch_visualization(self, epoch, tracked_images_data):
        """
        Add visualization data for one epoch
        
        Parameters:
        - epoch: Epoch number
        - tracked_images_data: Dictionary containing tracked image data
            {index: {"original": original_tensor, "compressed": compressed_tensor, "class_id": class_id}}
        """
        self.epoch_visualizations[epoch] = tracked_images_data
    
    def save_server_visualization(self, round_num):
        """
        Save visualization results for all epochs on the server side
        
        Parameters:
        - round_num: Round number
        """
        if not self.epoch_visualizations:
            print(f"No visualization data for round {round_num} on server side")
            return
        
        # For each tracked image
        tracked_indices = self.image_tracker.get_tracked_indices()
        
        for index, class_id in tracked_indices.items():
            # Collect data for this image across all epochs
            all_epochs_data = []
            
            for epoch in sorted(self.epoch_visualizations.keys()):
                epoch_data = self.epoch_visualizations[epoch]
                if index in epoch_data:
                    all_epochs_data.append((epoch, epoch_data[index]))
            
            if not all_epochs_data:
                continue
            
            # Save visualization results for this image
            self._save_single_image_server_visualization(
                index, class_id, round_num, all_epochs_data
            )
        
        # Clear cache for current round
        self.epoch_visualizations.clear()
    
    def _save_single_image_server_visualization(self, index, class_id, round_num, all_epochs_data):
        """
        Save server-side visualization results for a single image
        
        Parameters:
        - index: Image index
        - class_id: Class ID
        - round_num: Round number
        - all_epochs_data: List containing (epoch, data) tuples
        """
        # Get original image dimensions (from first epoch's data)
        first_epoch_data = all_epochs_data[0][1]
        orig_img = first_epoch_data["original"]
        orig_h, orig_w = orig_img.shape[2], orig_img.shape[3]
        
        # Prepare image data for all rows
        all_rows_images = []
        all_rows_titles = []
        
        # For each epoch
        for epoch, data in all_epochs_data:
            orig_img_epoch = data["original"]
            comp_img_epoch = data["compressed"]
            
            # Get compressed image dimensions
            comp_h, comp_w = comp_img_epoch.shape[2], comp_img_epoch.shape[3]
            
            # Create three visualization images
            # 1. Original image upsampled to 32x32 (for display) - using nearest neighbor interpolation
            upsampled_bilinear = torch.nn.functional.interpolate(
                orig_img_epoch,
                size=(32, 32),
                mode='bilinear',
                align_corners=False  # Nearest neighbor interpolation typically doesn't need align_corners
            )
            
            # 2. Encoder compressed image (maintain compressed size)
            compressed = comp_img_epoch.clone()
            
            # 3. Encoder compressed image upsampled to 32x32 - using nearest neighbor interpolation
            compressed_upsample = torch.nn.functional.interpolate(
                comp_img_epoch,
                size=(32, 32),
                mode='bilinear',
                align_corners=False  # Nearest neighbor interpolation typically doesn't need align_corners
            )
            
            # Add to row data
            all_rows_images.extend([upsampled_bilinear, compressed, compressed_upsample])
            all_rows_titles.extend([
                f"Epoch {epoch}: Bilinear Upsample\nSize: 32x32",
                f"Epoch {epoch}: Encoder Compressed\nSize: {comp_h}x{comp_w}",
                f"Epoch {epoch}: Compressed Upsample\nSize: 32x32"
            ])
        
        # Calculate total number of rows (one row per epoch)
        num_epochs = len(all_epochs_data)
        
        # Create figure (multiple rows, three columns)
        fig, axes = plt.subplots(num_epochs, 3, figsize=(15, 5 * num_epochs))
        
        # If only one epoch, adjust axes shape
        if num_epochs == 1:
            axes = [axes]
        
        # Fill images for each epoch
        for row in range(num_epochs):
            for col in range(3):
                ax = axes[row][col] if num_epochs > 1 else axes[col]
                img_idx = row * 3 + col
                
                if img_idx < len(all_rows_images):
                    img = all_rows_images[img_idx]
                    title = all_rows_titles[img_idx]
                    
                    # Process image data
                    if isinstance(img, torch.Tensor):
                        img = img.detach().cpu().numpy()
                    
                    # Adjust image dimension order
                    if len(img.shape) == 4:  # [batch, channels, height, width]
                        img = img[0]  # Take first image
                    
                    if len(img.shape) == 3:
                        if img.shape[0] == 1:  # Single channel image
                            img = img[0]  # [height, width]
                        elif img.shape[0] == 3:  # RGB image
                            img = img.transpose(1, 2, 0)  # [height, width, channels]
                    
                    # Display image
                    if len(img.shape) == 2:  # Grayscale image
                        ax.imshow(img, cmap='gray')
                    else:  # RGB image
                        # Normalize to [0, 1] range
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                        ax.imshow(img)
                    
                    ax.set_title(title, fontsize=10)
                    ax.axis('off')
        
        # Only save the overall summary image, not individual epoch images
        # Get save path (without using epoch parameter)
        save_dir = self.image_tracker.get_save_path(index, class_id, round_num, is_server=True)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save summary image
        save_path = os.path.join(save_dir, f"server_all_epochs.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save summary metadata
        metadata = {
            "index": int(index),
            "class_id": int(class_id),
            "round": int(round_num),
            "num_epochs": num_epochs,
            "original_size": f"{orig_h}x{orig_w}",
            "target_compressed_size": f"{self.compressed_size}x{self.compressed_size}",
            "epochs": [epoch for epoch, _ in all_epochs_data],
            "timestamp": str(np.datetime64('now'))
        }
        
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Server-side visualization results for index {index} (class {class_id}) saved to: {save_dir}")