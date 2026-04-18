import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
from PIL import Image as PILImage
from matplotlib.backends.backend_pdf import PdfPages
from .image_tracker import ImageTracker


class ClientVisualizer:
    """
    Client Visualizer: Responsible for client-side visualization
    """
    def __init__(self, image_tracker, config):
        """
        Initialize the client visualizer
        
        Parameters:
        - image_tracker: Image tracker instance
        - config: Configuration dictionary
        """
        self.image_tracker = image_tracker
        self.config = config
        self.compressed_size = config.get("compressed_image_size", 24)
    
    def save_client_visualization(self, original_images, compressed_images, labels,
                                 client_indices, client_id, round_num, dataset):
        """
        Save client visualization results
        
        Parameters:
        - original_images: Original images [batch, channels, height, width]
        - compressed_images: Compressed images [batch, channels, comp_h, comp_w]
        - labels: Labels
        - client_indices: List of client data indices
        - client_id: Client ID
        - round_num: Round number
        - dataset: Dataset object (used to get images for specific indices)
        """
        # Get the images this client is responsible for tracking
        responsibility = self.image_tracker.get_client_responsibility(client_indices)
        
        if not responsibility:
            print(f"Client {client_id} has no images to track")
            return
        
        print(f"Client {client_id} is responsible for tracking {len(responsibility)} images: {list(responsibility.keys())}")
        
        # Visualize each responsible image
        for idx, class_id in responsibility.items():
            # Find the position of this image in the client's data
            if idx not in client_indices:
                continue
            
            position_in_client = client_indices.index(idx)
            
            # Get the data for this image
            if position_in_client < len(original_images):
                orig_img = original_images[position_in_client:position_in_client+1]
                comp_img = compressed_images[position_in_client:position_in_client+1]
                label = labels[position_in_client:position_in_client+1]
                
                # Save visualization results
                self._save_single_image_visualization(
                    orig_img, comp_img, label, idx, class_id, 
                    client_id, round_num
                )
    
    def _save_single_image_visualization(self, orig_img, comp_img, label, 
                                        index, class_id, client_id, round_num):
        """
        Save visualization results for a single image
        
        Parameters:
        - orig_img: Original image [1, channels, height, width]
        - comp_img: Compressed image [1, channels, comp_h, comp_w]
        - label: Label [1]
        - index: Image index
        - class_id: Class ID
        - client_id: Client ID
        - round_num: Round number
        """
        # Get save path
        save_dir = self.image_tracker.get_save_path(index, class_id, round_num, is_server=False)
        os.makedirs(save_dir, exist_ok=True)
        
        # Get image dimensions
        orig_h, orig_w = orig_img.shape[2], orig_img.shape[3]
        comp_h, comp_w = comp_img.shape[2], comp_img.shape[3]
        
        # Create four visualization images
        # 1. Original image
        orig_display = orig_img.clone()
        
        # 2. Image directly downscaled to encoder output size (maintaining compressed dimensions) - using nearest neighbor interpolation
        orig_downsampled = torch.nn.functional.interpolate(
            orig_img,
            size=(comp_h, comp_w),
            mode='nearest'
            # Note: nearest neighbor interpolation does not support align_corners parameter
        )
        
        # 3. Encoder compressed image (maintaining compressed dimensions)
        comp_display = comp_img.clone()
        
        # Create image grid (single row, four columns)
        images_to_save = [orig_display, orig_downsampled, comp_display]
        titles = [
            f"Original\nSize: {orig_h}x{orig_w}",
            f"Direct Downsample\nSize: {comp_h}x{comp_w}",
            f"Encoder Compressed\nSize: {comp_h}x{comp_w}"
        ]
        
        # Save images
        save_path = os.path.join(save_dir, f"client_visualization.png")
        self._save_image_grid(images_to_save, titles, save_path, nrow=3, figsize=(15, 5))
        
        # Save metadata
        metadata = {
            "index": int(index),
            "class_id": int(class_id),
            "client_id": int(client_id),
            "round": int(round_num),
            "original_size": f"{orig_h}x{orig_w}",
            "compressed_size": f"{comp_h}x{comp_w}",
            "target_compressed_size": f"{self.compressed_size}x{self.compressed_size}",
            "label": int(label[0].item()) if hasattr(label[0], 'item') else int(label[0]),
            "timestamp": str(np.datetime64('now'))
        }
        
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Client {client_id} saved visualization for index {index} (class {class_id}) to: {save_dir}")
        
        # Save compressed image for subsequent PDF generation
        self._save_compressed_image_for_pdf(orig_img, comp_img, index, class_id, round_num)
    
    def _save_image_grid(self, images, titles, save_path, nrow=3, figsize=(15, 5)):
        """
        Save image grid
        
        Parameters:
        - images: List of images
        - titles: List of titles
        - save_path: Save path
        - nrow: Number of images per row
        - figsize: Figure size
        """
        # Ensure save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create figure
        fig, axes = plt.subplots(1, len(images), figsize=figsize)
        if len(images) == 1:
            axes = [axes]
        
        for i, (img, title) in enumerate(zip(images, titles)):
            ax = axes[i]
            
            # Process image data
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            
            # Adjust image dimension order
            if len(img.shape) == 4:  # [batch, channels, height, width]
                img = img[0]  # Take the first image
            
            if len(img.shape) == 3:
                if img.shape[0] == 1:  # Single-channel image
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
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _save_compressed_image_for_pdf(self, orig_img, comp_img, index, class_id, round_num):
        """
        Save compressed image as JPG file for subsequent PDF generation
        
        Parameters:
        - orig_img: Original image [1, channels, height, width]
        - comp_img: Compressed image [1, channels, comp_h, comp_w]
        - index: Image index
        - class_id: Class ID
        - round_num: Round number
        """
        # Get save path
        base_dir = self.config.get("visualization_save_dir", "./fedsumup_visualizations")
        dataset_name = self.config.get("Dataset", "CIFAR10")
        compressed_size = self.compressed_size
        
        # Create JPG image save directory
        jpg_data_dir = os.path.join(
            base_dir,
            f"fedsumup_visualization_{dataset_name}",
            f"compressed_{compressed_size}",
            "jpg_data"
        )
        os.makedirs(jpg_data_dir, exist_ok=True)
        
        # Save original image as JPG
        orig_img_np = orig_img.detach().cpu().numpy()
        if len(orig_img_np.shape) == 4:
            orig_img_np = orig_img_np[0]  # Take the first image
        
        # Adjust image dimension order
        if len(orig_img_np.shape) == 3:
            if orig_img_np.shape[0] == 1:  # Single-channel image
                orig_img_np = orig_img_np[0]  # [height, width]
            elif orig_img_np.shape[0] == 3:  # RGB image
                orig_img_np = orig_img_np.transpose(1, 2, 0)  # [height, width, channels]
        
        # Normalize to [0, 255] range
        if orig_img_np.max() > orig_img_np.min():
            orig_img_np = (orig_img_np - orig_img_np.min()) / (orig_img_np.max() - orig_img_np.min() + 1e-8) * 255
        orig_img_np = orig_img_np.astype(np.uint8)
        
        # Save original image
        orig_img_path = os.path.join(jpg_data_dir, f"{index}_{class_id}_round_{round_num}_original.jpg")
        if len(orig_img_np.shape) == 2:  # Grayscale image
            PILImage.fromarray(orig_img_np, mode='L').save(orig_img_path)
        else:  # RGB image
            PILImage.fromarray(orig_img_np, mode='RGB').save(orig_img_path)
        
        # Save compressed image as JPG
        comp_img_np = comp_img.detach().cpu().numpy()
        if len(comp_img_np.shape) == 4:
            comp_img_np = comp_img_np[0]  # Take the first image
        
        # Adjust image dimension order
        if len(comp_img_np.shape) == 3:
            if comp_img_np.shape[0] == 1:  # Single-channel image
                comp_img_np = comp_img_np[0]  # [height, width]
            elif comp_img_np.shape[0] == 3:  # RGB image
                comp_img_np = comp_img_np.transpose(1, 2, 0)  # [height, width, channels]
        
        # Normalize to [0, 255] range
        if comp_img_np.max() > comp_img_np.min():
            comp_img_np = (comp_img_np - comp_img_np.min()) / (comp_img_np.max() - comp_img_np.min() + 1e-8) * 255
        comp_img_np = comp_img_np.astype(np.uint8)
        
        # Save compressed image
        comp_img_path = os.path.join(jpg_data_dir, f"{index}_{class_id}_round_{round_num}_compressed.jpg")
        if len(comp_img_np.shape) == 2:  # Grayscale image
            PILImage.fromarray(comp_img_np, mode='L').save(comp_img_path)
        else:  # RGB image
            PILImage.fromarray(comp_img_np, mode='RGB').save(comp_img_path)
        
        # Save downsampled image (second column) - using nearest neighbor interpolation
        # Get compressed image dimensions
        comp_h, comp_w = comp_img.shape[2], comp_img.shape[3]
        
        # Convert original image to PIL image
        if len(orig_img_np.shape) == 2:  # Grayscale image
            orig_pil = PILImage.fromarray(orig_img_np, mode='L')
            # Downsample to compressed dimensions
            downsampled_img = orig_pil.resize((comp_w, comp_h), PILImage.NEAREST)
            downsampled_np = np.array(downsampled_img)
        else:  # RGB image
            orig_pil = PILImage.fromarray(orig_img_np, mode='RGB')
            # Downsample to compressed dimensions
            downsampled_img = orig_pil.resize((comp_w, comp_h), PILImage.NEAREST)
            downsampled_np = np.array(downsampled_img)
        
        # Save downsampled image
        downsampled_img_path = os.path.join(jpg_data_dir, f"{index}_{class_id}_round_{round_num}_downsampled.jpg")
        if len(downsampled_np.shape) == 2:  # Grayscale image
            PILImage.fromarray(downsampled_np, mode='L').save(downsampled_img_path)
        else:  # RGB image
            PILImage.fromarray(downsampled_np, mode='RGB').save(downsampled_img_path)
        
        # Save metadata as JSON file
        metadata = {
            "index": int(index),
            "class_id": int(class_id),
            "round": int(round_num),
            "original_image_path": orig_img_path,
            "downsampled_image_path": downsampled_img_path,
            "compressed_image_path": comp_img_path,
            "original_size": f"{orig_img.shape[2]}x{orig_img.shape[3]}",
            "downsampled_size": f"{comp_h}x{comp_w}",
            "compressed_size": f"{comp_img.shape[2]}x{comp_img.shape[3]}"
        }
        
        metadata_path = os.path.join(jpg_data_dir, f"{index}_{class_id}_round_{round_num}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def generate_summary_pdf(self, total_rounds, data_files=None):
        """
        Generate summary PDF, all content on one page, each image for each class with numbering
        Load image data from JPG files
        
        Parameters:
        - total_rounds: Total number of rounds
        - data_files: Optional list of data file paths, if None load from default directory
        """
        # Get save path
        base_dir = self.config.get("visualization_save_dir", "./fedsumup_visualizations")
        dataset_name = self.config.get("Dataset", "CIFAR10")
        compressed_size = self.compressed_size
        
        # PDF output path
        pdf_output_path = os.path.join(
            base_dir,
            f"fedsumup_visualization_{dataset_name}",
            f"compressed_{compressed_size}",
            "summary_visualization.pdf"
        )
        
        # JPG data directory
        jpg_data_dir = os.path.join(
            base_dir,
            f"fedsumup_visualization_{dataset_name}",
            f"compressed_{compressed_size}",
            "jpg_data"
        )
        
        # Check if JPG data directory exists
        if not os.path.exists(jpg_data_dir):
            print(f"JPG data directory does not exist: {jpg_data_dir}")
            return
        
        # Collect all metadata JSON files
        metadata_files = []
        for filename in os.listdir(jpg_data_dir):
            if filename.endswith("_metadata.json"):
                metadata_files.append(os.path.join(jpg_data_dir, filename))
        
        if not metadata_files:
            print("No metadata files found")
            return
        
        # Organize data by class and index
        class_data = {}
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            index = metadata["index"]
            class_id = metadata["class_id"]
            round_num = metadata["round"]
            original_image_path = metadata["original_image_path"]
            downsampled_image_path = metadata["downsampled_image_path"]
            compressed_image_path = metadata["compressed_image_path"]
            
            if class_id not in class_data:
                class_data[class_id] = {}
            
            if index not in class_data[class_id]:
                class_data[class_id][index] = {}
            
            # Store image paths and metadata
            class_data[class_id][index][round_num] = {
                "metadata": metadata,
                "original_image_path": original_image_path,
                "downsampled_image_path": downsampled_image_path,
                "compressed_image_path": compressed_image_path
            }
        
        # Create PDF - all content on one page
        with PdfPages(pdf_output_path) as pdf:
            # Set total width to 90mm (convert to inches: 90mm / 25.4 = 3.543 inches)
            total_width_inch = 90 / 25.4
            
            # Get all classes
            all_classes = sorted(class_data.keys())
            if not all_classes:
                print("No class data found")
                return
            
            # Each class displays 2 images, each image has 4 types
            # Total columns: 8 columns (2 images × 4 types)
            n_cols = 8
            # Number of rows: number of classes
            n_rows = len(all_classes)
            
            # Calculate figure size
            # Total width fixed at 90mm (3.543 inches)
            # Height calculated based on number of rows
            fig_width = total_width_inch
            # Each row height is approximately 1/8 of