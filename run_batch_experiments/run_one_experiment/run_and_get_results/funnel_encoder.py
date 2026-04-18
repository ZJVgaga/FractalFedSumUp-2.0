import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import math

class ConvPoolBlock(nn.Module):
    """
    Enhanced Convolution-Pooling Suite: Output size halved
    
    Enhanced design: Uses multiple convolutional layers to increase network depth and parameter count, improving performance
    Structure: Conv1 → Norm → ReLU → Conv2 → Norm → ReLU → MaxPool
    """
    def __init__(self, in_channels, out_channels, use_norm=True, use_act=True, num_convs=2,target_size=None):
        super().__init__()
        layers = []
        
        # First convolutional layer: Different input channels
        current_channels = in_channels
        for i in range(num_convs):
            # First convolution uses in_channels, subsequent ones use out_channels
            if i == 0:
                layers.append(nn.Conv2d(current_channels, out_channels, 
                                        kernel_size=7, stride=1, padding=3))
                current_channels = out_channels
            elif i == 1:
                layers.append(nn.Conv2d(current_channels, current_channels, 
                                        kernel_size=5, stride=1, padding=2)) 

            elif i == 2:
                layers.append(nn.Conv2d(current_channels, current_channels, 
                                        kernel_size=5, stride=1, padding=2)) 
            

            elif i == 3:
                layers.append(nn.Conv2d(current_channels, current_channels, 
                                        kernel_size=3, stride=1, padding=1)) 
            

            else:
                layers.append(nn.Conv2d(current_channels, current_channels, 
                                        kernel_size=3, stride=1, padding=1)) 
            
            # Normalization
            if use_norm:
                layers.append(nn.GroupNorm(current_channels, current_channels, affine=True))
            
            # Activation
            if use_act:
                layers.append(nn.ReLU(inplace=True))
        
        # ========== Adaptive Pooling (Replaces stride convolution) ==========
        if target_size is not None:
            layers.append(nn.AdaptiveAvgPool2d((target_size, target_size)))
        
        self.block = nn.Sequential(*layers)
        self.num_convs = num_convs
    
    def forward(self, x):
        return self.block(x)

class FunnelEncoder(nn.Module):
    def __init__(self, 
                 input_channels: int = 3,
                 input_size: int = 32,
                 output_size: int = 8,
                 num_classes: int = 10,
                 hidden_dim: int = 128,
                 compress_to_rgb: bool = True,
                 num_convs_per_block: int = 2):
        super().__init__()
        
        self.output_size = output_size
        self.latent_dim = hidden_dim
        self.compress_to_rgb = compress_to_rgb
        self.input_size = input_size
        self.num_convs_per_block = num_convs_per_block
        
        print(f"\n[FunnelEncoder] Input size {input_size} → Output size {output_size}")
        print(f"  Convolutional layers per block: {num_convs_per_block}")
        print(f"  Hidden layer channels: {hidden_dim}")
        
        # ========== Simplification: Only one ConvPoolBlock needed, using adaptive pooling to control output size ==========
        self.encoder = ConvPoolBlock(
            in_channels=input_channels,
            out_channels=hidden_dim,
            use_norm=True,
            use_act=True,
            num_convs=num_convs_per_block,
            target_size=output_size  # 🔑 Directly specify target size
        )
        
        # Compress to RGB
        if compress_to_rgb:
            self.rgb_compressor = nn.Conv2d(hidden_dim, 3, kernel_size=1)
        
        # Channel attention
        self.use_attention = True
        if self.use_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(hidden_dim, max(hidden_dim // 4, 4)),
                nn.ReLU(inplace=True),
                nn.Linear(max(hidden_dim // 4, 4), hidden_dim),
                nn.Sigmoid()
            )
        
        self.final_channels = hidden_dim
        
        # Verify output size
        with torch.no_grad():
            was_training = self.training
            self.eval()
            
            try:
                dummy = torch.randn(2, input_channels, input_size, input_size)
                dummy_out = self.encoder(dummy)
                actual_out_size = dummy_out.shape[-1]
                print(f"  Actual output size: {actual_out_size}×{actual_out_size}")
                
                if actual_out_size != output_size:
                    print(f"  Warning: Actual output size {actual_out_size} ≠ Target output size {output_size}")
            finally:
                if was_training:
                    self.train()
        
        # Print network structure
        print(f"  Final channel count: {hidden_dim}")
        print(f"  Encoder layers: 1 ConvPoolBlock")
        print(f"  Actual output size: {actual_out_size if 'actual_out_size' in locals() else output_size}")
    
    def encode(self, x: torch.Tensor, output_size: int = None) -> torch.Tensor:
        """Encoder forward propagation"""
        features = self.encoder(x)
        
        # Channel attention
        if self.use_attention:
            attention_weights = self.channel_attention(features)
            features = features * attention_weights.view(-1, self.final_channels, 1, 1)
        
        # Compress to RGB
        if self.compress_to_rgb:
            output = torch.tanh(self.rgb_compressor(features))
        else:
            output = features
        
        return output
    
    def forward(self, x: torch.Tensor, output_size: int = None) -> torch.Tensor:
        """Forward propagation (same as encode)"""
        return self.encode(x, output_size)
    
    def get_info_bottleneck_size(self, output_size: int = None) -> int:
        """Returns total information amount I (channels × height × width)"""
        out_sz = output_size if output_size is not None else self.output_size
        channels = 3 if self.compress_to_rgb else self.final_channels
        return channels * out_sz * out_sz
    
    def get_supported_output_sizes(self) -> list:
        """Returns list of supported output sizes (any positive integer, adaptive pooling supports all)"""
        # Adaptive pooling supports any output size, return common values
        supported = [1, 2, 4, 8, 16, 32]
        return [sz for sz in supported if sz <= self.input_size]
    
    def visualize_conv_parameters(self, save_dir=None, show_plots=False):
        """Visualize convolutional layer parameter values (unchanged)"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Collect all convolutional layers
        conv_layers = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))
        
        if not conv_layers:
            print("No convolutional layers found")
            return {}
        
        print(f"\n[FunnelEncoder] Found {len(conv_layers)} convolutional layers")
        
        stats = {}
        
        # Create visualization for each convolutional layer
        for i, (layer_name, conv_layer) in enumerate(conv_layers):
            print(f"  Visualizing layer {i+1}: {layer_name}")
            print(f"    Weight shape: {conv_layer.weight.shape}")
            print(f"    Bias shape: {conv_layer.bias.shape if conv_layer.bias is not None else 'No bias'}")
            
            # Get weight data
            weights = conv_layer.weight.detach().cpu().numpy()
            bias = conv_layer.bias.detach().cpu().numpy() if conv_layer.bias is not None else None
            
            # Calculate statistics
            weight_stats = {
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'min': float(np.min(weights)),
                'max': float(np.max(weights)),
                'shape': list(weights.shape)
            }
            
            if bias is not None:
                bias_stats = {
                    'mean': float(np.mean(bias)),
                    'std': float(np.std(bias)),
                    'min': float(np.min(bias)),
                    'max': float(np.max(bias)),
                    'shape': list(bias.shape)
                }
            else:
                bias_stats = None
            
            stats[layer_name] = {
                'weights': weight_stats,
                'bias': bias_stats
            }
            
            # Create visualization chart
            fig = self._create_conv_visualization_figure(
                weights, bias, layer_name, i+1, conv_layer
            )
            
            # Save chart
            if save_dir:
                save_path = os.path.join(save_dir, f"conv_layer_{i+1}_{layer_name.replace('.', '_')}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"    Chart saved to: {save_path}")
            
            # Display chart
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
        
        # Create summary chart
        self._create_summary_visualization(stats, save_dir, show_plots)
        
        return stats
    
    def _create_conv_visualization_figure(self, weights, bias, layer_name, layer_idx, conv_layer):
        """Create visualization chart for a single convolutional layer (unchanged)"""
        # Weight shape: [out_channels, in_channels, kernel_h, kernel_w]
        out_channels, in_channels, kernel_h, kernel_w = weights.shape
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'Convolutional Layer Visualization: {layer_name} (Layer {layer_idx})', fontsize=16, y=0.98)
        
        # 1. Weight distribution histogram
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(weights.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Weight Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Weight Distribution\nShape: {weights.shape}')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'Mean: {np.mean(weights):.4f}\nStd: {np.std(weights):.4f}\nMin: {np.min(weights):.4f}\nMax: {np.max(weights):.4f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Bias distribution histogram (if exists)
        ax2 = plt.subplot(2, 3, 2)
        if bias is not None:
            ax2.hist(bias.flatten(), bins=30, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Bias Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Bias Distribution\nShape: {bias.shape}')
            bias_stats_text = f'Mean: {np.mean(bias):.4f}\nStd: {np.std(bias):.4f}'
            ax2.text(0.02, 0.98, bias_stats_text, transform=ax2.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        else:
            ax2.text(0.5, 0.5, 'No Bias', ha='center', va='center', fontsize=14)
            ax2.set_title('Bias Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Kernel visualization (show first few kernels)
        ax3 = plt.subplot(2, 3, 3)
        # Select first few input channels of the first output channel
        num_kernels_to_show = min(4, in_channels)
        kernel_grid = np.zeros((kernel_h * num_kernels_to_show, kernel_w))
        
        for k in range(num_kernels_to_show):
            kernel = weights[0, k, :, :]  # First output channel, k-th input channel
            row_start = k * kernel_h
            row_end = (k + 1) * kernel_h
            kernel_grid[row_start:row_end, :] = kernel
        
        im = ax3.imshow(kernel_grid, cmap='RdBu_r', aspect='auto')
        ax3.set_title(f'Kernel Examples (Output Channel 0, First {num_kernels_to_show} Input Channels)')
        ax3.set_xlabel('Width')
        ax3.set_ylabel(f'Height × {num_kernels_to_show}')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. Output channel weight mean heatmap
        ax4 = plt.subplot(2, 3, 4)
        # Calculate weight mean for each output channel
        channel_means = np.mean(weights, axis=(1, 2, 3))
        if len(channel_means) > 1:
            # Create heatmap
            heatmap_data = channel_means.reshape(-1, 1)
            im2 = ax4.imshow(heatmap_data, cmap='viridis', aspect='auto')
            ax4.set_title(f'Output Channel Weight Mean Heatmap\n({out_channels} channels)')
            ax4.set_xlabel('')
            ax4.set_ylabel('Output Channel Index')
            ax4.set_yticks(range(out_channels))
            ax4.set_yticklabels(range(out_channels))
            plt.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)
        else:
            ax4.text(0.5, 0.5, f'Single Output Channel\nMean: {channel_means[0]:.4f}', 
                    ha='center', va='center', fontsize=12)
            ax4.set_title('Output Channel Weight Mean')
        
        # 5. Input channel weight mean heatmap
        ax5 = plt.subplot(2, 3, 5)
        # Calculate weight mean for each input channel
        input_means = np.mean(weights, axis=(0, 2, 3))
        if len(input_means) > 1:
            heatmap_data = input_means.reshape(-1, 1)
            im3 = ax5.imshow(heatmap_data, cmap='plasma', aspect='auto')
            ax5.set_title(f'Input Channel Weight Mean Heatmap\n({in_channels} channels)')
            ax5.set_xlabel('')
            ax5.set_ylabel('Input Channel Index')
            ax5.set_yticks(range(in_channels))
            ax5.set_yticklabels(range(in_channels))
            plt.colorbar(im3, ax=ax5, fraction=0.046, pad=0.04)
        else:
            ax5.text(0.5, 0.5, f'Single Input Channel\nMean: {input_means[0]:.4f}', 
                    ha='center', va='center', fontsize=12)
            ax5.set_title('Input Channel Weight Mean')
        
        # 6. Convolutional layer information
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        info_text = (
            f'Convolutional Layer Info:\n'
            f'• Name: {layer_name}\n'
            f'• Index: {layer_idx}\n'
            f'• Input Channels: {in_channels}\n'
            f'• Output Channels: {out_channels}\n'
            f'• Kernel Size: {kernel_h}×{kernel_w}\n'
            f'• Stride: {conv_layer.stride}\n'
            f'• Padding: {conv_layer.padding}\n'
            f'• Bias: {"Yes" if conv_layer.bias is not None else "No"}\n'
            f'• Total Parameters: {conv_layer.weight.numel() + (conv_layer.bias.numel() if conv_layer.bias is not None else 0):,}'
        )
        ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
    
    def _create_summary_visualization(self, stats, save_dir, show_plots):
        """Create summary visualization for all convolutional layers (unchanged)"""
        if not stats:
            return
        
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle('FunnelEncoder Convolutional Layers Parameter Summary', fontsize=16, y=0.98)
        
        # 1. Weight mean comparison across layers
        ax1 = plt.subplot(2, 2, 1)
        layer_names = list(stats.keys())
        weight_means = [stats[name]['weights']['mean'] for name in layer_names]
        weight_stds = [stats[name]['weights']['std'] for name in layer_names]
        
        x_pos = range(len(layer_names))
        bars = ax1.bar(x_pos, weight_means, yerr=weight_stds, capsize=5, 
                      alpha=0.7, color='