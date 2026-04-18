import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import math

class ConvPoolBlock(nn.Module):
    """
    增强版卷积-池化套装：输出尺寸减半
    
    增强设计：使用多个卷积层增加网络深度和参数数量，提升性能
    结构：Conv1 → Norm → ReLU → Conv2 → Norm → ReLU → MaxPool
    """
    def __init__(self, in_channels, out_channels, use_norm=True, use_act=True, num_convs=2,target_size=None):
        super().__init__()
        layers = []
        
        # 第一个卷积层：输入通道数不同
        current_channels = in_channels
        for i in range(num_convs):
            # 第一个卷积使用 in_channels，后续使用 out_channels
            if i == 0:
                layers.append(nn.Conv2d(current_channels, out_channels, 
                                        kernel_size=7, stride=1, padding=3))
                current_channels = out_channels
            elif i == 1:
                layers.append(nn.Conv2d(current_channels, current_channels, 
                                        kernel_size=5, stride=1, padding=2)) #fashionmnist 17-8,cifar10 11-5，bloodmnist 第一层，21, stride=1, padding=10，后面几层7-1-3取得了很好的效果            

            elif i == 2:
                layers.append(nn.Conv2d(current_channels, current_channels, 
                                        kernel_size=5, stride=1, padding=2)) #fashionmnist 17-8,cifar10 11-5，bloodmnist 第一层，21, stride=1, padding=10，后面几层7-1-3取得了很好的效果
            

            elif i == 3:
                layers.append(nn.Conv2d(current_channels, current_channels, 
                                        kernel_size=3, stride=1, padding=1)) #fashionmnist 17-8,cifar10 11-5，bloodmnist 第一层，21, stride=1, padding=10，后面几层7-1-3取得了很好的效果
            

            else:
                layers.append(nn.Conv2d(current_channels, current_channels, 
                                        kernel_size=3, stride=1, padding=1)) #fashionmnist 17-8,cifar10 11-5，bloodmnist 第一层，21, stride=1, padding=10，后面几层7-1-3取得了很好的效果
            
            # 归一化
            if use_norm:
                layers.append(nn.GroupNorm(current_channels, current_channels, affine=True))
            
            # 激活
            if use_act:
                layers.append(nn.ReLU(inplace=True))
        
        # ========== 自适应池化（替代 stride 卷积） ==========
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
        
        print(f"\n[FunnelEncoder] 输入尺寸 {input_size} → 输出尺寸 {output_size}")
        print(f"  每块卷积层数: {num_convs_per_block}")
        print(f"  隐藏层通道数: {hidden_dim}")
        
        # ========== 简化：只需要一个 ConvPoolBlock，用自适应池化控制输出尺寸 ==========
        self.encoder = ConvPoolBlock(
            in_channels=input_channels,
            out_channels=hidden_dim,
            use_norm=True,
            use_act=True,
            num_convs=num_convs_per_block,
            target_size=output_size  # 🔑 直接指定目标尺寸
        )
        
        # 压缩到RGB
        if compress_to_rgb:
            self.rgb_compressor = nn.Conv2d(hidden_dim, 3, kernel_size=1)
        
        # 通道注意力
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
        
        # 验证输出尺寸
        with torch.no_grad():
            was_training = self.training
            self.eval()
            
            try:
                dummy = torch.randn(2, input_channels, input_size, input_size)
                dummy_out = self.encoder(dummy)
                actual_out_size = dummy_out.shape[-1]
                print(f"  实际输出尺寸: {actual_out_size}×{actual_out_size}")
                
                if actual_out_size != output_size:
                    print(f"  警告: 实际输出尺寸 {actual_out_size} ≠ 目标输出尺寸 {output_size}")
            finally:
                if was_training:
                    self.train()
        
        # 打印网络结构
        print(f"  最终通道数: {hidden_dim}")
        print(f"  编码器层数: 1 个 ConvPoolBlock")
        print(f"  实际输出尺寸: {actual_out_size if 'actual_out_size' in locals() else output_size}")
    
    def encode(self, x: torch.Tensor, output_size: int = None) -> torch.Tensor:
        """编码器前向传播"""
        features = self.encoder(x)
        
        # 通道注意力
        if self.use_attention:
            attention_weights = self.channel_attention(features)
            features = features * attention_weights.view(-1, self.final_channels, 1, 1)
        
        # 压缩到RGB
        if self.compress_to_rgb:
            output = torch.tanh(self.rgb_compressor(features))
        else:
            output = features
        
        return output
    
    def forward(self, x: torch.Tensor, output_size: int = None) -> torch.Tensor:
        """前向传播（同 encode）"""
        return self.encode(x, output_size)
    
    def get_info_bottleneck_size(self, output_size: int = None) -> int:
        """返回 I 的总信息量（通道 × 高 × 宽）"""
        out_sz = output_size if output_size is not None else self.output_size
        channels = 3 if self.compress_to_rgb else self.final_channels
        return channels * out_sz * out_sz
    
    def get_supported_output_sizes(self) -> list:
        """返回支持的输出尺寸列表（任意正整数，自适应池化都支持）"""
        # 自适应池化支持任意输出尺寸，返回常见值
        supported = [1, 2, 4, 8, 16, 32]
        return [sz for sz in supported if sz <= self.input_size]
    
    def visualize_conv_parameters(self, save_dir=None, show_plots=False):
        """可视化卷积层的参数值（保持不变）"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 收集所有卷积层
        conv_layers = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))
        
        if not conv_layers:
            print("没有找到卷积层")
            return {}
        
        print(f"\n[FunnelEncoder] 找到 {len(conv_layers)} 个卷积层")
        
        stats = {}
        
        # 为每个卷积层创建可视化
        for i, (layer_name, conv_layer) in enumerate(conv_layers):
            print(f"  可视化第 {i+1} 层: {layer_name}")
            print(f"    权重形状: {conv_layer.weight.shape}")
            print(f"    偏置形状: {conv_layer.bias.shape if conv_layer.bias is not None else '无偏置'}")
            
            # 获取权重数据
            weights = conv_layer.weight.detach().cpu().numpy()
            bias = conv_layer.bias.detach().cpu().numpy() if conv_layer.bias is not None else None
            
            # 计算统计信息
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
            
            # 创建可视化图表
            fig = self._create_conv_visualization_figure(
                weights, bias, layer_name, i+1, conv_layer
            )
            
            # 保存图表
            if save_dir:
                save_path = os.path.join(save_dir, f"conv_layer_{i+1}_{layer_name.replace('.', '_')}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"    已保存图表到: {save_path}")
            
            # 显示图表
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
        
        # 创建汇总图表
        self._create_summary_visualization(stats, save_dir, show_plots)
        
        return stats
    
    def _create_conv_visualization_figure(self, weights, bias, layer_name, layer_idx, conv_layer):
        """创建单个卷积层的可视化图表（保持不变）"""
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
        """创建所有卷积层的汇总可视化（保持不变）"""
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
                      alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Convolutional Layer')
        ax1.set_ylabel('Weight Mean')
        ax1.set_title('Weight Mean Comparison Across Layers')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'L{i+1}' for i in range(len(layer_names))], rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for i, (bar, mean, std) in enumerate(zip(bars, weight_means, weight_stds)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Weight range comparison across layers
        ax2 = plt.subplot(2, 2, 2)
        weight_mins = [stats[name]['weights']['min'] for name in layer_names]
        weight_maxs = [stats[name]['weights']['max'] for name in layer_names]
        
        for i, (w_min, w_max) in enumerate(zip(weight_mins, weight_maxs)):
            ax2.plot([i, i], [w_min, w_max], 'o-', linewidth=3, markersize=8)
        
        ax2.set_xlabel('Convolutional Layer')
        ax2.set_ylabel('Weight Value Range')
        ax2.set_title('Weight Range Comparison Across Layers')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'L{i+1}' for i in range(len(layer_names))], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Parameter count comparison
        ax3 = plt.subplot(2, 2, 3)
        param_counts = []
        for name in layer_names:
            weight_shape = stats[name]['weights']['shape']
            # Calculate weight parameter count
            weight_params = np.prod(weight_shape)
            # Calculate bias parameter count
            if stats[name]['bias']:
                bias_shape = stats[name]['bias']['shape']
                bias_params = np.prod(bias_shape)
            else:
                bias_params = 0
            param_counts.append(weight_params + bias_params)
        
        bars3 = ax3.bar(x_pos, param_counts, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('Convolutional Layer')
        ax3.set_ylabel('Parameter Count')
        ax3.set_title('Parameter Count Comparison Across Layers')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'L{i+1}' for i in range(len(layer_names))], rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for i, (bar, count) in enumerate(zip(bars3, param_counts)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(param_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        # 4. Bias statistics (if any)
        ax4 = plt.subplot(2, 2, 4)
        bias_means = []
        bias_layers = []
        
        for i, name in enumerate(layer_names):
            if stats[name]['bias']:
                bias_means.append(stats[name]['bias']['mean'])
                bias_layers.append(f'L{i+1}')
        
        if bias_means:
            bars4 = ax4.bar(range(len(bias_means)), bias_means, alpha=0.7, 
                           color='orange', edgecolor='black')
            ax4.set_xlabel('Convolutional Layer (with bias)')
            ax4.set_ylabel('Bias Mean')
            ax4.set_title('Bias Mean Comparison Across Layers')
            ax4.set_xticks(range(len(bias_means)))
            ax4.set_xticklabels(bias_layers, rotation=45)
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add values on bars
            for i, (bar, mean) in enumerate(zip(bars4, bias_means)):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(bias_means)*0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'All convolutional layers have no bias', 
                    ha='center', va='center', fontsize=14)
            ax4.set_title('Bias Statistics')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save summary chart
        if save_dir:
            save_path = os.path.join(save_dir, "conv_layers_summary.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"    Saved summary chart to: {save_path}")
        
        # Show chart
        if show_plots:
            plt.show()
        else:
            plt.close(fig)


def get_funnel_encoder_for_fedsumup(dataset_info: dict, config: dict) -> FunnelEncoder:
    """为 FedSumUp 获取漏斗编码器（兼容旧版接口）"""
    input_channels = dataset_info.get("channel", 3)
    input_size = dataset_info.get("im_size", (32, 32))[0]
    output_size = config.get("compressed_image_size", 8)
    num_classes = dataset_info.get("num_classes", 10)
    hidden_dim = config.get("encoder_hidden_dim", 128)
    compress_to_rgb = config.get("compress_to_rgb", True)
    num_convs_per_block = config.get("num_convs_per_block", 5)  # 新增参数，默认为2
    
    print(f"\n{'='*60}")
    print(f"漏斗编码器配置（增强版ConvPool套装）")
    print(f"{'='*60}")
    print(f"  输入: {input_channels} 通道，{input_size}×{input_size}")
    print(f"  输出: {3 if compress_to_rgb else hidden_dim} x {output_size} x {output_size}")
    print(f"  每块卷积层数: {num_convs_per_block}")
    print(f"  特性: 使用增强版ConvPool套装，每块包含{num_convs_per_block}个卷积层")
    
    encoder = FunnelEncoder(
        input_channels=input_channels,
        input_size=input_size,
        output_size=output_size,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        compress_to_rgb=compress_to_rgb,
        num_convs_per_block=num_convs_per_block
    )
    
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\n参数量统计:")
    print(f"  总参数: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    freeze_encoder = config.get("freeze_vae", False)
    if freeze_encoder:
        print("  冻结编码器参数")
        for param in encoder.parameters():
            param.requires_grad = False
    
    return encoder


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("ConvPool 套装版本漏斗编码器测试")
    print("=" * 60)
    
    # 测试1: 固定输入尺寸 32×32，不同输出尺寸
    print("\n" + "=" * 40)
    print("测试1: 输入 32×32")
    print("=" * 40)
    
    for out_sz in [16, 8, 4, 2, 1]:
        print(f"\n--- 输出尺寸 {out_sz}×{out_sz} ---")
        model = FunnelEncoder(input_size=32, output_size=out_sz, hidden_dim=64)
        x = torch.randn(2, 3, 32, 32)
        y = model.encode(x)
        print(f"输入 32×32 → 输出 {y.shape}")
        print(f"信息容量: {model.get_info_bottleneck_size()} 个数值")
    
    # 测试2: 不同输入尺寸，固定输出 8×8
    print("\n" + "=" * 40)
    print("测试2: 不同输入尺寸 → 输出 8×8")
    print("=" * 40)
    
    for in_sz in [32, 64, 128]:
        print(f"\n--- 输入尺寸 {in_sz}×{in_sz} ---")
        model = FunnelEncoder(input_size=in_sz, output_size=8, hidden_dim=64)
        x = torch.randn(2, 3, in_sz, in_sz)
        y = model.encode(x)
        print(f"输入 {in_sz}×{in_sz} → 输出 {y.shape}")
    
    # 测试3: 接口兼容性测试
    print("\n" + "=" * 40)
    print("测试3: 接口兼容性测试")
    print("=" * 40)
    
    dataset_info = {"channel": 3, "im_size": (32, 32), "num_classes": 10}
    config = {"compressed_image_size": 8, "encoder_hidden_dim": 64, "compress_to_rgb": True}
    
    encoder = get_funnel_encoder_for_fedsumup(dataset_info, config)
    x = torch.randn(4, 3, 32, 32)
    y = encoder.encode(x)
    print(f"\n最终测试通过！输出形状: {y.shape}")
    
    # 测试4: 动态输出尺寸（现在会给出警告）
    print("\n" + "=" * 40)
    print("测试4: 动态输出尺寸（ConvPool 版本不支持动态变化）")
    print("=" * 40)
    
    model = FunnelEncoder(input_size=32, output_size=8, hidden_dim=64)
    x = torch.randn(2, 3, 32, 32)
    
    for out_sz in [4, 2, 1]:
        y = model.encode(x, output_size=out_sz)
        print(f"请求动态输出 {out_sz}×{out_sz} → 实际输出 {y.shape}")
    
    # 测试5: 训练性测试
    print("\n" + "=" * 40)
    print("测试5: 训练性测试")
    print("=" * 40)
    
    model = FunnelEncoder(input_size=32, output_size=8, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    x = torch.randn(4, 3, 32, 32)
    y = model.encode(x)
    loss = y.mean()
    loss.backward()
    optimizer.step()
    
    print(f"前向传播输出形状: {y.shape}")
    print(f"损失值: {loss.item():.6f}")
    print(f"梯度更新成功！模型可训练。")
    
    # 测试6: 卷积层参数可视化测试
    print("\n" + "=" * 40)
    print("测试6: 卷积层参数可视化测试")
    print("=" * 40)
    
    model = FunnelEncoder(input_size=32, output_size=8, hidden_dim=128)
    
    save_dir = "./test_conv_visualization"
    
    print(f"\n开始可视化卷积层参数...")
    stats = model.visualize_conv_parameters(save_dir=save_dir, show_plots=False)
    
    if stats:
        print(f"\n可视化完成！")
        print(f"所有可视化图表已保存到: {save_dir}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
