import os
import json


class ImageTracker:
    """
    图片追踪器：负责选择和管理需要追踪的图片
    """
    def __init__(self, config, dataset_info, train_indices, train_labels):
        """
        初始化图片追踪器
        
        Parameters:
        - config: 配置字典
        - dataset_info: 数据集信息
        - train_indices: 训练数据索引列表
        - train_labels: 训练数据标签列表
        """
        self.config = config
        self.dataset_info = dataset_info
        self.train_indices = train_indices
        self.train_labels = train_labels
        
        # 获取数据集信息
        self.dataset_name = config.get("Dataset", "CIFAR10")
        self.num_classes = dataset_info.get("num_classes", 10)
        self.compressed_size = config.get("compressed_image_size", 24)
        
        # 选择需要追踪的图片
        self.tracked_indices = self._select_tracked_images()
        
        # 打印追踪信息
        print(f"ImageTracker: 选择了 {len(self.tracked_indices)} 张图片进行追踪")
        for idx, class_id in self.tracked_indices.items():
            print(f"  - 索引 {idx}: 类别 {class_id}")
    
    def _select_tracked_images(self):
        """
        从train_indices中对每一个类别取两张图片获得其index
        返回字典：{index: class_id}
        """
        tracked_indices = {}
        
        # 按类别分组索引
        class_to_indices = {}
        for idx, label in zip(self.train_indices, self.train_labels):
            class_to_indices.setdefault(label, []).append(idx)
        
        # 对每一个类别取前两张图片
        for class_id in range(self.num_classes):
            if class_id in class_to_indices and class_to_indices[class_id]:
                # 取该类别的前两张图片
                for i in range(min(2, len(class_to_indices[class_id]))):
                    selected_idx = class_to_indices[class_id][i]
                    tracked_indices[selected_idx] = class_id
        
        return tracked_indices
    
    def get_tracked_indices(self):
        """获取追踪的图片索引"""
        return self.tracked_indices
    
    def get_client_responsibility(self, client_indices):
        """
        检查客户端是否负责追踪某些图片
        
        Parameters:
        - client_indices: 客户端的数据索引列表
        
        Returns:
        - 字典：{index: class_id}，该客户端负责追踪的图片
        """
        responsibility = {}
        for idx, class_id in self.tracked_indices.items():
            if idx in client_indices:
                responsibility[idx] = class_id
        
        return responsibility
    
    def get_save_path(self, index, class_id, round_num, is_server=False, epoch=None):
        """
        获取保存路径
        
        Parameters:
        - index: 图片索引
        - class_id: 类别ID
        - round_num: 轮次
        - is_server: 是否为服务器端
        - epoch: 服务器epoch（仅服务器端需要）
        
        Returns:
        - 保存路径
        """
        # 基础路径
        base_dir = self.config.get("visualization_save_dir", "./fedsumup_visualizations")
        
        # 根据用户要求的文件夹结构
        # fedsumup_visualization_{datasetname}/compressed_{size}/{index}_{class}/round/client或server_epochs
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
