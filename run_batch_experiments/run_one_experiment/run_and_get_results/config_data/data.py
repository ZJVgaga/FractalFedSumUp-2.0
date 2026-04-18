"""
数据配置模块
"""

DATA_CONFIG = {
    "content": "Data_Config",
    "type": "key",
    "children": [
        {
            "content": "Dataset",
            "choice": "CIFAR10",
            "type": "key",
            "children": [
                {"content": "MNIST", "type": "value"},
                {"content": "CIFAR10", "type": "value"},
                {"content": "CIFAR100", "type": "value"},
                {"content": "FashionMNIST", "type": "value"},
                 {"content":"MedMNIST-BloodMNIST", "type": "value"},
                
            ]
        },
        {
            "content": "Data_Parameters",
            "type": "key",
            "children": [
                {"content": "alpha", "type": "key", "children": [{"content": 0.5, "type": "value"}]},
                {"content": "batch_size", "type": "key", "children": [{"content": 256, "type": "value"}]},
                {"content": "images_per_class", "type": "key", "children": [{"content": 30, "type": "value"}]},  # 数据合成的基本参数值
                {"content": "hyperparameter_experiment", "type": "key", "children": [{"content": False, "type": "value"}]},
            ]
        }
    ]
}
