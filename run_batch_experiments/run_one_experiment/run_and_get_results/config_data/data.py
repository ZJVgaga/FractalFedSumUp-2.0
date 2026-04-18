"""
Data Configuration Module
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
                {"content": "images_per_class", "type": "key", "children": [{"content": 30, "type": "value"}]},  # Basic parameter value for data synthesis
                {"content": "hyperparameter_experiment", "type": "key", "children": [{"content": False, "type": "value"}]},
            ]
        }
    ]
}