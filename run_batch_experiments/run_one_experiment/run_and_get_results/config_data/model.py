"""
模型配置模块
"""

MODEL_CONFIG = {
    "content": "Model",
    "type": "key",
    "choice": "ConvNet",  # 默认选择 ConvNet 模型
    "children": [
        {
            "content": "ConvNet",
            "type": "value",
            "children": [
                {"content": "learning_rate", "type": "key", "children": [{"content": 0.01, "type": "value"}]},
                {"content": "weight_decay", "type": "key", "children": [{"content": 0.0005, "type": "value"}]},
                {"content": "momentum", "type": "key", "children": [{"content": 0.9, "type": "value"}]},
                {"content": "train_model_epochs", "type": "key", "children": [{"content": 5, "type": "value"}]},
                {"content": "train_batch_size", "type": "key", "children": [{"content": 256, "type": "value"}]},
                {"content": "vae_target_params_m", "type": "key", "children": [{"content": 1.0, "type": "value"}]},
            ]
        }
    ]
}
