"""
Attack Configuration Module
Contains configurations related to attacks such as MIA (Membership Inference Attack)
"""

ATTACK_CONFIG = {
    "content": "Attack_Config",
    "type": "key",
    "children": [
        {
            "content": "attack_mode",
            "type": "key",
            "choice": "MIA",  # Attack disabled by default
            "children": [
                {"content": "None", "type": "value"},  # Do not enable attack
                {"content": "MIA", "type": "value"}    # Enable MIA attack
            ]
        },
        {
            "content": "MIA_Parameters",
            "type": "key",
            "children": [
                {
                    "content": "MIA_data_target_ratio",
                    "type": "key",
                    "children": [{"content": 0.7, "type": "value"}]  # Default target dataset ratio
                },
                {
                    "content": "shadow_model_num",
                    "type": "key",
                    "children": [{"content": 3, "type": "value"}]  # Number of shadow models
                },
                {
                    "content": "MIA_attack_model",
                    "type": "key",
                    "choice": "Logits_Inference",
                    "children": [
                        {"content": "Logits_Inference", "type": "value"},
                        {"content": "LiRA", "type": "value"}
                    ]
                },
                {
                    "content": "MIA_train_epochs",
                    "type": "key",
                    "children": [{"content": 50, "type": "value"}]  # MIA model training epochs
                },
                {
                    "content": "MIA_batch_size",
                    "type": "key",
                    "children": [{"content": 64, "type": "value"}]  # MIA model batch size
                },
                {
                    "content": "MIA_learning_rate",
                    "type": "key",
                    "children": [{"content": 0.001, "type": "value"}]  # MIA model learning rate
                },
                {
                    "content": "attack_epochs",
                    "type": "key",
                    "children": [{"content": 50, "type": "value"}]  # Attack model training epochs
                },
                {
                    "content": "attack_batch_size",
                    "type": "key",
                    "children": [{"content": 10, "type": "value"}]  # Attack model batch size
                },
                {
                    "content": "attack_learning_rate",
                    "type": "key",
                    "children": [{"content": 0.001, "type": "value"}]  # Attack model learning rate
                },
                {
                    "content": "attack_model",
                    "type": "key",
                    "choice": "fc",
                    "children": [
                        {"content": "fc", "type": "value"},  # Fully connected network
                        {"content": "ConvNet", "type": "value"}  # Convolutional network
                    ]
                }
            ]
        }
    ]
}