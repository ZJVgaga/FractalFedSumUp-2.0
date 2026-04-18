"""
联邦学习配置模块
"""

FEDERATED_CONFIG = {
    "content": "Federated_Learning_Config",
    "type": "key",
    "choice": "FedSumUp",
    "children": [
        {
            "content": "FedAvg",
            "type": "value",
            "children": [
                {"content": "dp_mechanism", "type": "key", "children": [{"content":   "no_dp", "type": "value"}]},# 'laplace'  "gaussian"   "no_dp"
                {"content": "dp_epsilon", "type": "key", "children": [{"content": 5.0, "type": "value"}]},
                {"content": "dp_delta", "type": "key", "children": [{"content": 1e-5, "type": "value"}]},
                {"content": "dp_clip", "type": "key", "children": [{"content": 1.0, "type": "value"}]},
                {"content": "dp_noise_scale", "type": "key", "children": [{"content": 0.1, "type": "value"}]},
                {"content": "dp_clip_level", "type": "key", "children": [{"content": "batch", "type": "value"}]},
                {"content": "dp_element_wise_rand", "type": "key", "children": [{"content": True, "type": "value"}]},
            ]
        },
        {
            "content": "FedProx",
            "type": "value",
            "children": [
                {"content": "fedprox_mu", "type": "key", "children": [{"content": 0.01, "type": "value"}]}
            ]
        },
        {
            "content": "CollabDM",
            "type": "value",
            "children": [
                {"content": "collabdm_mode", "type": "key", "children": [{"content": "real", "type": "value"}]},
                {"content": "collabdm_iterations", "type": "key", "children": [{"content": 50, "type": "value"}]},
            ]
        },
        {
            "content": "FedAdam",
            "type": "value",
            "children": [
                {"content": "adam_momentum_beta1", "type": "key", "children": [{"content": 0.9, "type": "value"}]},
                {"content": "adam_momentum_learning_rate", "type": "key", "children": [{"content": 0.01, "type": "value"}]},
                {"content": "adam_momentum_beta2", "type": "key", "children": [{"content": 0.999, "type": "value"}]},
                {"content": "adam_momentum_epsilon", "type": "key", "children": [{"content": 1e-8, "type": "value"}]}
            ]
        },
        {
            "content": "FedSD2C",
            "type": "value",
            "children": [
                {"content": "sd2c_num_crop", "type": "key", "children": [{"content": 1, "type": "value"}]},  # 根据 '--fedsd2c_num_crop' 参数
                {"content": "sd2c_foulier_alpha", "type": "key", "children": [{"content": 0.1, "type": "value"}]},
                {"content": "sd2c_iterations", "type": "key", "children": [{"content": 50, "type": "value"}]},
            ]
        },
        {
            "content": "FedSumUp",
            "type": "value",
            "children": [
                {"content": "sumup_num_crop", "type": "key", "children": [{"content": 1, "type": "value"}]},  # 根据 '--fedsd2c_num_crop' 参数
                # 新流程参数：压缩图像大小（替代原来的k参数）
                {"content": "compressed_image_size", "type": "key", "children": [{"content": 24, "type": "value"}]},
                {"content": "data_representation_ratio", "type": "key", "children": [{"content": 0.5, "type": "value"}]},
                {"content": "sumup_utility_ratio", "type": "key", "children": [{"content": 1, "type": "value"}]},
                
            ]
        },
        {
            "content": "FedSum",
            "type": "value",
            "children": [
                {"content": "sumup_num_crop", "type": "key", "children": [{"content": 1, "type": "value"}]},
                {"content": "sumup_training_free", "type": "key", "children": [{"content": True, "type": "value"}]},
                {"content": "sumup_iterations_vae", "type": "key", "children": [{"content": 50, "type": "value"}]},
                {"content": "sumup_iterations_img", "type": "key", "children": [{"content": 50, "type": "value"}]},
            ]
        },
        {
            "content": "FedDM",
            "type": "value",
            "children": [
                
                {"content": "dm_client_sumup_epochs", "type": "key", "children": [{"content": 50, "type": "value"}]},
                {"content": "dm_model_noise", "type": "key", "children": [{"content": 5, "type": "value"}]},
                {"content": "dm_client_sumup_batch_size", "type": "key", "children": [{"content": 256, "type": "value"}]},
                {"content": "dm_image_learning_rate", "type": "key", "children": [{"content": 1, "type": "value"}]},
                {"content": "dm_data_template_mode", "type": "key", "children": [{"content": "real", "type": "value"}]},
                {"content": "dm_synth_ratio", "type": "key", "children": [{"content": 0.5, "type": "value"}]},
            ]
        },
    ]
}
