"""
攻击配置模块
包含MIA（Membership Inference Attack）等攻击相关的配置
"""

ATTACK_CONFIG = {
    "content": "Attack_Config",
    "type": "key",
    "children": [
        {
            "content": "attack_mode",
            "type": "key",
            "choice": "MIA",  # 默认不启用攻击
            "children": [
                {"content": "None", "type": "value"},  # 不启用攻击
                {"content": "MIA", "type": "value"}    # 启用MIA攻击
            ]
        },
        {
            "content": "MIA_Parameters",
            "type": "key",
            "children": [
                {
                    "content": "MIA_data_target_ratio",
                    "type": "key",
                    "children": [{"content": 0.7, "type": "value"}]  # 默认目标数据集比例
                },
                {
                    "content": "shadow_model_num",
                    "type": "key",
                    "children": [{"content": 3, "type": "value"}]  # 阴影模型数量
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
                    "children": [{"content": 50, "type": "value"}]  # MIA模型训练轮数
                },
                {
                    "content": "MIA_batch_size",
                    "type": "key",
                    "children": [{"content": 64, "type": "value"}]  # MIA模型批大小
                },
                {
                    "content": "MIA_learning_rate",
                    "type": "key",
                    "children": [{"content": 0.001, "type": "value"}]  # MIA模型学习率
                },
                {
                    "content": "attack_epochs",
                    "type": "key",
                    "children": [{"content": 50, "type": "value"}]  # 攻击模型训练轮数
                },
                {
                    "content": "attack_batch_size",
                    "type": "key",
                    "children": [{"content": 10, "type": "value"}]  # 攻击模型批量大小
                },
                {
                    "content": "attack_learning_rate",
                    "type": "key",
                    "children": [{"content": 0.001, "type": "value"}]  # 攻击模型学习率
                },
                {
                    "content": "attack_model",
                    "type": "key",
                    "choice": "fc",
                    "children": [
                        {"content": "fc", "type": "value"},  # 全连接网络
                        {"content": "ConvNet", "type": "value"}  # 卷积网络
                    ]
                }
            ]
        }
    ]
}
