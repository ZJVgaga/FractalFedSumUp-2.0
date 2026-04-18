"""
调度配置模块
"""

SCHEDULE_CONFIG = {
    "content": "Schedule_Config",
    "type": "key",
    "children": [
        {"content": "experiment_index", "type": "key", "children": [{"content": 33, "type": "value"}]},
        {"content": "experiment_times", "type": "key", "children": [{"content": 5, "type": "value"}]},
        {"content": "client_num", "type": "key", "children": [{"content": 10, "type": "value"}]},
        {"content": "communication_rounds", "type": "key", "children": [{"content":50, "type": "value"}]},
        {"content": "join_ratio", "type": "key", "children": [{"content": 1.0, "type": "value"}]},
        {"content": "eval_gap", "type": "key", "children": [{"content": 1, "type": "value"}]},
        {"content": "seed", "type": "key", "children": [{"content": 520, "type": "value"}]},
        {"content": "compute_budget_per_client_per_round", "type": "key", "children": [{"content": 3e9, "type": "value"}]},  # 每个客户端每轮的计算预算（浮点运算次数）
        {"content": "enable_compute_budget", "type": "key", "children": [{"content": True, "type": "value"}]},  # 是否启用计算预算
        {"content": "budget_check_frequency", "type": "key", "children": [{"content": 1, "type": "value"}]}  # 预算检查频率（每N个epoch检查一次）
    ]
}
