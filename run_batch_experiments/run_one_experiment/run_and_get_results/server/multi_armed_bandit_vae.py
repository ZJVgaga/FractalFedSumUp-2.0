"""
多臂老虎机VAE更新器
根据奖励分数S（交叉熵损失）调整VAE参数
S越大（预测越差）→ 生成的新VAE与旧VAE差异越大
S越小（预测越好）→ 生成的新VAE与旧VAE差异越小
"""

import torch
import torch.nn as nn
import copy
import numpy as np
import random


class MultiArmedBanditVAE:
    """
    多臂老虎机VAE更新器
    输入：当前VAE、奖励分数S（交叉熵损失）
    输出：新的VAE参数
    """
    
    def __init__(self, base_vae, config=None):
        """
        初始化多臂老虎机VAE更新器
        
        参数:
        - base_vae: 基础VAE模型
        - config: 配置参数
        """
        self.base_vae = base_vae
        self.config = config or {}
        
        # 多臂老虎机参数
        self.exploration_rate = self.config.get("exploration_rate", 0.3)  # 探索率
        self.exploitation_rate = self.config.get("exploitation_rate", 0.7)  # 利用率
        self.max_perturbation = self.config.get("max_perturbation", 0.5)  # 最大扰动
        self.min_perturbation = self.config.get("min_perturbation", 0.01)  # 最小扰动
        
        # 历史记录
        self.reward_history = []
        self.vae_history = []
        
        self.logger = None
        
    def set_logger(self, logger):
        """设置日志记录器"""
        self.logger = logger
    
    def update_vae(self, current_vae, reward_score, current_round):
        """
        根据奖励分数更新VAE参数
        
        参数:
        - current_vae: 当前VAE模型
        - reward_score: 奖励分数（交叉熵损失）
        - current_round: 当前轮次
        
        返回:
        - new_vae: 新的VAE模型
        """
        if self.logger:
            self.logger.info(f"多臂老虎机：第{current_round}轮，奖励分数S={reward_score:.4f}")
        
        # 保存历史记录
        self.reward_history.append(reward_score)
        self.vae_history.append(copy.deepcopy(current_vae))
        
        # 计算扰动幅度：S越大，扰动越大
        # 归一化奖励分数（假设损失在0-10之间）
        normalized_reward = min(max(reward_score, 0.0), 10.0) / 10.0
        
        # 计算扰动幅度：S越大，扰动越大
        perturbation_magnitude = self.min_perturbation + normalized_reward * (self.max_perturbation - self.min_perturbation)
        
        if self.logger:
            self.logger.info(f"多臂老虎机：归一化奖励={normalized_reward:.4f}，扰动幅度={perturbation_magnitude:.4f}")
        
        # 创建新的VAE（深拷贝当前VAE）
        new_vae = copy.deepcopy(current_vae)
        
        # 根据探索/利用策略更新参数
        if random.random() < self.exploration_rate:
            # 探索：随机扰动参数
            new_vae = self._explore(new_vae, perturbation_magnitude, current_round)
        else:
            # 利用：基于奖励分数调整参数
            new_vae = self._exploit(new_vae, reward_score, perturbation_magnitude, current_round)
        
        return new_vae
    
    def _explore(self, vae, perturbation_magnitude, current_round):
        """
        探索策略：随机扰动VAE参数
        
        参数:
        - vae: VAE模型
        - perturbation_magnitude: 扰动幅度
        - current_round: 当前轮次
        
        返回:
        - 扰动后的VAE
        """
        if self.logger:
            self.logger.info(f"多臂老虎机：第{current_round}轮使用探索策略")
        
        with torch.no_grad():
            for name, param in vae.named_parameters():
                if param.requires_grad:
                    # 生成随机扰动
                    perturbation = torch.randn_like(param) * perturbation_magnitude
                    
                    # 应用扰动
                    param.add_(perturbation)
                    
                    # 可选：对某些层进行特殊处理
                    if "latent" in name.lower():
                        # 潜在层扰动可以更大一些
                        extra_perturbation = torch.randn_like(param) * perturbation_magnitude * 0.5
                        param.add_(extra_perturbation)
        
        return vae
    
    def _exploit(self, vae, reward_score, perturbation_magnitude, current_round):
        """
        利用策略：基于奖励分数调整VAE参数
        
        参数:
        - vae: VAE模型
        - reward_score: 奖励分数
        - perturbation_magnitude: 扰动幅度
        - current_round: 当前轮次
        
        返回:
        - 调整后的VAE
        """
        if self.logger:
            self.logger.info(f"多臂老虎机：第{current_round}轮使用利用策略")
        
        # 如果有历史记录，可以基于历史奖励调整
        if len(self.reward_history) > 1:
            # 计算奖励趋势
            recent_rewards = self.reward_history[-3:] if len(self.reward_history) >= 3 else self.reward_history
            reward_trend = np.mean(np.diff(recent_rewards)) if len(recent_rewards) > 1 else 0
            
            if self.logger:
                self.logger.info(f"多臂老虎机：奖励趋势={reward_trend:.4f}")
            
            # 如果奖励在下降（损失在增加），需要更大的变化
            if reward_trend > 0:  # 损失在增加
                perturbation_magnitude *= 1.5
                if self.logger:
                    self.logger.info(f"多臂老虎机：奖励下降，增加扰动幅度到{perturbation_magnitude:.4f}")
            elif reward_trend < -0.1:  # 损失在显著下降
                perturbation_magnitude *= 0.7
                if self.logger:
                    self.logger.info(f"多臂老虎机：奖励上升，减少扰动幅度到{perturbation_magnitude:.4f}")
        
        with torch.no_grad():
            for name, param in vae.named_parameters():
                if param.requires_grad:
                    # 基于参数梯度方向调整（如果有梯度信息）
                    if hasattr(param, 'grad') and param.grad is not None:
                        # 沿着梯度方向调整（减少损失的方向）
                        perturbation = -param.grad * perturbation_magnitude * 0.1
                    else:
                        # 随机扰动
                        perturbation = torch.randn_like(param) * perturbation_magnitude * 0.5
                    
                    # 应用扰动
                    param.add_(perturbation)
        
        return vae
    
    def get_statistics(self):
        """获取多臂老虎机统计信息"""
        if len(self.reward_history) == 0:
            return "无历史记录"
        
        stats = {
            "total_rounds": len(self.reward_history),
            "avg_reward": np.mean(self.reward_history) if self.reward_history else 0,
            "min_reward": min(self.reward_history) if self.reward_history else 0,
            "max_reward": max(self.reward_history) if self.reward_history else 0,
            "last_reward": self.reward_history[-1] if self.reward_history else 0,
            "exploration_rate": self.exploration_rate,
            "exploitation_rate": self.exploitation_rate,
        }
        
        return stats


def create_multi_armed_bandit_for_vae(vae, config=None, logger=None):
    """
    创建多臂老虎机VAE更新器
    
    参数:
    - vae: VAE模型
    - config: 配置参数
    - logger: 日志记录器
    
    返回:
    - MultiArmedBanditVAE实例
    """
    bandit = MultiArmedBanditVAE(vae, config)
    if logger:
        bandit.set_logger(logger)
    
    return bandit


# 测试代码
if __name__ == "__main__":
    print("测试多臂老虎机VAE更新器")
    
    # 创建一个简单的VAE模型用于测试
    class SimpleVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(10, 5)
            self.decoder = nn.Linear(5, 10)
        
        def encode(self, x):
            return self.encoder(x), None, None
        
        def decode(self, z):
            return self.decoder(z)
    
    # 创建测试VAE
    test_vae = SimpleVAE()
    
    # 创建多臂老虎机
    config = {
        "exploration_rate": 0.3,
        "exploitation_rate": 0.7,
        "max_perturbation": 0.5,
        "min_perturbation": 0.01,
    }
    
    bandit = create_multi_armed_bandit_for_vae(test_vae, config)
    
    # 测试更新
    print(f"初始VAE参数: {list(test_vae.parameters())[0].data[0][:5]}")
    
    # 模拟不同奖励分数
    test_rewards = [2.5, 1.8, 3.2, 0.9, 0.5]
    
    for i, reward in enumerate(test_rewards):
        print(f"\n第{i+1}轮，奖励分数S={reward}")
        new_vae = bandit.update_vae(test_vae, reward, i+1)
        print(f"更新后VAE参数: {list(new_vae.parameters())[0].data[0][:5]}")
        
        # 更新当前VAE
        test_vae = new_vae
    
    # 获取统计信息
    stats = bandit.get_statistics()
    print(f"\n多臂老虎机统计信息: {stats}")
