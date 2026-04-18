import copy

import io
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import softmax
from ...network_tools import random_pertube_model

class PerLabelDatasetNonIID():
    def __init__(self, trainset, train_indices, classes, channel, device,seed):  
        self.images_all = []
        self.seed=seed
        labels_all = []
        self.indices_class = {c: [] for c in classes}  # 存储 self.images_all 的索引
        self.original_indices_map = []  # 记录 self.images_all 索引与原始数据集索引的映射
        self.device = device

        # 使用 DataLoader 安全地遍历数据集
        data_loader = DataLoader(trainset, batch_size=1, shuffle=False, sampler=SubsetRandomSampler(train_indices))

        for i, (image, label) in enumerate(data_loader):
            if label.item() not in classes:
                continue
            self.images_all.append(image)
            labels_all.append(label.item())
            self.indices_class[label.item()].append(len(self.images_all) - 1)  # 存储 self.images_all 的本地索引
            self.original_indices_map.append(train_indices[i])  # 记录原始数据集索引

        # 转换为张量
        self.images_all = torch.cat(self.images_all, dim=0).to(device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)
    def __len__(self):
        return self.images_all.shape[0]
    def get_all_class_c_images(self, target_class):
        # 获取类别为 target_class 的所有图像及其在原始数据集中的索引
        all_c_imgs = []
        all_c_indices = []

        # 遍历 indices_class[target_class] 获取图像和原始索引
        for local_idx in self.indices_class.get(target_class, []):
            all_c_imgs.append(self.images_all[local_idx])
            original_idx = self.original_indices_map[local_idx]
            all_c_indices.append(original_idx)

        # 将图像堆叠为一个张量
        all_c_imgs = torch.stack(all_c_imgs) if all_c_imgs else torch.tensor([]).to(self.device)
        return all_c_imgs
    def get_random_images(self, n):  # get n random images
        np.random.seed(self.seed)
        idx_shuffle = np.random.permutation(range(self.images_all.shape[0]))[:n]
        return self.images_all[idx_shuffle]
    def get_images(self, c, n, avg=False):  # get n random images from class c
        np.random.seed(self.seed)
        #print(f"Function called with parameters: class={c}, number={n}, average={avg}")
        if not avg:
            indices = self.indices_class[c]
            #print(f"Indices for class {c}: {indices}")
            if len(indices) >= n:
                idx_shuffle = np.random.choice(indices, n, replace=False)
                #print(f"Selected {n} unique indices without replacement: {idx_shuffle}")
            else:
                idx_shuffle = np.random.choice(indices, n, replace=True)
                #print(f"Selected {n} indices with replacement: {idx_shuffle}")
            selected_images = self.images_all[idx_shuffle]
            #print(f"Selected images shape: {selected_images.shape}")
            return selected_images
        else:
            sampled_imgs = []
            batch_size = 5  # 平均化时使用的样本数量
            sampled_indices = []  # 用于存储平均化后图像对应的原始索引
            for _ in range(n):
                if len(self.indices_class[c]) >= 5:
                    idx = np.random.choice(self.indices_class[c], 5, replace=False)
                else:
                    idx = np.random.choice(self.indices_class[c], 5, replace=True)
                sampled_imgs.append(torch.mean(self.images_all[idx], dim=0, keepdim=True))
                # 添加参与平均计算的所有索引
                sampled_indices.extend(idx.tolist())

            sampled_imgs = torch.cat(sampled_imgs, dim=0).cuda()
            
            return sampled_imgs