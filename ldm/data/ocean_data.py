import netCDF4
import numpy as np
import torch
from torch.utils.data import Dataset

class OceanDataset(Dataset):
    def __init__(self, file_path, depth, start_index=None, end_index=None, **kwargs):
        """
        初始化海洋数据集
        :param file_path: .nc 文件路径
        :param depth: 提取的深度
        :param start_index: 起始时间索引
        :param end_index: 结束时间索引
        """
        super().__init__()
        self.dataset = netCDF4.Dataset(file_path)
        self.depth = depth
        
        # 提取数据
        to_surface = self.dataset['analysed_sst'][:]  # 表层温度数据
        ugo_surface = self.dataset['uo'][:]  # 表层UGO数据
        vgo_surface = self.dataset['vo'][:]  # 表层VGO数据
        zo_surface = self.dataset['zos_glor'][:]  # ZO数据
        labels = self.dataset['thetao_glor'][:, self.depth, :, :]  # 目标深度温度

        # 切片处理
        if start_index is not None or end_index is not None:
            start_index = start_index or 0
            end_index = end_index or len(to_surface)
            to_surface = to_surface[start_index:end_index]
            ugo_surface = ugo_surface[start_index:end_index]
            vgo_surface = vgo_surface[start_index:end_index]
            zo_surface = zo_surface[start_index:end_index]
            labels = labels[start_index:end_index]

        # 将四个二维数组整合为一个四维数组 (b, c, h, w)
        self.data = np.stack((to_surface, ugo_surface, vgo_surface, zo_surface), axis=1)
        self.labels = labels

    def __getitem__(self, index):
        """
        根据索引获取时间点的数据和标签
        :param index: 数据索引（时间点）
        :return: 字典格式的数据和标签
        """
        sample = self.data[index]  # 获取第 index 个时间点的数据
        label = self.labels[index]  # 获取第 index 个时间点的标签
        
        # LDM框架期望返回字典格式的数据
        return {
            "image": torch.tensor(sample, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32)
        }

    def __len__(self):
        return len(self.data)
    
    def close(self):
        """关闭数据集释放资源"""
        if hasattr(self, 'dataset'):
            self.dataset.close()