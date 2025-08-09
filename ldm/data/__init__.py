# 在ldm/data/__init__.py中添加
from .ocean_data import OceanDataset
__all__ = [
    "OceanDataset",
    # ... 其他已有的数据集类
]