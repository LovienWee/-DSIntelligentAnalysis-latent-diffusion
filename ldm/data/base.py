from abc import abstractmethod
from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset
# 在base.py文件开头添加
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
class OceanDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, file_path, depth, batch_size, 
                 train_ratio=0.8, num_workers=None, **kwargs):
        super().__init__()
        self.file_path = file_path
        self.depth = depth
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.dataset = None

    def prepare_data(self):
        # 不需要在此处加载数据
        pass

    def setup(self, stage=None):
        # 创建完整数据集
        full_dataset = OceanDataset(self.file_path, self.depth)
        
        # 划分训练集和验证集
        total_size = len(full_dataset)
        train_size = int(total_size * self.train_ratio)
        val_size = total_size - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        
class Txt2ImgIterableBaseDataset(IterableDataset):
    '''
    Define an interface to make the IterableDatasets for text2img data chainable
    '''
    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass