import os
import torch
import scipy.io as sio
import numpy as np
from torch.utils.data.dataset import Dataset
import random
from ranksvm import get_dynamic_image
from tqdm import tqdm

class TrainDataset(Dataset):
    def __init__(self, image_path1 = '/MMFJL_Net/data/MRI_PET_152_182_152_split/train/',
image_path2 = '/MMFJL_Net/data/MRI_PET_152_182_152_split/test/',
included_labels=('CN', 'MCI'), indices=None,random_state=4, use_augmentation=False):
        """
        :param image_paths: 包含多个数据路径的列表
        :param included_labels: 包含的标签类别
        :param indices: 使用的数据索引（可选）
        """
        self.image_paths = [image_path1,image_path2]
        self.included_labels = included_labels
        self.random_state = random_state
        self.use_augmentation = use_augmentation
        random.seed(self.random_state) 

        label_map = {'CN': 0, 'AD': 1, 'MCI': 2}

        if not all(label in label_map for label in included_labels):
            raise ValueError("Included labels must be one of 'CN', 'AD', 'MCI'")

        # 初始化文件名和标签列表
        self.file_names = []
        self.labels = []
        # 遍历所有路径，加载数据
        for image_path in self.image_paths:
            # 筛选符合条件的文件名
            all_files = os.listdir(image_path)
            files = [f for f in all_files if f[0] in {str(label_map[label]) for label in included_labels}]
            self.file_names.extend([os.path.join(image_path, f) for f in files])  # 保存完整路径
            self.labels.extend([self.get_label_from_filename(f) for f in files])
        # 打乱数据和标签
        combined = list(zip(self.file_names, self.labels))
        random.shuffle(combined)
        self.file_names[:], self.labels[:] = zip(*combined)
        # 如果提供了索引，则只使用这些索引对应的数据
        if indices is not None:
            self.file_names = [self.file_names[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
        self.data = []  # 存储所有加载的数据
        for file_name in tqdm(self.file_names, desc="Loading data to memory"):
            img = sio.loadmat(file_name)
            tensor_data = torch.from_numpy(img['data']).float()
            self.data.append(tensor_data)

    def get_label_from_filename(self, filename):

        label_char = filename[0]
        
        # 原始文件前缀映射
        if label_char == '0':
            original_label = 'CN'
        elif label_char == '1':
            original_label = 'AD'
        elif label_char == '2':
            original_label = 'MCI'
        else:
            raise ValueError(f"无效的文件名前缀: {label_char}")

        if set(self.included_labels) == {'CN', 'AD'}:
            return {'CN': 0, 'AD': 1}.get(original_label)
        
        elif set(self.included_labels) == {'CN', 'MCI'}:
            return {'CN': 0, 'MCI': 1}.get(original_label)
        
        elif set(self.included_labels) == {'AD', 'MCI'}:
            return {'AD': 1, 'MCI': 0}.get(original_label)
        
        elif set(self.included_labels) == {'CN', 'AD', 'MCI'}:
            return {'CN': 0, 'AD': 1, 'MCI': 2}.get(original_label)
        
        else:
            raise ValueError(f"不支持的标签组合: {self.included_labels}")
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        if self.use_augmentation:
                tensor_data = data[0]  
                start_x = (tensor_data.shape[0] - 110) // 2
                start_y = (tensor_data.shape[1] - 110) // 2
                start_z = (tensor_data.shape[2] - 110) // 2
               
                tensor_data = tensor_data[
                    start_x:start_x+110, 
                    start_y:start_y+110, 
                    start_z:start_z+110
                ]
               
                tensor_data = np.transpose(tensor_data, (2, 0, 1))  
                tensor_data = tensor_data[..., np.newaxis]          
                tensor_data = get_dynamic_image(tensor_data.numpy())
                tensor_data = np.expand_dims(tensor_data,0)
                tensor_data = np.concatenate([tensor_data,tensor_data,tensor_data], 0)
                tensor_data = torch.from_numpy(tensor_data).float()
                return tensor_data,label
        else:
            return data, label
     
class TestDataset(Dataset):
    def __init__(self, image_path1 = '/MMFJL_Net/data/MRI_PET_152_182_152_split/train/',
image_path2 = '/MMFJL_Net/data/MRI_PET_152_182_152_split/test/',
included_labels=('CN', 'MCI'), indices=None,random_state=4, use_augmentation=False):
        """
        :param image_paths: 包含多个数据路径的列表
        :param included_labels: 包含的标签类别
        :param indices: 使用的数据索引（可选）
        """
        self.image_paths = [image_path1,image_path2]
        self.included_labels = included_labels
        self.random_state = random_state
        self.use_augmentation = use_augmentation
        random.seed(self.random_state)  # 设置随机种子

        # 标签到数值的映射
        label_map = {'CN': 0, 'AD': 1, 'MCI': 2}

        # 确保所有included_labels都在label_map中
        if not all(label in label_map for label in included_labels):
            raise ValueError("Included labels must be one of 'CN', 'AD', 'MCI'")

        # 初始化文件名和标签列表
        self.file_names = []
        self.labels = []
        self.data_sources = []  # 新增：存储数据来源编号
        # 遍历所有路径，加载数据
        for image_path in self.image_paths:
            # 筛选符合条件的文件名
            all_files = os.listdir(image_path)
            # 只保留ADNI-GO(0)和ADNI-2(2)的数据
            files = [f for f in all_files 
                    if f.split('_')[0] in {str(label_map[label]) for label in included_labels}
                    and len(f.split('_')) >= 2 
                    and int(f.split('_')[1]) in {0, 2}]  # 0=ADNI-GO, 2=ADNI-2
            
            # 保存完整路径
            self.file_names.extend([os.path.join(image_path, f) for f in files])
            
            # 获取标签和数据来源
            for f in files:
                parts = f.split('_')
                self.labels.append(self.get_label_from_filename(f))
                self.data_sources.append(int(parts[1]))  
        # 打乱数据和标签
        combined = list(zip(self.file_names, self.labels))
        random.shuffle(combined)
        self.file_names[:], self.labels[:] = zip(*combined)
        if indices is not None:
            self.file_names = [self.file_names[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
        self.data = []  
        for file_name in tqdm(self.file_names, desc="Loading data to memory"):
            img = sio.loadmat(file_name)
            tensor_data = torch.from_numpy(img['data']).float()
            self.data.append(tensor_data)

    def get_label_from_filename(self, filename):
        """
        根据文件名和当前任务类型确定标签值
        严格遵循以下规则：
        - CN-AD分类: CN=0, AD=1, MCI=None
        - CN-MCI分类: CN=0, MCI=1, AD=None
        - AD-MCI分类: AD=1, MCI=0, CN=None 
        - CN-AD-MCI分类: CN=0, AD=1, MCI=2
        """
        label_char = filename[0]
        
        if label_char == '0':
            original_label = 'CN'
        elif label_char == '1':
            original_label = 'AD'
        elif label_char == '2':
            original_label = 'MCI'
        else:
            raise ValueError(f"无效的文件名前缀: {label_char}")

        if set(self.included_labels) == {'CN', 'AD'}:
            return {'CN': 0, 'AD': 1}.get(original_label)
        
        elif set(self.included_labels) == {'CN', 'MCI'}:
            return {'CN': 0, 'MCI': 1}.get(original_label)
        
        elif set(self.included_labels) == {'AD', 'MCI'}:
            return {'AD': 1, 'MCI': 0}.get(original_label)
        
        elif set(self.included_labels) == {'CN', 'AD', 'MCI'}:
            return {'CN': 0, 'AD': 1, 'MCI': 2}.get(original_label)
        
        else:
            raise ValueError(f"不支持的标签组合: {self.included_labels}")
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        if self.use_augmentation:
                tensor_data = data[0] 
                start_x = (tensor_data.shape[0] - 110) // 2
                start_y = (tensor_data.shape[1] - 110) // 2
                start_z = (tensor_data.shape[2] - 110) // 2

                tensor_data = tensor_data[
                    start_x:start_x+110, 
                    start_y:start_y+110, 
                    start_z:start_z+110
                ]
                tensor_data = np.transpose(tensor_data, (2, 0, 1))  
                tensor_data = tensor_data[..., np.newaxis]          
                tensor_data = get_dynamic_image(tensor_data.numpy())
                tensor_data = np.expand_dims(tensor_data,0)
                tensor_data = np.concatenate([tensor_data,tensor_data,tensor_data], 0)
                tensor_data = torch.from_numpy(tensor_data).float()
                return tensor_data,label
        else:
            return data, label