import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class UIChartDataset(Dataset):
    def __init__(self, root_dir="data", task="cognitive_load", image_size=(128, 128)):
        self.root_dir = os.path.join(root_dir, task)
        self.image_size = image_size
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        
        for cls_name in self.classes:
            cls_folder = os.path.join(self.root_dir, cls_name)
            if os.path.isdir(cls_folder):
                for img_path in glob.glob(os.path.join(cls_folder, "*.png")):
                    self.samples.append((img_path, self.class_to_idx[cls_name]))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize(self.image_size)
        
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1)) # (3, H, W)
        
        img_tensor = torch.tensor(img_np, dtype=torch.float32)
        img_tensor = (img_tensor - 0.5) / 0.5
        
        return img_tensor, label
