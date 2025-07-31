import os
from torch.utils.data import Dataset
from PIL import Image

class AnimeFaceDataset(Dataset):
    """
    自定义数据集类，适配新的目录结构:
    root_dir -> anime_id -> char_id -> image.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.img_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        if not root_dir or not os.path.isdir(root_dir):
            return
            
        current_idx = 0
        anime_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        for anime_id in anime_dirs:
            anime_folder = os.path.join(root_dir, anime_id)
            char_dirs = sorted([d for d in os.listdir(anime_folder) if os.path.isdir(os.path.join(anime_folder, d))])
            
            for char_id in char_dirs:
                class_name = f"{anime_id}/{char_id}"
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = current_idx
                    self.idx_to_class[current_idx] = class_name
                    current_idx += 1
                
                label = self.class_to_idx[class_name]
                char_folder = os.path.join(anime_folder, char_id)
                
                for img_name in os.listdir(char_folder):
                    img_path = os.path.join(char_folder, img_name)
                    if os.path.isfile(img_path):
                        self.img_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label 