import os
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
import torchvision.transforms.functional as TF
from tqdm import tqdm
import shutil
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from app.core.models import ArcFaceModel
from app.core.datasets import AnimeFaceDataset

import faiss

@dataclass
class ClusteringConfig:
    """聚类配置类"""
    eps: float = 0.4
    min_samples: int = 2
    metric: str = 'cosine'
    min_samples_for_clustering: int = 5


@dataclass
class DatabaseConfig:
    """数据库配置类"""
    db_path: str = "feature_db.npy"
    class_path: str = "class_to_idx.json"
    meta_path: str = 'meta/character_meta_restructured.json'
    batch_size: int = 512
    num_workers: int = 12


class FeatureExtractor:
    """特征提取器类"""
    
    def __init__(self, model: ArcFaceModel, device: torch.device, config: DatabaseConfig):
        self.model = model
        self.device = device
        self.config = config
        
    def extract_features(self, dataloader: DataLoader) -> Tuple[np.ndarray, List[int]]:
        """批量提取特征，并进行水平翻转增强 (TTA)"""
        self.model.eval()
        features = []
        labels = []

        with torch.no_grad():
            for imgs, labs in tqdm(dataloader, desc="提取特征中 (翻转增强)"):
                imgs = imgs.to(self.device)

                # 1. 原始图像特征
                feats_orig = self.model(imgs)

                # 2. 水平翻转图像特征
                imgs_flipped = TF.hflip(imgs)
                feats_flipped = self.model(imgs_flipped)

                # 3. 特征融合：将原始特征和翻转特征进行平均，然后重新归一化
                combined_feats = (feats_orig + feats_flipped) / 2.0
                final_feats = F.normalize(combined_feats, dim=1)

                features.append(final_feats.cpu().numpy())
                labels.extend(labs.numpy())

        features = np.vstack(features)
        return features, labels


class ClusteringProcessor:
    """聚类处理器类"""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        
    def process_character_features(self, feats: np.ndarray, character_name: str = "未知") -> np.ndarray:
        """处理单个角色的特征，生成多中心特征"""
        if len(feats) < self.config.min_samples_for_clustering:
            # 样本过少，直接求平均值
            centers = np.mean(feats, axis=0, keepdims=True)
        else:
            # 使用DBSCAN聚类
            centers = self._cluster_features(feats, character_name)
            
        # L2归一化
        return centers / np.linalg.norm(centers, axis=1, keepdims=True)
    
    def _cluster_features(self, feats: np.ndarray, character_name: str) -> np.ndarray:
        """使用DBSCAN对特征进行聚类"""
        db = DBSCAN(
            eps=self.config.eps, 
            min_samples=self.config.min_samples, 
            metric=self.config.metric
        ).fit(feats)
        cluster_labels = db.labels_
        
        centers = []
        unique_labels = set(cluster_labels)
        
        # 1. 处理非噪声点形成的簇，计算簇中心
        for k in unique_labels:
            if k == -1:  # 跳过噪声标签
                continue
            
            class_members = feats[cluster_labels == k]
            center = np.mean(class_members, axis=0)
            centers.append(center)
            
        # 2. 将所有噪声点（离群点）本身作为独立的特征中心
        noise_indices = np.where(cluster_labels == -1)[0]
        if noise_indices.size > 0:
            centers.extend(list(feats[noise_indices]))
            
        if not centers:
            # 如果所有点都是噪声，则使用全局平均值
            print(f"警告: 角色 '{character_name}' 的所有样本点均为噪声，将使用全局平均值作为唯一中心。")
            centers.append(np.mean(feats, axis=0))
            
        return np.stack(centers)


class DatabaseManager:
    """数据库管理器类"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.meta_loader = MetaDataLoader(config.meta_path)
        
    def load_database(self) -> Tuple[Dict, Dict]:
        """加载特征数据库和类别映射"""
        if os.path.exists(self.config.db_path) and os.path.exists(self.config.class_path):
            feature_db = np.load(self.config.db_path, allow_pickle=True).item()
            with open(self.config.class_path, "r", encoding="utf-8") as f:
                class_to_idx = json.load(f)
            print("已成功加载现有的特征数据库和类别映射。")
            return feature_db, class_to_idx
        else:
            print("未找到现有数据库，将创建新的数据库。")
            return {}, {}
    
    def save_database(self, feature_db: Dict, class_to_idx: Dict) -> None:
        """保存特征数据库和类别映射"""
        np.save(self.config.db_path, feature_db)
        with open(self.config.class_path, "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, ensure_ascii=False, indent=4)
        print("\n特征数据库和类别映射保存完成！")
    
    def get_character_name(self, label_name: str) -> str:
        """获取角色的可读名称"""
        try:
            anime_id, char_id = label_name.split('/')
            anime_name = self.meta_loader.get_anime_name(anime_id)
            char_name = self.meta_loader.get_character_name(anime_id, char_id)
            return f"{anime_name} - {char_name}"
        except:
            return label_name


class MetaDataLoader:
    """元数据加载器类"""
    
    def __init__(self, meta_path: str):
        self.meta_path = meta_path
        self._meta_info = None
        
    def _load_meta(self) -> Dict:
        """加载元数据"""
        if self._meta_info is None:
            self._meta_info = {}
            if os.path.exists(self.meta_path):
                with open(self.meta_path, 'r', encoding='utf-8') as f:
                    self._meta_info = json.load(f)
        return self._meta_info
    
    def get_anime_name(self, anime_id: str) -> str:
        """获取动漫名称"""
        meta_info = self._load_meta()
        return meta_info.get(anime_id, {}).get("name", anime_id)
    
    def get_character_name(self, anime_id: str, char_id: str) -> str:
        """获取角色名称"""
        meta_info = self._load_meta()
        return meta_info.get(anime_id, {}).get("characters", {}).get(char_id, {}).get("name", char_id)


class FileManager:
    """文件管理器类"""
    
    @staticmethod
    def move_character_files(class_names: List[str], source_dir: str, dest_dir: str) -> None:
        """移动角色文件从源目录到目标目录"""
        print(f"\n正在将已处理角色从 {source_dir} 移动到 {dest_dir}...")
        
        for class_name in class_names:
            anime_id, char_id = class_name.split('/')
            source_path = os.path.join(source_dir, anime_id, char_id)
            dest_anime_path = os.path.join(dest_dir, anime_id)
            dest_path = os.path.join(dest_anime_path, char_id)
            
            if os.path.exists(source_path):
                try:
                    os.makedirs(dest_anime_path, exist_ok=True)
                    if os.path.exists(dest_path):
                        print(f"警告：目标目录 '{dest_path}' 已存在。将进行覆盖。")
                        shutil.rmtree(dest_path)
                    shutil.move(source_path, dest_path)
                    print(f"  - 已移动: {class_name}")
                except Exception as e:
                    print(f"  - 移动 '{class_name}' 失败: {e}")
            else:
                print(f"  - 警告: 源目录 '{source_path}' 未找到，无法移动。")
        
        # 清理空的动漫文件夹
        FileManager._cleanup_empty_dirs(source_dir)
        print("移动操作完成。")
    
    @staticmethod
    def move_image_files(new_images_by_char: Dict[str, List[str]], 
                        new_character_dir: str, 
                        train_dir: str) -> None:
        """移动图片文件"""
        print("正在归档已处理的样本文件...")
        
        for char_name, new_image_paths in new_images_by_char.items():
            anime_id, c_id = char_name.split('/')
            old_image_dir = os.path.join(train_dir, anime_id, c_id)
            new_image_dir = os.path.join(new_character_dir, anime_id, c_id)

            os.makedirs(old_image_dir, exist_ok=True)
            for img_path in new_image_paths:
                if not os.path.exists(img_path): 
                    continue
                    
                filename = os.path.basename(img_path)
                dest_path = os.path.join(old_image_dir, filename)
                
                if os.path.exists(dest_path):
                    name, ext = os.path.splitext(filename)
                    timestamp = int(time.time() * 1000)
                    new_filename = f"{name}_{timestamp}{ext}"
                    dest_path = os.path.join(old_image_dir, new_filename)
                
                try:
                    shutil.move(img_path, dest_path)
                except Exception as e:
                    print(f"  - 移动文件 '{filename}' 失败: {e}")
            
            try:
                if os.path.exists(new_image_dir) and os.path.isdir(new_image_dir):
                    shutil.rmtree(new_image_dir)
            except Exception as e:
                print(f"删除目录 '{new_image_dir}' 失败: {e}")

        FileManager._cleanup_empty_dirs(new_character_dir)
        print("文件归档完成。")
    
    @staticmethod
    def _cleanup_empty_dirs(directory: str) -> None:
        """清理空目录"""
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path) and not os.listdir(item_path):
                os.rmdir(item_path)


class FeatureDatabaseBuilder:
    """特征数据库构建器类"""
    
    def __init__(self, 
                 model: ArcFaceModel, 
                 device: torch.device,
                 db_config: DatabaseConfig = None,
                 clustering_config: ClusteringConfig = None):
        self.db_config = db_config or DatabaseConfig()
        self.clustering_config = clustering_config or ClusteringConfig()
        
        self.extractor = FeatureExtractor(model, device, self.db_config)
        self.clustering_processor = ClusteringProcessor(self.clustering_config)
        self.db_manager = DatabaseManager(self.db_config)
    
    def build_database(self, dataset: AnimeFaceDataset) -> None:
        """构建并保存特征数据库"""
        dataloader = DataLoader(
            dataset, 
            batch_size=self.db_config.batch_size, 
            shuffle=False, 
            num_workers=self.db_config.num_workers, 
            persistent_workers=True
        )
        
        features, labels = self.extractor.extract_features(dataloader)
        features_by_label = defaultdict(list)
        for feat, label in zip(features, labels):
            features_by_label[label].append(feat)

        feature_db = {}
        print("\n使用 DBSCAN 聚类生成多中心特征...")
        
        for label, feats in tqdm(features_by_label.items(), desc="处理角色特征"):
            feats = np.stack(feats)
            label_name = dataset.idx_to_class.get(label, "未知/未知")
            character_name = self.db_manager.get_character_name(label_name)
            
            centers = self.clustering_processor.process_character_features(feats, character_name)
            feature_db[label] = centers

        self.db_manager.save_database(feature_db, dataset.class_to_idx)
    
    def add_new_characters(self, 
                          transform: transforms.Compose,
                          new_character_dir: str = "./data/new") -> Tuple[str, int]:
        """向数据库添加新角色"""
        if not os.path.exists(new_character_dir):
            raise FileNotFoundError(f"提供的路径 '{new_character_dir}' 不存在。")

        # 加载现有数据库
        feature_db, class_to_idx = self.db_manager.load_database()
        
        # 创建新角色数据集
        new_dataset = AnimeFaceDataset(root_dir=new_character_dir, transform=transform)
        if not new_dataset.img_paths:
            raise ValueError(f"在 '{new_character_dir}' 中没有找到任何图片。")

        # 过滤新角色
        original_labels = list(class_to_idx.keys())
        new_class_names = [name for name in new_dataset.class_to_idx.keys() 
                          if name not in original_labels]

        if not new_class_names:
            return "所有检测到的角色均已存在于数据库中，无需添加。", 0
        
        print(f"检测到新角色: {', '.join(new_class_names)}")

        # 提取特征
        new_dataloader = DataLoader(new_dataset, batch_size=self.db_config.batch_size, 
                                  shuffle=False, num_workers=0)
        new_features, new_labels_idx = self.extractor.extract_features(new_dataloader)
        
        # 按标签组织特征
        new_features_by_label_idx = defaultdict(list)
        for feat, label_idx in zip(new_features, new_labels_idx):
            new_features_by_label_idx[label_idx].append(feat)

        # 处理新角色特征
        highest_existing_label_id = max(class_to_idx.values()) if class_to_idx else -1
        result_data = ""
        updated_count = 0
        
        for class_name in new_class_names:
            new_local_idx = new_dataset.class_to_idx[class_name]
            feats = np.stack(new_features_by_label_idx[new_local_idx])
            
            centers = self.clustering_processor.process_character_features(feats, class_name)
            
            # 分配新ID并更新
            highest_existing_label_id += 1
            new_global_id = highest_existing_label_id
            feature_db[new_global_id] = centers
            class_to_idx[class_name] = new_global_id
            
            result_data += f"已成功为角色 '{class_name}' (ID: {new_global_id}) 添加 {len(centers)} 个特征中心。\n"
            updated_count += 1

        # 保存数据库
        self.db_manager.save_database(feature_db, class_to_idx)
        
        # 移动文件
        train_dir = "./data/train"
        os.makedirs(train_dir, exist_ok=True)
        FileManager.move_character_files(new_class_names, new_character_dir, train_dir)
        
        return result_data, updated_count
    
    def update_existing_characters(self, 
                                 transform: transforms.Compose,
                                 new_character_dir: str = "./data/new",
                                 train_dir: str = "./data/train") -> Tuple[str, int]:
        """更新现有角色的特征"""
        if not os.path.exists(new_character_dir):
            raise FileNotFoundError(f"提供的路径 '{new_character_dir}' 不存在。")

        # 加载数据库
        feature_db, class_to_idx = self.db_manager.load_database()
        
        # 扫描新角色
        all_new_chars = self._scan_new_characters(new_character_dir)
        characters_to_update = [char for char in all_new_chars if char in class_to_idx]
        newly_found_chars = [char for char in all_new_chars if char not in class_to_idx]

        if newly_found_chars:
            print(f"检测到新角色: {', '.join(newly_found_chars)}。请使用 '添加新角色' 功能。")

        if not characters_to_update:
            return f"在 '{new_character_dir}' 中没有找到需要更新的现有角色文件夹。", 0
        
        print(f"\n准备批量更新以下 {len(characters_to_update)} 个角色: {', '.join(characters_to_update)}")

        # 聚合图片路径
        all_image_paths, all_labels, new_images_by_char = self._aggregate_image_paths(
            characters_to_update, class_to_idx, new_character_dir, train_dir
        )

        if not all_image_paths:
            raise ValueError("未能收集到任何待处理的图片。")
        
        print(f"共找到 {len(all_image_paths)} 张图片。开始批量提取特征...")

        # 提取特征
        update_dataset = AnimeFaceDataset(root_dir=None, transform=transform)
        update_dataset.img_paths = all_image_paths
        update_dataset.labels = all_labels
        
        update_dataloader = DataLoader(update_dataset, batch_size=self.db_config.batch_size, 
                                     shuffle=False, num_workers=0)
        all_features, all_processed_labels = self.extractor.extract_features(update_dataloader)
        
        # 按标签分组特征
        features_by_label = defaultdict(list)
        for feat, label in zip(all_features, all_processed_labels):
            features_by_label[label].append(feat)

        print("特征提取完成。开始为每个角色更新特征中心...")
        
        # 更新特征
        result_data = ""
        updated_count = 0
        
        for char_name in characters_to_update:
            label_id = class_to_idx[char_name]
            
            if label_id not in features_by_label:
                print(f"- 警告: 未能为角色 '{char_name}' 提取到任何特征，跳过更新。")
                continue
                
            feats = np.stack(features_by_label[label_id])
            centers = self.clustering_processor.process_character_features(feats, char_name)
            
            feature_db[label_id] = centers
            result_data += f"- 角色 '{char_name}' 的特征已更新，现有 {len(centers)} 个特征中心。\n"
            updated_count += 1

        # 保存并移动文件
        if updated_count > 0:
            self.db_manager.save_database(feature_db, class_to_idx)
            FileManager.move_image_files(new_images_by_char, new_character_dir, train_dir)
            print("\n批量更新操作成功完成！")
        else:
            print("\n没有角色被成功更新。")
            
        return result_data, updated_count
    
    def rebuild_database(self, transform: transforms.Compose, train_dir: str = "./data/train") -> None:
        """重建整个特征数据库"""
        print(f"\n模式: 重建整个数据库, 来源: '{train_dir}'")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"目录 '{train_dir}' 不存在。")
            
        dataset = AnimeFaceDataset(train_dir, transform=transform)
        if not dataset.img_paths:
            raise ValueError(f"在 '{train_dir}' 中未找到任何图片。")
            
        self.build_database(dataset)
    
    def _scan_new_characters(self, new_character_dir: str) -> List[str]:
        """扫描新角色目录"""
        all_new_chars = []
        for anime_id in os.listdir(new_character_dir):
            anime_path = os.path.join(new_character_dir, anime_id)
            if not os.path.isdir(anime_path): 
                continue
            for char_id in os.listdir(anime_path):
                char_path = os.path.join(anime_path, char_id)
                if not os.path.isdir(char_path): 
                    continue
                all_new_chars.append(f"{anime_id}/{char_id}")
        return all_new_chars
    
    def _aggregate_image_paths(self, 
                             characters_to_update: List[str],
                             class_to_idx: Dict,
                             new_character_dir: str,
                             train_dir: str) -> Tuple[List[str], List[int], Dict[str, List[str]]]:
        """聚合图片路径"""
        all_image_paths = []
        all_labels = []
        new_images_by_char = defaultdict(list)
        
        for char_name in characters_to_update:
            label_id = class_to_idx[char_name]
            anime_id, c_id = char_name.split('/')
            
            old_image_dir = os.path.join(train_dir, anime_id, c_id)
            new_image_dir = os.path.join(new_character_dir, anime_id, c_id)
            
            current_char_paths = []
            if os.path.exists(old_image_dir):
                current_char_paths.extend([
                    os.path.join(old_image_dir, f) 
                    for f in os.listdir(old_image_dir) 
                    if os.path.isfile(os.path.join(old_image_dir, f))
                ])
            
            new_image_paths = [
                os.path.join(new_image_dir, f) 
                for f in os.listdir(new_image_dir) 
                if os.path.isfile(os.path.join(new_image_dir, f))
            ]
            current_char_paths.extend(new_image_paths)
            new_images_by_char[char_name] = new_image_paths
            
            all_image_paths.extend(current_char_paths)
            all_labels.extend([label_id] * len(current_char_paths))
        
        return all_image_paths, all_labels, new_images_by_char


# 兼容性函数 - 保持原有接口
def extract_features(model, dataloader, device):
    """兼容性函数"""
    extractor = FeatureExtractor(model, device, DatabaseConfig())
    return extractor.extract_features(dataloader)


def build_and_save_feature_database(model, dataset, device, save_path="feature_db.npy", class_path="class_to_idx.json"):
    """兼容性函数"""
    config = DatabaseConfig(db_path=save_path, class_path=class_path)
    builder = FeatureDatabaseBuilder(model, device, config)
    builder.build_database(dataset)


def add_character_to_database(model, device, transform, new_character_dir="./data/new", db_path="feature_db.npy", class_path="class_to_idx.json"):
    """兼容性函数"""
    config = DatabaseConfig(db_path=db_path, class_path=class_path)
    builder = FeatureDatabaseBuilder(model, device, config)
    return builder.add_new_characters(transform, new_character_dir)


def update_characters_in_database(model, device, transform, new_character_dir="./data/new", train_dir="./data/train", db_path="feature_db.npy", class_path="class_to_idx.json"):
    """兼容性函数"""
    config = DatabaseConfig(db_path=db_path, class_path=class_path)
    builder = FeatureDatabaseBuilder(model, device, config)
    return builder.update_existing_characters(transform, new_character_dir, train_dir)


def rebuild_full_database(model, device, transform, train_dir="./data/train"):
    """兼容性函数"""
    builder = FeatureDatabaseBuilder(model, device)
    builder.rebuild_database(transform, train_dir)

def build_faiss_index(app_instance):
    """
    根据 app context 中当前的 feature_db 构建 Faiss 索引。
    :param app_instance: Flask app object (or current_app).
    """
    try:
        app_instance.logger.info("开始构建Faiss索引...")
        if not hasattr(app_instance, 'feature_db') or not app_instance.feature_db:
             app_instance.faiss_index = None
             app_instance.logger.warning("feature_db 为空或不存在，跳过Faiss索引构建。")
             return

        all_features_list = []
        app_instance.faiss_index_to_label = []

        all_labels_in_db = set(app_instance.feature_db.keys())
        idx_to_class_local = {v: k for k,v in app_instance.class_to_idx.items()}

        for i in range(len(idx_to_class_local)):
            if i in all_labels_in_db:
                centers = app_instance.feature_db[i]
                if centers.ndim == 1:
                    centers = centers.reshape(1, -1)
                
                for feature_vector in centers:
                    all_features_list.append(feature_vector)
                    app_instance.faiss_index_to_label.append(i)

        if not all_features_list:
            app_instance.faiss_index = None
            app_instance.logger.warning("特征数据库为空，Faiss索引未创建。")
        else:
            all_features_np = np.vstack(all_features_list).astype('float32')
            dimension = all_features_np.shape[1]
            
            app_instance.faiss_index = faiss.IndexFlatIP(dimension)
            app_instance.faiss_index.add(all_features_np)
            app_instance.logger.info(f"Faiss索引成功创建。向量总数: {app_instance.faiss_index.ntotal}")

    except Exception as e:
        app_instance.faiss_index = None
        app_instance.logger.error(f"构建Faiss索引时出错: {e}", exc_info=True)

# 默认变换
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
