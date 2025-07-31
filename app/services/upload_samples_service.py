import os
import json
import base64
import io
import time
from PIL import Image
from flask import current_app


class UploadSamplesService:
    """上传样本服务类，处理角色样本上传相关的业务逻辑"""
    
    def __init__(self):
        self.meta_path = current_app.config.get("CHARACTER_META_PATH")
        self.train_data_dir = current_app.config.get("TRAIN_DATA_DIR")
        self.new_data_dir = current_app.config.get("NEW_DATA_DIR")
        
    
    def get_all_animes(self):
        """获取所有动漫作品的列表"""
        if not os.path.exists(self.meta_path):
            return []
        
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        return sorted([info["name"] for info in meta.values()])
    
    def get_characters_by_anime(self, anime_name):
        """根据动漫作品名称获取角色列表（带预览图）"""
        if not anime_name:
            raise ValueError("未提供动漫名称")
        
        if not os.path.exists(self.meta_path):
            return []
        
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        characters = []
        
        # 找到匹配的动漫ID
        target_anime_id = None
        target_anime_info = None
        for anime_id, anime_info in meta.items():
            if anime_info.get("name") == anime_name:
                target_anime_id = anime_id
                target_anime_info = anime_info
                break
        
        if not target_anime_info:
            return []
        
        for c_id, char_info in target_anime_info.get("characters", {}).items():
            char_folder = os.path.join(self.train_data_dir, target_anime_id, c_id)
            preview_b64 = self._generate_preview_image(char_folder)
            
            composite_id = f"{target_anime_id}/{c_id}"
            characters.append({
                "id": composite_id,
                "name": char_info.get("name", "未知"),
                "preview_b64": preview_b64,
            })
        
        return sorted(characters, key=lambda x: x["name"])
    
    def _generate_preview_image(self, char_folder):
        """为角色生成预览图"""
        if not os.path.exists(char_folder):
            return None
        
        try:
            first_img = next(
                (f for f in os.listdir(char_folder)
                 if os.path.isfile(os.path.join(char_folder, f))),
                None
            )
            
            if first_img:
                with Image.open(os.path.join(char_folder, first_img)) as img:
                    img.thumbnail((64, 64))
                    buf = io.BytesIO()
                    img.convert("RGB").save(buf, format="JPEG")
                    return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            current_app.logger.error(f"为角色文件夹 {char_folder} 生成预览图失败: {e}")
        
        return None
    
    def upload_sample_for_existing_character(self, character_id, image_b64):
        """为已有角色上传样本图片"""
        if not character_id:
            raise ValueError("未提供角色ID (character_id)")
        
        if not image_b64:
            raise ValueError("未提供图片数据 (image_b64)")
        
        # 角色ID现在是 'anime_id/c_id' 格式
        if "/" not in character_id:
            raise ValueError("提供的角色ID格式无效")
        
        anime_id, c_id = character_id.split("/", 1)
        target_dir = os.path.join(self.new_data_dir, anime_id, c_id)
        os.makedirs(target_dir, exist_ok=True)
        
        # 解码 base64 并保存图片
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        
        # 为防止文件名冲突，使用时间戳
        new_filename = f"sample_{int(time.time() * 1000)}.png"
        save_path = os.path.join(target_dir, new_filename)
        image.save(save_path, "PNG")
        
        return new_filename
    
    def create_new_character(self, anime_name, char_name, image_b64):
        """创建新角色并保存第一个样本"""
        if not all([anime_name, char_name, image_b64]):
            raise ValueError("缺少参数 (anime_name, character_name, image_b64)")
        
        # 加载元数据
        meta = self._load_meta_data()
        
        # 查找或创建 anime_id
        target_anime_id = self._find_or_create_anime_id(meta, anime_name)
        
        # 检查角色是否已存在
        self._check_character_exists(meta, target_anime_id, char_name)
        
        # 生成新的角色ID
        new_char_id = self._generate_new_character_id(meta, target_anime_id)
        
        # 更新元数据
        meta[target_anime_id]["characters"][new_char_id] = {"name": char_name}
        
        # 创建文件夹并保存图片
        new_filename = self._save_character_image(target_anime_id, new_char_id, image_b64)
        
        # 保存元数据文件
        self._save_meta_data(meta)
        
        # 重载全局变量
        current_app.character_meta = meta
        
        return f"{target_anime_id}/{new_char_id}", new_filename
    
    def _load_meta_data(self):
        """加载元数据文件"""
        if not os.path.exists(self.meta_path):
            return {}
        
        with open(self.meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _find_or_create_anime_id(self, meta, anime_name):
        """查找或创建动漫ID"""
        # 查找现有动漫
        for an_id, an_info in meta.items():
            if an_info.get("name") == anime_name:
                return an_id
        
        # 创建新动漫ID
        max_anime_id_num = 0
        for key in meta.keys():
            if key.startswith("anime_"):
                max_anime_id_num = max(max_anime_id_num, int(key.split("_")[1]))
        
        new_anime_id = f"anime_{max_anime_id_num + 1:05d}"
        meta[new_anime_id] = {"name": anime_name, "characters": {}}
        
        return new_anime_id
    
    def _check_character_exists(self, meta, anime_id, char_name):
        """检查角色是否已存在"""
        anime_chars = meta[anime_id]["characters"]
        for c_info in anime_chars.values():
            if c_info.get("name") == char_name:
                raise ValueError(f"角色 '{char_name}' 已存在于作品 '{meta[anime_id]['name']}' 中")
    
    def _generate_new_character_id(self, meta, anime_id):
        """生成新的角色ID"""
        anime_chars = meta[anime_id]["characters"]
        max_char_id_num = 0
        
        for key in anime_chars.keys():
            if key.startswith("c_"):
                max_char_id_num = max(max_char_id_num, int(key.split("_")[1]))
        
        return f"c_{max_char_id_num + 1:05d}"
    
    def _save_character_image(self, anime_id, char_id, image_b64):
        """保存角色图片"""
        target_dir = os.path.join(self.new_data_dir, anime_id, char_id)
        os.makedirs(target_dir, exist_ok=True)
        
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        new_filename = f"sample_{int(time.time() * 1000)}.png"
        save_path = os.path.join(target_dir, new_filename)
        image.save(save_path, "PNG")
        
        return new_filename
    
    def _save_meta_data(self, meta):
        """保存元数据文件"""
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=4) 