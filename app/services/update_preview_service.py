import os
import json
import base64
import io
import shutil
import re
from PIL import Image
from flask import current_app


class UpdatePreviewService:
    """处理数据库更新预览相关的业务逻辑"""
    
    def __init__(self):
        self.new_data_dir = current_app.config.get("NEW_DATA_DIR")
        self.train_data_dir = current_app.config.get("TRAIN_DATA_DIR")
        self.character_meta_path = current_app.config.get("CHARACTER_META_PATH")
        self.character_meta = current_app.character_meta
        self.class_to_idx = current_app.class_to_idx
    
    def get_folder_image_previews(self, folder_path):
        """生成文件夹中所有图片的缩略图预览，包括文件名"""
        previews = []
        image_extensions = (".png", ".jpg", ".jpeg", ".webp")
        
        if not os.path.exists(folder_path):
            return previews

        # 优先显示非隐藏文件
        filenames = sorted([f for f in os.listdir(folder_path) if not f.startswith(".")])

        for img_file in filenames:
            if img_file.lower().endswith(image_extensions):
                try:
                    img_path = os.path.join(folder_path, img_file)
                    with Image.open(img_path) as img:
                        img.thumbnail((64, 64))
                        buf = io.BytesIO()
                        img.convert("RGB").save(buf, format="JPEG", quality=80)
                        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                        previews.append({"filename": img_file, "b64": b64_str})
                except Exception as e:
                    current_app.logger.error(f"Error generating preview for {img_path}: {e}")
        
        return previews
    
    def scan_new_data_directory(self):
        """扫描新数据目录，获取待更新和新增的角色信息"""
        updates = []
        additions = []
        
        # class_to_idx 的 key 现在是 'anime_id/c_id'
        existing_ids = set(self.class_to_idx.keys())

        if not os.path.exists(self.new_data_dir):
            return updates, additions

        # 遍历 anime 文件夹
        for anime_id in os.listdir(self.new_data_dir):
            anime_folder = os.path.join(self.new_data_dir, anime_id)
            if not os.path.isdir(anime_folder):
                continue
            
            anime_info_from_meta = self.character_meta.get(anime_id, {})
            anime_name_from_meta = anime_info_from_meta.get("name", anime_id)

            # 遍历 character 文件夹
            for char_id in os.listdir(anime_folder):
                char_folder = os.path.join(anime_folder, char_id)
                if not os.path.isdir(char_folder) or not os.listdir(char_folder):
                    continue

                composite_id = f"{anime_id}/{char_id}"
                
                char_info_from_meta = anime_info_from_meta.get("characters", {}).get(char_id, {})
                char_name_from_meta = char_info_from_meta.get("name", char_id)

                previews = self.get_folder_image_previews(char_folder)
                if not previews:
                    continue

                item_data = {
                    "id": composite_id,
                    "name": char_name_from_meta,
                    "anime": anime_name_from_meta,
                    "previews": previews,
                    "image_count": len(previews),
                }

                if composite_id in existing_ids:
                    # 这是更新
                    existing_char_folder = os.path.join(self.train_data_dir, anime_id, char_id)
                    item_data["existing_previews"] = self.get_folder_image_previews(existing_char_folder)
                    updates.append(item_data)
                else:
                    # 这是新增
                    additions.append(item_data)

        return updates, additions
    
    def group_characters_by_anime(self, character_list):
        """按动漫分组角色列表"""
        # 按角色名称排序
        character_list.sort(key=lambda x: x["name"])

        grouped = {}
        for char in character_list:
            anime_name = char.get("anime", "未知作品")
            if anime_name not in grouped:
                grouped[anime_name] = []
            grouped[anime_name].append(char)
        
        return grouped
    
    def get_database_changes_preview(self):
        """获取数据库变更预览，按动漫分组"""
        updates, additions = self.scan_new_data_directory()
        
        updates_grouped = self.group_characters_by_anime(updates)
        additions_grouped = self.group_characters_by_anime(additions)
        
        return {
            "updates_grouped": updates_grouped,
            "additions_grouped": additions_grouped
        }
    
    def validate_character_id(self, char_id):
        """验证角色ID格式"""
        if not char_id or "/" not in char_id:
            return False, "缺少或无效的角色ID"
        
        try:
            anime_id, c_id = char_id.split('/', 1)
            if not (re.match(r"^anime_\d{5}$", anime_id) and re.match(r"^c_\d{5}$", c_id)):
                return False, "无效的角色ID格式"
            return True, (anime_id, c_id)
        except ValueError:
            return False, "角色ID格式错误"
    
    def delete_pending_character(self, char_id):
        """删除待处理角色"""
        # 验证角色ID
        is_valid, result = self.validate_character_id(char_id)
        if not is_valid:
            return False, result
        
        anime_id, c_id = result
        
        # 删除 data/new 目录下的文件夹
        char_path = os.path.join(self.new_data_dir, anime_id, c_id)
        parent_dir = os.path.dirname(char_path)

        if not os.path.isdir(char_path):
            return False, "待处理的角色目录不存在或已被删除"

        try:
            shutil.rmtree(char_path)
            current_app.logger.info(f"已删除 new 目录下的待处理文件夹: {char_path}")
            
            # 清理空的父级动漫文件夹
            if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                os.rmdir(parent_dir)
                current_app.logger.info(f"已删除空的 new 动漫文件夹: {parent_dir}")
        except Exception as e:
            current_app.logger.error(f"删除文件夹 {char_path} 时出错: {e}")
            return False, f"删除文件夹失败: {e}"
        
        # 检查是否为新角色，如果是则从元数据中删除
        train_char_folder = os.path.join(self.train_data_dir, anime_id, c_id)
        if not os.path.isdir(train_char_folder):
            current_app.logger.info(f"检测到 {char_id} 是一个仅存在于 new 目录中的角色，将从元数据中删除。")
            meta_update_result = self._remove_character_from_meta(anime_id, c_id, char_id)
            if not meta_update_result[0]:
                return True, f"成功删除角色文件夹，但更新元数据时出错: {meta_update_result[1]}"
        
        return True, f"成功删除待处理角色: {char_id}"
    
    def _remove_character_from_meta(self, anime_id, c_id, char_id):
        """从元数据中移除角色"""
        try:
            with open(self.character_meta_path, "r+", encoding="utf-8") as f:
                meta = json.load(f)
                if anime_id in meta and c_id in meta[anime_id].get("characters", {}):
                    del meta[anime_id]["characters"][c_id]
                    # 如果这是最后一个角色，删除动漫条目
                    if not meta[anime_id]["characters"]:
                        del meta[anime_id]
                    
                    f.seek(0)
                    json.dump(meta, f, ensure_ascii=False, indent=4)
                    f.truncate()
                    current_app.character_meta = meta  # 重新加载元数据到应用上下文
                    current_app.logger.info(f"已成功从元数据中移除角色 {char_id}")
                    return True, None
        except Exception as e:
            current_app.logger.error(f"从元数据中删除角色 {char_id} 时出错: {e}")
            return False, str(e) 