import base64
import io
import json
import os
import re
import shutil
from threading import Lock

from PIL import Image

# 使用线程锁来确保对元数据文件的并发写操作是安全的
meta_lock = Lock()


class GalleryService:
    """封装所有与画廊相关的业务逻辑和数据操作"""

    def __init__(self, character_meta_path, train_data_dir, new_data_dir, logger):
        self.character_meta_path = character_meta_path
        self.train_data_dir = train_data_dir
        self.new_data_dir = new_data_dir
        self.logger = logger
        self.image_extensions = (".png", ".jpg", ".jpeg", ".webp", ".gif")

    # --- 元数据核心操作 ---

    def _load_meta(self):
        """加载并返回元数据。如果文件不存在或解析失败，则返回空字典。"""
        if not os.path.exists(self.character_meta_path):
            self.logger.warning(f"元数据文件不存在: {self.character_meta_path}")
            return {}
        try:
            with open(self.character_meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"加载或解析元数据文件失败: {e}")
            return {}

    def _save_meta(self, meta):
        """将元数据写回文件。"""
        try:
            with meta_lock:
                with open(self.character_meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=4)
            return True
        except IOError as e:
            self.logger.error(f"保存元数据文件失败: {e}")
            return False

    # --- 动漫列表和统计 ---

    def get_anime_list_data(self):
        """获取用于展示的动漫列表及其元数据统计信息。"""
        meta = self._load_meta()
        anime_list = []
        for anime_id, anime_info in meta.items():
            anime_name = anime_info.get("name", anime_id)
            anime_list.append(
                {"name": anime_name, "count": len(anime_info.get("characters", {}))}
            )

        sorted_animes = sorted(
            anime_list, key=lambda item: (item["name"] == "未分类", item["name"])
        )
        total_animes = len(meta)
        total_characters = sum(
            len(info.get("characters", {})) for info in meta.values()
        )

        return {
            "all_animes": sorted_animes,
            "total_animes": total_animes,
            "total_characters": total_characters,
        }

    def get_total_images_stats(self):
        """计算所有已训练角色的图片总数。这是一个耗时的操作。"""
        meta = self._load_meta()
        total_images = 0
        for anime_id, anime_info in meta.items():
            for char_id in anime_info.get("characters", {}).keys():
                char_folder = os.path.join(self.train_data_dir, anime_id, char_id)
                if os.path.isdir(char_folder):
                    try:
                        image_count = len(
                            [
                                f
                                for f in os.listdir(char_folder)
                                if not f.startswith(".")
                                and f.lower().endswith(self.image_extensions)
                            ]
                        )
                        total_images += image_count
                    except OSError as e:
                        self.logger.warning(
                            f"无法读取角色目录 {char_folder} 的图片数量: {e}"
                        )
        return total_images

    # --- 角色和图片详情 ---

    def get_characters_by_anime(self, anime_name):
        """根据动漫名称获取其下所有角色的详细信息，合并 train 和 new 目录。"""
        meta = self._load_meta()
        target_anime_id, target_anime_info = None, None
        for anime_id, anime_info in meta.items():
            if anime_info.get("name") == anime_name:
                target_anime_id = anime_id
                target_anime_info = anime_info
                break

        if not target_anime_info:
            return []

        characters_data = []
        for c_id, char_info in target_anime_info.get("characters", {}).items():
            char_name = char_info.get("name", "未知")
            existing_folder = os.path.join(self.train_data_dir, target_anime_id, c_id)
            pending_folder = os.path.join(self.new_data_dir, target_anime_id, c_id)

            existing_files, existing_count = self._get_image_files(existing_folder)
            pending_files, pending_count = self._get_image_files(pending_folder)
            total_count = existing_count + pending_count

            if total_count == 0:
                continue

            preview_b64 = self._generate_preview_b64(
                pending_folder if pending_count > 0 else existing_folder,
                pending_files[0] if pending_count > 0 else existing_files[0],
                f"{target_anime_id}/{c_id}",
            )

            characters_data.append(
                {
                    "id": f"{target_anime_id}/{c_id}",
                    "name": char_name,
                    "preview_b64": preview_b64,
                    "image_count": total_count,
                }
            )

        return sorted(characters_data, key=lambda x: x["name"])

    def get_character_images(self, character_id):
        """获取指定角色的所有图片，区分为 'existing' 和 'pending'。"""
        anime_id, c_id = character_id.split("/", 1)
        existing_folder = os.path.join(self.train_data_dir, anime_id, c_id)
        pending_folder = os.path.join(self.new_data_dir, anime_id, c_id)

        return {
            "existing": self._get_images_with_previews_from_dir(existing_folder),
            "pending": self._get_images_with_previews_from_dir(pending_folder),
        }

    # --- 写操作 (增/删/改) ---

    def rename_character(self, character_id, new_name):
        """重命名角色。"""
        with meta_lock:
            meta = self._load_meta()
            anime_id, c_id = character_id.split("/", 1)

            if anime_id not in meta or c_id not in meta[anime_id].get("characters", {}):
                return False, "角色ID不存在", 404

            # 检查重名
            for char_key, char_info in meta[anime_id]["characters"].items():
                if char_key != c_id and char_info.get("name") == new_name:
                    msg = f"角色 '{new_name}' 已存在于作品 '{meta[anime_id]['name']}' 中"
                    return False, msg, 409

            meta[anime_id]["characters"][c_id]["name"] = new_name
            if self._save_meta(meta):
                return True, "角色已成功重命名！", 200
            else:
                return False, "写入元数据文件失败", 500

    def delete_character_and_images(self, character_id):
        """完全删除一个角色，包括其所有图片和元数据。"""
        anime_id, c_id = character_id.split("/", 1)
        train_folder = os.path.join(self.train_data_dir, anime_id, c_id)
        new_folder = os.path.join(self.new_data_dir, anime_id, c_id)

        # 1. 删除文件目录
        try:
            self._cleanup_dir(train_folder)
            self._cleanup_dir(new_folder)
        except Exception as e:
            self.logger.error(f"删除角色目录时出错 ({character_id}): {e}")
            return False, f"删除角色文件时出错: {e}", 500

        # 2. 从元数据中删除
        with meta_lock:
            meta = self._load_meta()
            if anime_id in meta and c_id in meta[anime_id].get("characters", {}):
                del meta[anime_id]["characters"][c_id]
                if not meta[anime_id]["characters"]:
                    del meta[anime_id]
                    self.logger.info(f"动漫 '{anime_id}' 已无角色，已从元数据中移除。")

                if self._save_meta(meta):
                    return True, "角色已成功删除。", 200
                else:
                    return False, "更新元数据时出错", 500
            return True, "角色目录已删除，元数据中未找到对应条目。", 200 # 即使元数据没有也算成功

    def delete_character_images(self, character_id, filenames, location):
        """
        从指定位置('train' or 'new')删除一个或多个图片。
        删除后会进行清理，如果角色在所有位置的图片都已清空，则会从元数据中移除该角色。
        """
        anime_id, c_id = character_id.split("/", 1)
        base_dir = (
            self.train_data_dir if location == "train" else self.new_data_dir
        )
        char_dir = os.path.join(base_dir, anime_id, c_id)

        deleted_count = 0
        errors = []
        for filename in filenames:
            if ".." in filename or filename.startswith("/"):
                errors.append(f"检测到无效文件名: {filename}")
                continue
            
            file_path = os.path.join(char_dir, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    errors.append(f"删除 {filename} 失败: {e}")
            else:
                errors.append(f"文件 {filename} 不存在")

        # --- 清理和元数据更新 ---
        character_was_deleted = self._cleanup_after_image_deletion(anime_id, c_id)

        return {
            "deleted_count": deleted_count,
            "errors": errors,
            "character_deleted": character_was_deleted,
        }

    # --- 辅助方法 ---
    def _get_image_files(self, directory_path):
        """获取目录中的所有图片文件列表和数量。"""
        if not os.path.isdir(directory_path):
            return [], 0
        files = sorted(
            [
                f
                for f in os.listdir(directory_path)
                if f.lower().endswith(self.image_extensions)
            ]
        )
        return files, len(files)

    def _generate_preview_b64(self, dir_path, filename, char_ref):
        """为单个图片生成Base64编码的预览图。"""
        try:
            with Image.open(os.path.join(dir_path, filename)) as img:
                img.thumbnail((128, 128))
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=85)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            self.logger.error(f"为角色 {char_ref} 生成预览图失败 ({filename}): {e}")
            return None

    def _get_images_with_previews_from_dir(self, directory_path):
        """获取目录中所有图片的预览信息。"""
        images_data = []
        if not os.path.isdir(directory_path):
            return images_data
        
        files, _ = self._get_image_files(directory_path)
        for img_file in files:
            b64_str = self._generate_preview_b64(directory_path, img_file, directory_path)
            images_data.append({"filename": img_file, "b64": b64_str})
        return images_data

    def _cleanup_dir(self, dir_path):
        """删除指定目录，并尝试删除空的父目录。"""
        if os.path.isdir(dir_path):
            parent_dir = os.path.dirname(dir_path)
            shutil.rmtree(dir_path)
            self.logger.info(f"已删除文件夹: {dir_path}")
            if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                os.rmdir(parent_dir)
                self.logger.info(f"已删除空的父文件夹: {parent_dir}")
    
    def _cleanup_after_image_deletion(self, anime_id, c_id):
        """
        在图片被删除后，检查并清理空目录。
        如果一个角色的 train 和 new 目录都空了，就从元数据中删除该角色。
        """
        train_char_folder = os.path.join(self.train_data_dir, anime_id, c_id)
        new_char_folder = os.path.join(self.new_data_dir, anime_id, c_id)

        # 清理各自的空目录
        if os.path.isdir(train_char_folder) and not os.listdir(train_char_folder):
            self._cleanup_dir(train_char_folder)
        if os.path.isdir(new_char_folder) and not os.listdir(new_char_folder):
             self._cleanup_dir(new_char_folder)

        # 检查是否两个目录都已不存在，如果是，则更新元数据
        is_train_gone = not os.path.exists(train_char_folder)
        is_new_gone = not os.path.exists(new_char_folder)

        if is_train_gone and is_new_gone:
            self.logger.info(f"角色 {anime_id}/{c_id} 的目录均已空，将从元数据中删除。")
            with meta_lock:
                meta = self._load_meta()
                if anime_id in meta and c_id in meta[anime_id].get("characters", {}):
                    del meta[anime_id]["characters"][c_id]
                    if not meta[anime_id]["characters"]:
                        del meta[anime_id]
                    if self._save_meta(meta):
                        self.logger.info(f"已成功从元数据中移除角色 {anime_id}/{c_id}")
                        return True
                    else:
                        self.logger.error(f"从元数据中移除角色 {anime_id}/{c_id} 时保存失败")
        return False 