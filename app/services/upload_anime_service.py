import os
import json
import threading
import re
import shutil
import tempfile
import time
import zipfile
from typing import List, Tuple, Dict, Any
from flask import current_app


class UploadAnimeService:
    """动漫角色上传服务类"""
    
    def __init__(self,logger):
        self.meta_lock = threading.Lock()
        self.image_extensions = (".png", ".jpg", ".jpeg", ".webp", ".gif")
        self.id_pattern = re.compile(r"id_\d{5}")
        self.logger = logger
    
    def extract_zip_with_encoding_fix(self, zip_stream, extract_to_dir: str) -> None:
        """
        安全解压ZIP文件，处理文件名编码问题
        
        :param zip_stream: ZIP文件流
        :param extract_to_dir: 解压目标目录
        """
        os.makedirs(extract_to_dir, exist_ok=True)
        with zipfile.ZipFile(zip_stream, "r") as zip_ref:
            for member in zip_ref.infolist():
                filename = member.filename
                if not (member.flag_bits & 0x800):  # 如果不是UTF-8标志
                    try:
                        filename = member.filename.encode("cp437").decode("gbk")
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass

                target_path = os.path.join(extract_to_dir, filename)

                # 安全检查：防止ZipSlip路径遍历
                real_extract_path = os.path.realpath(extract_to_dir)
                real_target_path = os.path.realpath(target_path)

                if not real_target_path.startswith(real_extract_path):
                    self.logger.warning(f"检测到不安全的路径，跳过解压文件: {member.filename}")
                    continue

                if member.is_dir():
                    os.makedirs(target_path, exist_ok=True)
                else:
                    parent_dir = os.path.dirname(target_path)
                    os.makedirs(parent_dir, exist_ok=True)

                    with zip_ref.open(member, "r") as source_file, open(target_path, "wb") as target_file:
                        shutil.copyfileobj(source_file, target_file)
    
    def analyze_character_directories(self, extract_path: str) -> List[Dict[str, Any]]:
        """
        分析解压后的目录结构，识别角色文件夹
        
        :param extract_path: 解压目录路径
        :return: 角色目录信息列表
        """
        char_dirs = []
        
        for root, dirs, files in os.walk(extract_path):
            if "__MACOSX" in root:
                continue

            if any(f.lower().endswith(self.image_extensions) for f in files) and not self.id_pattern.match(os.path.basename(root)):
                char_name = os.path.basename(root)
                parent_dir = os.path.dirname(root)

                relative_path_from_extract = os.path.relpath(parent_dir, extract_path)

                if relative_path_from_extract == ".":
                    anime_name = "未分类"
                else:
                    anime_name = os.path.basename(parent_dir)

                char_dirs.append({
                    "anime": anime_name,
                    "char": char_name,
                    "path": root,
                })
        
        return char_dirs
    
    def get_character_paths(self, extract_path: str) -> List[str]:
        """
        获取角色文件夹路径列表
        
        :param extract_path: 解压目录路径
        :return: 角色文件夹路径列表
        """
        char_dirs_paths = []
        
        for root, dirs, files in os.walk(extract_path):
            if "__MACOSX" in root:
                continue

            is_char_dir = any(f.lower().endswith(self.image_extensions) for f in files)
            if is_char_dir and not self.id_pattern.match(os.path.basename(root)):
                char_dirs_paths.append(root)
        
        return char_dirs_paths
    
    def load_metadata(self) -> Tuple[Dict[str, Any], List[str]]:
        """
        加载元数据文件
        
        :return: (元数据字典, 日志行列表)
        """
        log_lines = []
        meta_path = current_app.config.get("CHARACTER_META_PATH")
        
        if not os.path.exists(meta_path):
            meta = {}
            log_lines.append("元数据文件不存在，将创建新的。")
        else:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        
        return meta, log_lines
    
    def save_metadata(self, meta: Dict[str, Any]) -> bool:
        """
        保存元数据文件
        
        :param meta: 元数据字典
        :return: 是否保存成功
        """
        try:
            with open(current_app.config.get("CHARACTER_META_PATH"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            self.logger.error(f"保存元数据文件失败: {e}")
            return False
    
    def find_or_create_anime_id(self, anime_name: str, meta: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        查找或创建动漫ID
        
        :param anime_name: 动漫名称
        :param meta: 元数据字典
        :return: (动漫ID, 日志行列表)
        """
        log_lines = []
        
        # 查找现有动漫ID
        target_anime_id = None
        for an_id, an_info in meta.items():
            if an_info.get("name") == anime_name:
                target_anime_id = an_id
                break
        
        # 如果不存在则创建新的
        if not target_anime_id:
            max_anime_id_num = 0
            for key in meta.keys():
                if key.startswith("anime_"):
                    max_anime_id_num = max(max_anime_id_num, int(key.split("_")[1]))
            target_anime_id = f"anime_{max_anime_id_num + 1:05d}"
            meta[target_anime_id] = {"name": anime_name, "characters": {}}
            log_lines.append(f"创建新动漫条目: '{anime_name}' -> {target_anime_id}")
        
        return target_anime_id, log_lines
    
    def find_existing_character_id(self, char_name: str, anime_chars: Dict[str, Any]) -> str:
        """
        查找现有角色ID
        
        :param char_name: 角色名称
        :param anime_chars: 动漫角色字典
        :return: 角色ID或None
        """
        for c_id, c_info in anime_chars.items():
            if c_info.get("name") == char_name:
                return c_id
        return None
    
    def handle_duplicate_character(self, dir_path: str, anime_name: str, char_name: str, 
                                 target_anime_id: str, existing_char_id: str) -> Tuple[int, List[str]]:
        """
        处理重复角色
        
        :param dir_path: 角色目录路径
        :param anime_name: 动漫名称
        :param char_name: 角色名称
        :param target_anime_id: 动漫ID
        :param existing_char_id: 现有角色ID
        :return: (移动文件数量, 日志行列表)
        """
        log_lines = []
        log_lines.append(f"检测到重复角色: '{anime_name}/{char_name}' (ID: {target_anime_id}/{existing_char_id})。")
        
        # 移动到data/new目录
        target_path_for_duplicate = os.path.join(
            current_app.config.get("NEW_DATA_DIR"), target_anime_id, existing_char_id
        )
        os.makedirs(target_path_for_duplicate, exist_ok=True)

        files_moved_count = 0
        for item in os.listdir(dir_path):
            source_item_path = os.path.join(dir_path, item)
            if os.path.isfile(source_item_path):
                dest_item_path = os.path.join(target_path_for_duplicate, os.path.basename(source_item_path))
                # 防覆盖
                if os.path.exists(dest_item_path):
                    base, ext = os.path.splitext(dest_item_path)
                    dest_item_path = f"{base}_{int(time.time()*1000)}{ext}"
                
                shutil.move(source_item_path, dest_item_path)
                files_moved_count += 1
        
        log_lines.append(f"-> 已将 {files_moved_count} 张图片作为补充样本添加到 {target_path_for_duplicate}")
        
        # 清理空的源目录
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            log_lines.append(f"警告: 无法删除源目录 {dir_path}: {e}")
        
        return files_moved_count, log_lines
    
    def create_new_character(self, dir_path: str, anime_name: str, char_name: str, 
                           target_anime_id: str, target_dir: str, meta: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        创建新角色
        
        :param dir_path: 角色目录路径
        :param anime_name: 动漫名称
        :param char_name: 角色名称
        :param target_anime_id: 动漫ID
        :param target_dir: 目标目录
        :param meta: 元数据字典
        :return: (目标路径, 日志行列表)
        """
        log_lines = []
        
        # 创建新角色ID
        anime_chars = meta[target_anime_id].get("characters", {})
        max_char_id_num = 0
        for key in anime_chars.keys():
            if key.startswith("c_"):
                max_char_id_num = max(max_char_id_num, int(key.split("_")[1]))
        
        new_char_id = f"c_{max_char_id_num + 1:05d}"
        meta[target_anime_id]["characters"][new_char_id] = {"name": char_name}

        target_anime_folder = os.path.join(target_dir, target_anime_id)
        os.makedirs(target_anime_folder, exist_ok=True)
        target_path = os.path.join(target_anime_folder, new_char_id)
        
        shutil.move(dir_path, target_path)
        log_lines.append(f"处理成功: '{anime_name}/{char_name}' -> {target_path}")
        
        return target_path, log_lines
    
    def process_and_rename_character_folders(self, character_paths: List[str], target_dir: str) -> Tuple[bool, str]:
        """
        处理并重命名角色文件夹
        
        :param character_paths: 角色文件夹路径列表
        :param target_dir: 目标目录
        :return: (是否成功, 日志信息)
        """
        log_lines = []
        
        with self.meta_lock:
            meta, meta_logs = self.load_metadata()
            log_lines.extend(meta_logs)

        processed_count = 0
        for dir_path in character_paths:
            try:
                char_name = os.path.basename(dir_path)
                anime_name = os.path.basename(os.path.dirname(dir_path))
                
                # 跳过已经是ID格式的目录
                if re.match(r"^(anime|c)_\d{5}$", char_name) or re.match(r"^(anime|c)_\d{5}$", anime_name):
                    log_lines.append(f"跳过已是ID格式的目录: {dir_path}")
                    continue

                # 查找或创建动漫ID
                target_anime_id, anime_logs = self.find_or_create_anime_id(anime_name, meta)
                log_lines.extend(anime_logs)

                # 检查重复角色
                anime_chars = meta[target_anime_id].get("characters", {})
                existing_char_id = self.find_existing_character_id(char_name, anime_chars)

                if existing_char_id:
                    # 处理重复角色
                    _, duplicate_logs = self.handle_duplicate_character(
                        dir_path, anime_name, char_name, target_anime_id, existing_char_id
                    )
                    log_lines.extend(duplicate_logs)
                    processed_count += 1
                    continue

                # 创建新角色
                _, new_char_logs = self.create_new_character(
                    dir_path, anime_name, char_name, target_anime_id, target_dir, meta
                )
                log_lines.extend(new_char_logs)
                processed_count += 1

            except Exception as e:
                log_lines.append(f"处理失败: {dir_path}, 错误: {e}")

        log_lines.append(f"\n处理完成。共成功处理 {processed_count} 个角色。")

        # 保存元数据
        with self.meta_lock:
            if self.save_metadata(meta):
                log_lines.append(f"成功更新元数据文件: {current_app.config.get('CHARACTER_META_PATH')}")
            else:
                log_lines.append("错误: 更新元数据文件失败")
                return False, "\n".join(log_lines)

        return True, "\n".join(log_lines)
    
    def process_zip_file(self, file_stream, mode: str) -> Tuple[bool, str]:
        """
        处理ZIP文件的完整流程
        
        :param file_stream: 文件流
        :param mode: 处理模式 ('train' 或 'new')
        :return: (是否成功, 日志信息)
        """
        temp_dir = None
        try:
            # 1. 解压文件
            temp_dir = tempfile.mkdtemp(prefix="rename_proc_")
            extract_path = os.path.join(temp_dir, "extracted")
            self.extract_zip_with_encoding_fix(file_stream, extract_path)

            # 2. 分析目录结构
            char_dirs_paths = self.get_character_paths(extract_path)

            if not char_dirs_paths:
                return False, "未找到有效的角色文件夹。请确保目录结构为 动漫名/角色名/图片... 或 角色名/图片..."

            # 3. 设置目标目录并处理
            target_directory = (
                current_app.config.get("TRAIN_DATA_DIR") if mode == "train" 
                else current_app.config.get("NEW_DATA_DIR")
            )
            os.makedirs(target_directory, exist_ok=True)

            success, log_output = self.process_and_rename_character_folders(char_dirs_paths, target_directory)
            return success, log_output

        except Exception as e:
            return False, f"处理文件时发生意外错误: {e}"
        finally:
            # 清理临时文件夹
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


# 服务实例将在app.py中创建 