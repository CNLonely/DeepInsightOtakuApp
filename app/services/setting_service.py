import os
import json
import uuid
import random
import platform
from PIL import Image
import torch
import ultralytics
import numpy as np
import cv2
from app.config import DEFAULT_CONFIG, load_config, save_config


class SettingService:
    """封装所有与设置相关的业务逻辑和数据操作"""

    def __init__(self, app_config, logger):
        self.app_config = app_config
        self.logger = logger
        self.backgrounds_dir = os.path.join("templates", "backgrounds")
        self.last_sequential_index = -1

    # --- 配置管理 ---

    def get_current_config(self):
        """获取当前应用配置"""
        return self.app_config

    def update_config(self, data):
        """更新配置并保存到文件"""
        if not data:
            raise ValueError("请求体为空")

        # 从文件加载原始配置
        try:
            with open(os.path.join('config', 'config.json'), 'r', encoding='utf-8') as f:
                new_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            new_config = DEFAULT_CONFIG.copy()

        # 增量更新配置
        self._update_config_values(new_config, data)
        
        # 保存到文件
        save_config(new_config)
        return new_config

    def _update_config_values(self, config, data):
        """更新配置中的具体值"""
        # 项目名称
        if "project_name" in data:
            config["project_name"] = str(data["project_name"]).strip()

        # 识别阈值
        if "recognition_threshold" in data:
            val = float(data["recognition_threshold"])
            if not (0.0 <= val <= 1.0):
                raise ValueError("识别阈值必须在 0.0 和 1.0 之间")
            config["recognition_threshold"] = val

        # 修正豁免阈值
        if "correction_override_threshold" in data:
            val = float(data["correction_override_threshold"])
            if not (0.0 <= val <= 1.0):
                raise ValueError("修正豁免阈值必须在 0.0 和 1.0 之间")
            config["correction_override_threshold"] = val

        # 最大识别数量
        if "max_faces" in data:
            val = int(data["max_faces"])
            if not (val > 0):
                raise ValueError("最大识别数量必须大于 0")
            config["max_faces"] = val

        # 上传压缩
        if "enable_upload_compression" in data:
            config["enable_upload_compression"] = bool(data["enable_upload_compression"])

        # 识别修正
        if "use_recognition_correction" in data:
            config["use_recognition_correction"] = bool(data["use_recognition_correction"])

        # UI透明度
        if "glass_opacity" in data:
            val = float(data["glass_opacity"])
            if not (0.0 <= val <= 1.0):
                raise ValueError("UI透明度必须在 0.0 和 1.0 之间")
            config["glass_opacity"] = val

        # 随机扫描速度
        if "animation_speed_random" in data:
            val = int(data["animation_speed_random"])
            if not (100 <= val <= 5000):
                raise ValueError("随机扫描速度必须在 100 和 5000 毫秒之间")
            config["animation_speed_random"] = val

        # 锁定目标扫描速度
        if "animation_speed_target" in data:
            val = int(data["animation_speed_target"])
            if not (100 <= val <= 5000):
                raise ValueError("锁定目标扫描速度必须在 100 和 5000 毫秒之间")
            config["animation_speed_target"] = val

        # 识别后端
        if "recognition_backend" in data:
            val = str(data["recognition_backend"]).lower()
            if val not in ["pytorch", "onnx"]:
                raise ValueError("识别后端必须是 'pytorch' 或 'onnx'")
            config["recognition_backend"] = val

        # 背景设置
        if "background" in data:
            self._update_background_config(config, data["background"])

    def _update_background_config(self, config, bg_data):
        """更新背景配置"""
        if "background" not in config:
            config["background"] = {}
        
        config["background"]["type"] = bg_data.get("type", config["background"].get("type"))
        config["background"]["color"] = bg_data.get("color", config["background"].get("color"))
        config["background"]["mode"] = bg_data.get("mode", config["background"].get("mode"))
        config["background"]["image"] = bg_data.get("image", config["background"].get("image"))

    # --- 模型文件状态 ---

    def get_model_file_status(self):
        """检查关键模型文件的存在性"""
        onnx_path = self.app_config.get("ONNX_MODEL_PATH")
        pytorch_path = self.app_config.get("RECOGNITION_MODEL_PATH")
        
        return {
            "onnx_exists": os.path.exists(onnx_path),
            "pytorch_exists": os.path.exists(pytorch_path)
        }

    # --- 背景管理 ---

    def get_backgrounds(self):
        """获取所有已上传的背景图片列表"""
        os.makedirs(self.backgrounds_dir, exist_ok=True)

        current_uploaded_images = self.app_config.get("background", {}).get("uploaded_images", [])
        
        # 过滤掉已不存在于文件系统中的图片
        valid_images = [img_path for img_path in current_uploaded_images if os.path.exists(img_path)]

        # 如果列表被修改，则更新配置文件
        if len(valid_images) != len(current_uploaded_images):
            disk_config = load_config()
            if "background" not in disk_config:
                disk_config["background"] = DEFAULT_CONFIG["background"].copy()

            disk_config["background"]["uploaded_images"] = valid_images
            save_config(disk_config)
            self.app_config.update(disk_config)

        return valid_images

    def upload_background(self, file):
        """上传新的背景图片"""
        if not file or file.filename == "":
            raise ValueError("文件无效")

        # 从文件名中分离扩展名
        _, ext = os.path.splitext(file.filename)
        if not ext:
            ext = ".png"

        # 生成唯一的文件名
        filename = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(self.backgrounds_dir, filename)

        file.save(save_path)

        # 更新配置
        disk_config = load_config()
        
        if "background" not in disk_config:
            disk_config["background"] = DEFAULT_CONFIG["background"].copy()

        if "uploaded_images" not in disk_config["background"]:
            disk_config["background"]["uploaded_images"] = []

        relative_path = os.path.join(self.backgrounds_dir, filename).replace("\\", "/")
        disk_config["background"]["uploaded_images"].append(relative_path)
        save_config(disk_config)
        
        self.app_config.update(disk_config)

        return relative_path

    def delete_background(self, filepath):
        """删除指定的背景图片"""
        # 安全性检查
        if not filepath.startswith(self.backgrounds_dir.replace("\\", "/")):
            raise ValueError("无效的路径")

        if os.path.exists(filepath):
            os.remove(filepath)

        # 更新配置
        disk_config = load_config()

        if "background" in disk_config and "uploaded_images" in disk_config["background"]:
            disk_config["background"]["uploaded_images"] = [
                img for img in disk_config["background"]["uploaded_images"] if img != filepath
            ]

        # 如果删除的是当前选中的固定图片，则重置为默认
        if disk_config.get("background", {}).get("image") == filepath:
            disk_config["background"]["image"] = DEFAULT_CONFIG["background"]["image"]

        save_config(disk_config)
        self.app_config.update(disk_config)

    # --- 动态背景选择 ---

    def get_active_background(self):
        """根据配置决定当前应该展示哪个背景"""
        bg_config = self.app_config.get("background", DEFAULT_CONFIG["background"])

        if bg_config.get("type") == "color":
            return {"type": "color", "value": bg_config.get("color", "#f0f2f5")}

        # 图片类型
        mode = bg_config.get("mode", "fixed")
        images = bg_config.get("uploaded_images", [])

        # 过滤掉不存在的图片
        valid_images = [img for img in images if os.path.exists(img)]
        if not valid_images:
            return self._get_default_background()

        image_url = self._select_background_image(mode, valid_images, bg_config)
        
        # 路径修正: 确保返回的路径不包含 'templates/' 前缀
        corrected_image_url = image_url.replace("templates/", "", 1)
        
        return {"type": "image", "value": corrected_image_url}

    def _get_default_background(self):
        """获取默认背景"""
        os.makedirs(os.path.dirname(DEFAULT_CONFIG["background"]["image"]), exist_ok=True)
        if not os.path.exists(DEFAULT_CONFIG["background"]["image"]):
            try:
                Image.new("RGB", (100, 100), color="grey").save(
                    DEFAULT_CONFIG["background"]["image"]
                )
            except:
                pass

        raw_path = DEFAULT_CONFIG["background"]["image"]
        corrected_path = raw_path.replace("templates/", "", 1)
        return {"type": "image", "value": corrected_path}

    def _select_background_image(self, mode, valid_images, bg_config):
        """根据模式选择背景图片"""
        if mode == "fixed":
            fixed_image = bg_config.get("image", valid_images[0])
            return fixed_image if fixed_image in valid_images else valid_images[0]
        elif mode == "random":
            return random.choice(valid_images)
        elif mode == "sequential":
            self.last_sequential_index = (self.last_sequential_index + 1) % len(valid_images)
            return valid_images[self.last_sequential_index]
        else:
            return valid_images[0]

    # --- 系统信息 ---

    def get_system_info(self):
        """获取系统、Python和库的版本信息"""
        # 模块版本
        torch_version = torch.__version__
        yolo_version = ultralytics.__version__
        numpy_version = np.__version__
        pillow_version = Image.__version__
        cv2_version = cv2.__version__

        # GPU 信息
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpus.append(torch.cuda.get_device_name(i))

        # CPU 信息
        cpu_info = {"model": platform.processor(), "cores": os.cpu_count()}

        return {
            "python_version": platform.python_version(),
            "torch_version": torch_version,
            "yolo_version": yolo_version,
            "numpy_version": numpy_version,
            "pillow_version": pillow_version,
            "cv2_version": cv2_version,
            "gpus": gpus,
            "cpu": cpu_info,
        } 