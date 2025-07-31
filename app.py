from flask import Flask
import os
import json
import numpy as np
import torch
from ultralytics import YOLO
import time

# 配置导入
from app.config import (
    load_config, save_config, character_meta_path, model_checkpoint_path,
    feature_db_path, class_map_path, yolo_model_path, train_data_dir_for_rename,
    new_data_dir_for_rename, onnx_model_path, onnx_yolo_model_path
)

# 蓝图导入
from app.controllers.views import views_bp
from app.controllers.auth import auth_bp, login_manager
from app.controllers.logs_api import logs_api_bp
from app.controllers.gallery import gallery_bp
from app.controllers.statistics import stats_bp
from app.controllers.recognize import recognize_bp
from app.controllers.bilibot import bilibot_bp
from app.controllers.auto_test import auto_test_bp
from app.controllers.upload_anime import upload_anime_bp
from app.controllers.upload_samples import upload_samples_bp
from app.controllers.update_preview import update_preview_bp
from app.controllers.reload_db import reload_db_bp
from app.controllers.setting import setting_bp

# 服务导入
from app.services.gallery_service import GalleryService
from app.services.bilibot_service import BilibotService
from app.services.recognition_service import RecognitionService
from app.services.auto_test_service import AutoTestService
from app.services.upload_anime_service import UploadAnimeService
from app.services.upload_samples_service import UploadSamplesService
from app.services.update_preview_service import UpdatePreviewService

# 核心模块导入
from app.core.recognizer import Recognizer
from app.core.models import ArcFaceModel
from app.core.feature_extractor import build_faiss_index

# 日志导入
from app.log import setup_logger, logger

start_time = time.time()


def create_app():
    """创建Flask应用"""
    app = Flask(__name__, static_folder='templates')
    
    # 配置日志
    app.logger = logger
    with app.app_context():
        setup_logger()

    # 注册所有蓝图
    blueprints = [
        (views_bp, None),
        (auth_bp, "/auth"),
        (logs_api_bp, "/api/admin"),
        (gallery_bp, None),
        (stats_bp, None),
        (recognize_bp, None),
        (bilibot_bp, None),
        (auto_test_bp, None),
        (upload_anime_bp, None),
        (upload_samples_bp, None),
        (update_preview_bp, None),
        (setting_bp, None),
        (reload_db_bp, None)
    ]
    
    for blueprint, url_prefix in blueprints:
        app.register_blueprint(blueprint, url_prefix=url_prefix)

    # 加载配置
    config = load_config()
    app.config.update(config)
    
    # 注入模型路径配置
    model_paths = {
        "YOLO_MODEL_PATH": yolo_model_path,
        "RECOGNITION_MODEL_PATH": model_checkpoint_path,
        "FEATURE_DB_PATH": feature_db_path,
        "CLASS_MAP_PATH": class_map_path,
        "ONNX_MODEL_PATH": onnx_model_path,
        "ONNX_YOLO_MODEL_PATH": onnx_yolo_model_path,
        "CHARACTER_META_PATH": character_meta_path,
        "TRAIN_DATA_DIR": train_data_dir_for_rename,
        "NEW_DATA_DIR": new_data_dir_for_rename
    }
    app.config.update(model_paths)

    # 生成secret_key
    if "secret_key" not in app.config or not app.config["secret_key"]:
        app.config["secret_key"] = os.urandom(24).hex()
        config["secret_key"] = app.config["secret_key"]
        save_config(config)
    
    app.secret_key = app.config["secret_key"]
    login_manager.init_app(app)

    # 注册基础服务
    app.gallery_service = GalleryService(
        character_meta_path=character_meta_path,
        train_data_dir=train_data_dir_for_rename,
        new_data_dir=new_data_dir_for_rename,
        logger=app.logger,
    )

    # 模板上下文处理器
    @app.context_processor
    def inject_config():
        return dict(config=app.config)
    
    @app.context_processor
    def inject_active_background():
        from app.services.setting_service import SettingService
        setting_service = SettingService(app.config, app.logger)
        return {"active_background": setting_service.get_active_background()}
    
    return app


def load_models_and_data(app):
    """加载所有模型和数据"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = app.config.get("recognition_backend", "pytorch")
    
    app.logger.info(f"使用设备: {device}, 后端: {backend.upper()}")
    
    # 加载识别模型
    model = None
    onnx_session = None
    
    if backend == "onnx":
        try:
            import onnxruntime
            onnx_path = app.config.get("ONNX_MODEL_PATH")
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX模型文件未找到: {onnx_path}")
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
            onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
            app.logger.info(f"ONNX模型加载成功: {onnx_session.get_providers()}")
        except Exception as e:
            app.logger.error(f"ONNX模型加载失败: {e}")
            raise
    else:
        # PyTorch模型
        checkpoint_path = app.config.get("RECOGNITION_MODEL_PATH")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"模型文件未找到: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        with open(class_map_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        
        model = ArcFaceModel(num_classes=len(class_to_idx), use_stn=True).to(device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        if "margin.weight" in state_dict and state_dict["margin.weight"].shape != model.margin.weight.shape:
            del state_dict["margin.weight"]
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        app.logger.info("PyTorch模型加载成功")

    # 加载数据
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    
    # 特征数据库
    feature_db = {}
    if os.path.exists(feature_db_path) and os.path.getsize(feature_db_path) > 0:
        try:
            feature_db_raw = np.load(feature_db_path, allow_pickle=True).item()
            feature_db = {int(k): v for k, v in feature_db_raw.items()}
        except EOFError:
            app.logger.error("特征数据库文件损坏，初始化为空")
    else:
        app.logger.warning("特征数据库未找到，初始化为空")

    # 角色元数据
    with open(app.config.get("CHARACTER_META_PATH"), "r", encoding="utf-8") as f:
        character_meta = json.load(f)

    # YOLO检测器
    yolo_path = app.config.get("ONNX_YOLO_MODEL_PATH") if backend == "onnx" else yolo_model_path
    if backend == "onnx" and not os.path.exists(yolo_path):
        app.logger.warning("YOLO ONNX模型未找到，回退到PyTorch")
        yolo_path = yolo_model_path
    
    first_model = YOLO(yolo_path)
    app.logger.info(f"YOLO检测器加载成功: {yolo_path}")

    # 附加到app对象
    app.model = model
    app.onnx_session = onnx_session
    app.feature_db = feature_db
    app.class_to_idx = class_to_idx
    app.character_meta = character_meta
    app.device = device
    app.first_model = first_model


def initialize_services(app):
    """初始化所有服务"""
    with app.app_context():
        # 加载模型和数据
        app.logger.info("开始加载模型和数据...")
        load_models_and_data(app)
        
        # 构建Faiss索引
        build_faiss_index(app)
        
        # 初始化识别服务
        recognizer = Recognizer(
            model=app.model, onnx_session=app.onnx_session,
            class_to_idx=app.class_to_idx, character_meta=app.character_meta,
            device=app.device, first_model=app.first_model,
            faiss_index=app.faiss_index, faiss_index_to_label=app.faiss_index_to_label,
            logger=app.logger, config=app.config
        )
        app.recognition_service = RecognitionService(recognizer=recognizer, logger=app.logger)
        
        # 初始化其他服务
        app.auto_test_service = AutoTestService(recognition_service=app.recognition_service, logger=app.logger)
        app.bilibot_service = BilibotService(config=app.config.get('bilibot', {}), logger=app.logger, recognition_service=app.recognition_service)
        app.upload_anime_service = UploadAnimeService(logger=app.logger)
        app.upload_samples_service = UploadSamplesService()
        app.update_preview_service = UpdatePreviewService()
        
        # B站机器人自动登录
        bilibot_config = app.config.get('bilibot', {})
        if 'bilibot_cookies' in bilibot_config:
            app.logger.info("检测到B站Cookie，尝试自动登录...")
            app.bilibot_service.login_with_cookies(bilibot_config['bilibot_cookies'])
        
        app.logger.info("所有服务初始化完成")


def configure_bilibili_api():
    """配置Bilibili API"""
    try:
        from bilibili_api import select_client, request_settings
        select_client("curl_cffi")
        request_settings.set("impersonate", "chrome110")
        logger.info("Bilibili API配置完成")
    except ImportError:
        logger.warning("bilibili_api模块未安装，跳过配置")
    except Exception as e:
        logger.warning(f"Bilibili API配置失败: {e}")


# 创建应用实例
app = create_app()


def main():
    """主函数"""
    configure_bilibili_api()
    initialize_services(app)
    
    end_time = time.time()
    app.logger.info(f"应用初始化完成，耗时: {end_time - start_time:.2f}秒")
    
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
