from flask import current_app, request, jsonify, Blueprint
from flask_login import login_required
import os
import json
import base64
import io
import shutil
import re
from PIL import Image
import numpy as np
from app.core.feature_extractor import (
    update_characters_in_database,
    add_character_to_database,
    ArcFaceModel as ExtractorArcFaceModel,  
    transform as extractor_transform,
)
from app.core.feature_extractor import build_faiss_index

reload_db_bp = Blueprint('reload_db', __name__)


@reload_db_bp.route("/api/run_update_and_reload", methods=["POST"])
@login_required
def run_update_and_reload():
    """
    运行特征更新脚本，并热重载数据库和类别映射。
    """
    try:
        # --- 加载用于特征提取的模型 ---
        extractor_model_path = current_app.model
        device = current_app.device
        data, updated_count = update_characters_in_database(extractor_model_path, device, extractor_transform,new_character_dir=current_app.config.get("NEW_DATA_DIR"), train_dir=current_app.config.get("TRAIN_DATA_DIR"), db_path=current_app.config.get("FEATURE_DB_PATH"),class_path=current_app.config.get("CLASS_MAP_PATH"))

        current_app.logger.info(f"数据库更新任务日志:\n{data}\n\n已更新 {updated_count} 个角色。")

        # 热重载数据库和元数据到 current_app
        current_app.logger.info("正在热重载特征数据...")
        feature_db_raw = np.load(current_app.config.get("FEATURE_DB_PATH"), allow_pickle=True).item()
        current_app.feature_db = {int(k): v for k, v in feature_db_raw.items()}

        with open(current_app.config.get("CLASS_MAP_PATH"), "r", encoding="utf-8") as f:
            current_app.class_to_idx = json.load(f)

        with open(current_app.config.get("CHARACTER_META_PATH"), "r", encoding="utf-8") as f:
            current_app.character_meta = json.load(f)
        current_app.logger.info("特征数据热重载完成。")

        # 重建Faiss索引
        build_faiss_index(current_app)

        # 通知识别服务热重载数据
        current_app.recognition_service.reload_data(
            class_to_idx=current_app.class_to_idx,
            character_meta=current_app.character_meta,
            faiss_index=current_app.faiss_index,
            faiss_index_to_label=current_app.faiss_index_to_label
        )

        return jsonify({"success": True, "log": data})

    except Exception as e:
        current_app.logger.error(f"更新数据库时发生错误: {e}", exc_info=True)
        return jsonify({"success": False, "log": str(e)}), 500


@reload_db_bp.route("/api/admin/rename/add_new_characters", methods=["POST"])
@login_required
def api_add_new_characters_to_db():
    """
    运行脚本，将 train 目录中未被索引的新角色添加到数据库。
    """
    try:
        extractor_model_path = current_app.model
        device = current_app.device
        data, updated_count = add_character_to_database(extractor_model_path, device, extractor_transform,new_character_dir=current_app.config.get("NEW_DATA_DIR"), db_path=current_app.config.get("FEATURE_DB_PATH"),class_path=current_app.config.get("CLASS_MAP_PATH"))

        current_app.logger.info(f"新角色添加任务日志:\n{data}\n\n已更新 {updated_count} 个角色。")

        # 热重载数据库和元数据到 current_app
        current_app.logger.info("正在热重载特征数据...")
        feature_db_raw = np.load(current_app.config.get("FEATURE_DB_PATH"), allow_pickle=True).item()
        current_app.feature_db = {int(k): v for k, v in feature_db_raw.items()}

        with open(current_app.config.get("CLASS_MAP_PATH"), "r", encoding="utf-8") as f:
            current_app.class_to_idx = json.load(f)

        with open(current_app.config.get("CHARACTER_META_PATH"), "r", encoding="utf-8") as f:
            current_app.character_meta = json.load(f)
        current_app.logger.info("特征数据热重载完成。")

        # 重建Faiss索引
        build_faiss_index(current_app)

        # 通知识别服务热重载数据
        current_app.recognition_service.reload_data(
            class_to_idx=current_app.class_to_idx,
            character_meta=current_app.character_meta,
            faiss_index=current_app.faiss_index,
            faiss_index_to_label=current_app.faiss_index_to_label
        )

        return jsonify({"success": True, "log": data})

    except Exception as e:
        current_app.logger.error(f"添加新角色到数据库时发生错误: {e}", exc_info=True)
        return jsonify({"success": False, "log": str(e)}), 500