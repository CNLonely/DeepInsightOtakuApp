from flask import current_app, request, jsonify, Blueprint
from flask_login import login_required
import os
import tempfile
import shutil

# 创建蓝图
upload_anime_bp = Blueprint('upload_anime', __name__)

# --------------------------- 管理后台 - 上传动漫 --------------------------- #


@upload_anime_bp.route("/api/admin/rename/upload", methods=["POST"])
@login_required
def rename_upload_zip():
    """
    上传并分析ZIP文件，返回角色目录信息
    """
    if "file" not in request.files:
        return jsonify({"error": "未收到文件"}), 400

    file = request.files["file"]
    if file.filename == "" or not file.filename.lower().endswith(".zip"):
        return jsonify({"error": "请上传一个zip压缩文件"}), 400

    try:
        temp_dir = tempfile.mkdtemp(prefix="rename_")
        zip_path = os.path.join(temp_dir, file.filename)
        file.save(zip_path)

        extract_path = os.path.join(temp_dir, "extracted")

        # 使用服务层解压文件
        with open(zip_path, "rb") as zip_stream:
            current_app.upload_anime_service.extract_zip_with_encoding_fix(zip_stream, extract_path)

        # 使用服务层分析目录结构
        char_dirs = current_app.upload_anime_service.analyze_character_directories(extract_path)

        return jsonify({
            "success": True,
            "message": "成功解压并分析文件。",
            "temp_dir": temp_dir,
            "characters": char_dirs,
        })

    except Exception as e:
        # 确保在出错时清理临时文件夹
        if "temp_dir" in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return jsonify({"error": f"处理zip文件失败: {e}"}), 500


@upload_anime_bp.route("/api/admin/rename/process", methods=["POST"])
@login_required
def rename_process_folders():
    """
    处理已分析的文件夹列表
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空"}), 400

    temp_dir = data.get("temp_dir")
    character_paths = [item["path"] for item in data.get("characters", [])]
    mode = data.get("mode")

    if not all([temp_dir, character_paths, mode]):
        return jsonify({"error": "缺少参数(temp_dir, characters, mode)"}), 400

    if not os.path.isdir(temp_dir):
        return jsonify({"error": "临时目录不存在或已失效"}), 400

    if mode == "train":
        target_directory = current_app.config.get("TRAIN_DATA_DIR")
    elif mode == "new":
        target_directory = current_app.config.get("NEW_DATA_DIR")
    else:
        return jsonify({"error": f"无效的处理模式: {mode}"}), 400

    os.makedirs(target_directory, exist_ok=True)

    # 使用服务层处理文件夹
    success, log = current_app.upload_anime_service.process_and_rename_character_folders(
        character_paths, target_directory
    )

    # 清理临时文件夹
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        log += f"\n\n警告: 清理临时目录 {temp_dir} 失败: {e}"

    return jsonify({"success": success, "log": log})


@upload_anime_bp.route("/api/admin/rename/process_zip", methods=["POST"])
@login_required
def rename_process_zip():
    """
    接收一个zip文件和模式，完成所有处理步骤并返回日志
    """
    if "file" not in request.files:
        return jsonify({"success": False, "log": "错误: 未收到文件"}), 400

    file = request.files["file"]
    mode = request.form.get("mode")

    if file.filename == "" or not file.filename.lower().endswith(".zip"):
        return jsonify({
            "success": False,
            "log": f"错误: 无效的文件 '{file.filename}'。请上传 .zip 文件。"
        }), 400

    if mode not in ["train", "new"]:
        return jsonify({"success": False, "log": f"错误: 无效的处理模式 '{mode}'"}), 400

    # 使用服务层处理ZIP文件
    success, log_output = current_app.upload_anime_service.process_zip_file(file.stream, mode)

    return jsonify({"success": success, "log": log_output})
