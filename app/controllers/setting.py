from flask import current_app, request, jsonify, Blueprint
from flask_login import login_required
from app.services.setting_service import SettingService

setting_bp = Blueprint('setting', __name__)


@setting_bp.route("/api/admin/settings", methods=["GET", "POST"])
@login_required
def api_admin_settings():
    """设置管理API"""
    setting_service = SettingService(current_app.config, current_app.logger)
    
    if request.method == "GET":
        # 获取当前配置
        return jsonify(setting_service.get_current_config())

    if request.method == "POST":
        data = request.get_json()
        
        try:
            # 更新配置
            new_config = setting_service.update_config(data)
            # 同步到current_app.config
            current_app.config.update(new_config)
            
            return jsonify({"success": True, "message": "设置已成功保存！"})
        except ValueError as e:
            return jsonify({"success": False, "message": f"数据格式错误: {e}"}), 400
        except Exception as e:
            current_app.logger.error(f"保存设置失败: {e}", exc_info=True)
            return jsonify({"success": False, "message": f"保存失败: {e}"}), 500


@setting_bp.route("/api/admin/model_file_status")
@login_required
def api_admin_model_file_status():
    """检查模型文件状态"""
    setting_service = SettingService(current_app.config, current_app.logger)
    status = setting_service.get_model_file_status()
    return jsonify(status)


# --------------------------- 背景 API --------------------------- #

@setting_bp.route("/api/admin/backgrounds", methods=["GET"])
@login_required
def get_backgrounds():
    """获取所有已上传的背景图片列表"""
    setting_service = SettingService(current_app.config, current_app.logger)
    backgrounds = setting_service.get_backgrounds()
    return jsonify(backgrounds)


@setting_bp.route("/api/admin/backgrounds/upload", methods=["POST"])
@login_required
def upload_background():
    """上传新的背景图片"""
    setting_service = SettingService(current_app.config, current_app.logger)
    
    if "file" not in request.files:
        return jsonify({"success": False, "message": "未收到文件"}), 400

    file = request.files["file"]
    
    try:
        filepath = setting_service.upload_background(file)
        return jsonify({
            "success": True,
            "message": "背景图片上传成功!",
            "filepath": filepath,
        })
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"上传背景图片失败: {e}", exc_info=True)
        return jsonify({"success": False, "message": f"保存文件失败: {e}"}), 500


@setting_bp.route("/api/admin/backgrounds", methods=["DELETE"])
@login_required
def delete_background():
    """删除指定的背景图片"""
    setting_service = SettingService(current_app.config, current_app.logger)
    
    data = request.get_json()
    if not data or "filepath" not in data:
        return jsonify({"success": False, "message": "请求无效，缺少文件路径"}), 400

    filepath = data["filepath"]
    
    try:
        setting_service.delete_background(filepath)
        return jsonify({"success": True, "message": "背景图片已删除"})
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"删除背景图片失败: {e}", exc_info=True)
        return jsonify({"success": False, "message": f"删除失败: {e}"}), 500


# --------------------------- 动态背景选择 --------------------------- #

@setting_bp.context_processor
def inject_active_background():
    """将动态背景注入到所有模板"""
    setting_service = SettingService(current_app.config, current_app.logger)
    return {"active_background": setting_service.get_active_background()}


@setting_bp.route("/api/admin/system_info")
@login_required
def get_system_info():
    """获取系统信息"""
    setting_service = SettingService(current_app.config, current_app.logger)
    
    try:
        system_info = setting_service.get_system_info()
        return jsonify(system_info)
    except Exception as e:
        current_app.logger.error(f"获取系统信息失败: {e}", exc_info=True)
        return jsonify({"error": "获取系统信息时发生错误"}), 500
