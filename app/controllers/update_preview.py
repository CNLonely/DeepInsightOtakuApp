from flask import current_app, request, jsonify, Blueprint
from flask_login import login_required

update_preview_bp = Blueprint('update_preview', __name__)


@update_preview_bp.route("/api/admin/preview_database_changes")
@login_required
def preview_database_changes():
    """
    提供待更新或新增角色的预览，按动漫分组。
    从 data/new 目录扫描所有待处理项目。
    """
    try:
        result = current_app.update_preview_service.get_database_changes_preview()
        return jsonify(result)
    except Exception as e:
        current_app.logger.error(f"获取数据库变更预览时出错: {e}")
        return jsonify({"error": "获取预览失败"}), 500


@update_preview_bp.route("/api/admin/delete_pending_character", methods=["POST"])
@login_required
def delete_pending_character():
    """
    删除待处理角色文件夹，如果角色不存在于训练数据中，也从元数据中删除。
    """
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "请求体为空"}), 400

    char_id = data.get("character_id")  # composite_id: 'anime_id/c_id'
    
    if not char_id:
        return jsonify({"success": False, "message": "缺少角色ID"}), 400

    try:
        success, message = current_app.update_preview_service.delete_pending_character(char_id)
        
        if success:
            return jsonify({"success": True, "message": message})
        else:
            return jsonify({"success": False, "message": message}), 400
            
    except Exception as e:
        current_app.logger.error(f"删除待处理角色时出错: {e}")
        return jsonify({"success": False, "message": f"删除失败: {e}"}), 500