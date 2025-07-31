from flask import current_app, request, jsonify, Blueprint
from flask_login import login_required

upload_samples_bp = Blueprint('upload_samples', __name__)


@upload_samples_bp.route("/api/get_animes")
def get_animes():
    """获取所有动漫作品的列表"""
    try:
        animes = current_app.upload_samples_service.get_all_animes()
        return jsonify(animes)
    except Exception as e:
        current_app.logger.error(f"获取动漫列表失败: {e}")
        return jsonify({"error": "获取动漫列表失败"}), 500


@upload_samples_bp.route("/api/get_characters_by_anime")
def get_characters_by_anime():
    """根据动漫作品名称获取角色列表（带预览图）"""
    anime_name = request.args.get("anime")
    if not anime_name:
        return jsonify({"error": "未提供动漫名称"}), 400

    try:
        characters = current_app.upload_samples_service.get_characters_by_anime(anime_name)
        return jsonify(characters)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"获取角色列表失败: {e}")
        return jsonify({"error": "获取角色列表失败"}), 500


@upload_samples_bp.route("/api/upload_samples", methods=["POST"])
@login_required
def upload_samples():
    """接收为已有角色补充的样本图片(Base64)"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空或不是有效的JSON"}), 400

    character_id = data.get("character_id")
    image_b64 = data.get("image_b64")

    try:
        new_filename = current_app.upload_samples_service.upload_sample_for_existing_character(character_id, image_b64)
        
        return jsonify({
            "message": f"成功为角色 {character_id} 上传 1 个文件。",
            "file": new_filename
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"上传样本失败: {e}")
        return jsonify({"error": f"处理图片失败: {e}"}), 500


@upload_samples_bp.route("/api/admin/create_new_character", methods=["POST"])
@login_required
def create_new_character():
    """创建新角色并保存第一个样本"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "请求体为空"}), 400

    anime_name = data.get("anime_name", "").strip()
    char_name = data.get("character_name", "").strip()
    image_b64 = data.get("image_b64")

    try:
        new_character_id, new_filename = current_app.upload_samples_service.create_new_character(anime_name, char_name, image_b64)
        
        return jsonify({
            "success": True,
            "message": f"成功创建新角色 '{char_name}' 并保存了 1 个样本。请稍后运行数据库更新以提取特征。",
            "new_character_id": new_character_id,
        })
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"创建新角色时出错: {e}")
        return jsonify({
            "success": False,
            "message": f"创建新角色时发生服务器错误: {e}"
        }), 500