
import os
import json
import base64
import io
import re
import shutil

from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required
from PIL import Image

from app.services.gallery_service import GalleryService


gallery_bp = Blueprint('gallery', __name__, url_prefix='/api/admin')


@gallery_bp.route("/gallery/anime_list")
@login_required
def get_gallery_anime_list():
    """获取画廊所需的所有动漫名称列表及其角色数量，支持分页和搜索"""
    gallery_service = current_app.gallery_service
    
    # 从服务层获取完整数据
    data = gallery_service.get_anime_list_data()
    if not data or not data["all_animes"]:
        return jsonify(
            {
                "animes": [],
                "total_pages": 0,
                "current_page": 1,
                "total_animes": 0,
                "total_characters": 0,
            }
        )
        
    all_animes = data["all_animes"]
    
    # --- 在控制器层处理搜索和分页 ---
    search_term = request.args.get("search", "").lower()
    if search_term:
        filtered_animes = [
            anime for anime in all_animes if search_term in anime["name"].lower()
        ]
    else:
        filtered_animes = all_animes

    page = request.args.get("page", 1, type=int)
    per_page = 20
    total_items = len(filtered_animes)
    total_pages = (total_items + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page

    paginated_animes = filtered_animes[start:end]

    return jsonify(
        {
            "animes": paginated_animes,
            "total_pages": total_pages,
            "current_page": page,
            "total_animes": data["total_animes"],
            "total_characters": data["total_characters"],
        }
    )


@gallery_bp.route("/gallery/stats")
@login_required
def get_gallery_stats():
    """获取全局统计信息，如图片总数。这是一个可能较慢的独立接口。"""
    gallery_service = current_app.gallery_service
    total_images = gallery_service.get_total_images_stats()
    return jsonify({"total_images": total_images})


@gallery_bp.route("/gallery/characters_by_anime")
@login_required
def get_gallery_characters_by_anime():
    """
    根据动漫名称获取其下所有角色, 并合并 train 和 new 目录中的信息。
    """
    anime_name = request.args.get("anime")
    if not anime_name:
        return jsonify({"error": "未提供动漫名称"}), 400
    
    gallery_service = current_app.gallery_service
    characters_data = gallery_service.get_characters_by_anime(anime_name)
    
    return jsonify(characters_data)


@gallery_bp.route("/character_images/<path:character_id>")
@login_required
def get_character_images(character_id):
    """
    获取指定角色的所有图片样本, 包括已存在(train)和待处理(new)的。
    """
    if "/" not in character_id:
        return jsonify({"error": "无效的角色ID格式"}), 400

    gallery_service = current_app.gallery_service
    images_data = gallery_service.get_character_images(character_id)
    
    return jsonify(images_data)


@gallery_bp.route("/character_image", methods=["DELETE"])
@login_required
def delete_character_image():
    """删除指定的角色样本图片(支持批量), 并清理空目录和空的元数据。"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "请求体为空"}), 400

    character_id = data.get("character_id")  # 'anime_id/c_id'
    filenames = data.get("filenames", [])

    if not character_id or not filenames:
        return jsonify({"success": False, "message": "缺少参数"}), 400

    if "/" not in character_id:
        return jsonify({"success": False, "message": "无效的角色ID格式"}), 400

    gallery_service = current_app.gallery_service
    result = gallery_service.delete_character_images(
        character_id, filenames, location="train"
    )

    if not result["errors"]:
        return jsonify({
            "success": True, 
            "message": f"成功删除 {result['deleted_count']} 张图片。",
            "character_deleted": result["character_deleted"]
        })
    else:
        return jsonify({
            "success": result['deleted_count'] > 0,
            "message": f"操作完成，删除 {result['deleted_count']} 张图片。详情请查看错误列表。",
            "errors": result["errors"],
            "character_deleted": result["character_deleted"]
        })


@gallery_bp.route("/character/rename", methods=["POST"])
@login_required
def rename_character():
    """重命名一个角色"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "请求体为空"}), 400

    character_id = data.get("character_id")
    new_name = data.get("new_name", "").strip()

    if not all([character_id, new_name]):
        return (
            jsonify({"success": False, "message": "缺少参数 (character_id, new_name)"}),
            400,
        )
        
    if "/" not in character_id:
        return jsonify({"success": False, "message": "无效的角色ID格式"}), 400

    gallery_service = current_app.gallery_service
    success, message, status_code = gallery_service.rename_character(character_id, new_name)

    response_data = {"success": success, "message": message}
    if success:
        response_data["new_name"] = new_name
        # 重新加载全局变量以保证数据同步
        current_app.character_meta = gallery_service._load_meta()

    return jsonify(response_data), status_code
    

@gallery_bp.route("/character", methods=["DELETE"])
@login_required
def delete_character():
    """永久删除一个角色及其所有图片, 包括 train 和 new 目录, 并清理空的父目录。"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "请求体为空"}), 400

    character_id = data.get("character_id")
    if not character_id or "/" not in character_id:
        return jsonify({"success": False, "message": "缺少或无效的角色ID"}), 400
    
    gallery_service = current_app.gallery_service
    success, message, status_code = gallery_service.delete_character_and_images(character_id)

    if success:
        # 更新 app context 中的 meta
        current_app.character_meta = gallery_service._load_meta()

    return jsonify({"success": success, "message": message}), status_code


@gallery_bp.route("/pending_character_image", methods=["DELETE"])
@login_required
def delete_pending_character_image():
    """Deletes one or more images from a pending character folder in data/new, and cleans up empty dirs."""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "请求体为空"}), 400

    character_id = data.get("character_id")
    filenames = data.get("filenames")

    if not all([character_id, filenames]):
        return (
            jsonify({"success": False, "message": "缺少参数 (character_id, filenames)"}),
            400,
        )

    if not isinstance(filenames, list):
        filenames = [filenames]

    if "/" not in character_id:
        return jsonify({"success": False, "message": "无效的角色ID格式"}), 400
        
    gallery_service = current_app.gallery_service
    result = gallery_service.delete_character_images(
        character_id, filenames, location="new"
    )

    if result["character_deleted"]:
        current_app.character_meta = gallery_service._load_meta()

    if not result["errors"]:
        return jsonify({
            "success": True, 
            "message": f"成功删除 {result['deleted_count']} 个文件。",
            "character_deleted": result["character_deleted"]
        })
    else:
        return jsonify({
            "success": result['deleted_count'] > 0,
            "message": f"操作完成，删除 {result['deleted_count']} 张图片。详情请查看错误列表。",
            "errors": result["errors"],
            "character_deleted": result["character_deleted"]
        }) 