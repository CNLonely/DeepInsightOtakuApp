from flask import render_template, Blueprint
from flask_login import login_required
import datetime

# 创建一个蓝图对象
views_bp = Blueprint('views', __name__, template_folder='../templates')

@views_bp.context_processor
def inject_current_year():
    """向所有模板注入当前年份"""
    return {'current_year': datetime.datetime.now().year}

@views_bp.route("/")
def index():
    """首页：上传图片"""
    return render_template("frontend/index.html")

@views_bp.route("/admin")
@login_required
def admin():
    """管理页面"""
    return render_template("backend/manage.html")

@views_bp.route("/admin/settings")
@login_required
def admin_settings():
    """设置页面"""
    return render_template("backend/settings.html")

@views_bp.route("/admin/batch_import")
@login_required
def admin_batch_import():
    """重命名与索引工具页面"""
    return render_template("backend/batch_import.html")

@views_bp.route("/admin/api")
@login_required
def admin_api_docs():
    """API文档页面"""
    return render_template("backend/api_docs.html")

@views_bp.route("/admin/gallery")
@login_required
def admin_gallery():
    """角色画廊页面"""
    return render_template("backend/character_gallery.html")

@views_bp.route("/admin/auto_test")
@login_required
def admin_auto_test():
    """自动测试页面"""
    return render_template("backend/auto_test.html")

@views_bp.route("/admin/bilibot")
@login_required
def bilibot_page():
    """渲染B站机器人管理页面"""
    return render_template("backend/bilibot.html")

import json
import os

@views_bp.route("/contributors")
def contributors():
    """贡献榜页面"""
    json_path = os.path.join('templates', 'img', 'data.json')
    contributors_data = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 将字典转换为列表，并为每个贡献者添加一个默认状态
            for key, value in data.items():
                contributor = {
                    'img': value.get('img'),
                    'name': value.get('name', '匿名'),
                    'status': value.get('status', '基础开发者')
                }
                contributors_data.append(contributor)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading contributors data: {e}")
        # 即使出错，也渲染一个空列表的页面
        pass

    return render_template("frontend/contributors.html", contributors=contributors_data) 