from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import json
import os
import threading
from app.config import USERS_DB_PATH

auth_bp = Blueprint('auth', __name__)

# --- Flask-Login 初始化 ---
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "请登录以访问此页面。"
login_manager.login_message_category = "info"

# --------------------------- 全局锁 --------------------------- #
# 用于保护 users.json 文件的读写，防止多线程竞争条件
users_lock = threading.Lock()

def load_users():
    """从JSON文件加载用户数据"""
    if not os.path.exists(USERS_DB_PATH):
        # 如果文件不存在，创建并添加默认的admin用户
        default_password_hash = generate_password_hash("123456", method='pbkdf2:sha256')
        default_users = {"admin": default_password_hash}
        with open(USERS_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(default_users, f, ensure_ascii=False, indent=4)
        return default_users
    
    try:
        with open(USERS_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # 如果文件损坏或无法读取，也返回默认用户
        default_password_hash = generate_password_hash("123456", method='pbkdf2:sha256')
        return {"admin": default_password_hash}

def save_users(users_data):
    """保存用户数据到JSON文件"""
    with open(USERS_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(users_data, f, ensure_ascii=False, indent=4)

# 确保文件在启动时被创建
load_users()

class User(UserMixin):
    def __init__(self, id, password_hash):
        self.id = id
        self.password_hash = password_hash

    @staticmethod
    def check_password_static(password_hash, password):
        return check_password_hash(password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    users = load_users()
    if user_id in users:
        return User(user_id, password_hash=users[user_id])
    return None

@auth_bp.before_app_request
def check_default_password():
    # 确保用户已登录，且不在修改密码页面或静态文件路由
    if current_user.is_authenticated and request.endpoint not in ['auth.login', 'auth.change_password', 'auth.logout', 'static']:
        # 检查是否为默认密码
        if User.check_password_static(current_user.password_hash, "123456"):
            flash("首次登录或密码为默认密码，请修改您的密码。", "warning")
            return redirect(url_for('auth.change_password'))

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('views.admin'))
    
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        remember_me = request.form.get("remember") == "1"
        
        users = load_users()
        user_password_hash = users.get(username)

        if user_password_hash and User.check_password_static(user_password_hash, password):
            user = User(username, password_hash=user_password_hash)
            login_user(user, remember=remember_me)
            current_app.logger.info(f"用户 '{username}' 登录成功。")
            # 登录后重定向到之前尝试访问的页面，或后台主页
            next_page = request.args.get('next')
            return redirect(next_page or url_for('views.admin'))
        else:
            flash("用户名或密码无效。", "danger")
            current_app.logger.warning(f"用户 '{username}' 尝试使用无效凭证登录。")
            
    return render_template("backend/login.html")

@auth_bp.route("/logout")
@login_required
def logout():
    username = current_user.id
    logout_user()
    current_app.logger.info(f"用户 '{username}' 已成功注销。")
    flash("您已成功注销。", "success")
    return redirect(url_for('auth.login'))

@auth_bp.route("/admin/change_password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        new_password = request.form.get("new_password")
        confirm_password = request.form.get("confirm_password")

        if not new_password or len(new_password) < 6:
            flash("新密码不能为空且长度至少为6位。", "danger")
            return redirect(url_for('auth.change_password'))
        
        if new_password != confirm_password:
            flash("两次输入的密码不一致。", "danger")
            return redirect(url_for('auth.change_password'))

        # 加上锁来保证读写操作的原子性
        with users_lock:
            users = load_users()
            users[current_user.id] = generate_password_hash(new_password, method='pbkdf2:sha256')
            save_users(users)
        
        username = current_user.id
        # 强制用户重新登录以刷新会话
        logout_user()
        current_app.logger.info(f"用户 '{username}' 已成功修改密码并被强制注销。")
        flash("密码已成功修改，请使用新密码重新登录。", "success")
        return redirect(url_for('auth.login'))

    return render_template("backend/change_password.html")


# --------------------------- 用户管理 API --------------------------- #
@auth_bp.route("/api/admin/users", methods=["GET", "POST", "DELETE"])
@login_required
def manage_users():
    # 将 users = load_users() 移动到锁内部，确保每次操作都基于最新数据
    if request.method == "GET":
        with users_lock:
            users = load_users()
            return jsonify(list(users.keys()))

    if request.method == "POST":
        data = request.get_json()
        if not data or 'username' not in data or not data['username'].strip():
            return jsonify({"success": False, "message": "用户名不能为空"}), 400
        
        username = data['username'].strip()

        with users_lock:
            users = load_users()
            if username in users:
                return jsonify({"success": False, "message": "用户名已存在"}), 409
            
            users[username] = generate_password_hash("123456", method='pbkdf2:sha256')
            save_users(users)
            current_app.logger.info(f"管理员 '{current_user.id}' 创建了新用户 '{username}'。")
            # 返回完整的用户列表，以便前端直接更新
            return jsonify({
                "success": True, 
                "message": f"用户 '{username}' 创建成功，默认密码为 123456。",
                "users": list(users.keys())
            })

    if request.method == "DELETE":
        data = request.get_json()
        if not data or 'username' not in data:
            return jsonify({"success": False, "message": "未提供要删除的用户名"}), 400
        
        username = str(data['username']).strip()  # 强制转为字符串并去除空格

        with users_lock:
            users = load_users()
            if username not in users:
                return jsonify({"success": False, "message": "用户不存在"}), 404

            if len(users) <= 1:
                return jsonify({"success": False, "message": "系统至少需要保留一个管理员账户"}), 400
            
            if username == current_user.id:
                return jsonify({"success": False, "message": "不能删除当前登录的账户"}), 400

            del users[username]
            save_users(users)
            current_app.logger.info(f"管理员 '{current_user.id}' 删除了用户 '{username}'。")
             # 返回完整的用户列表，以便前端直接更新
            return jsonify({
                "success": True, 
                "message": f"用户 '{username}' 已被删除。",
                "users": list(users.keys())
            }) 