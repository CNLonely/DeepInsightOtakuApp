from flask import Blueprint, render_template, request, jsonify, current_app, session, send_file
from flask_login import login_required
import asyncio
import requests
import io
import base64
import qrcode
import uuid

from app.services.bilibot_service import get_bot_instance
from app.config import save_bilibot_config, load_bilibot_config, DEFAULT_BILIBOT_CONFIG
# 不再需要直接从控制器调用 login_v2
# from bilibili_api import login_v2, Credential, exceptions

bilibot_bp = Blueprint("bilibot", __name__)

# 全局二维码会话存储已移至 BilibotService
# qrcode_sessions = {}


@bilibot_bp.route("/api/bilibot/status")
@login_required
def api_get_status():
    """获取B站机器人状态"""
    bot = get_bot_instance()
    return jsonify(bot.get_status())

@bilibot_bp.route("/api/bilibot/control", methods=["POST"])
@login_required
def control_bilibot():
    """启动或停止机器人"""
    action = request.json.get('action')
    bot = get_bot_instance()
    if action == 'start':
        bot.start()
        return jsonify({"success": True, "message": "机器人已启动"})
    elif action == 'stop':
        bot.stop()
        return jsonify({"success": True, "message": "机器人已停止"})
    return jsonify({"success": False, "message": "无效操作"}), 400

@bilibot_bp.route("/api/bilibot/login", methods=["POST"])
@login_required
def api_login():
    """使用Cookie登录"""
    data = request.get_json()
    cookies_str = data.get('cookies')
    if not cookies_str:
        return jsonify({"success": False, "message": "未提供Cookie"}), 400

    # 解析Cookie字符串为字典
    cookies = {}
    try:
        for item in cookies_str.split(';'):
            item = item.strip()
            if not item: continue
            key, value = item.split('=', 1)
            cookies[key.strip()] = value.strip()
    except ValueError:
        return jsonify({"success": False, "message": "Cookie格式错误"}), 400
        
    bot = get_bot_instance()
    
    # 手动提取必要的key
    required_keys = ["SESSDATA", "bili_jct", "buvid3", "DedeUserID"]
    login_cookies = {key: cookies.get(key) for key in required_keys}
    
    if not all(login_cookies.values()):
        return jsonify({"success": False, "message": f"Cookie缺少必要的字段，需要: {', '.join(required_keys)}"}), 400

    success = bot.login_with_cookies(login_cookies)
    
    if success:
        # 保存有效cookie到主配置
        bilibot_config = load_bilibot_config()
        bilibot_config['bilibot_cookies'] = login_cookies
        save_bilibot_config(bilibot_config)
        current_app.config['bilibot'].update(bilibot_config)
        bot.update_config(bilibot_config)
        
        return jsonify({"success": True, "message": "登录成功！", "user_info": bot.user_info})
    else:
        return jsonify({"success": False, "message": "登录失败，请检查Cookie是否正确或已过期。"}), 401


@bilibot_bp.route("/api/bilibot/logout", methods=["POST"])
@login_required
def api_logout():
    """登出并清除Cookie"""
    bot = get_bot_instance()
    bot.logout() # 调用服务层方法来清理内部状态
    
    # 控制器负责处理应用级别的配置和持久化
    bilibot_config = load_bilibot_config()
    if 'bilibot_cookies' in bilibot_config:
        del bilibot_config['bilibot_cookies']
        save_bilibot_config(bilibot_config)
        
        # 更新 app.config 以同步状态
        current_app.config['bilibot'].update(bilibot_config)
        # 通知服务配置已变更（虽然此处是删除，但仍是好习惯）
        bot.update_config(current_app.config['bilibot']) 
        
    return jsonify({"success": True, "message": "已登出"})

@bilibot_bp.route("/api/bilibot/config", methods=["GET", "POST"])
@login_required
def api_bilibot_config():
    """获取或更新Bilibot的配置"""
    bot = get_bot_instance()
    
    if request.method == "GET":
        # 总是从文件加载最新配置，以确保数据一致性
        return jsonify(load_bilibot_config())

    if request.method == "POST":
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "请求体为空"}), 400
        
        try:
            # 加载现有配置
            bilibot_config = load_bilibot_config()

            # 安全地更新每个字段，仅当它在请求中被提供且不为None时
            if data.get('enabled') is not None:
                bilibot_config['enabled'] = bool(data['enabled'])
            if data.get('polling_interval_base') is not None:
                bilibot_config['polling_interval_base'] = int(data['polling_interval_base'])
            if data.get('polling_interval_jitter') is not None:
                bilibot_config['polling_interval_jitter'] = int(data['polling_interval_jitter'])
            if data.get('trigger_keyword') is not None:
                bilibot_config['trigger_keyword'] = str(data['trigger_keyword'])
            if data.get('confidence_threshold') is not None:
                bilibot_config['confidence_threshold'] = float(data['confidence_threshold'])
            if data.get('use_obfuscation') is not None:
                bilibot_config['use_obfuscation'] = bool(data['use_obfuscation'])
            
            # 对于列表，也检查类型
            if 'reply_templates' in data and isinstance(data.get('reply_templates'), list):
                bilibot_config['reply_templates'] = data['reply_templates']
            if 'random_embellishments' in data and isinstance(data.get('random_embellishments'), list):
                bilibot_config['random_embellishments'] = data['random_embellishments']

            # 保存到文件
            save_bilibot_config(bilibot_config)
            
            # 热更新 bot service 和 app.config
            current_app.config['bilibot'].update(bilibot_config)
            bot.update_config(bilibot_config)

            return jsonify({"success": True, "message": "配置已更新。"})
        
        except (ValueError, TypeError) as e:
            return jsonify({"success": False, "message": f"数据格式错误: {e}"}), 400
        except Exception as e:
            return jsonify({"success": False, "message": f"保存配置失败: {e}"}), 500

@bilibot_bp.route("/api/bilibot/login/qrcode")
@login_required
def api_bilibot_login_qrcode():
    """获取B站登录二维码"""
    bot = get_bot_instance()
    login_session_id = str(uuid.uuid4())
    session['bili_login_session_id'] = login_session_id
    
    try:
        # 调用服务层方法生成二维码
        img_str = bot.generate_login_qrcode(login_session_id)
        return jsonify({"success": True, "image": img_str})
    except Exception as e:
        current_app.logger.error(f"生成二维码时发生错误: {e}")
        return jsonify({"success": False, "message": f"生成二维码失败: {e}"}), 500


@bilibot_bp.route("/api/bilibot/login/check")
@login_required
def api_bilibot_login_check():
    """轮询二维码扫描状态"""
    login_session_id = session.get('bili_login_session_id')
    if not login_session_id:
        return jsonify({"success": False, "code": -1, "message": "会话已过期，请刷新二维码"})

    bot = get_bot_instance()
    result = bot.check_login_qrcode_state(login_session_id)

    # 如果登录成功，控制器负责保存cookie并清理session
    if result.get("success"):
        bilibot_config = load_bilibot_config()
        bilibot_config['bilibot_cookies'] = result.get("credential")
        save_bilibot_config(bilibot_config)
        
        # 热更新 app.config 和服务内部的 config
        current_app.config['bilibot'].update(bilibot_config)
        bot.update_config(bilibot_config)
        
        # 清理会话
        session.pop('bili_login_session_id', None)

    # 如果会话已终结（成功、失败、超时），也需要清理 session
    if result.get("code") not in [-2]: # -2 是 "等待中"
        session.pop('bili_login_session_id', None)

    # 直接将服务层返回的结果作为JSON响应
    return jsonify(result)


@bilibot_bp.route("/api/bilibot/avatar_proxy")
@login_required
def bilibot_avatar_proxy():
    """
    一个服务器端代理，用于绕过B站头像的防盗链。
    """
    avatar_url = request.args.get('url')
    if not avatar_url:
        return "Missing URL", 400

    try:
        # 设置一个看起来合法的Referer
        headers = {
            'Referer': 'https://www.bilibili.com/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(avatar_url, headers=headers, stream=True)
        response.raise_for_status()
        
        # 将获取到的图片数据流式传输回客户端
        image_data = io.BytesIO(response.content)
        return send_file(image_data, mimetype=response.headers.get('Content-Type', 'image/jpeg'))

    except requests.RequestException as e:
        # 如果请求失败，可以返回一个默认的占位符图片或错误信息
        current_app.logger.error(f"[BiliBot] 头像代理失败: {e}")
        return "Failed to fetch image", 500