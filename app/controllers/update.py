#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新控制器 - 提供更新检测和下载功能
"""

from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required
import json
import threading
import time

# 创建蓝图
update_bp = Blueprint('update', __name__)

# 全局变量存储更新进度
update_progress = {
    'is_updating': False,
    'progress': 0,
    'message': '',
    'result': None
}

def progress_callback(progress, message):
    """更新进度回调函数"""
    global update_progress
    update_progress['progress'] = progress
    update_progress['message'] = message

@update_bp.route('/api/admin/update/check', methods=['GET'])
@login_required
def check_updates():
    """检查是否有可用更新"""
    try:
        from app.services.update_service import UpdateService
        
        update_service = UpdateService(current_app.config, current_app.logger)
        result = update_service.check_for_updates()
        
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"检查更新异常: {e}")
        return jsonify({
            'has_update': False,
            'error': f'检查更新失败: {str(e)}'
        }), 500

@update_bp.route('/api/admin/update/history', methods=['GET'])
@login_required
def get_update_history():
    """获取更新历史"""
    try:
        from app.services.update_service import UpdateService
        
        update_service = UpdateService(current_app.config, current_app.logger)
        history = update_service.get_update_history()
        
        return jsonify({
            'success': True,
            'history': history
        })
        
    except Exception as e:
        current_app.logger.error(f"获取更新历史异常: {e}")
        return jsonify({
            'success': False,
            'error': f'获取更新历史失败: {str(e)}'
        }), 500

@update_bp.route('/api/admin/update/perform', methods=['POST'])
@login_required
def perform_update():
    """执行更新"""
    global update_progress
    
    # 检查是否正在更新
    if update_progress['is_updating']:
        return jsonify({
            'success': False,
            'error': '更新正在进行中，请稍后再试'
        }), 400
    
    try:
        from app.services.update_service import UpdateService
        
        # 重置更新状态
        update_progress['is_updating'] = True
        update_progress['progress'] = 0
        update_progress['message'] = '准备更新...'
        update_progress['result'] = None
        
        # 获取当前应用的配置和日志记录器
        app_config = current_app.config.copy()
        app_logger = current_app.logger
        
        # 在后台线程中执行更新
        def update_thread():
            global update_progress
            try:
                update_service = UpdateService(app_config, app_logger)
                result = update_service.perform_update(progress_callback)
                update_progress['result'] = result
            except Exception as e:
                update_progress['result'] = {
                    'success': False,
                    'error': f'更新异常: {str(e)}'
                }
            finally:
                update_progress['is_updating'] = False
        
        # 启动更新线程
        thread = threading.Thread(target=update_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': '更新已开始，请查看进度'
        })
        
    except Exception as e:
        update_progress['is_updating'] = False
        current_app.logger.error(f"启动更新异常: {e}")
        return jsonify({
            'success': False,
            'error': f'启动更新失败: {str(e)}'
        }), 500

@update_bp.route('/api/admin/update/progress', methods=['GET'])
@login_required
def get_update_progress():
    """获取更新进度"""
    global update_progress
    
    return jsonify({
        'is_updating': update_progress['is_updating'],
        'progress': update_progress['progress'],
        'message': update_progress['message'],
        'result': update_progress['result']
    })



@update_bp.route('/api/admin/update/status', methods=['GET'])
@login_required
def get_update_status():
    """获取更新状态信息"""
    try:
        from app.services.update_service import UpdateService
        
        update_service = UpdateService(current_app.config, current_app.logger)
        
        # 获取当前版本信息
        current_version = current_app.config.get('version', '1.0.0')
        
        # 获取GitHub仓库信息
        github_repo = current_app.config.get('github_repo', 'CNLonely/DeepInsightOtakuApp')
        
        return jsonify({
            'success': True,
            'current_version': current_version,
            'github_repo': github_repo,
            'last_check': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        current_app.logger.error(f"获取更新状态异常: {e}")
        return jsonify({
            'success': False,
            'error': f'获取状态失败: {str(e)}'
        }), 500 