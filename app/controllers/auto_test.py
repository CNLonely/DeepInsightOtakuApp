from flask import Blueprint, jsonify, current_app, Response, stream_with_context
from flask_login import login_required

# 创建蓝图
auto_test_bp = Blueprint('auto_test', __name__)


@auto_test_bp.route("/api/admin/auto_test", methods=["GET"])
@login_required
def api_admin_auto_test():
    """
    自动遍历 /data/test/ 下所有图片，按规则统计动漫名、图片数、总人脸数、识别正确数、准确率。
    实时返回进度和结果。
    """
    # 从 current_app 获取服务实例
    auto_test_service = current_app.auto_test_service

    def generate():
        """生成流式响应"""
        for data in auto_test_service.run_auto_test():
            yield data

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@auto_test_bp.route("/api/admin/auto_test_info", methods=["GET"])
@login_required
def api_admin_auto_test_info():
    """获取自动测试的基本信息"""
    # 从 current_app 获取服务实例
    auto_test_service = current_app.auto_test_service
    return jsonify(auto_test_service.get_test_info()) 