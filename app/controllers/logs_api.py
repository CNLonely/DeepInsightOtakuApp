import os
import time
from flask import Blueprint, Response, stream_with_context, current_app, jsonify
from flask_login import login_required

from app.log import get_latest_log_file
from flask import current_app

logs_api_bp = Blueprint("logs_api", __name__)

@logs_api_bp.route("/logs/stream")
@login_required
def logs_stream():
    """
    使用 Server-Sent Events (SSE) 流式传输最新的日志。
    这个版本通过定期检查文件修改时间来决定是否推送新日志，
    比简单地 `tail -f` 更具鲁棒性，尤其是在日志文件被轮换或删除时。
    """
    def generate():
        log_file = get_latest_log_file()
        last_position = 0

        # 1. 发送历史日志 (采用更简单可靠的方式)
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    # 直接读取所有行，取最后200行。对于单日日志文件来说足够高效。
                    lines = f.readlines()
                    initial_logs = lines[-200:]
                    for line in initial_logs:
                        yield f"data: {line.strip()}\n\n"
                    
                    # 更新读取位置到文件末尾
                    last_position = f.tell()
        except Exception as e:
            current_app.logger.error(f"Error sending initial logs: {e}", exc_info=True)
            yield f"data: [SYSTEM] Error reading initial log file: {e}\n\n"

        # 2. 进入实时监控循环 (Tailing)
        while True:
            try:
                if os.path.exists(log_file):
                    current_size = os.path.getsize(log_file)
                    
                    # 文件被截断或轮换
                    if current_size < last_position:
                        last_position = 0
                        yield "data: [SYSTEM] Log file has been rotated or truncated. Restarting stream.\n\n"

                    # 文件有新内容
                    if current_size > last_position:
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            f.seek(last_position)
                            for line in f:
                                yield f"data: {line.strip()}\n\n"
                            last_position = f.tell()
                else:
                    # 如果日志文件消失了，等待它再次出现
                    yield "data: [SYSTEM] Waiting for log file to be created...\n\n"
                    last_position = 0 # 重置位置

            except Exception as e:
                error_msg = f"[SYSTEM] An error occurred in log stream: {e}"
                current_app.logger.error(error_msg, exc_info=True)
                yield f"data: {error_msg}\n\n"
                time.sleep(5)
            
            time.sleep(1)

    return Response(stream_with_context(generate()), mimetype="text/event-stream") 