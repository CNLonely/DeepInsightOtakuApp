from flask import Blueprint, request, jsonify, send_file, current_app, Response, stream_with_context
from PIL import Image
import io
import base64
import json
import time

from app.controllers.statistics import increment_recognition_count
from app.utils.image_renderer import render_cards_to_img_bytes
from app.core.recognizer import TooManyFacesError

recognize_bp = Blueprint('recognize', __name__)

@recognize_bp.route("/api/recognize", methods=["POST"])
def api_recognize():
    """
    前端异步上传文件后调用该接口。
    - 正常情况返回识别JSON。
    - 如果请求参数 stream_progress=1，则使用SSE流式返回进度。
    """
    t_total_start = time.perf_counter()

    if "file" not in request.files:
        return jsonify({"error": "未收到文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    stream_progress = request.form.get("stream_progress") == "1"
    
    # 从 service locator (current_app) 获取服务实例
    recognition_service = current_app.recognition_service

    try:
        use_correction_str = request.form.get("use_correction")
        anime_only_str = request.form.get("anime_only")
        use_correction_override = None
        if use_correction_str in ["0", "1"]:
            use_correction_override = bool(int(use_correction_str))

        anime_only = anime_only_str == "1"
        
        t_decode_start = time.perf_counter()
        image_pil = Image.open(file.stream).convert("RGB")
        t_decode_end = time.perf_counter()
        current_app.logger.info(f"[API_PROFILING] Image decoding (Image.open): {(t_decode_end - t_decode_start) * 1000:.2f} ms")

        if not stream_progress:
            # --- 经典模式 ---
            t_rec_start = time.perf_counter()
            faces = recognition_service.recognize_image(
                image_pil,
                use_correction_override=use_correction_override,
                anime_only=anime_only,
                include_images=False,
            )
            t_rec_end = time.perf_counter()
            current_app.logger.info(f"[API_PROFILING] detect_and_recognize call: {(t_rec_end - t_rec_start) * 1000:.2f} ms")
            
            increment_recognition_count()

            t_total_end = time.perf_counter()
            current_app.logger.info(f"[API_PROFILING] Total request time: {(t_total_end - t_total_start) * 1000:.2f} ms")
            return jsonify({"faces": faces})
        else:
            # --- 流式响应模式 ---
            def generate():
                # 调用服务层的流式方法
                yield from recognition_service.recognize_image_stream(
                    image_pil,
                    use_correction_override=use_correction_override,
                    anime_only=anime_only
                )
                # 识别计数在流完成后进行
                increment_recognition_count()

            return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except TooManyFacesError as e:
        return (
            jsonify({"error": f"图片人数超出限制 (检测到 {e.count} 人, 上限 {e.limit} 人)。"}),
            413,
        )
    except Exception as e:
        current_app.logger.error(f"Error in api_recognize: {e}", exc_info=True)
        return jsonify({"error": f"无法解析或处理图片: {e}"}), 400


@recognize_bp.route("/api/generate_image", methods=["POST"])
def api_generate_image():
    """
    接收上传的图片，识别后，将结果渲染成一张图片并返回。
    """
    if "file" not in request.files:
        return jsonify({"error": "未收到文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    recognition_service = current_app.recognition_service

    try:
        t_total_start = time.perf_counter()

        use_correction_str = request.form.get("use_correction")
        use_correction_override = None
        if use_correction_str in ["0", "1"]:
            use_correction_override = bool(int(use_correction_str))

        t_decode_start = time.perf_counter()
        image_pil = Image.open(file.stream).convert("RGB")
        t_decode_end = time.perf_counter()
        current_app.logger.info(f"[API_PROFILING] Image decoding (Image.open): {(t_decode_end - t_decode_start) * 1000:.2f} ms")
        
        t_rec_start = time.perf_counter()
        faces = recognition_service.recognize_image(
            image_pil,
            use_correction_override=use_correction_override,
            include_images=True,
        )
        t_rec_end = time.perf_counter()
        current_app.logger.info(f"[API_PROFILING] detect_and_recognize call: {(t_rec_end - t_rec_start) * 1000:.2f} ms")

        increment_recognition_count()
    except TooManyFacesError as e:
        return (
            jsonify({"error": f"图片人数超出限制 (检测到 {e.count} 人, 上限 {e.limit} 人)。"}),
            413,
        )
    except Exception as e:
        current_app.logger.error(f"Error in api_generate_image during recognition: {e}", exc_info=True)
        return jsonify({"error": f"无法解析或处理图片: {e}"}), 400

    if not faces:
        return jsonify({"error": "未检测到人脸"}), 404

    try:
        image_bytes = render_cards_to_img_bytes(faces)
    except Exception as e:
        current_app.logger.error(f"Error in api_generate_image during rendering: {e}", exc_info=True)
        return jsonify({"error": "生成结果图片失败"}), 500

    if not image_bytes:
        return jsonify({"error": "生成结果图片失败"}), 500

    t_total_end = time.perf_counter()
    current_app.logger.info(f"[API_PROFILING] Total request time (generate_image): {(t_total_end - t_total_start) * 1000:.2f} ms")
    return send_file(
        io.BytesIO(image_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name="result.png",
    ) 