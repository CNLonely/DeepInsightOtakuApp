import json
import time
from PIL import Image

from app.core.recognizer import Recognizer, TooManyFacesError

class RecognitionService:
    def __init__(self, recognizer: Recognizer, logger):
        self.recognizer = recognizer
        self.logger = logger

    def recognize_image_stream(self, image_pil: Image.Image, use_correction_override: bool = None, anime_only: bool = False):
        """
        以流式响应的方式处理图像识别，用于SSE。
        """
        t_stream_start = time.perf_counter()
        try:
            # 1. 先进行人脸检测并立即返回人脸框
            t_detect_start = time.perf_counter()
            detection_results = self.recognizer._run_face_detection(image_pil)
            t_detect_end = time.perf_counter()
            self.logger.info(f"[API_PROFILING_STREAM] Initial detection (_run_face_detection): {(t_detect_end - t_detect_start) * 1000:.2f} ms")
            
            first_boxes, _, _ = detection_results
            boxes_list = first_boxes.tolist() if hasattr(first_boxes, 'tolist') else first_boxes
            
            # yield 'detected' 事件
            yield f"event: detected\ndata: {json.dumps({'boxes': boxes_list})}\n\n"

            # 2. 然后继续进行完整的识别
            t_rec_start = time.perf_counter()
            faces = self.recognizer.detect_and_recognize(
                image_pil,
                include_images=False,
                anime_only=anime_only,
                use_correction_override=use_correction_override,
                detection_results=detection_results,
            )
            t_rec_end = time.perf_counter()
            self.logger.info(f"[API_PROFILING_STREAM] Main recognition (detect_and_recognize): {(t_rec_end - t_rec_start) * 1000:.2f} ms")
            
            # 3. yield 'completed' 事件
            yield f"event: completed\ndata: {json.dumps({'faces': faces})}\n\n"
            
            t_stream_end = time.perf_counter()
            self.logger.info(f"[API_PROFILING_STREAM] Total stream processing time: {(t_stream_end - t_stream_start) * 1000:.2f} ms")

        except TooManyFacesError as e:
            error_data = {"error": f"图片人数超出限制 (检测到 {e.count} 人, 上限 {e.limit} 人)。"}
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
        except Exception as e:
            self.logger.error(f"Error in recognition stream: {e}", exc_info=True)
            error_data = {"error": f"无法解析或处理图片: {str(e)}"}
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

    def recognize_image(self, image_pil: Image.Image, use_correction_override: bool = None, anime_only: bool = False, include_images: bool = False):
        """
        以普通（非流式）方式处理图像识别。
        """
        return self.recognizer.detect_and_recognize(
            image_pil,
            include_images=include_images,
            anime_only=anime_only,
            use_correction_override=use_correction_override,
        )

    def reload_data(self, class_to_idx, character_meta, faiss_index, faiss_index_to_label):
        """调用底层识别器的热重载方法"""
        self.recognizer.reload_data(class_to_idx, character_meta, faiss_index, faiss_index_to_label)
        self.logger.info("[RecognitionService] 数据已通知底层识别器进行重载。") 