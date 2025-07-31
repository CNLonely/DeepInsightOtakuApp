import json
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
from torchvision import ops
from torchvision.transforms import functional as TF
import cv2
from collections import Counter
# from flask import current_app # 移除对Flask的强依赖
from app.config import (
    LEFT_RATIO,
    TOP_RATIO,
    RIGHT_RATIO,
    BOTTOM_RATIO,
)

from app.core.models import ArcFaceModel 

import io
import base64
import time


def crop_face(image_pil, box):
    left, top, right, bottom = map(int, box)
    return image_pil.crop((left, top, right, bottom))

def expand_box(box, image_size, scale=1.2):
    """将box按中心放大scale倍，自动裁剪到图片边界"""
    left, top, right, bottom = box
    w = right - left
    h = bottom - top
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    new_w = w * scale
    new_h = h * scale
    new_left = max(0, int(cx - new_w / 2))
    new_top = max(0, int(cy - new_h / 2))
    new_right = min(image_size[0], int(cx + new_w / 2))
    new_bottom = min(image_size[1], int(cy + new_h / 2))
    return [new_left, new_top, new_right, new_bottom]

class TooManyFacesError(Exception):
    """自定义异常，用于表示检测到的人脸过多。"""
    def __init__(self, count, limit):
        self.count = count
        self.limit = limit
        super().__init__(f"Detected {count} faces, which exceeds the limit of {limit}.")

def resize_with_padding(img: Image.Image, target_size=(112, 112)) -> Image.Image:
    """保持完整内容：等比缩放后在四周填充空白至固定尺寸，不裁剪原图"""
    ratio = min(target_size[0] / img.width, target_size[1] / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    paste_x = (target_size[0] - new_size[0]) // 2
    paste_y = (target_size[1] - new_size[1]) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    return new_img


class Recognizer:
    def __init__(self, model, onnx_session, class_to_idx, character_meta, device, first_model, faiss_index, faiss_index_to_label, logger, config):
        self.model = model
        self.onnx_session = onnx_session
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.character_meta = character_meta
        self.device = device
        self.first_model = first_model
        self.faiss_index = faiss_index
        self.faiss_index_to_label = faiss_index_to_label
        self.logger = logger
        self.config = config

    def reload_data(self, class_to_idx, character_meta, faiss_index, faiss_index_to_label):
        """热重载识别器所需的数据"""
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.character_meta = character_meta
        self.faiss_index = faiss_index
        self.faiss_index_to_label = faiss_index_to_label
        self.logger.info("[Recognizer] 核心数据已成功热重载。")

    # ------------------- Refactored Helper Methods -------------------

    def _run_detection_and_batch_recognition(self, image_pil: Image.Image, detection_results: tuple = None):
        """
        运行人脸检测和批量识别，返回中间结果。
        这是v1和v2算法的共同初步步骤。
        """
        if detection_results is None:
            first_boxes, first_confs, first_classes = self._run_face_detection(image_pil)
        else:
            first_boxes, first_confs, first_classes = detection_results

        if len(first_boxes) == 0:
            return None

        high_conf_indices = [i for i, conf in enumerate(first_confs) if conf >= 0.7]
        if not high_conf_indices:
            return None
            
        high_conf_boxes = first_boxes[high_conf_indices]

        recognition_results_list = self._recognize_faces_batch(image_pil, high_conf_boxes)

        return {
            "recognition_results_list": recognition_results_list,
            "first_boxes": first_boxes,
            "first_classes": first_classes,
            "high_conf_indices": high_conf_indices,
        }

    def _process_initial_results(self, preliminary_results: dict):
        """
        将原始识别结果处理成标准的中间数据结构列表。
        """
        if not preliminary_results:
            return []

        recognition_results_list = preliminary_results["recognition_results_list"]
        first_boxes = preliminary_results["first_boxes"]
        first_classes = preliminary_results["first_classes"]
        high_conf_indices = preliminary_results["high_conf_indices"]

        processed_faces = []
        for i, recognition_data in enumerate(recognition_results_list):
            original_index = high_conf_indices[i]
            box1 = first_boxes[original_index]
            
            top_results = recognition_data["top_results"]
            top_k_info = []
            for char_id, char_score in top_results:
                char_name, char_anime = self.get_character_details(char_id)
                if "未命名" in char_name:
                    char_name = "未命名"
                top_k_info.append(
                    {
                        "identity": char_id, "name": char_name, "anime": char_anime,
                        "score": round(float(char_score), 3),
                    }
                )
            
            first_class = (
                int(first_classes[original_index])
                if first_classes is not None and original_index < len(first_classes)
                else None
            )

            processed_info = {
                "first_class": first_class,
                "bounding_box": [float(c) for c in box1],
                "recognition_box": [float(c) for c in recognition_data["expanded_box"]],
                "resolution": recognition_data["resolution"],
                "top_k": top_k_info,
                "image_pil_for_encoding": recognition_data["refined_face_for_rec"],
                "image_pil_original_crop": recognition_data["original_crop_pil"],
            }
            processed_faces.append(processed_info)
        return processed_faces

    def _assemble_final_face_info(self, face_data: dict, include_images: bool = True):
        """
        从处理过的数据组装单个面部的最终JSON可序列化字典。
        """
        final_top_result = face_data["top_k"][0]
        identity = final_top_result["identity"]
        score = final_top_result["score"]
        name = final_top_result["name"]
        anime = final_top_result["anime"]

        self.logger.info(f"最终识别结果: {identity} - {name} - {anime} - {score}")

        face_info = {
            "identity": identity, "score": score, "name": name, "anime": anime,
            "first_class": face_data["first_class"],
            "bounding_box": face_data["bounding_box"],
            "recognition_box": face_data["recognition_box"],
            "resolution": face_data["resolution"],
            "top_k": face_data["top_k"],
            "is_corrected": face_data.get("is_corrected", False),
        }

        if include_images:
            buf_display = io.BytesIO()
            face_data["image_pil_for_encoding"].save(buf_display, format="PNG")
            img_display_b64 = base64.b64encode(buf_display.getvalue()).decode()

            buf_original = io.BytesIO()
            face_data["image_pil_original_crop"].save(buf_original, format="PNG")
            img_original_b64 = base64.b64encode(buf_original.getvalue()).decode()

            face_info["image"] = img_display_b64
            face_info["image_original"] = img_original_b64

        return face_info

    # ------------------- Public API Methods -------------------

    def get_character_details(self, identity: str):
        """从元数据中获取角色名和动漫名(适配新结构)"""
        meta = self.character_meta
        name = "未知角色"
        anime = "未知作品"
        if identity != "未知角色" and "/" in identity:
            anime_id, char_id = identity.split("/", 1)
            anime_info = meta.get(anime_id, {})
            if anime_info:
                anime = anime_info.get("name", anime_id)
                char_info = anime_info.get("characters", {}).get(char_id, {})
                if char_info:
                    name = char_info.get("name", char_id)
        return name, anime


    def _run_face_detection(self, image_pil: Image.Image):
        """
        运行YOLO人脸检测，包括NMS，并返回过滤后的结果。
        """
        t_start = time.perf_counter()

        first_model = self.first_model
        max_faces_limit = self.config.get("max_faces", 20)

        t_pre_start = time.perf_counter()
        img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        t_pre_end = time.perf_counter()
        self.logger.info(f"[PROFILING] Pre-processing (cvtColor): {(t_pre_end - t_pre_start) * 1000:.2f} ms")

        t_infer_start = time.perf_counter()
        first_results = first_model(img_bgr, conf=0.3, iou=0.5, verbose=False)[0]
        t_infer_end = time.perf_counter()
        self.logger.info(f"[PROFILING] YOLOv8 Inference (incl. internal NMS): {(t_infer_end - t_infer_start) * 1000:.2f} ms")

        detected_faces_count = len(first_results.boxes)
        if detected_faces_count > max_faces_limit:
            raise TooManyFacesError(count=detected_faces_count, limit=max_faces_limit)

        if detected_faces_count == 0:
            self.logger.info(f"[PROFILING] Total face detection time: {(time.perf_counter() - t_start) * 1000:.2f} ms. No faces found.")
            return [], [], []

        t_post_start = time.perf_counter()
        boxes = first_results.boxes.xyxy.cpu().numpy()
        confs = first_results.boxes.conf.cpu().numpy()
        classes = (
            first_results.boxes.cls.cpu().numpy()
            if hasattr(first_results.boxes, "cls") and first_results.boxes.cls is not None
            else None
        )
        t_post_end = time.perf_counter()
        self.logger.info(f"[PROFILING] Post-processing (Tensor to Numpy): {(t_post_end - t_post_start) * 1000:.2f} ms")

        if len(boxes) == 0:
            self.logger.info(f"[PROFILING] Total face detection time: {(time.perf_counter() - t_start) * 1000:.2f} ms. No faces after tensor conversion.")
            return [], [], []
        
        t_nms_start = time.perf_counter()
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(confs, dtype=torch.float32)
        keep_idxs = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
        t_nms_end = time.perf_counter()
        self.logger.info(f"[PROFILING] Manual torchvision NMS: {(t_nms_end - t_nms_start) * 1000:.2f} ms. (Note: This might be redundant)")


        final_boxes = boxes[keep_idxs]
        final_confs = confs[keep_idxs]
        final_classes = classes[keep_idxs] if classes is not None else None

        if final_boxes.ndim == 1:
            final_boxes = np.expand_dims(final_boxes, 0)
        
        final_confs = np.atleast_1d(final_confs)
        if final_classes is not None:
            final_classes = np.atleast_1d(final_classes)

        self.logger.info(f"[PROFILING] Total face detection time: {(time.perf_counter() - t_start) * 1000:.2f} ms. Found {len(final_boxes)} faces.")
        return final_boxes, final_confs, final_classes


    def _recognize_faces_batch(self, image_pil: Image.Image, boxes: list):
        """
        对多个检测框进行批量人脸识别。
        """
        model = self.model
        onnx_session = self.onnx_session
        device = self.device
        recognition_threshold = self.config.get("recognition_threshold", 0.5)
        use_low_res_enhancement = True 
        k = 5

        face_images_for_rec = []
        expanded_boxes = []
        original_crops = []

        for box in boxes:
            original_crop_pil = crop_face(image_pil, box)
            w, h = original_crop_pil.size
            x1, y1, x2, y2 = box

            x1_n = max(0, x1 - LEFT_RATIO * w)
            y1_n = max(0, y1 - TOP_RATIO * h)
            x2_n = min(image_pil.size[0], x2 + RIGHT_RATIO * w)
            y2_n = min(image_pil.size[1], y2 + BOTTOM_RATIO * h)
            expanded_box = [x1_n, y1_n, x2_n, y2_n]

            face_img_for_rec = crop_face(image_pil, expanded_box)
            refined_face_for_rec = resize_with_padding(face_img_for_rec)

            original_crops.append(original_crop_pil)
            expanded_boxes.append(expanded_box)
            face_images_for_rec.append(refined_face_for_rec)

        if not face_images_for_rec:
            return []

        all_augmented_tensors = []
        num_augs_per_face = 0
        basic_transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

        for face_img in face_images_for_rec:
            augs_for_this_face = []
            augs_for_this_face.append(face_img)
            augs_for_this_face.append(TF.adjust_brightness(face_img, 1.2))
            if use_low_res_enhancement:
                scales = [0.3, 0.4]
                for scale in scales:
                    w, h = face_img.size
                    low_res_img = face_img.resize(
                        (int(w * scale), int(h * scale)), Image.Resampling.BILINEAR
                    )
                    enhanced_img = low_res_img.resize((w, h), Image.Resampling.BILINEAR)
                    augs_for_this_face.append(enhanced_img)

            if num_augs_per_face == 0:
                num_augs_per_face = len(augs_for_this_face)

            for aug in augs_for_this_face:
                all_augmented_tensors.append(basic_transform(aug))

        img_tensors_batch = torch.stack(all_augmented_tensors)
        if onnx_session:
            ort_inputs = {onnx_session.get_inputs()[0].name: img_tensors_batch.cpu().numpy()}
            all_feats_np = onnx_session.run(None, ort_inputs)[0]
        else:
            img_tensors_batch = img_tensors_batch.to(device)
            with torch.no_grad():
                all_feats_tensor = model(img_tensors_batch)
            all_feats_tensor = F.normalize(all_feats_tensor, p=2, dim=1)
            all_feats_np = all_feats_tensor.cpu().numpy()

        faiss_index = self.faiss_index
        if faiss_index is None:
            self.logger.warning("Faiss index is not available. Recognition skipped.")
            empty_result = {
                "top_results": [("未知角色", -1.0)], "expanded_box": [0,0,0,0],
                "refined_face_for_rec": Image.new("RGB", (112, 112)), "original_crop_pil": Image.new("RGB", (1, 1)),
                "resolution": (0, 0),
            }
            return [empty_result] * len(boxes)

        faiss_index_to_label = self.faiss_index_to_label
        idx_to_class = self.idx_to_class

        try:
            all_scores, all_indices = faiss_index.search(all_feats_np.astype('float32'), k)
        except Exception as e:
            self.logger.error(f"Error during Faiss batch search: {e}", exc_info=True)
            empty_result = {
                "top_results": [("未知角色", -1.0)], "expanded_box": [0,0,0,0],
                "refined_face_for_rec": Image.new("RGB", (112, 112)), "original_crop_pil": Image.new("RGB", (1, 1)),
                "resolution": (0, 0),
            }
            return [empty_result] * len(boxes)

        num_faces = len(boxes)
        all_scores = all_scores.reshape(num_faces, num_augs_per_face, k)
        all_indices = all_indices.reshape(num_faces, num_augs_per_face, k)

        all_faces_results = []
        for i in range(num_faces):
            character_scores = {}
            face_scores = all_scores[i]
            face_indices = all_indices[i]

            for aug_idx in range(num_augs_per_face):
                for res_idx in range(k):
                    if face_indices[aug_idx, res_idx] == -1:
                        continue
                    
                    label_int = faiss_index_to_label[face_indices[aug_idx, res_idx]]
                    score = face_scores[aug_idx, res_idx]
                    label_str = str(label_int)
                    
                    if label_str not in character_scores or score > character_scores[label_str]:
                        character_scores[label_str] = score
            
            sorted_scores = sorted(
                character_scores.items(), key=lambda item: item[1], reverse=True
            )

            top_k_results = []
            for label, score in sorted_scores[:k]:
                identity = idx_to_class.get(int(label))
                if identity:
                    top_k_results.append((identity, float(score)))

            if not top_k_results or top_k_results[0][1] < recognition_threshold:
                if top_k_results:
                    final_results = [("未知角色", top_k_results[0][1])] + top_k_results
                else:
                    final_results = [("未知角色", -1.0)]
            else:
                final_results = top_k_results
            all_faces_results.append(final_results)

        recognition_datas = []
        for i in range(num_faces):
            recognition_data = {
                "top_results": all_faces_results[i], "expanded_box": expanded_boxes[i],
                "refined_face_for_rec": face_images_for_rec[i], "original_crop_pil": original_crops[i],
                "resolution": original_crops[i].size,
            }
            recognition_datas.append(recognition_data)

        return recognition_datas

    def detect_and_recognize(
        self,
        image_pil: Image.Image,
        include_images: bool = True,
        anime_only: bool = False,
        use_correction_override: bool = None,
        detection_results: tuple = None,
    ):
        """
        识别调度器：根据配置选择使用哪个版本的识别函数。
        """
        if anime_only:
            self.logger.info("正在进行主导动漫分析...")
            return self.detect_and_recognize_v2(
                image_pil,
                include_images=False,
                anime_only=True,
                detection_results=detection_results,
            )

        # 确定修正模式：API传入的 use_correction_override 参数具有最高优先级。
        if use_correction_override is not None:
            # 如果API明确指定了模式，则使用它
            use_correction = use_correction_override
            self.logger.info(f"API 强制设定修正模式为: {'开启' if use_correction else '关闭'}")
        else:
            # 否则，回退到全局配置文件中的设置
            use_correction = self.config.get("use_recognition_correction", True)

        if use_correction:
            self.logger.info("正在使用 V2 (全局修正) 识别算法...")
            return self.detect_and_recognize_v2(
                image_pil, include_images, detection_results=detection_results
            )
        else:
            self.logger.info("正在使用 V1 (经典) 识别算法...")
            return self.detect_and_recognize_v1(
                image_pil, include_images, detection_results=detection_results
            )


    def detect_and_recognize_v1(
        self,
        image_pil: Image.Image, 
        include_images: bool = True,
        detection_results: tuple = None,
    ):
        """
        V1: 经典识别流程。
        """
        preliminary_results = self._run_detection_and_batch_recognition(image_pil, detection_results)
        
        processed_faces = self._process_initial_results(preliminary_results)
        if not processed_faces:
            return []

        faces_info = []
        for face_data in processed_faces:
            final_info = self._assemble_final_face_info(face_data, include_images)
            final_info.pop("is_corrected", None)  # V1没有修正概念
            faces_info.append(final_info)

        faces_info.sort(key=lambda x: x["score"], reverse=True)
        return faces_info


    def detect_and_recognize_v2(
        self,
        image_pil: Image.Image,
        include_images: bool = True,
        anime_only: bool = False,
        detection_results: tuple = None,
    ):
        """
        V2: 全局修正识别流程。
        """
        preliminary_results = self._run_detection_and_batch_recognition(image_pil, detection_results)
        
        initial_faces_info = self._process_initial_results(preliminary_results)
        if not initial_faces_info:
            return {} if anime_only else []

        all_anime_candidates = []
        for face in initial_faces_info:
            for res in face["top_k"]:
                if res["identity"] != "未知角色" and res["anime"] != "未知作品":
                    all_anime_candidates.append(res["anime"])

        if anime_only:
            anime_scores_map = {}
            BACK_FACE_CLASS_INDEX = 1

            for face in initial_faces_info:
                is_back_face = face.get("first_class") == BACK_FACE_CLASS_INDEX
                for res in face["top_k"]:
                    if res["identity"] != "未知角色" and res["anime"] != "未知作品":
                        anime = res["anime"]
                        score = res["score"]
                        if is_back_face:
                            score = max(0.1, score - 0.2)

                        if anime not in anime_scores_map:
                            anime_scores_map[anime] = []
                        anime_scores_map[anime].append(score)

            if not anime_scores_map:
                return {}
            
            sorted_animes = sorted(
                anime_scores_map.items(), key=lambda item: len(item[1]), reverse=True
            )

            top_animes = sorted_animes[:2]
            anime_results = []
            for anime, scores in top_animes:
                scores.sort(reverse=True)
                anime_results.append(
                    {
                        "anime": anime, "count": len(scores),
                        "scores": ", ".join([f"{s:.3f}" for s in scores]),
                    }
                )
            return anime_results

        dominant_anime = None
        if all_anime_candidates:
            anime_counts = Counter(all_anime_candidates)
            top_animes = anime_counts.most_common(2)
            self.logger.info(top_animes)

            if len(top_animes) == 1 or (
                len(top_animes) > 1 and top_animes[0][1] > top_animes[1][1]
            ):
                if top_animes[0][0] != "未知作品":
                    dominant_anime = top_animes[0][0]

        if dominant_anime:
            self.logger.info(f"全局分析完成，主导动漫为: {dominant_anime}")
            correction_override_threshold = self.config.get(
                "correction_override_threshold", 0.85
            )

            for face in initial_faces_info:
                face["is_corrected"] = False
                if not face["top_k"]:
                    continue
                current_top_result = face["top_k"][0]
                if current_top_result["identity"] == "未知角色":
                    pass
                elif current_top_result["score"] >= correction_override_threshold:
                    self.logger.info(
                        f"识别结果 '{current_top_result['name']}' 置信度({current_top_result['score']:.3f}) >= 阈值({correction_override_threshold})，跳过全局修正。"
                    )
                    continue

                current_top_anime = current_top_result["anime"]
                if current_top_anime != dominant_anime:
                    best_alternative = None
                    for candidate in face["top_k"]:
                        if candidate["anime"] == dominant_anime:
                            if (
                                best_alternative is None
                                or candidate["score"] > best_alternative["score"]
                            ):
                                best_alternative = candidate

                    if best_alternative:
                        self.logger.info(
                            f"修正: 将角色 {face['top_k'][0]['name']} 修正为 {best_alternative['name']} (分数: {best_alternative['score']})"
                        )
                        face["top_k"].sort(
                            key=lambda x: x["identity"] == best_alternative["identity"],
                            reverse=True,
                        )
                        face["is_corrected"] = True

        faces_info = []
        for face_data in initial_faces_info:
            final_info = self._assemble_final_face_info(face_data, include_images)
            faces_info.append(final_info)

        faces_info.sort(key=lambda x: x["score"], reverse=True)
        return faces_info

