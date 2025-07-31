import os
import json
from PIL import Image
from typing import Dict, List, Generator, Any


class AutoTestService:
    def __init__(self, recognition_service, logger):
        self.recognition_service = recognition_service
        self.logger = logger
        self.LOW_RES_THRESHOLD = 80 * 80

    def get_test_info(self) -> Dict[str, int]:
        """获取测试目录的基本信息"""
        test_dir = os.path.join("data", "test")
        image_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        files = (
            [f for f in os.listdir(test_dir) if f.lower().endswith(image_exts)]
            if os.path.exists(test_dir)
            else []
        )
        anime_map = {}
        for fname in files:
            if "@" in fname:
                anime = fname.split("@")[0]
                anime_map.setdefault(anime, []).append(fname)
        return {"total_files": len(files), "total_animes": len(anime_map)}

    def _parse_filename_gt(self, fname: str) -> int:
        """从文件名解析真实标签数量"""
        try:
            gt_count = int(fname.split("@")[-1].split(".")[0])
            return gt_count
        except (ValueError, IndexError):
            return 0

    def _get_image_files(self) -> List[str]:
        """获取测试目录下的所有图片文件"""
        test_dir = os.path.join("data", "test")
        if not os.path.exists(test_dir):
            raise FileNotFoundError("测试目录不存在")

        image_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        files = [f for f in os.listdir(test_dir) if f.lower().endswith(image_exts)]
        
        if len(files) == 0:
            raise ValueError("测试目录下没有图片")
        
        return files

    def _group_files_by_anime(self, files: List[str]) -> Dict[str, List[str]]:
        """按动漫名分组文件"""
        anime_map = {}
        for fname in files:
            if "@" in fname:
                anime = fname.split("@")[0]
                anime_map.setdefault(anime, []).append(fname)
        return anime_map

    def _initialize_result_structure(self, anime_map: Dict[str, List[str]]) -> Dict[str, Any]:
        """初始化结果数据结构"""
        result = {
            "total_files": sum(len(files) for files in anime_map.values()),
            "animes": {},
            "progress": 0,
            "done": False,
            "overall": {
                # Precision-based stats
                "precision_total_detected": 0,
                "precision_total_correct": 0,
                "low_res_detected": 0,
                "low_res_correct": 0,
                "normal_res_detected": 0,
                "normal_res_correct": 0,
                # Recall-based stats (based on filename GT)
                "recall_total_gt": 0,
                "recall_total_correct": 0,
            },
        }

        # 初始化每个动漫的统计数据
        for anime in anime_map.keys():
            result["animes"][anime] = {
                "files": len(anime_map[anime]),
                "gt_total": 0,  # from filename
                "correct": 0,  # total correct recognitions
                "low_res_detected": 0,
                "low_res_correct": 0,
                "normal_res_detected": 0,
                "normal_res_correct": 0,
            }

        return result

    def _process_single_file(self, test_dir: str, fname: str, anime: str) -> Dict[str, Any]:
        """处理单个文件并返回统计结果"""
        path = os.path.join(test_dir, fname)
        gt_count = self._parse_filename_gt(fname)
        
        file_stats = {
            "gt_count": gt_count,
            "correct": 0,
            "low_res_detected": 0,
            "low_res_correct": 0,
            "normal_res_detected": 0,
            "normal_res_correct": 0,
        }

        try:
            img = Image.open(path).convert("RGB")
            faces = self.recognition_service.recognize_image(img, include_images=False)

            for face in faces:
                width, height = face.get("resolution", (0, 0))
                is_low_res = (width * height) < self.LOW_RES_THRESHOLD
                is_correct = face.get("anime") == anime

                if is_low_res:
                    file_stats["low_res_detected"] += 1
                    if is_correct:
                        file_stats["low_res_correct"] += 1
                else:
                    file_stats["normal_res_detected"] += 1
                    if is_correct:
                        file_stats["normal_res_correct"] += 1

                if is_correct:
                    file_stats["correct"] += 1

        except Exception as e:
            self.logger.error(f"处理文件 {fname} 时出错: {e}")

        return file_stats

    def _update_anime_stats(self, anime_stats: Dict[str, Any], file_stats: Dict[str, Any]):
        """更新动漫统计数据"""
        anime_stats["gt_total"] += file_stats["gt_count"]
        anime_stats["correct"] += file_stats["correct"]
        anime_stats["low_res_detected"] += file_stats["low_res_detected"]
        anime_stats["low_res_correct"] += file_stats["low_res_correct"]
        anime_stats["normal_res_detected"] += file_stats["normal_res_detected"]
        anime_stats["normal_res_correct"] += file_stats["normal_res_correct"]

    def _calculate_anime_accuracy(self, anime_stats: Dict[str, Any]):
        """计算动漫的准确率"""
        anime_stats["recall_accuracy"] = (
            round(anime_stats["correct"] / anime_stats["gt_total"] * 100, 2)
            if anime_stats["gt_total"] > 0
            else 0
        )
        anime_stats["low_res_precision"] = (
            round(
                anime_stats["low_res_correct"]
                / anime_stats["low_res_detected"]
                * 100,
                2,
            )
            if anime_stats["low_res_detected"] > 0
            else 0
        )
        anime_stats["normal_res_precision"] = (
            round(
                anime_stats["normal_res_correct"]
                / anime_stats["normal_res_detected"]
                * 100,
                2,
            )
            if anime_stats["normal_res_detected"] > 0
            else 0
        )

    def _update_overall_stats(self, result: Dict[str, Any]):
        """更新全局统计数据"""
        # For Precision cards
        sum_low_res_detected = sum(
            d.get("low_res_detected", 0) for d in result["animes"].values()
        )
        sum_low_res_correct = sum(
            d.get("low_res_correct", 0) for d in result["animes"].values()
        )
        sum_normal_res_detected = sum(
            d.get("normal_res_detected", 0) for d in result["animes"].values()
        )
        sum_normal_res_correct = sum(
            d.get("normal_res_correct", 0) for d in result["animes"].values()
        )

        result["overall"]["precision_total_detected"] = (
            sum_low_res_detected + sum_normal_res_detected
        )
        result["overall"]["precision_total_correct"] = (
            sum_low_res_correct + sum_normal_res_correct
        )
        result["overall"]["low_res_detected"] = sum_low_res_detected
        result["overall"]["low_res_correct"] = sum_low_res_correct
        result["overall"]["normal_res_detected"] = sum_normal_res_detected
        result["overall"]["normal_res_correct"] = sum_normal_res_correct

        # For Recall card (the one the user asked for)
        sum_gt_from_filenames = sum(
            d.get("gt_total", 0) for d in result["animes"].values()
        )
        sum_correct_for_recall = sum(
            d.get("correct", 0) for d in result["animes"].values()
        )
        result["overall"]["recall_total_gt"] = sum_gt_from_filenames
        result["overall"]["recall_total_correct"] = sum_correct_for_recall

    def run_auto_test(self) -> Generator[str, None, None]:
        """运行自动测试并返回流式结果"""
        try:
            # 获取文件列表
            files = self._get_image_files()
            anime_map = self._group_files_by_anime(files)
            result = self._initialize_result_structure(anime_map)
            
            test_dir = os.path.join("data", "test")
            tested_files = 0
            total_files = result["total_files"]

            # 处理每个动漫的每个文件
            for anime, anime_files in anime_map.items():
                for fname in anime_files:
                    # 处理单个文件
                    file_stats = self._process_single_file(test_dir, fname, anime)
                    
                    # 更新动漫统计数据
                    self._update_anime_stats(result["animes"][anime], file_stats)
                    
                    # 计算动漫准确率
                    self._calculate_anime_accuracy(result["animes"][anime])
                    
                    # 更新全局统计数据
                    self._update_overall_stats(result)
                    
                    tested_files += 1
                    result["progress"] = round(tested_files / total_files * 100, 2)
                    
                    # 返回当前进度
                    yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

            # 完成测试
            result["done"] = True
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

        except (FileNotFoundError, ValueError) as e:
            error_result = {"error": str(e)}
            yield f"data: {json.dumps(error_result, ensure_ascii=False)}\n\n"
        except Exception as e:
            self.logger.error(f"自动测试过程中发生错误: {e}", exc_info=True)
            error_result = {"error": f"自动测试失败: {str(e)}"}
            yield f"data: {json.dumps(error_result, ensure_ascii=False)}\n\n" 