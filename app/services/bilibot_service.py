import threading
import time
import re
import os
import json
import requests
import io
import av
import random
import asyncio
from av.error import FFmpegError
from collections import defaultdict
from flask import current_app

import base64

from bilibili_api import user, session, video, comment, Credential, login_v2


# --- 全局变量和锁 ---
# bot_instance 和 bot_lock 不再需要，因为实例由 app 管理
# qrcode_sessions 将被移入服务内部

class BilibotService:
    def __init__(self, config: dict, logger, recognition_service):
        """
        服务初始化.
        :param config: 包含 Bilibot 配置的字典.
        :param logger: 应用的 logger 实例.
        :param recognition_service: 识别服务实例.
        """
        self.logger = logger
        self.recognition_service = recognition_service
        self.credential = None
        self.user_info = None
        self.my_uid = None
        self.my_name = None
        
        self.running = False
        self.logged_in = False
        self._stop_event = threading.Event()
        self._thread = None
        self.qrcode_sessions = {} # 在服务内部管理二维码会话

        self.processed_ids_file = os.path.join("config", "processed_ids.json")
        self.processed_ids = self._load_processed_ids()

        # 从传入的配置初始化
        self.config = {} # 先初始化为空字典
        self.update_config(config)


    def update_config(self, new_config):
        """热更新配置"""
        self.config = new_config
        self.polling_interval_base = self.config.get('polling_interval_base', 15)
        self.polling_interval_jitter = self.config.get('polling_interval_jitter', 10)
        self.trigger_keyword = self.config.get('trigger_keyword', '识别动漫')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.55)
        self.reply_templates = self.config.get('reply_templates', ["模板错误"])
        self.random_embellishments = self.config.get('random_embellishments', [])
        self.use_obfuscation = self.config.get('use_obfuscation', True)
        self.logger.info(f"[BiliBot] 配置已更新: 轮询间隔 = {self.polling_interval_base}s ± {self.polling_interval_jitter}s, 触发关键词 = '{self.trigger_keyword}', 置信度阈值 = {self.confidence_threshold}")

    def get_status(self):
        """获取机器人当前状态"""
        return {
            "logged_in": self.logged_in,
            "running": self.running,
            "user_info": self.user_info,
            "config": self.config
        }

    def start(self):
        """启动机器人后台轮询线程"""
        if not self.logged_in:
            self.logger.warning("[BiliBot] 无法启动：尚未登录。")
            return
        if self.running:
            self.logger.warning("[BiliBot] 机器人已在运行中。")
            return
        
        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._main_loop_wrapper, daemon=True)
        self._thread.start()
        self.logger.info("[BiliBot] 机器人后台服务已启动。")

    def stop(self):
        """停止机器人后台轮询线程"""
        if not self.running:
            self.logger.warning("[BiliBot] 机器人尚未运行。")
            return
        
        self._stop_event.set()
        if self._thread:
            self._thread.join() # 等待线程结束
        self.running = False
        self.logger.info("[BiliBot] 机器人后台服务已停止。")

    def login_with_credential(self, credential: Credential):
        """使用凭据对象登录并初始化服务"""
        try:
            self.credential = credential
            # 验证凭据有效性
            user_info_data = asyncio.run(user.get_self_info(credential=self.credential))
            
            if not user_info_data or 'mid' not in user_info_data:
                self.logged_in = False
                self.logger.error("[BiliBot] 使用凭据登录失败，无法获取用户信息。")
                return False

            self.user_info = {
                "name": user_info_data['name'],
                "face": user_info_data['face'],
                "mid": user_info_data['mid']
            }
            self.my_uid = user_info_data['mid']
            self.my_name = user_info_data['name']
            self.logged_in = True
            self.logger.info(f"[BiliBot] 登录成功，账号: {self.my_name} (UID: {self.my_uid})")
            return True
        except Exception as e:
            self.logger.error(f"[BiliBot] 凭据登录时发生异常: {e}")
            self.logged_in = False
            return False

    def login_with_cookies(self, cookies):
        """通过Cookie字典创建凭据并登录"""
        credential = Credential(
            sessdata=cookies.get("SESSDATA"),
            bili_jct=cookies.get("bili_jct"),
            buvid3=cookies.get("buvid3"),
            dedeuserid=cookies.get("DedeUserID"),
        )
        return self.login_with_credential(credential)

    def logout(self):
        """登出并清理机器人状态"""
        if self.running:
            self.stop()
        
        self.credential = None
        self.logged_in = False
        self.user_info = None
        self.my_uid = None
        self.my_name = None
        self.logger.info("[BiliBot] 服务已登出，内部状态已清理。")

    def generate_login_qrcode(self, session_id: str):
        """生成登录二维码并返回 Base64 编码的图片字符串"""
        qr_login = login_v2.QrCodeLogin()
        self.qrcode_sessions[session_id] = qr_login

        asyncio.run(qr_login.generate_qrcode())
        
        img = qr_login.get_qrcode_picture()
        buffered = io.BytesIO(img.content)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_str

    def check_login_qrcode_state(self, session_id: str):
        """
        检查二维码登录状态.
        返回一个字典，包含:
        - success (bool): 操作是否成功 (登录成功被视为 success: True)
        - code (int): 状态码 (0: 成功, -1: 会话无效, -2: 等待扫码, -3: 二维码超时, -4: 凭据登录失败, -99: 未知错误)
        - message (str): 状态信息
        - user_info (dict, optional): 成功时返回用户信息
        """
        if session_id not in self.qrcode_sessions:
            return {"success": False, "code": -1, "message": "会话已过期，请刷新二维码"}

        qr_login = self.qrcode_sessions[session_id]

        try:
            state = asyncio.run(qr_login.check_state())

            if state == login_v2.QrCodeLoginEvents.DONE:
                credential = qr_login.get_credential()
                login_ok = self.login_with_credential(credential)

                if login_ok:
                    # 登录成功后，清理会话
                    del self.qrcode_sessions[session_id]
                    return {
                        "success": True, 
                        "code": 0, 
                        "message": "登录成功！", 
                        "user_info": self.user_info,
                        "credential": { # 返回凭据信息以便控制器保存
                            "SESSDATA": credential.sessdata,
                            "bili_jct": credential.bili_jct,
                            "buvid3": credential.buvid3,
                            "DedeUserID": credential.dedeuserid
                        }
                    }
                else:
                    del self.qrcode_sessions[session_id]
                    return {"success": False, "code": -4, "message": "通过凭据登录Bilibot服务失败"}

            elif state == login_v2.QrCodeLoginEvents.TIMEOUT:
                del self.qrcode_sessions[session_id]
                return {"success": False, "code": -3, "message": "二维码已过期"}
            else:
                return {"success": False, "code": -2, "message": "等待扫码或确认中..."}
                
        except Exception as e:
            self.logger.error(f"检查二维码状态时发生未知错误: {e}")
            if session_id in self.qrcode_sessions:
                del self.qrcode_sessions[session_id]
            return {"success": False, "code": -99, "message": f"未知错误: {e}"}


    def _load_processed_ids(self):
        """从文件加载已处理过的动态/评论ID"""
        if os.path.exists(self.processed_ids_file):
            try:
                with open(self.processed_ids_file, 'r') as f:
                    return set(json.load(f))
            except (json.JSONDecodeError, TypeError):
                self.logger.warning(f"[BiliBot] {self.processed_ids_file} 文件损坏，已重置。")
        return set()

    def _save_processed_ids(self):
        with open(self.processed_ids_file, 'w') as f:
            json.dump(list(self.processed_ids), f)

    def _main_loop_wrapper(self):
        """包装异步主循环，以便在同步的线程中运行"""
        try:
            asyncio.run(self._main_loop())
        except Exception as e:
            print(f"Error in Bilibot main loop: {e}")
            self.running = False


    async def _download_video_from_url(self, video_obj: video.Video, bvid: str, cid: int):
        """使用bilibili-api-python下载视频流"""
        try:
            # 1. 获取下载信息, 传入 cid
            download_info = await video_obj.get_download_url(cid=cid)
            # 2. 使用解析器
            detector = video.VideoDownloadURLDataDetecter(data=download_info)
            # 3. 获取最佳视频流, 设置最高画质为 1080P
            streams = detector.detect_best_streams(video_max_quality=video.VideoQuality._1080P)
            
            video_url = None
            if streams:
                # 对于FLV/MP4流，列表只有一个元素。对于DASH，第一个通常是视频。
                video_url = streams[0].url

            if not video_url:
                self.logger.error(f"[BiliBot] 无法从视频 {bvid} 中解析出下载链接。")
                return None
            
            headers = {
                'Referer': f'https://www.bilibili.com/video/{bvid}',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(video_url, stream=True, headers=headers)
            response.raise_for_status()
            
            video_stream = io.BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                video_stream.write(chunk)
            video_stream.seek(0)
            return video_stream
        except Exception as e:
            self.logger.error(f"[BiliBot] 下载视频 BVID:{bvid} 失败: {e}")
            return None

    def _extract_frames_from_video(self, video_stream, bvid):
        """从内存中的视频流中提取帧"""
        screenshots = []
        try:
            with av.open(video_stream) as container:
                stream = container.streams.video[0]
                stream.codec_context.skip_frame = 'NONREF'
                
                total_duration_sec = float(stream.duration * stream.time_base)
                if total_duration_sec == 0:
                    self.logger.warning(f"[BiliBot] 视频 {bvid} 时长为0，跳过处理。")
                    return [], 0

                if total_duration_sec <= 60:
                    interval_sec = 2
                elif total_duration_sec <= 300:
                    interval_sec = 5
                else:
                    interval_sec = 5
                
                self.logger.info(f"[BiliBot] 视频 {bvid} (时长 {total_duration_sec:.2f}s), 每 {interval_sec}s 采样一帧。")
                
                for frame in container.decode(stream):
                    if frame.time >= len(screenshots) * interval_sec:
                        img = frame.to_image()
                        screenshots.append({'timestamp': frame.time, 'image': img})
                
                return screenshots, interval_sec

        except FFmpegError as e:
            self.logger.error(f"[BiliBot] FFmpeg处理视频 {bvid} 时出错: {e}")
            return [], 0
        except Exception as e:
            self.logger.error(f"[BiliBot] 提取帧时发生未知错误: {e}")
            return [], 0


    def _recognize_and_generate_timeline(self, screenshots_data, interval_sec):
        if not screenshots_data:
            return "抱歉，未能截取到任何有效视频帧。", []

        all_recognition_results = []
        self.logger.info(f"[BiliBot] 开始批量识别 {len(screenshots_data)} 张截图...")
        for shot in screenshots_data:
            timestamp = shot['timestamp']
            img = shot['image']
            try:
                # 调用注入的识别服务，而不是直接调用函数
                results_list = self.recognition_service.recognize_image(img, include_images=False, anime_only=True)
                if results_list:
                    all_recognition_results.append((timestamp, results_list))
            except Exception as e:
                self.logger.error(f"[BiliBot] 识别时间戳 {timestamp} 的图像时出错: {e}")
        
        self.logger.info(f"[BiliBot] 批量识别完成，现在开始聚合数据...")

        anime_occurrences = defaultdict(list)
        for timestamp, results_list in all_recognition_results:
            if results_list and isinstance(results_list, list) and len(results_list) > 0:
                top_result = results_list[0]
                anime_name = top_result.get("anime")
                scores_str = top_result.get("scores")
                count = top_result.get("count", 0)
                
                if anime_name and anime_name != "未知作品" and scores_str:
                    try:
                        scores = [float(s.strip()) for s in scores_str.split(',')]
                        if scores:
                            max_score = max(scores)
                            if max_score >= self.confidence_threshold:
                                anime_occurrences[anime_name].append({
                                    'timestamp': timestamp,
                                    'count': count
                                })
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"[BiliBot] 解析分数时出错: '{scores_str}', 错误: {e}")

        anime_timestamps = defaultdict(list)
        for anime, occurrences in anime_occurrences.items():
            if len(occurrences) == 1 and occurrences[0]['count'] == 1:
                self.logger.info(f"[BiliBot] 过滤掉单次低置信度结果: {anime}")
                continue
            for occ in occurrences:
                anime_timestamps[anime].append(occ['timestamp'])

        final_segments = []
        for anime, timestamps in anime_timestamps.items():
            if timestamps:
                start_time = min(timestamps)
                end_time = max(timestamps)
                final_segments.append({
                    "anime": anime,
                    "start": start_time,
                    "end": end_time + 1 
                })

        final_segments.sort(key=lambda x: x['start'])

        filtered_segments = []
        for segment_to_check in final_segments:
            is_subset = False
            for other_segment in final_segments:
                if segment_to_check is other_segment:
                    continue
                if (other_segment['start'] <= segment_to_check['start'] and
                    other_segment['end'] >= segment_to_check['end']):
                    if (other_segment['start'] < segment_to_check['start'] or
                        other_segment['end'] > segment_to_check['end']):
                        is_subset = True
                        self.logger.info(f"[BiliBot] 过滤子集: 《{segment_to_check['anime']}》({segment_to_check['start']}-{segment_to_check['end']}) 被 《{other_segment['anime']}》({other_segment['start']}-{segment_to_check['end']}) 包含。")
                        break
            if not is_subset:
                filtered_segments.append(segment_to_check)
        
        if not filtered_segments:
            self.logger.info(f"[BiliBot] Timeline generation failed. Timestamps found: {json.dumps(dict(anime_timestamps))}")
            return "抱歉，未能识别出任何时长超过一帧的有效动漫片段。", []

        timeline_parts = []
        for i, segment in enumerate(filtered_segments):
            start_str = time.strftime('%M:%S', time.gmtime(segment['start']))
            end_str = time.strftime('%M:%S', time.gmtime(segment['end']))
            line = f"{i+1}. 《{segment['anime']}》: {start_str} ~ {end_str}"
            timeline_parts.append(line)
        
        # The return signature is (str, list), so we return an empty list for characters
        return "\n".join(timeline_parts), filtered_segments


    async def _handle_at_message(self, at_msg):
        if at_msg['item']['type'] not in ['reply', 'comment']:
            return

        item_id = at_msg['id']
        if item_id in self.processed_ids:
            return
        
        user_info = at_msg['user']
        content = at_msg['item']['source_content']

        if self.trigger_keyword not in content:
            return

        self.logger.info(f"[BiliBot] 检测到来自 {user_info['nickname']} 的召唤 (item_id: {item_id})")

        bvid_match = re.search(r'BV[1-9A-HJ-NP-Za-km-z]{10}', at_msg['item']['uri'])
        if not bvid_match:
            self.logger.warning(f"[BiliBot] 无法从at消息中解析出BVID: {at_msg['item']['uri']}")
            return
        bvid = bvid_match.group(0)

        try:
            v = video.Video(bvid=bvid, credential=self.credential)
            video_info_data = await v.get_info()
            video_title = video_info_data['title']
            cid = video_info_data['cid']

            video_stream = await self._download_video_from_url(v, bvid, cid)
            if not video_stream: return
            
            screenshots, interval_sec = self._extract_frames_from_video(video_stream, bvid)
            
            final_timeline, _ = self._recognize_and_generate_timeline(screenshots, interval_sec)
            
            if final_timeline.startswith("抱歉"):
                reply_text = f"您好，视频《{video_title or '未知'}》分析完成。\n{final_timeline}"
            else:
                template = random.choice(self.reply_templates)
                embellishment = random.choice(self.random_embellishments) if self.random_embellishments else ""
                reply_text = template.format(
                    video_title=video_title or "当前视频",
                    timeline=final_timeline,
                    nickname=user_info['nickname']
                ) + " " + embellishment

            if self.use_obfuscation:
                delay = random.uniform(3, 12)
                self.logger.info(f"[BiliBot] 准备回复，将随机延迟 {delay:.2f} 秒...")
                await asyncio.sleep(delay)

            root_rpid = at_msg['item']['root_id']
            parent_rpid = at_msg['item']['source_id']
            
            # 发送评论并检查结果
            comment_result = await comment.send_comment(
                text=f"@{user_info['nickname']} {reply_text}",
                oid=video_info_data['aid'],
                type_=comment.CommentResourceType.VIDEO,
                root=root_rpid,
                parent=parent_rpid,
                credential=self.credential
            )
            # 成功的回复会包含 rpid (reply id)，以此为准
            print(comment_result)
            if comment_result and comment_result.get('rpid'):
                self.logger.info(f"[BilibiliBot] 成功回复 {user_info['nickname']} 的评论。")
            else:
                self.logger.error(f"[BilibiliBot] 回复评论失败，返回结果: {comment_result}")

            # 发送私信并检查结果
            pm_result = await session.send_msg(
                receiver_id=user_info['mid'],
                msg_type=session.EventType.TEXT,
                content=reply_text,
                credential=self.credential
            )
            # 根据观察，私信成功时 code 也为 0
            if pm_result and pm_result.get('code') == 0:
                self.logger.info(f"[BilibiliBot] 成功向 {user_info['nickname']} 发送私信。")
            else:
                self.logger.error(f"[BilibiliBot] 发送私信失败，返回结果: {pm_result}")

            self.processed_ids.add(item_id)
            self._save_processed_ids()
            
        except Exception as e:
            self.logger.error(f"[BiliBot] 处理at消息时发生严重错误: {e}", exc_info=True)


    async def _main_loop(self):
        self.logger.info("[BiliBot] 主循环已启动。")

        while not self._stop_event.is_set():
            try:
                self.logger.debug(f"[BiliBot_DIAG] 开始新一轮轮询...")
                at_messages = await session.get_at(credential=self.credential)

                #print(at_messages)
                
                if at_messages and at_messages.get('items'):
                    for msg in at_messages['items']:
                        await self._handle_at_message(msg)
                
                self._save_processed_ids()
                
                # --- 使用随机化间隔 ---
                jitter = random.uniform(-self.polling_interval_jitter, self.polling_interval_jitter)
                wait_time = max(5, self.polling_interval_base + jitter) # 确保等待时间不少于5秒
                
                self.logger.debug(f"[BiliBot] 轮询结束，将等待 {wait_time:.2f} 秒...")

            except Exception as e:
                self.logger.error(f"[BiliBot] 轮询时发生未知严重错误: {e}", exc_info=True)
                
                # 即使出错也等待，避免快速失败循环
                wait_time = self.polling_interval_base 
            
            await asyncio.sleep(wait_time)
        
        self.logger.info("[BiliBot] 主循环已正常退出。")

def get_bot_instance():
    """从 Flask app context 获取 BilibotService 单例"""
    if not hasattr(current_app, 'bilibot_service'):
        raise RuntimeError("BilibotService 未在 Flask app 中初始化。")
    return current_app.bilibot_service