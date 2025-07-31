import os
import json


# --------------------------- 全局配置 --------------------------- #
CONFIG_PATH = "config/config.json"
USERS_DB_PATH = "config/users.json"
STATS_FILE = 'config/stats.json'
DEFAULT_CONFIG = {  
    "project_name": "动漫角色识别工具",
    "secret_key": "",
    "recognition_threshold": 0.5,
    "correction_override_threshold": 0.85,  # 新增：高置信度修正豁免阈值
    "max_faces": 20,
    "use_recognition_correction": True, # 新增：是否启用全局修正算法
    "enable_upload_compression": True,
    "recognition_backend": "pytorch",  # 新增：识别后端，可选 "pytorch" 或 "onnx"
    "glass_opacity": 0.50,
    "animation_speed_random": 500,  # 新增：随机扫描动画速度（毫秒）
    "animation_speed_target": 800,  # 新增：锁定目标后扫描速度（毫秒）
    "background": {
        "type": "image",
        "image": "static/backgrounds/default.png",
        "color": "#f0f2f5",
        "mode": "fixed",
        "uploaded_images": ["static/backgrounds/default.png"]
    }
}


# --------------------------- B站机器人配置 --------------------------- #
BILIBOT_CONFIG_PATH = "config/bilibot_config.json"
DEFAULT_BILIBOT_CONFIG = {
    "enabled": False,
    "polling_interval_base": 15,
    "polling_interval_jitter": 10,
    "trigger_keyword": "识别动漫",
    "confidence_threshold": 0.55,
    "reply_templates": [
        "您要找的是不是：\n【《{video_title}》】\n{timeline}\n\n召唤AI不易，一键三连支持下up主吧~[打call]",
        "来啦来啦！本次识别结果如下：\n【视频：《{video_title}》】\n{timeline}\n\n内容由 {nickname} 召唤，觉得有用的话给up主和我都点个赞吧！[喜欢]",
        "滴！识别卡！\n【《{video_title}》】\n{timeline}\n\n结果准确的话，不妨给up主一个三连哦！[星星眼]",
        "报告！在《{video_title}》中发现目标：\n{timeline}\n\n本次任务由 {nickname} 指派。",
        "分析完成，结果如下。\n作品：《{video_title}》\n时间轴：\n{timeline}",
        "AI为您服务~\n视频《{video_title}》的识别结果是：\n{timeline}",
        "Hi, {nickname}！这是你要的结果：\n{timeline}\n\n出自视频：《{video_title}》",
        "识别到啦！\n{timeline}\n\n视频是《{video_title}》，感谢 {nickname} 的召唤！"
    ],
    "random_embellishments": [
        "[doge]", "[喜欢]", "[脱单doge]", "[OK]", "[思考]", "[大笑]", "[偷笑]",
        "w(ﾟДﾟ)w", "!", "...", "呀~", "awa", "(｡･ω･｡)ﾉ♡", "ovo", "~~"
    ],
    "use_obfuscation": True
}


def load_bilibot_config():
    """加载B站机器人配置文件，如果不存在则创建"""
    if not os.path.exists(BILIBOT_CONFIG_PATH):
        with open(BILIBOT_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_BILIBOT_CONFIG, f, ensure_ascii=False, indent=4)
        return DEFAULT_BILIBOT_CONFIG
    
    try:
        with open(BILIBOT_CONFIG_PATH, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
            config = DEFAULT_BILIBOT_CONFIG.copy()
            config.update(user_config)
            # --- 向后兼容旧的 polling_interval ---
            if 'polling_interval' in config and 'polling_interval_base' not in user_config:
                config['polling_interval_base'] = config['polling_interval']
            # 兼容旧版单模板配置
            if 'reply_template' in config and isinstance(config['reply_template'], str):
                config['reply_templates'] = [config['reply_template']]
                del config['reply_template']
            return config
    except (json.JSONDecodeError, IOError):
        return DEFAULT_BILIBOT_CONFIG

def save_bilibot_config(config_data):
    """保存B站机器人配置到JSON文件"""
    with open(BILIBOT_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4)


def load_config():
    """加载JSON配置文件"""
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=4)
        
        # 首次创建主配置时，也加载B站配置
        config = DEFAULT_CONFIG.copy()
        config['bilibot'] = load_bilibot_config()
        return config
    
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
            # 合并默认配置，以防旧的配置文件缺少新字段
            config = DEFAULT_CONFIG.copy()
            # 深层合并 background 字典
            if 'background' in config and 'background' in user_config:
                 config['background'].update(user_config['background'])
                 del user_config['background'] # 避免重复更新
            
            config.update(user_config)
            
            # 将B站配置合并到主配置中
            config['bilibot'] = load_bilibot_config()

            return config
    except (json.JSONDecodeError, IOError):
        # 出错时也加载B站配置
        config = DEFAULT_CONFIG.copy()
        config['bilibot'] = load_bilibot_config()
        return config

def save_config(config_data):
    """保存主配置到JSON文件（不包括B站配置）"""
    # 确保不会意外地将bilibot配置写入主文件
    config_to_save = config_data.copy()
    if 'bilibot' in config_to_save:
        del config_to_save['bilibot']
        
    # 移除其他已知非JSON序列化的Flask内部密钥
    keys_to_remove = [
        'PERMANENT_SESSION_LIFETIME'
    ]
    for key in keys_to_remove:
        if key in config_to_save:
            del config_to_save[key]
        
    # 写入文件
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=4)

# --------------------------- 核心路径配置 --------------------------- #
# 用户可上传的目录
train_data_dir_for_rename = os.path.join("data", "train")
new_data_dir_for_rename = os.path.join("data", "new")

# 模型与元数据
character_meta_path = os.path.join("models", "character_meta_restructured.json")

yolo_model_path = os.path.join("models", "pytorch", "yolo_model.pt")
model_checkpoint_path = os.path.join("models", "pytorch", "model.pth")

feature_db_path = os.path.join("models", "feature_db.npy")
class_map_path = os.path.join("models","class_to_idx.json")

# -- ONNX 模型路径 --
onnx_yolo_model_path = os.path.join("models", "onnx", "yolo_model.onnx")
onnx_model_path = os.path.join("models", "onnx", "model.onnx")


# --------------------------- 裁剪区域扩增比例 --------------------------- #
LEFT_RATIO = 0
TOP_RATIO = 0
RIGHT_RATIO = 0
BOTTOM_RATIO = 0 

version = "0.0.1"
github_repo = "CNLonely/DeepInsightOtakuApp"