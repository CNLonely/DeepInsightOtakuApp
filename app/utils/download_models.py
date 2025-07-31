import os
import sys
from pathlib import Path
from packaging.version import parse as parse_version, InvalidVersion
import shutil

# 动态导入库
try:
    from huggingface_hub import HfApi
    from huggingface_hub import snapshot_download as hf_snapshot_download
    from huggingface_hub.utils import HfHubHTTPError as HfError
    from modelscope.hub.api import HubApi as MsHubApi
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装所需库: pip install huggingface_hub modelscope 'packaging>=20.0'")
    sys.exit(1)


# --- 配置区 ---

# -- 统一下载目录 --
LOCAL_MODEL_DIR = "models"

# -- Hugging Face 配置 --
HF_REPO_NAME = "CNLonely/deepinsightotakumodel"

# -- ModelScope 配置 --
MS_REPO_NAME = "CNLonely/deepinsightotaku"

# -- 通用配置 --
# 注意：两个列表中的文件结构需要与上传脚本中的保持一致
FILES_TO_DOWNLOAD_HF = [
    # PyTorch 模型文件
    "pytorch/yolo_model.pt",
    "pytorch/model.pth",

    # ONNX 模型文件 (请根据你的实际情况修改本地路径)
    "onnx/model.onnx",
    "onnx/yolo_model.onnx",

    # 共享的元数据文件
    "class_to_idx.json",
    "feature_db.npy",
    
    "character_meta_restructured.json"
]

FILES_TO_DOWNLOAD_MS = FILES_TO_DOWNLOAD_HF + ["version.txt"] # MS 多一个版本文件

# --- 配置区结束 ---

# === Hugging Face 版本的函数 ===
def _get_hf_version_key(tag: str):
    """
    将版本标签转换为可供排序的元组。
    这使得脚本能理解 'v1.0' 和 'beta0.0.1' 这样的自定义格式。
    'beta' 版本被认为是比正式版更早的版本。
    例如: 'v1.0' -> (2, Version('1.0')), 'beta0.0.1' -> (1, Version('0.0.1'))
    """
    tag_lower = tag.lower()
    
    # 定义版本的“阶段”，数字越大越新
    stage_map = {'beta': 1}
    stage = 2  # 默认为正式版 (final release)
    version_str = tag_lower

    if tag_lower.startswith('v'):
        version_str = tag_lower[1:]
    elif tag_lower.startswith('beta'):
        stage = stage_map['beta']
        version_str = tag_lower[4:]
    
    # 确保版本号部分不为空，且看起来像一个版本号
    if not version_str or not version_str[0].isdigit():
        return None

    try:
        # 使用标准库来解析版本号的数字部分
        parsed_version = parse_version(version_str)
        return (stage, parsed_version)
    except InvalidVersion:
        return None

def get_latest_hf_tag(api: HfApi, repo_id: str):
    """获取远程仓库中最新的版本标签，支持自定义格式"""
    try:
        refs = api.list_repo_refs(repo_id=repo_id)
        all_tags = [ref.name for ref in refs.tags]
        
        if not all_tags:
            print("  - 仓库中没有找到任何版本标签。")
            return None
        
        print(f"  - 发现 {len(all_tags)} 个远程标签: {all_tags}")

        valid_versions = []
        for tag in all_tags:
            key = _get_hf_version_key(tag)
            if key:
                valid_versions.append((key, tag))
            else:
                print(f"  - ⚠️ 警告: 忽略无法解析的版本标签 '{tag}'")
        
        if not valid_versions:
            print("  - 未找到任何可识别的版本标签。")
            return None
            
        # 使用我们自定义的key来进行降序排序
        valid_versions.sort(key=lambda x: x[0], reverse=True)
        
        latest_tag = valid_versions[0][1]
        print(f"  - 找到最新远程版本 (Tag): {latest_tag}")
        return latest_tag
        
    except HfError as e:
        if e.response.status_code == 404:
            print(f"❌ 错误: 找不到仓库 '{repo_id}'。请检查 REPO_NAME 是否正确。")
        else:
            print(f"❌ 访问仓库时出错: {e}")
        return None
    except Exception as e:
        print(f"❌ 获取最新版本时发生未知错误: {e}")
        return None

# === ModelScope 版本的函数 ===
def get_latest_ms_version(api: MsHubApi, repo_id: str):
    """通过下载version.txt获取ModelScope仓库的最新版本"""
    try:
        # ModelScope 的 snapshot_download 更适合下载单个文件
        # 它会在本地缓存中寻找或下载文件，并返回顶层目录的路径
        model_dir = ms_snapshot_download(repo_id, allow_patterns=["version.txt"])
        version_file = Path(model_dir) / "version.txt"

        if not version_file.is_file():
             print(f"  - ⚠️ 在仓库 '{repo_id}' 中找不到 version.txt 文件。")
             return None

        remote_version = version_file.read_text().strip()
        print(f"  - 找到最新远程版本 (version.txt): {remote_version}")
        return remote_version
    except Exception as e:
        # 捕获一个通用的异常，并检查错误信息来判断是不是“文件不存在”
        if 'Not Found' in str(e) or 'does not exist' in str(e):
             print(f"  - ⚠️ 在仓库 '{repo_id}' 中找不到 version.txt 文件。")
             return None
        print(f"❌ 获取 ModelScope 版本时出错: {e}")
        return None

# === 通用函数 ===
def get_local_version(model_dir: str):
    """获取本地已下载模型的版本"""
    version_file = Path(model_dir) / ".version"
    if version_file.is_file():
        local_tag = version_file.read_text().strip()
        print(f"  - 找到本地版本: {local_tag}")
        return local_tag
    print("  - 未找到本地版本记录。")
    return None

def save_local_version(model_dir: str, tag: str):
    """保存下载好的模型版本到本地"""
    os.makedirs(model_dir, exist_ok=True)
    version_file = Path(model_dir) / ".version"
    version_file.write_text(tag)
    print(f"  - 已将本地版本更新为: {tag}")

def cleanup_download_dir(local_dir: str, valid_files: list):
    """清理下载目录中由库创建的临时文件和文件夹。"""
    print(f"\n🧹 开始清理下载目录: {local_dir}")
    local_path = Path(local_dir)
    if not local_path.is_dir():
        return

    # 1. 创建一个所有有效文件及其父目录的集合，以便保留
    files_to_keep = set()
    # 明确地将 .version 文件加入保留列表
    files_to_keep.add(local_path / ".version")

    for f in valid_files:
        # 将文件本身加入保留列表
        files_to_keep.add(local_path / f)
        # 将文件的所有上级父目录也加入保留列表
        for parent in (local_path / f).parents:
            if parent == local_path:
                break
            files_to_keep.add(parent)
            
    # 2. 遍历下载目录中的所有项目
    cleaned_count = 0
    for item in local_path.iterdir():
        # 如果项目不在保留列表中
        if item not in files_to_keep:
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                    print(f"  - 已删除临时文件夹: {item.name}")
                else:
                    item.unlink()
                    print(f"  - 已删除临时文件: {item.name}")
                cleaned_count += 1
            except Exception as e:
                print(f"  - ⚠️ 清理 {item.name} 时出错: {e}")

    if cleaned_count == 0:
        print("  - 目录已是最新状态，无需清理。")


def check_and_download(platform: str, repo_id: str, local_dir: str, files_to_download: list, get_latest_version_func, api):
    """通用下载流程"""
    print(f"\n🚀 开始检查和下载模型到 '{local_dir}'...")
    
    latest_version = get_latest_version_func(api, repo_id)
    if not latest_version:
        print("🛑 无法确定远程版本，下载中止。")
        return

    local_version = get_local_version(local_dir)
    files_missing = any(not (Path(local_dir) / f).is_file() for f in files_to_download if f != "version.txt")

    if local_version == latest_version and not files_missing:
        print(f"\n✅ 你已拥有最新版本 ({latest_version})，无需下载。")
        return

    if local_version != latest_version:
        print(f"\n🔄 检测到新版本，准备从 '{local_version or 'N/A'}' 更新到 '{latest_version}'。")
    elif files_missing:
        print(f"\n🔍 检测到本地文件不完整，准备下载版本 '{latest_version}'。")

    print("\nDownloading files...")
    try:
        # 根据平台选择正确的下载函数
        if platform == 'hf':
            hf_snapshot_download(
                repo_id=repo_id,
                revision=latest_version,
                allow_patterns=files_to_download,
                local_dir=local_dir,
                local_dir_use_symlinks=False  # HF特有的参数
            )
        elif platform == 'ms':
            ms_snapshot_download(
                repo_id=repo_id,
                revision='master',  # 始终从 master 分支下载最新文件
                allow_patterns=files_to_download,
                local_dir=local_dir
            )
        print("    ✅ 所有文件下载/更新成功")
    except Exception as e:
        print(f"    ❌ 下载失败: {e}")
        return

    save_local_version(local_dir, latest_version)
    
    # 在所有操作成功后调用清理函数
    final_files_list = [f for f in files_to_download if f != 'version.txt']
    cleanup_download_dir(local_dir, final_files_list)

    print(f"\n🎉 任务完成。文件保存在: {Path(local_dir).resolve()}")


if __name__ == "__main__":
    platform_choice = ""
    while platform_choice not in ["1", "2"]:
        platform_choice = input("请选择要从哪个平台下载:\n  1: Hugging Face\n  2: ModelScope\n请输入 (1/2): ")

    if platform_choice == "1":
        check_and_download(
            platform='hf',
            repo_id=HF_REPO_NAME,
            local_dir=LOCAL_MODEL_DIR,
            files_to_download=FILES_TO_DOWNLOAD_HF,
            get_latest_version_func=get_latest_hf_tag,
            api=HfApi()
        )
    elif platform_choice == "2":
        check_and_download(
            platform='ms',
            repo_id=MS_REPO_NAME,
            local_dir=LOCAL_MODEL_DIR,
            files_to_download=FILES_TO_DOWNLOAD_MS,
            get_latest_version_func=get_latest_ms_version,
            api=MsHubApi()
        ) 