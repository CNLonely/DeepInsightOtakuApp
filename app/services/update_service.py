#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新检测服务 - 检查GitHub上的最新版本和下载更新
"""

import os
import json
import requests
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin
import zipfile
import tempfile
import shutil


class UpdateService:
    def __init__(self, config: Dict, logger):
        """
        初始化更新服务
        
        Args:
            config: 应用配置
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        
        # GitHub配置
        self.repo = config.get('github_repo', 'CNLonely/DeepInsightOtakuApp')
        self.api_base = "https://api.github.com"
        self.manifest_branch = "manifest"
        
        # 本地版本信息
        self.current_version = config.get('version', '0.0.0')
        self.app_root = Path(__file__).parent.parent.parent
        
        # 缓存
        self._cached_manifest = None
        self._cached_release = None
        
    def get_file_sha256(self, file_path: str) -> str:
        """计算文件的SHA256哈希值"""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"计算文件SHA256失败 {file_path}: {e}")
            return ""
    
    def get_manifest_from_github(self) -> Optional[Dict]:
        """从GitHub获取最新的文件清单"""
        try:
            # 使用gh-proxy.com代理访问GitHub
            url = f"https://gh-proxy.com/https://github.com/{self.repo}/raw/{self.manifest_branch}/file_manifest.json"
            
            self.logger.info(f"正在获取文件清单: {url}")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                manifest = response.json()
                self.logger.info(f"成功获取文件清单，版本: {manifest.get('version', 'unknown')}")
                return manifest
            else:
                self.logger.warning(f"获取文件清单失败: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"获取文件清单异常: {e}")
            return None
    
    def get_update_history(self) -> List[Dict]:
        """获取更新历史"""
        try:
            url = f"https://gh-proxy.com/https://github.com/{self.repo}/raw/{self.manifest_branch}/update_history.json"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                history = response.json()
                return history.get('updates', [])
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"获取更新历史异常: {e}")
            return []
    
    def check_for_updates(self) -> Dict:
        """检查是否有可用更新"""
        try:
            # 获取远程清单
            remote_manifest = self.get_manifest_from_github()
            if not remote_manifest:
                return {
                    'has_update': False,
                    'error': '无法获取远程版本信息'
                }
            
            remote_version = remote_manifest.get('version', '0.0.0')
            update_notes = remote_manifest.get('update_notes', '')
            
            # 比较版本
            has_update = self._compare_versions(remote_version, self.current_version)
            
            if has_update:
                # 分析需要更新的文件
                files_to_update = self._analyze_files_to_update(remote_manifest)
                
                return {
                    'has_update': True,
                    'current_version': self.current_version,
                    'remote_version': remote_version,
                    'update_notes': update_notes,
                    'files_to_update': files_to_update,
                    'total_files': len(remote_manifest.get('files', {})),
                    'update_size_mb': self._calculate_update_size(files_to_update)
                }
            else:
                return {
                    'has_update': False,
                    'current_version': self.current_version,
                    'remote_version': remote_version,
                    'message': '已是最新版本'
                }
                
        except Exception as e:
            self.logger.error(f"检查更新异常: {e}")
            return {
                'has_update': False,
                'error': f'检查更新失败: {str(e)}'
            }
    
    def _compare_versions(self, version1: str, version2: str) -> bool:
        """比较版本号，返回version1是否大于version2"""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # 补齐版本号长度
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            for i in range(max_len):
                if v1_parts[i] > v2_parts[i]:
                    return True
                elif v1_parts[i] < v2_parts[i]:
                    return False
            
            return False  # 版本相同
            
        except Exception as e:
            self.logger.error(f"版本比较异常: {e}")
            return False
    
    def _analyze_files_to_update(self, remote_manifest: Dict) -> List[Dict]:
        """分析需要更新的文件"""
        files_to_update = []
        remote_files = remote_manifest.get('files', {})
        
        for file_path, file_info in remote_files.items():
            local_path = self.app_root / file_path
            
            # 检查文件是否存在
            if not local_path.exists():
                files_to_update.append({
                    'path': file_path,
                    'action': 'download',
                    'reason': '文件不存在',
                    'size': file_info.get('size', 0)
                })
                continue
            
            # 检查文件哈希
            local_sha256 = self.get_file_sha256(str(local_path))
            remote_sha256 = file_info.get('sha256', '')
            
            if local_sha256 != remote_sha256:
                files_to_update.append({
                    'path': file_path,
                    'action': 'update',
                    'reason': '文件内容已更改',
                    'size': file_info.get('size', 0)
                })
        
        return files_to_update
    
    def _calculate_update_size(self, files_to_update: List[Dict]) -> float:
        """计算更新包大小（MB）"""
        total_size = sum(file_info.get('size', 0) for file_info in files_to_update)
        return round(total_size / (1024 * 1024), 2)
    
    def download_file(self, file_path: str, download_url: str) -> bool:
        """下载单个文件"""
        try:
            local_path = self.app_root / file_path
            
            # 创建目录
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 修复下载URL中的路径分隔符
            fixed_url = download_url.replace('\\', '/')
            
            # 下载文件
            self.logger.info(f"正在下载: {file_path}")
            response = requests.get(fixed_url, stream=True, timeout=60)
            
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                self.logger.info(f"下载完成: {file_path}")
                return True
            else:
                self.logger.error(f"下载失败 {file_path}: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"下载文件异常 {file_path}: {e}")
            return False
    
    def perform_update(self, progress_callback=None) -> Dict:
        """执行更新"""
        try:
            # 获取远程清单
            remote_manifest = self.get_manifest_from_github()
            if not remote_manifest:
                return {
                    'success': False,
                    'error': '无法获取远程版本信息'
                }
            
            # 分析需要更新的文件
            files_to_update = self._analyze_files_to_update(remote_manifest)
            
            if not files_to_update:
                return {
                    'success': True,
                    'message': '无需更新',
                    'updated_files': 0
                }
            
            # 创建临时目录
            download_dir = self.app_root / "download"
            backup_dir = self.app_root / "backup"
            
            # 清理之前的临时目录
            if download_dir.exists():
                import shutil
                shutil.rmtree(download_dir)
            if backup_dir.exists():
                import shutil
                shutil.rmtree(backup_dir)
            
            # 创建新的临时目录
            download_dir.mkdir(exist_ok=True)
            backup_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"创建临时目录: {download_dir}, {backup_dir}")
            
            # 下载更新文件到临时目录
            updated_files = 0
            failed_files = []
            
            for i, file_info in enumerate(files_to_update):
                file_path = file_info['path']
                download_url = remote_manifest['files'][file_path]['download_url']
                
                # 更新进度
                if progress_callback:
                    progress = (i + 1) / len(files_to_update) * 100
                    progress_callback(progress, f"正在下载: {file_path}")
                
                # 下载文件到临时目录
                if self.download_file_to_temp(file_path, download_url, download_dir):
                    updated_files += 1
                else:
                    failed_files.append(file_path)
            
            if failed_files:
                # 有文件下载失败，清理临时目录
                self._cleanup_temp_dirs(download_dir, backup_dir)
                return {
                    'success': False,
                    'error': f'部分文件下载失败: {", ".join(failed_files)}',
                    'failed_files': failed_files
                }
            
            # 备份原文件
            if progress_callback:
                progress_callback(50, "正在备份原文件...")
            
            backup_failed = self._backup_original_files(files_to_update, backup_dir)
            if backup_failed:
                # 备份失败，清理临时目录
                self._cleanup_temp_dirs(download_dir, backup_dir)
                return {
                    'success': False,
                    'error': f'备份文件失败: {", ".join(backup_failed)}',
                    'failed_files': backup_failed
                }
            
            # 替换文件
            if progress_callback:
                progress_callback(75, "正在替换文件...")
            
            replace_failed = self._replace_files(files_to_update, download_dir)
            if replace_failed:
                # 替换失败，尝试恢复备份
                self._restore_from_backup(backup_dir)
                self._cleanup_temp_dirs(download_dir, backup_dir)
                return {
                    'success': False,
                    'error': f'替换文件失败: {", ".join(replace_failed)}',
                    'failed_files': replace_failed
                }
            
            # 更新本地版本号
            if updated_files > 0:
                self._update_local_version(remote_manifest['version'])
            
            # 清理临时目录
            if progress_callback:
                progress_callback(90, "正在清理临时文件...")
            
            self._cleanup_temp_dirs(download_dir, backup_dir)
            
            # 更新进度
            if progress_callback:
                progress_callback(100, "更新完成")
            
            return {
                'success': True,
                'updated_files': updated_files,
                'failed_files': [],
                'new_version': remote_manifest['version']
            }
            
        except Exception as e:
            # 发生异常时清理临时目录
            try:
                self._cleanup_temp_dirs(download_dir, backup_dir)
            except:
                pass
            
            self.logger.error(f"执行更新异常: {e}")
            return {
                'success': False,
                'error': f'更新失败: {str(e)}'
            }
    
    def download_file_to_temp(self, file_path: str, download_url: str, download_dir: Path) -> bool:
        """下载文件到临时目录"""
        try:
            # 修复下载URL中的路径分隔符
            fixed_url = download_url.replace('\\', '/')
            
            # 创建临时文件路径
            temp_file_path = download_dir / file_path
            temp_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 下载文件
            self.logger.info(f"正在下载到临时目录: {file_path}")
            response = requests.get(fixed_url, stream=True, timeout=60)
            
            if response.status_code == 200:
                with open(temp_file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                self.logger.info(f"下载完成: {file_path}")
                return True
            else:
                self.logger.error(f"下载失败 {file_path}: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"下载文件异常 {file_path}: {e}")
            return False
    
    def _backup_original_files(self, files_to_update: List[Dict], backup_dir: Path) -> List[str]:
        """备份原文件"""
        failed_files = []
        
        for file_info in files_to_update:
            file_path = file_info['path']
            local_path = self.app_root / file_path
            
            if local_path.exists():
                try:
                    # 创建备份路径
                    backup_path = backup_dir / file_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 复制文件到备份目录
                    import shutil
                    shutil.copy2(local_path, backup_path)
                    self.logger.info(f"备份文件: {file_path}")
                    
                except Exception as e:
                    self.logger.error(f"备份文件失败 {file_path}: {e}")
                    failed_files.append(file_path)
        
        return failed_files
    
    def _replace_files(self, files_to_update: List[Dict], download_dir: Path) -> List[str]:
        """替换文件"""
        failed_files = []
        
        for file_info in files_to_update:
            file_path = file_info['path']
            local_path = self.app_root / file_path
            temp_file_path = download_dir / file_path
            
            try:
                # 确保目标目录存在
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 复制临时文件到目标位置
                import shutil
                shutil.copy2(temp_file_path, local_path)
                self.logger.info(f"替换文件: {file_path}")
                
            except Exception as e:
                self.logger.error(f"替换文件失败 {file_path}: {e}")
                failed_files.append(file_path)
        
        return failed_files
    
    def _restore_from_backup(self, backup_dir: Path):
        """从备份恢复文件"""
        try:
            if not backup_dir.exists():
                return
            
            # 遍历备份目录中的所有文件
            for backup_file in backup_dir.rglob("*"):
                if backup_file.is_file():
                    # 计算相对路径
                    relative_path = backup_file.relative_to(backup_dir)
                    target_path = self.app_root / relative_path
                    
                    # 恢复文件
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(backup_file, target_path)
                    self.logger.info(f"恢复文件: {relative_path}")
            
            self.logger.info("文件恢复完成")
            
        except Exception as e:
            self.logger.error(f"恢复文件异常: {e}")
    
    def _cleanup_temp_dirs(self, download_dir: Path, backup_dir: Path):
        """清理临时目录"""
        try:
            import shutil
            
            if download_dir.exists():
                shutil.rmtree(download_dir)
                self.logger.info(f"清理下载目录: {download_dir}")
            
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
                self.logger.info(f"清理备份目录: {backup_dir}")
                
        except Exception as e:
            self.logger.error(f"清理临时目录异常: {e}")
    
    def _create_backup(self) -> Optional[Path]:
        """创建备份（已废弃，使用新的备份机制）"""
        # 这个方法现在被新的备份机制替代
        return None
    
    def _update_local_version(self, new_version: str):
        """更新本地版本号"""
        try:
            # 更新配置文件
            config_path = self.app_root / "config" / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                config['app_version'] = new_version
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                self.current_version = new_version
                self.logger.info(f"版本已更新为: {new_version}")
                
        except Exception as e:
            self.logger.error(f"更新版本号异常: {e}")
    
 