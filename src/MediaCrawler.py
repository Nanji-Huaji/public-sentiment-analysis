import importlib
import pathlib
import runpy
import subprocess
import sys
from types import ModuleType
from typing import Any, Dict, List

import json

import os

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent / "MediaCrawler"


class MediaCrawler:
    def __init__(
        self,
        platform: str,
        keywords: List[str],
        time: int = 30,
        max_crawl_note: int = 50,
        **kwargs,
    ):
        self.platform = platform
        self.keywords = keywords
        self.config: Dict[str, Any] = {
            "PLATFORM": platform,
            "KEYWORDS": ",".join(self.keywords),
            "CRAWLER_TYPE": "search",
            "SAVE_DATA_OPTION": "csv",
            "HEADLESS": True,
            "ENABLE_GET_COMMENTS": True,
            "MAX_CONCURRENCY_NUM": 1,
            "MAX_CRAWL_NOTE": max_crawl_note,
        }
        self.configure(**kwargs)

    def configure(self, **kwargs) -> None:
        """在运行前动态更新配置"""
        for k, v in kwargs.items():
            self.config[k.upper()] = v

    def crawl(self, login_type: str = "qrcode", **kwargs) -> dict | None:
        """真正启动爬虫；默认扫码登录"""
        if kwargs:
            self.configure(**kwargs)
        self._apply_config_to_module()  # 1. 把参数写回 base_config.py
        return self._run_main_py(login_type)  # 2. 执行 main.py

    def __call__(self, *args, **kwargs) -> dict | None:
        return self.crawl(*args, **kwargs)

    def get_crawler_type(self) -> str:
        return self.config.get("CRAWLER_TYPE", "search")

    def run_file(self, file_path: str):
        """如果你想直接执行 MediaCrawler 里任何单独脚本，可用该方法"""
        self._apply_config_to_module()
        return runpy.run_path(file_path)  # 返回脚本的 globals()

    def _apply_config_to_module(self) -> None:
        sys.path.insert(0, str(_PROJECT_ROOT))  # 保证能 import 到 MediaCrawler
        base_cfg: ModuleType = importlib.import_module("config.base_config")

        for k, v in self.config.items():
            setattr(base_cfg, k, v)  # 动态写回
        importlib.reload(base_cfg)  # 防止重复 import 不生效

    def _run_main_py(self, login_type: str) -> dict | None:
        RESULT_PREFIX = "@@RESULT@@ "
        cmd = [
            "conda",
            "run",
            "-n",
            "media_crawler",
            "python",
            str(_PROJECT_ROOT / "main.py"),
            "--platform",
            self.platform,
            "--lt",
            login_type,
            "--type",
            self.get_crawler_type(),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        # 1. 先把非结果的日志原样打印出来（可选）
        for line in completed.stdout.splitlines():
            if not line.startswith(RESULT_PREFIX):
                print(line)
        # 2. 判断退出码
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr)

        # 3. 抓取带前缀的那一行
        result_line = next(
            (
                line[len(RESULT_PREFIX) :].strip()
                for line in completed.stdout.splitlines()
                if line.startswith(RESULT_PREFIX)
            ),
            None,
        )
        return json.loads(result_line) if result_line else None

    @staticmethod
    def get_valid_csv_file_path(csv_dict: dict) -> list | None:
        return_list = []
        for save_type, save_file_path in csv_dict.items():
            if os.path.exists(save_file_path):
                return_list.append(save_file_path)
        if return_list:
            return return_list
        else:
            return None
