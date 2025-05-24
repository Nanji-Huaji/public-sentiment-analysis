import importlib
import pathlib
import runpy
import subprocess
import sys
from types import ModuleType
from typing import Any, Dict, List

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent / "MediaCrawler"


class MediaCrawler:
    def __init__(
        self,
        platform: str,
        keywords: List[str],
        time: float = 30.0,
        max_crawl_note: int = 50,
        **kwargs,
    ):
        self.platform = platform
        self.keywords = keywords
        self.config: Dict[str, Any] = {
            "PLATFORM": platform,
            "KEYWORDS": ",".join(self.keywords),
            "CRAWLER_TYPE": "search",
            "SAVE_DATA_OPTION": "json",
            "HEADLESS": True,
            "ENABLE_GET_COMMENTS": True,
            "MAX_CONCURRENCY_NUM": 1,
            "TIMEOUT": time,
            "MAX_CRAWL_NOTE": max_crawl_note,
        }
        self.configure(**kwargs)

    def configure(self, **kwargs) -> None:
        """在运行前动态更新配置"""
        for k, v in kwargs.items():
            self.config[k.upper()] = v  # 统一转成大写，方便与 base_config 对应

    def crawl(self, login_type: str = "qrcode", **kwargs) -> str | None:
        """真正启动爬虫；默认扫码登录"""
        if kwargs:
            self.configure(**kwargs)
        self._apply_config_to_module()  # 1. 把参数写回 base_config.py
        return self._run_main_py(login_type)  # 2. 执行 main.py

    def __call__(self, *args, **kwargs) -> str | None:  # 语法糖
        return self.crawl(*args, **kwargs)

    def get_crawler_type(self) -> str:
        return self.config.get("CRAWLER_TYPE", "search")

    def run_file(self, file_path: str) -> str | None:
        """如果你想直接执行 MediaCrawler 里任何单独脚本，可用该方法"""
        self._apply_config_to_module()
        return runpy.run_path(file_path)  # 返回脚本的 globals()

    def _apply_config_to_module(self) -> None:
        sys.path.insert(0, str(_PROJECT_ROOT))  # 保证能 import 到 MediaCrawler
        base_cfg: ModuleType = importlib.import_module("config.base_config")

        for k, v in self.config.items():
            setattr(base_cfg, k, v)  # 动态写回
        importlib.reload(base_cfg)  # 防止重复 import 不生效

    def _run_main_py(self, login_type: str) -> str | None:
        cmd = [
            sys.executable,
            str(_PROJECT_ROOT / "main.py"),
            "--platform",
            self.platform,
            "--lt",
            login_type,
            "--type",
            self.get_crawler_type(),
        ]
        # 进一步的启动参数（如 --save json）可在这里追加
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr)
        return completed.stdout  # 或者返回数据文件路径
