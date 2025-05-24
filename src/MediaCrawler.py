import importlib
import pathlib
import runpy
import subprocess
import sys
from types import ModuleType
from typing import Any, Dict, List


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent / "MediaCrawler"


class MediaCrawler:
    def __init__(self, platform: str, keywords: list[str], time: float = 30.0, max_crawl_note: int = 50, **kwargs):
        self.platform = platform
        self.keywords = keywords
        self.time = time
        self.config = {
            "PLATFORM": platform,
            "KEYWORDS": ",".join(self.keywords),
            "CRAWLER_TYPE": "search",
            "SAVE_DATA_OPTION": "csv",  # 默认保存为JSON
            "HEADLESS": True,  # 默认无头模式
            "ENABLE_GET_COMMENTS": True,  # 默认爬取评论
            "MAX_CONCURRENCY_NUM": 1,  # 默认并发数
        }

    def crawl(self, *args, **kwargs) -> str | None:
        pass

    def configure(self, **kwargs) -> None:
        pass

    def get_crawler_type(self) -> str:
        pass

    def __call__(self, *args, **kwargs) -> str | None:
        return self.crawl(*args, **kwargs)

    def run_file(self, file_path: str) -> str | None:
        pass
