import os
import sys

# 让 tests 能直接 import ris_nlsim（项目根目录加入 sys.path）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    """测试临时目录放在仓库内 _scratch/pytest_tmp（不写系统临时目录）；按进程号分目录，并发运行互不干扰。"""
    if getattr(config.option, "basetemp", None) is None:
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_scratch", "pytest_tmp")
        os.makedirs(root, exist_ok=True)
        config.option.basetemp = os.path.join(root, f"run-{os.getpid()}")
