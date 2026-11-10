#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查当前 Python 环境是否满足项目运行条件。

用法：
    python check_env.py
"""
import platform
import sys

import cv2
import torch
import ultralytics


def main() -> None:
    print("=" * 56)
    print("Python 版本 :", sys.version.split()[0])
    print("解释器路径 :", sys.executable)
    print("操作系统   :", platform.platform())
    print("-" * 56)
    print("PyTorch    :", torch.__version__)
    print("CUDA 可用  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("显卡型号   :", torch.cuda.get_device_name(0))
    print("OpenCV     :", cv2.__version__)
    print("Ultralytics:", ultralytics.__version__)
    print("=" * 56)


if __name__ == "__main__":
    main()
