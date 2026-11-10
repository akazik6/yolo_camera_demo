#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用同一套 YOLO 模型检测单张图片，便于在没有摄像头的环境中验证流程。

用法：
    python detect_image.py --source assets/bus.jpg
    python detect_image.py --source assets/bus.jpg --conf 0.4 --output outputs
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 单张图片检测")
    parser.add_argument("--source", type=str, required=True, help="输入图片路径")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="模型权重文件")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--output", type=str, default="outputs", help="结果保存目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    results = model.predict(source=args.source, conf=args.conf, verbose=False)
    result = results[0]

    annotated = result.plot(line_width=2, font_size=12)
    out_path = output_dir / f"{Path(args.source).stem}_detected.jpg"
    cv2.imwrite(str(out_path), annotated)

    names = result.names
    labels = [names[int(c)] for c in result.boxes.cls.tolist()]
    counts = Counter(labels)

    print(f"[INFO] 图片：{args.source}")
    print(f"[INFO] 共检出 {len(labels)} 个目标：")
    for label, num in counts.most_common():
        print(f"        {label:<12s} x {num}")
    print(f"[INFO] 结果已保存：{out_path}")


if __name__ == "__main__":
    main()
