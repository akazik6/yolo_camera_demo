#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLO 摄像头实时目标检测 Demo。

功能：
    1. 打开本机摄像头（默认 0 号设备），逐帧读取画面；
    2. 使用 Ultralytics YOLO 预训练模型检测画面中的常见物体；
    3. 在画面上绘制检测框、类别名称和置信度，并叠加实时 FPS；
    4. 按 q 或 Esc 退出程序，按 s 保存当前带检测框的画面。

用法：
    python detect_camera.py                          # 默认摄像头 + yolov8n.pt
    python detect_camera.py --camera 1 --conf 0.4    # 指定摄像头与置信度阈值
    python detect_camera.py --model yolo11s.pt       # 更换模型
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="YOLO 实时摄像头目标检测")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="模型权重文件")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值，默认 0.5")
    parser.add_argument("--width", type=int, default=1280, help="采集画面宽度")
    parser.add_argument("--height", type=int, default=720, help="采集画面高度")
    parser.add_argument("--output", type=str, default="outputs", help="截图保存目录")
    return parser.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """打开摄像头并设置分辨率。

    Windows 下优先使用 DirectShow 后端，打开速度更快、兼容性更好；
    其他系统直接使用默认后端。
    """
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开摄像头 {index}，请检查：\n"
            "  1) 摄像头是否被其他程序占用；\n"
            "  2) 设备管理器中摄像头驱动是否正常；\n"
            "  3) 是否授予了应用访问摄像头的权限。"
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 正在加载模型：{args.model}")
    model = YOLO(args.model)  # 首次运行会自动下载权重

    print(f"[INFO] 正在打开摄像头：{args.camera}")
    cap = open_camera(args.camera, args.width, args.height)
    print("[INFO] 程序已启动：按 q / Esc 退出，按 s 保存截图")

    fps, frames, start = 0.0, 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] 读取画面失败，程序退出")
                break

            # 推理：conf 越小检出越多，但误检也会增加
            results = model.predict(frame, conf=args.conf, verbose=False)
            annotated = results[0].plot(line_width=2, font_size=12)  # BGR 图像

            # 统计并显示实时帧率
            frames += 1
            elapsed = time.time() - start
            if elapsed >= 0.5:
                fps = frames / elapsed
                frames, start = 0, time.time()
            cv2.putText(
                annotated, f"FPS: {fps:.1f}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA,
            )

            cv2.imshow("YOLO Camera Demo  (q/ESC: quit, s: save)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):        # q 或 Esc 退出
                break
            if key == ord("s"):              # s 保存当前帧
                img_path = output_dir / time.strftime("capture_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(str(img_path), annotated)
                print(f"[INFO] 截图已保存：{img_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] 资源已释放，程序结束")


if __name__ == "__main__":
    main()
