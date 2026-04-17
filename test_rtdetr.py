# ***********************************************************************************************
# created:          2026-04-16
#
# author:           chensong
#
# purpose:          RT-DETR Transformer 检测器测试脚本
#
# 使用说明:
# 1. 准备一张测试图片（猪只图片）
# 2. 运行：python test_rtdetr.py [图片路径]
# 3. 如果不指定图片路径，默认使用 'test.jpg'
# ************************************************************************************************/

import cv2
import sys
import time
import numpy as np
import requests
from io import BytesIO
from src.detection.object_detector import ObjectDetector
from src.detection.rt_detr_detector import RTDETRDetector

# 在线猪只测试图片（来自 COCO 数据集）
# TEST_PIG_IMAGE_URL = "https://images.cocodataset.org/val2017/000000000139.jpg"
TEST_PIG_IMAGE_URL = "test.png"  # 替换为实际测试图片路径

def compare_detectors(image_path):
    """
    对比 YOLO11 和 RT-DETR 的检测效果
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 错误：无法读取图像 {image_path}")
        print("\n请确保：")
        print("  1. 图片路径正确")
        print("  2. 图片格式支持 (jpg, png, bmp)")
        print("  3. 图片中包含猪只（COCO 类别 ID 16）")
        return

    print(f"✅ 图像大小：{image.shape[1]} x {image.shape[0]}")
    print("=" * 60)

    # 测试 YOLO11
    print("\n[1/2] 测试 YOLO11 检测器...")
    try:
        yolo_detector = ObjectDetector()
    except Exception as e:
        print(f"❌ YOLO11 初始化失败：{e}")
        print("请确保已安装 ultralytics: pip install ultralytics")
        return

    start_time = time.perf_counter()
    yolo_detections = yolo_detector.detect(image)
    yolo_time = (time.perf_counter() - start_time) * 1000

    print(f"  检测数量：{len(yolo_detections)}")
    print(f"  推理时间：{yolo_time:.2f}ms")

    if len(yolo_detections) == 0:
        print("  ⚠️  未检测到猪只，可能原因：")
        print("     - 图片中没有猪")
        print("     - 图片中猪的类别不是 COCO 数据集中的猪 (class_id=16)")
        print("     - 建议使用包含猪只的真实场景图片测试")
    else:
        for det in yolo_detections:
            print(f"    - {det['class_name']}: bbox={det['bbox']}, conf={det['confidence']:.3f}")

    # 可视化 YOLO11 结果
    yolo_result = yolo_detector.visualize(image.copy(), yolo_detections)
    cv2.putText(yolo_result, "YOLO11 Result", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 测试 RT-DETR
    print("\n[2/2] 测试 RT-DETR Transformer 检测器...")
    try:
        rtdetr_detector = RTDETRDetector()

        start_time = time.perf_counter()
        rtdetr_detections = rtdetr_detector.detect(image)
        rtdetr_time = (time.perf_counter() - start_time) * 1000

        print(f"  检测数量：{len(rtdetr_detections)}")
        print(f"  推理时间：{rtdetr_time:.2f}ms")
        for det in rtdetr_detections:
            print(f"    - {det['class_name']}: bbox={det['bbox']}, conf={det['confidence']:.3f}")

        # 可视化 RT-DETR 结果
        rtdetr_result = rtdetr_detector.visualize(image.copy(), rtdetr_detections)
        cv2.putText(rtdetr_result, "RT-DETR Result", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 保存对比结果
        # 将两张图片上下拼接
        combined_height = yolo_result.shape[0] * 2
        combined_image = np.zeros((combined_height, image.shape[1], 3), dtype=np.uint8)
        combined_image[0:image.shape[0], :] = yolo_result
        combined_image[image.shape[0]:, :] = rtdetr_result

        output_path = 'comparison_result.jpg'
        cv2.imwrite(output_path, combined_image)
        print(f"\n对比结果已保存：{output_path}")

        # 性能对比
        print("\n" + "=" * 60)
        print("性能对比：")
        print(f"  推理时间差异：RT-DETR 比 YOLO11 {'慢' if rtdetr_time > yolo_time else '快'} {abs(rtdetr_time - yolo_time):.2f}ms")
        print(f"  检测数量差异：RT-DETR 比 YOLO11 {'多' if len(rtdetr_detections) > len(yolo_detections) else '少'} {abs(len(rtdetr_detections) - len(yolo_detections))}个")

    except Exception as e:
        print(f"  RT-DETR 测试失败：{e}")
        print("  请确保已安装 ultralytics>=8.3.0")

def test_video_detection(video_path='test_video.mp4'):
    """
    测试视频检测（使用 RT-DETR）
    """
    print("测试视频检测...")
    detector = RTDETRDetector()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return

    frame_count = 0
    total_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        start_time = time.perf_counter()
        detections = detector.detect(frame)
        inference_time = (time.perf_counter() - start_time) * 1000
        total_time += inference_time

        # 可视化
        frame = detector.visualize(frame, detections)
        cv2.putText(frame, f"FPS: {1000/inference_time:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('RT-DETR Video Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    avg_fps = 1000 / (total_time / frame_count) if frame_count > 0 else 0
    print(f"\n视频检测完成:")
    print(f"  总帧数：{frame_count}")
    print(f"  平均 FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 默认使用在线猪只图片测试
        image_path = TEST_PIG_IMAGE_URL

    print("=" * 60)
    print("YOLO11 vs RT-DETR Transformer 检测器对比测试")
    print("=" * 60)

    # 如果是 URL，下载图片
    if image_path.startswith('http'):
        print(f"\n正在下载测试图片：{image_path}")
        try:
            response = requests.get(image_path, timeout=10)
            image = cv2.imdecode(BytesIO(response.content), cv2.IMREAD_COLOR)
            if image is not None:
                # 保存为本地文件
                local_path = 'test_pig.jpg'
                cv2.imwrite(local_path, image)
                print(f"✅ 图片已保存到：{local_path}")
                print(f"   图像大小：{image.shape[1]} x {image.shape[0]}")
                image_path = local_path
            else:
                print("❌ 无法解码下载的图片")
                #return
        except Exception as e:
            print(f"❌ 下载图片失败：{e}")
            print("\n请确保已安装 requests: pip install requests")
            #return

    compare_detectors(image_path)
