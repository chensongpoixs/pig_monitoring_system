# ***********************************************************************************************
# created:          2026-04-16
#
# author:           chensong
#
# purpose:          检测 test.png 中的猪只并保存结果
#
# ************************************************************************************************/

import cv2
import numpy as np
from ultralytics import YOLO

def detect_and_visualize(image_path, output_path='detected_pig_result.jpg'):
    """
    检测图片中的猪只并可视化结果
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    print(f"图像大小：{image.shape[1]} x {image.shape[0]}")
    print(f"正在加载 YOLO11 模型...")

    # 加载 YOLO11 模型
    model = YOLO('yolo11n.pt')

    # 检测（降低置信度阈值以捕获更多目标）
    print("正在检测猪只...")
    results = model(image, conf=0.2, iou=0.45)  # 降低阈值到 0.2

    # 处理结果
    detection_count = 0
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())

                # 获取类别名称
                class_name = model.names.get(cls_id, f"class_{cls_id}")

                # 获取边界框
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                detection_count += 1
                print(f"检测到：{class_name} (ID={cls_id}), 置信度={conf:.3f}, bbox=[{x1},{y1},{x2},{y2}]")

    # 可视化结果（使用 ultralytics 的内置可视化）
    print(f"\n正在保存结果到 {output_path}...")

    # 使用 save() 方法保存可视化结果
    results[0].save(output_path)
    print(f"✅ 结果已保存到：{output_path}")

    # 也保存原始标注结果
    annotated = results[0].plot()
    cv2.imwrite('annotated_pig.jpg', annotated)
    print(f"✅ 标注结果已保存到：annotated_pig.jpg")

    # 自定义可视化（更详细的标注）
    print("\n生成自定义可视化结果...")
    custom_result = image.copy()

    pig_detected = False
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                class_name = model.names.get(cls_id, f"class_{cls_id}")

                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                # 如果是猪（COCO 类别 ID 16）
                if cls_id == 16:
                    pig_detected = True
                    # 根据置信度设置颜色
                    if conf > 0.5:
                        color = (0, 255, 0)  # 高置信度 - 绿色
                        label = f'Pig: {conf:.2f}'
                    elif conf > 0.3:
                        color = (0, 255, 255)  # 中等置信度 - 黄色
                        label = f'Pig: {conf:.2f}'
                    else:
                        color = (0, 0, 255)  # 低置信度 - 红色
                        label = f'Pig: {conf:.2f}'

                    # 绘制边界框
                    cv2.rectangle(custom_result, (x1, y1), (x2, y2), color, 3)

                    # 绘制标签背景
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(custom_result, (x1, y1 - text_size[1] - 10),
                                 (x1 + text_size[0], y1), color, -1)
                    cv2.putText(custom_result, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    # 其他类别
                    color = (255, 0, 0)
                    cv2.rectangle(custom_result, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(custom_result, f'{class_name}: {conf:.2f}', (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if not pig_detected:
        cv2.putText(custom_result, "No Pig Detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        print("⚠️  未检测到猪只（COCO 类别 ID 16）")

    cv2.imwrite('custom_pig_detection.jpg', custom_result)
    print("✅ 自定义可视化结果已保存到：custom_pig_detection.jpg")

    return detection_count

if __name__ == "__main__":
    print("="*60)
    print("猪只检测工具")
    print("="*60)

    image_path = 'test.png'
    detect_and_visualize(image_path, 'detected_pig_result.jpg')
