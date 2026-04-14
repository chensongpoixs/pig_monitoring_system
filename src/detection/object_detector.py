import cv2
import torch
import numpy as np
from config.config import config

class ObjectDetector:
    def __init__(self):
        # 加载YOLOv5模型
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # 设置置信度阈值
        self.model.conf = config.CONFIDENCE_THRESHOLD
        # 设置非极大值抑制阈值
        self.model.iou = config.NMS_THRESHOLD
        # 只检测猪相关类别
        self.model.classes = [16]  # 16是COCO数据集中猪的类别ID
    
    def detect(self, frame):
        """
        检测视频帧中的目标
        
        Args:
            frame: 视频帧
            
        Returns:
            detections: 检测结果，包含目标的位置、类别和置信度
        """
        # 转换帧为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 使用模型进行检测
        results = self.model(frame_rgb)
        
        # 解析检测结果
        detections = []
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': 'pig' if int(cls) == 16 else 'unknown'
            })
        
        return detections
    
    def visualize(self, frame, detections):
        """
        在视频帧上可视化检测结果
        
        Args:
            frame: 视频帧
            detections: 检测结果
            
        Returns:
            frame: 带有检测结果的视频帧
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制类别和置信度
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
