# ***********************************************************************************************
# created:          2026-04-15
#
# author:           chensong
#
# purpose:          目标检测模块
#
#
#
#
#
#
# 输赢不重要，答案对你们有什么意义才重要。
#
# 光阴者，百代之过客也，唯有奋力奔跑，方能生风起时，是时代造英雄，英雄存在于时代。或许世人道你轻狂，可你本就年少啊。 看护好，自己的理想和激情。
#
#
# 我可能会遇到很多的人，听他们讲好2多的故事，我来写成故事或编成歌，用我学来的各种乐器演奏它。
# 然后还可能在一个国家遇到一个心仪我的姑娘，她可能会被我帅气的外表捕获，又会被我深邃的内涵吸引，在某个下雨的夜晚，她会全身淋透然后要在我狭小的住处换身上的湿衣服。
# 3小时候后她告诉我她其实是这个国家的公主，她愿意向父皇求婚。我不得已告诉她我是穿越而来的男主角，我始终要回到自己的世界。
# 然后我的身影慢慢消失，我看到她眼里的泪水，心里却没有任何痛苦，我才知道，原来我的心被丢掉了，我游历全世界的原因，就是要找回自己的本心。
# 于是我开始有意寻找各种各样失去心的人，我变成一块砖头，一颗树，一滴水，一朵白云，去听大家为什么会失去自己的本心。
# 我发现，刚出生的宝宝，本心还在，慢慢的，他们的本心就会消失，收到了各种黑暗之光的侵蚀。
# 从一次争论，到嫉妒和悲愤，还有委屈和痛苦，我看到一只只无形的手，把他们的本心扯碎，蒙蔽，偷走，再也回不到主人都身边。
# 我叫他本心猎手。他可能是和宇宙同在的级别 但是我并不害怕，我仔细回忆自己平淡的一生 寻找本心猎手的痕迹。
# 沿着自己的回忆，一个个的场景忽闪而过，最后发现，我的本心，在我写代码的时候，会回来。
# 安静，淡然，代码就是我的一切，写代码就是我本心回归的最好方式，我还没找到本心猎手，但我相信，顺着这个线索，我一定能顺藤摸瓜，把他揪出来。
# ************************************************************************************************/
import cv2
import numpy as np
from ultralytics import YOLO
from config.config import config

class ObjectDetector:
    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL)
        self.model.conf = config.CONFIDENCE_THRESHOLD
        self.model.iou = config.NMS_THRESHOLD
        self.device = config.YOLO_DEVICE

        self.pig_class_id = 16

    def classify_pig_type(self, bbox):
        """
        基于体型大小分类猪只类型

        Args:
            bbox: 边界框 [x1, y1, x2, y2]

        Returns:
            str: 'sow' (母猪) 或 'piglet' (小猪)
        """
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)

        if area > config.PIG_AREA_THRESHOLD:
            return 'sow'
        else:
            return 'piglet'

    def detect(self, frame):
        """
        检测视频帧中的目标

        Args:
            frame: 视频帧

        Returns:
            list: 检测结果列表
        """
        results = self.model(frame, verbose=False)

        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    if cls_id == self.pig_class_id:
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        conf = float(boxes.conf[i].item())
                        pig_type = self.classify_pig_type([x1, y1, x2, y2])

                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_id': cls_id,
                            'class_name': pig_type,
                            'area': (x2 - x1) * (y2 - y1),
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                        })

        return detections

    def get_sows_and_piglets(self, detections):
        """
        分离母猪和小猪

        Args:
            detections: 检测结果列表

        Returns:
            tuple: (sows, piglets)
        """
        sows = [d for d in detections if d['class_name'] == 'sow']
        piglets = [d for d in detections if d['class_name'] == 'piglet']
        return sows, piglets

    def visualize(self, frame, detections):
        """
        在视频帧上可视化检测结果

        Args:
            frame: 视频帧
            detections: 检测结果

        Returns:
            ndarray: 带有检测结果的视频帧
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']

            # 母猪用绿色，小猪用蓝色
            if class_name == 'sow':
                color = (0, 255, 0)
                label = f'Sow: {confidence:.2f}'
            else:
                color = (255, 0, 0)
                label = f'Piglet: {confidence:.2f}'

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 绘制类别和置信度
            cv2.putText(frame, label, (x1, y1 - 10),
                        config.DISPLAY_FONT, 0.5, color, 2)

            # 绘制中心点
            center = det['center']
            cv2.circle(frame, center, 3, color, -1)

        return frame
