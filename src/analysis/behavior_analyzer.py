import cv2
import numpy as np
from config.config import config

class BehaviorAnalyzer:
    def __init__(self):
        # 初始化状态
        self.farrowing_start_time = None
        self.crush_start_time = None
        self.prev_piglets = []
    
    def detect_farrowing(self, frame, pigs):
        """
        检测母猪分娩行为
        
        Args:
            frame: 视频帧
            pigs: 检测到的猪
            
        Returns:
            bool: 是否检测到分娩
        """
        if not pigs:
            return False
        
        # 计算猪的运动情况
        # 这里使用简单的运动检测方法，实际应用中可以使用更复杂的算法
        motion_score = 0
        
        # 检测是否有异常运动
        if motion_score > config.FARROWING_MOTION_THRESHOLD:
            if self.farrowing_start_time is None:
                self.farrowing_start_time = cv2.getTickCount()
            else:
                # 计算持续时间
                current_time = cv2.getTickCount()
                duration = (current_time - self.farrowing_start_time) / cv2.getTickFrequency()
                
                if duration > config.FARROWING_DURATION_THRESHOLD:
                    return True
        else:
            # 重置计时器
            self.farrowing_start_time = None
        
        return False
    
    def detect_crush(self, frame, pigs, piglets):
        """
        检测母猪压猪行为
        
        Args:
            frame: 视频帧
            pigs: 检测到的母猪
            piglets: 检测到的小猪
            
        Returns:
            bool: 是否检测到压猪
        """
        if not pigs or not piglets:
            return False
        
        # 检查每头小猪是否被母猪压住
        for piglet in piglets:
            piglet_x = (piglet['bbox'][0] + piglet['bbox'][2]) // 2
            piglet_y = (piglet['bbox'][1] + piglet['bbox'][3]) // 2
            
            for pig in pigs:
                pig_x1, pig_y1, pig_x2, pig_y2 = pig['bbox']
                
                # 检查小猪是否在母猪的 bounding box 内
                if pig_x1 < piglet_x < pig_x2 and pig_y1 < piglet_y < pig_y2:
                    if self.crush_start_time is None:
                        self.crush_start_time = cv2.getTickCount()
                    else:
                        # 计算持续时间
                        current_time = cv2.getTickCount()
                        duration = (current_time - self.crush_start_time) / cv2.getTickFrequency()
                        
                        if duration > config.CRUSH_DURATION_THRESHOLD:
                            return True
                    break
                else:
                    # 重置计时器
                    self.crush_start_time = None
        
        return False
    
    def analyze(self, frame, detections):
        """
        分析视频帧中的行为
        
        Args:
            frame: 视频帧
            detections: 检测结果
            
        Returns:
            dict: 分析结果
        """
        # 分离母猪和小猪
        pigs = []
        piglets = []
        
        for det in detections:
            # 简单的大小判断：小猪通常比母猪小
            bbox = det['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            if area > 5000:  # 假设面积大于5000的是母猪
                pigs.append(det)
            else:
                piglets.append(det)
        
        # 检测分娩
        farrowing_detected = self.detect_farrowing(frame, pigs)
        
        # 检测压猪
        crush_detected = self.detect_crush(frame, pigs, piglets)
        
        return {
            'pigs': pigs,
            'piglets': piglets,
            'farrowing_detected': farrowing_detected,
            'crush_detected': crush_detected
        }
