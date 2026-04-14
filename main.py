import cv2
import time
from src.detection.object_detector import ObjectDetector
from src.analysis.behavior_analyzer import BehaviorAnalyzer
from src.alarm.alarm_system import AlarmSystem
from src.utils.logger import logger
from config.config import config

class PigMonitoringSystem:
    def __init__(self):
        # 初始化各模块
        self.detector = ObjectDetector()
        self.analyzer = BehaviorAnalyzer()
        self.alarm = AlarmSystem()
        
        # 初始化视频捕获
        self.cap = cv2.VideoCapture(config.VIDEO_SOURCE)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {config.VIDEO_SOURCE}")
            raise Exception(f"Failed to open video source: {config.VIDEO_SOURCE}")
        
        # 设置显示窗口
        cv2.namedWindow('Pig Monitoring System', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pig Monitoring System', config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
        
        logger.info("Pig Monitoring System initialized successfully")
    
    def run(self):
        """
        运行监控系统
        """
        try:
            while True:
                # 读取视频帧
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame from video source")
                    break
                
                # 检测目标
                detections = self.detector.detect(frame)
                
                # 分析行为
                analysis_result = self.analyzer.analyze(frame, detections)
                
                # 触发报警
                if analysis_result['farrowing_detected']:
                    self.alarm.trigger_alarm('farrowing')
                elif analysis_result['crush_detected']:
                    self.alarm.trigger_alarm('crush')
                else:
                    self.alarm.reset_alarm()
                
                # 可视化结果
                frame = self.detector.visualize(frame, detections)
                
                # 显示状态
                status_text = "Status: Normal"
                status_color = (0, 255, 0)  # 绿色
                
                if analysis_result['farrowing_detected']:
                    status_text = "Status: Farrowing Detected"
                    status_color = (0, 255, 255)  # 黄色
                elif analysis_result['crush_detected']:
                    status_text = "Status: Crush Detected"
                    status_color = (0, 0, 255)  # 红色
                
                cv2.putText(frame, status_text, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
                # 显示猪和小猪的数量
                pig_count = len(analysis_result['pigs'])
                piglet_count = len(analysis_result['piglets'])
                cv2.putText(frame, f'Pigs: {pig_count}', (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'Piglets: {piglet_count}', (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 显示帧
                cv2.imshow('Pig Monitoring System', frame)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User exited the system")
                    break
                
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            # 清理资源
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Pig Monitoring System stopped")

if __name__ == "__main__":
    # 创建并运行系统
    system = PigMonitoringSystem()
    system.run()
