# ***********************************************************************************************
# created:          2026-04-15
#
# author:           chensong
#
# purpose:          主程序
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
import time
import numpy as np
from src.detection.object_detector import ObjectDetector
from src.detection.tracker import MultiObjectTracker
from src.analysis.behavior_analyzer import BehaviorAnalyzer
from src.alarm.alarm_system import AlarmSystem
from src.utils.logger import logger
from config.config import config

class PigMonitoringSystem:
    def __init__(self):
        # 初始化各模块
        logger.info("Initializing Pig Monitoring System...")

        # 目标检测器
        self.detector = ObjectDetector()
        logger.info("Object detector initialized (YOLOv8)")

        # 多目标跟踪器
        self.tracker = MultiObjectTracker()
        logger.info("Multi-object tracker initialized")

        # 行为分析器
        self.analyzer = BehaviorAnalyzer()
        logger.info("Behavior analyzer initialized")

        # 报警系统
        self.alarm = AlarmSystem()
        logger.info("Alarm system initialized")

        # 初始化视频捕获
        self.cap = cv2.VideoCapture(config.VIDEO_SOURCE)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {config.VIDEO_SOURCE}")
            raise Exception(f"Failed to open video source: {config.VIDEO_SOURCE}")

        # 设置视频属性
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_HEIGHT)

        # 设置显示窗口
        cv2.namedWindow('Pig Monitoring System', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pig Monitoring System', config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)

        # 统计信息
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = time.time()

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

                self.frame_count += 1

                # 计算FPS
                if self.frame_count % 30 == 0:
                    self.fps = 30 / (time.time() - self.fps_start_time)
                    self.fps_start_time = time.time()

                # 检测目标
                detections = self.detector.detect(frame)

                # 更新跟踪器
                tracked_objects = self.tracker.update(detections)

                # 分离母猪和小猪跟踪轨迹
                sow_tracks = self.tracker.get_sow_tracks()
                piglet_tracks = self.tracker.get_piglet_tracks()

                # 分析行为
                analysis_result = self.analyzer.analyze(frame, sow_tracks, piglet_tracks)

                # 触发报警
                if analysis_result['farrowing_detected']:
                    self.alarm.trigger_alarm('farrowing')
                elif analysis_result['crush_detected']:
                    self.alarm.trigger_alarm('crush')
                else:
                    self.alarm.reset_alarm()

                # 可视化结果
                frame = self._visualize_results(frame, analysis_result)

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

    def _visualize_results(self, frame, analysis_result):
        """
        可视化分析结果

        Args:
            frame: 视频帧
            analysis_result: 分析结果

        Returns:
            ndarray: 可视化后的帧
        """
        # 可视化检测结果
        all_detections = analysis_result['sows'] + analysis_result['piglets']
        frame = self.detector.visualize(frame, all_detections)

        # 确定状态颜色和文本
        if analysis_result['farrowing_detected']:
            status_text = "Status: FARROWING DETECTED"
            status_color = (0, 255, 255)
        elif analysis_result['crush_detected']:
            status_text = "Status: CRUSH DETECTED"
            status_color = (0, 0, 255)
        else:
            status_text = "Status: Normal"
            status_color = (0, 255, 0)

        # 绘制状态信息
        cv2.putText(frame, status_text, (20, 50),
                    config.DISPLAY_FONT, 1, status_color, 2)

        # 绘制压猪事件
        if analysis_result['crush_detected']:
            for event in analysis_result['crush_events']:
                bbox = event['piglet_bbox']
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              (0, 0, 255), 3)
                cv2.putText(frame, f"Crush! ID:{event['piglet_track_id']}",
                            (bbox[0], bbox[1] - 25),
                            config.DISPLAY_FONT, 0.5, (0, 0, 255), 2)

        # 绘制统计信息
        cv2.putText(frame, f"Sows: {analysis_result['sow_count']}", (20, 85),
                    config.DISPLAY_FONT, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Piglets: {analysis_result['piglet_count']}", (20, 115),
                    config.DISPLAY_FONT, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 145),
                    config.DISPLAY_FONT, 0.7, (255, 255, 255), 2)

        # 绘制分娩评分
        if analysis_result['farrowing_score'] > 0:
            score_text = f"Farrowing Score: {analysis_result['farrowing_score']:.2f}"
            cv2.putText(frame, score_text, (20, 175),
                        config.DISPLAY_FONT, 0.7, (255, 255, 0), 2)

        return frame

if __name__ == "__main__":
    # 创建并运行系统
    system = PigMonitoringSystem()
    system.run()
