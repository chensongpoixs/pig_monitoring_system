# ***********************************************************************************************
# created:          2026-04-15
#
# author:           chensong
#
# purpose:          行为分析模块
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
from collections import deque
from config.config import config

class BehaviorAnalyzer:
    def __init__(self):
        self.prev_gray = None
        self.motion_history = deque(maxlen=30)
        self.sow_posture_history = deque(maxlen=30)
        self.piglet_count_history = deque(maxlen=30)

        self.farrowing_start_time = None
        self.farrowing_score_history = deque(maxlen=30)

        self.crush_events = {}

        self.frame_count = 0
        self.optical_flow_skip = 2

    def calculate_optical_flow(self, prev_frame, current_frame, mask=None):
        """
        使用Farneback光流法计算运动强度

        Args:
            prev_frame: 上一帧灰度图
            current_frame: 当前帧灰度图
            mask: 感兴趣区域掩码

        Returns:
            tuple: (magnitude, angle, flow)
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame,
            current_frame,
            None,
            pyr_scale=config.FARNCBACK_PYR_SCALE,
            levels=config.FARNCBACK_LEVELS,
            winsize=config.FARNCBACK_WINSIZE,
            iterations=config.FARNCBACK_ITERATIONS,
            poly_n=config.FARNCBACK_POLY_N,
            poly_sigma=config.FARNCBACK_POLY_SIGMA,
            flags=0
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        return magnitude, angle, flow

    def calculate_region_motion(self, magnitude, bbox):
        """
        计算指定区域的平均运动强度

        Args:
            magnitude: 光流幅度图
            bbox: 边界框 [x1, y1, x2, y2]

        Returns:
            float: 平均运动强度
        """
        x1, y1, x2, y2 = bbox
        roi = magnitude[y1:y2, x1:x2]

        if roi.size == 0:
            return 0.0

        return float(np.mean(roi))

    def detect_farrowing(self, sow_tracks, frame):
        """
        检测分娩行为

        指标:
        1. 母猪频繁起身和卧下（基于位置变化）
        2. 母猪运动强度异常
        3. 小猪数量变化

        Args:
            sow_tracks: 母猪跟踪轨迹列表
            frame: 当前帧

        Returns:
            dict: 分娩检测结果
        """
        self.frame_count += 1

        if len(sow_tracks) == 0:
            return {'detected': False, 'score': 0.0, 'indicators': {}}

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is not None and self.frame_count % self.optical_flow_skip == 0:
            magnitude, _, _ = self.calculate_optical_flow(self.prev_gray, current_gray)

            total_motion = 0.0
            for track in sow_tracks:
                motion = self.calculate_region_motion(magnitude, track.bbox)
                total_motion += motion

            avg_motion = total_motion / len(sow_tracks) if sow_tracks else 0.0
            self.motion_history.append(avg_motion)
            self.prev_gray = current_gray.copy()
        elif self.prev_gray is None:
            self.prev_gray = current_gray.copy()
            self.motion_history.append(0.0)

        motion_variance = np.var(list(self.motion_history)) if len(self.motion_history) > 1 else 0.0
        motion_mean = np.mean(list(self.motion_history)) if len(self.motion_history) > 0 else 0.0

        posture_changes = 0
        posture_list = list(self.sow_posture_history)
        for i in range(1, len(posture_list)):
            prev_bbox = posture_list[i-1]
            curr_bbox = posture_list[i]
            prev_height = prev_bbox[3] - prev_bbox[1]
            curr_height = curr_bbox[3] - curr_bbox[1]
            if abs(prev_height - curr_height) > 30:
                posture_changes += 1

        if len(sow_tracks) > 0:
            self.sow_posture_history.append(sow_tracks[0].bbox)

        farrowing_score = 0.0
        indicators = {
            'motion_variance': motion_variance,
            'motion_mean': motion_mean,
            'posture_changes': posture_changes,
            'motion_history_len': len(self.motion_history)
        }

        if motion_variance > 50:
            farrowing_score += 0.3
            indicators['high_motion_variance'] = True

        if posture_changes > config.FARROWING_POSTURE_CHANGE_THRESHOLD:
            farrowing_score += 0.3
            indicators['frequent_posture_change'] = True

        if motion_mean > config.FARROWING_MOTION_THRESHOLD:
            farrowing_score += 0.4
            indicators['high_motion_intensity'] = True

        self.farrowing_score_history.append(farrowing_score)

        detected = farrowing_score > config.FARROWING_SCORE_THRESHOLD

        if detected:
            if self.farrowing_start_time is None:
                self.farrowing_start_time = cv2.getTickCount()
            else:
                current_time = cv2.getTickCount()
                duration = (current_time - self.farrowing_start_time) / cv2.getTickFrequency()
                indicators['duration'] = duration

                if duration < 10:
                    detected = False
        else:
            self.farrowing_start_time = None

        return {
            'detected': detected,
            'score': farrowing_score,
            'indicators': indicators
        }

    def detect_crush(self, sow_tracks, piglet_tracks):
        """
        检测压猪行为

        多维度判断:
        1. 小猪是否在母猪边界框内
        2. 小猪是否长时间静止
        3. 小猪是否在母猪下方（基于相对位置）

        Args:
            sow_tracks: 母猪跟踪轨迹列表
            piglet_tracks: 小猪跟踪轨迹列表

        Returns:
            list: 压猪事件列表
        """
        if len(sow_tracks) == 0 or len(piglet_tracks) == 0:
            return []

        crush_events = []

        for piglet_track in piglet_tracks:
            piglet_center = piglet_track.detection['center']
            piglet_bbox = piglet_track.bbox
            piglet_area = piglet_track.detection['area']

            # 检查小猪是否在母猪边界框内
            for sow_track in sow_tracks:
                sow_bbox = sow_track.bbox
                sow_area = sow_track.detection['area']

                x1, y1, x2, y2 = sow_bbox
                px, py = piglet_center

                # 检查是否在边界框内
                is_inside = (x1 < px < x2) and (y1 < py < y2)

                # 检查面积比例（母猪应该比小猪大很多）
                area_ratio = sow_area / piglet_area if piglet_area > 0 else 0
                is_size_appropriate = area_ratio > config.CRUSH_AREA_RATIO

                # 检查小猪是否静止
                is_stationary = piglet_track.stationary_frames > config.CRUSH_STATIONARY_FRAMES

                if is_inside and is_size_appropriate and is_stationary:
                    crush_id = piglet_track.track_id
                    if crush_id not in self.crush_events:
                        self.crush_events[crush_id] = {
                            'start_time': cv2.getTickCount(),
                            'track_id': crush_id,
                            'stationary_frames': piglet_track.stationary_frames,
                            'confidence': 0.0
                        }

                    # 计算持续时间
                    current_time = cv2.getTickCount()
                    duration = (current_time - self.crush_events[crush_id]['start_time']) / cv2.getTickFrequency()

                    # 更新置信度
                    confidence = min(1.0, duration / config.CRUSH_DURATION_THRESHOLD)
                    self.crush_events[crush_id]['confidence'] = confidence

                    if duration >= config.CRUSH_DURATION_THRESHOLD:
                        crush_events.append({
                            'piglet_track_id': crush_id,
                            'duration': duration,
                            'confidence': confidence,
                            'sow_bbox': sow_bbox,
                            'piglet_bbox': piglet_bbox
                        })
                else:
                    # 如果小猪不再被压，清除事件
                    if piglet_track.track_id in self.crush_events:
                        del self.crush_events[piglet_track.track_id]

        return crush_events

    def analyze(self, frame, sow_tracks, piglet_tracks):
        """
        综合分析行为

        Args:
            frame: 当前帧
            sow_tracks: 母猪跟踪轨迹列表
            piglet_tracks: 小猪跟踪轨迹列表

        Returns:
            dict: 分析结果
        """
        # 分娩检测
        farrowing_result = self.detect_farrowing(sow_tracks, frame)

        # 压猪检测
        crush_events = self.detect_crush(sow_tracks, piglet_tracks)

        return {
            'sows': [t.detection for t in sow_tracks],
            'piglets': [t.detection for t in piglet_tracks],
            'sow_count': len(sow_tracks),
            'piglet_count': len(piglet_tracks),
            'farrowing_detected': farrowing_result['detected'],
            'farrowing_score': farrowing_result['score'],
            'farrowing_indicators': farrowing_result['indicators'],
            'crush_events': crush_events,
            'crush_detected': len(crush_events) > 0
        }
