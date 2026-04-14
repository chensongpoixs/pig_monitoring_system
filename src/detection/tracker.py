# ***********************************************************************************************
# created:          2026-04-15
#
# author:           chensong
#
# purpose:          多目标跟踪模块
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
import numpy as np
from collections import deque
from config.config import config

class TrackState:
    """跟踪状态"""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class Track:
    """单个目标跟踪轨迹"""
    next_id = 0

    def __init__(self, detection):
        self.track_id = Track.next_id
        Track.next_id += 1

        self.detection = detection
        self.bbox = detection['bbox']
        self.class_name = detection['class_name']
        self.confidence = detection['confidence']

        self.state = TrackState.New
        self.history = deque(maxlen=30)
        self.stationary_frames = 0
        self.lost_frames = 0

        # 运动分析
        self.velocity = [0, 0]
        self.motion_intensity = 0.0

    def update(self, detection):
        """更新跟踪"""
        self.detection = detection
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']

        # 计算速度
        if len(self.history) > 0:
            prev_center = self.history[-1]['center']
            curr_center = detection['center']
            self.velocity = [
                curr_center[0] - prev_center[0],
                curr_center[1] - prev_center[1]
            ]
            self.motion_intensity = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)

        # 更新历史
        self.history.append({
            'bbox': self.bbox,
            'center': detection['center'],
            'velocity': self.velocity.copy(),
            'motion_intensity': self.motion_intensity
        })

        # 检查是否静止
        if self.motion_intensity < 1.0:
            self.stationary_frames += 1
        else:
            self.stationary_frames = 0

        self.state = TrackState.Tracked
        self.lost_frames = 0

    def mark_lost(self):
        """标记为丢失"""
        self.lost_frames += 1
        if self.lost_frames > config.TRACK_BUFFER:
            self.state = TrackState.Removed
        elif self.lost_frames > 0:
            self.state = TrackState.Lost

    def get_recent_motion(self, frames=30):
        """获取最近N帧的运动历史"""
        if len(self.history) == 0:
            return []
        return list(self.history)[-frames:]

class MultiObjectTracker:
    """多目标跟踪器"""

    def __init__(self):
        self.tracks = []
        self.next_track_id = 0
        self.frame_count = 0

    def update(self, detections):
        """
        更新跟踪器

        Args:
            detections: 当前帧的检测结果

        Returns:
            list: 更新后的跟踪轨迹列表
        """
        self.frame_count += 1

        if len(detections) == 0:
            for track in self.tracks:
                track.mark_lost()
            self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]
            return self.tracks

        matched_tracks, unmatched_detections = self._match(detections)

        matched_track_ids = set()
        for track_id, det_idx in matched_tracks:
            self.tracks[track_id].update(detections[det_idx])
            matched_track_ids.add(track_id)

        for det_idx in unmatched_detections:
            new_track = Track(detections[det_idx])
            self.tracks.append(new_track)

        for track in self.tracks:
            if track.track_id not in matched_track_ids:
                track.mark_lost()

        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        return self.tracks

    def _match(self, detections):
        """
        匹配检测和跟踪

        Args:
            detections: 检测结果

        Returns:
            tuple: (matched_pairs, unmatched_detection_indices)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections)))

        # 计算IOU矩阵
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, det['bbox'])

        # 贪心匹配
        matched = []
        unmatched_det = list(range(len(detections)))
        used_det = set()
        used_trk = set()

        # 按IOU从大到小排序
        indices = np.argsort(iou_matrix.flatten())[::-1]
        for idx in indices:
            i = idx // len(detections)
            j = idx % len(detections)

            if i in used_trk or j in used_det:
                continue

            if iou_matrix[i, j] < 0.3:
                break

            matched.append((i, j))
            used_trk.add(i)
            used_det.add(j)

        unmatched_det = [j for j in range(len(detections)) if j not in used_det]

        return matched, unmatched_det

    def _calculate_iou(self, bbox1, bbox2):
        """
        计算两个边界框的IOU

        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]

        Returns:
            float: IOU值
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # 计算并集
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def get_active_tracks(self):
        """获取活跃的跟踪轨迹"""
        return [t for t in self.tracks if t.state == TrackState.Tracked]

    def get_sow_tracks(self):
        """获取母猪跟踪轨迹"""
        return [t for t in self.tracks if t.state == TrackState.Tracked and t.class_name == 'sow']

    def get_piglet_tracks(self):
        """获取小猪跟踪轨迹"""
        return [t for t in self.tracks if t.state == TrackState.Tracked and t.class_name == 'piglet']
