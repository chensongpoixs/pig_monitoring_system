# ***********************************************************************************************
# created:          2026-04-15
#
# author:           chensong
#
# purpose:          系统配置文件
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

class Config:
    # 视频配置
    VIDEO_SOURCE = 0
    VIDEO_SOURCE = 'fftest_pig_motion.mp4'  # 替换为实际视频路径;
    VIDEO_WIDTH = 1280
    VIDEO_HEIGHT = 720
    FPS = 30

    # 目标检测配置
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4

    # YOLOv8模型配置
    YOLO_MODEL = 'yolo11n.pt'  # 升级为 YOLO11，性能优于 YOLOv8/v9/v10
    YOLO_DEVICE = 'cuda:0'  # 使用 GPU 加速，CPU 环境下改为'cpu'
    
    # RT-DETR Transformer 模型配置（推荐用于高精度场景）
    RTDETR_ENABLED = False  # 是否启用 RT-DETR
    RTDETR_MODEL = 'rtdetr-l.pt'  # 模型选择：rtdetr-s/l/x (s=快，l=平衡，x=精度)
    RTDETR_IMG_SIZE = 640  # 输入图像大小

    # 猪只分类配置（基于体型大小）
    PIG_AREA_THRESHOLD = 5000

    # 行为分析配置
    # 光流法配置
    FARNCBACK_PYR_SCALE = 0.5
    FARNCBACK_LEVELS = 3
    FARNCBACK_WINSIZE = 15
    FARNCBACK_ITERATIONS = 3
    FARNCBACK_POLY_N = 5
    FARNCBACK_POLY_SIGMA = 1.2

    # 分娩检测参数
    FARROWING_MOTION_THRESHOLD = 3.0
    FARROWING_DURATION_THRESHOLD = 60
    FARROWING_POSTURE_CHANGE_THRESHOLD = 5
    FARROWING_SCORE_THRESHOLD = 0.7

    # 压猪检测参数
    CRUSH_DISTANCE_THRESHOLD = 50
    CRUSH_DURATION_THRESHOLD = 5
    CRUSH_STATIONARY_FRAMES = 10
    CRUSH_AREA_RATIO = 5.0

    # 跟踪算法配置
    TRACKER_TYPE = 'bytetrack'
    TRACK_BUFFER = 30
    TRACK_THRESHOLD = 0.5

    # 报警配置
    ALARM_SOUND = True
    ALARM_DURATION = 10
    ALARM_COOLDOWN = 60

    # 日志配置
    LOG_FILE = 'pig_monitoring.log'
    LOG_LEVEL = 'INFO'

    # 显示配置
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720
    DISPLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX

    # 记录配置
    RECORD_ENABLED = False
    RECORD_PATH = 'records'
    RECORD_BUFFER_SIZE = 30

    # 数据库配置
    DB_ENABLED = False
    DB_PATH = 'detections.db'

    # 远程报警配置
    REMOTE_ALARM_ENABLED = False
    REMOTE_WEBHOOK_URL = ''

# 全局配置实例
config = Config()
