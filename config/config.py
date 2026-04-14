# 系统配置文件

class Config:
    # 视频配置
    VIDEO_SOURCE = 0  # 摄像头ID，0表示默认摄像头
    # VIDEO_SOURCE = 'path/to/video.mp4'  # 或者使用视频文件
    
    # 目标检测配置
    CONFIDENCE_THRESHOLD = 0.5  # 目标检测置信度阈值
    NMS_THRESHOLD = 0.4  # 非极大值抑制阈值
    
    # 行为分析配置
    # 分娩检测参数
    FARROWING_MOTION_THRESHOLD = 3.0  # 分娩时的运动阈值
    FARROWING_DURATION_THRESHOLD = 60  # 持续时间阈值（秒）
    
    # 压猪检测参数
    CRUSH_DISTANCE_THRESHOLD = 50  # 母猪与小猪的距离阈值（像素）
    CRUSH_DURATION_THRESHOLD = 5  # 持续时间阈值（秒）
    
    # 报警配置
    ALARM_SOUND = True  # 是否开启声音报警
    ALARM_DURATION = 10  # 报警持续时间（秒）
    
    # 日志配置
    LOG_FILE = 'pig_monitoring.log'  # 日志文件路径
    
    # 显示配置
    DISPLAY_WIDTH = 1280  # 显示宽度
    DISPLAY_HEIGHT = 720  # 显示高度
    
    # 模型配置
    MODEL_PATH = 'models/yolov5/yolov5s.pt'  # 预训练模型路径
    
    # 类别配置
    CLASSES = ['pig', 'piglet']  # 检测类别

# 创建配置实例
config = Config()
