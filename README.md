# 母猪分娩与压猪检测报警系统

## 项目简介

本项目是一个基于计算机视觉技术的智能监控系统，用于实时监测猪舍内母猪的分娩情况和压猪情况，当检测到异常时及时触发报警，帮助养殖人员及时发现和处理问题，提高养殖效率和小猪存活率。

## 功能特性

- **实时监控**：24小时不间断监控猪舍情况
- **目标检测**：使用YOLOv8检测视频中的母猪和小猪
- **母猪/小猪分类**：基于体型大小启发式分类母猪和小猪
- **多目标跟踪**：基于IOU的多目标跟踪算法，持续追踪每个个体
- **分娩检测**：使用Farneback光流法检测母猪运动，结合姿态变化分析
- **压猪检测**：基于位置关系、静止状态、面积比例多维度判断
- **自动报警**：当检测到异常时，通过声音报警
- **可视化显示**：实时显示监控画面和检测结果
- **日志记录**：记录系统运行状态和检测结果

## 技术栈

| 组件 | 技术 |
|------|------|
| **编程语言** | Python 3.7+ |
| **目标检测** | YOLOv8 (ultralytics) |
| **运动检测** | Farneback光流法 (OpenCV) |
| **目标跟踪** | 自定义IOU匹配跟踪器 |
| **深度学习框架** | PyTorch |
| **声音报警** | pyttsx3 |

## 系统要求

- Python 3.7+
- OpenCV 4.8.0+
- PyTorch 2.0+
- ultralytics (YOLOv8)
- pyttsx3 2.90+

## 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/chensongpoixs/pig_monitoring_system.git
   cd pig_monitoring_system
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置系统**
   编辑 `config/config.py` 文件，根据实际情况修改配置参数：
   - `VIDEO_SOURCE`：设置为摄像头ID或视频文件路径
   - `YOLO_MODEL`：YOLOv8模型路径 (默认: yolov8n.pt)
   - `PIG_AREA_THRESHOLD`：母猪/小猪分类面积阈值
   - `FARROWING_MOTION_THRESHOLD`：分娩检测运动阈值
   - `CRUSH_DURATION_THRESHOLD`：压猪检测持续时间阈值
   - 其他参数可根据实际情况调整

## 使用方法

1. **运行系统**
   ```bash
   python main.py
   ```

2. **操作说明**
   - 系统启动后会自动开始监控
   - 当检测到母猪分娩或压猪情况时，会触发声音报警
   - 按下 `q` 键退出系统

3. **查看日志**
   系统运行过程中，日志会记录在 `pig_monitoring.log` 文件中

## 项目结构

```
pig_monitoring_system/
├── config/
│   ├── __init__.py
│   └── config.py              # 配置文件
├── src/
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── object_detector.py # YOLOv8目标检测
│   │   └── tracker.py         # 多目标跟踪
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── behavior_analyzer.py # 行为分析（光流法）
│   ├── alarm/
│   │   ├── __init__.py
│   │   └── alarm_system.py    # 报警系统
│   └── utils/
│       ├── __init__.py
│       └── logger.py          # 日志模块
├── main.py                    # 主程序
├── requirements.txt           # 依赖库
├── PROJECT_PLAN.md            # 项目计划
├── PROJECT_ANALYSIS.md        # 项目分析
└── README.md                  # 说明文档
```

## 技术原理

### 1. 目标检测 (YOLOv8)
使用YOLOv8模型检测视频中的猪只，COCO数据集中猪的类别ID为16。检测后基于边界框面积进行启发式分类：
- 面积 > `PIG_AREA_THRESHOLD`：母猪
- 面积 <= `PIG_AREA_THRESHOLD`：小猪

### 2. 多目标跟踪
基于IOU的贪心匹配算法：
- 计算检测框与跟踪轨迹的IOU矩阵
- 按IOU从大到小贪心匹配
- 未匹配的检测创建新轨迹
- 未匹配的轨迹标记为丢失，超过阈值后移除

### 3. 分娩检测
使用Farneback光流法计算运动强度，结合以下指标综合评分：
- **运动方差**：运动强度的变化程度
- **姿态变化**：母猪高度变化频率
- **运动强度**：平均运动幅度

### 4. 压猪检测
多维度判断：
- 小猪中心点是否在母猪边界框内
- 小猪是否长时间静止
- 母猪与小猪面积比例是否合理

## 核心算法流程

```
视频帧 → YOLOv8检测 → 母猪/小猪分类 → 多目标跟踪 → 行为分析 → 报警触发
                                    ↓
                              光流法运动检测
                                    ↓
                              姿态变化分析
                                    ↓
                              位置关系判断
```

## 配置参数说明

### 目标检测参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `YOLO_MODEL` | yolov8n.pt | YOLOv8模型路径 |
| `CONFIDENCE_THRESHOLD` | 0.5 | 检测置信度阈值 |
| `NMS_THRESHOLD` | 0.4 | NMS阈值 |
| `PIG_AREA_THRESHOLD` | 5000 | 母猪/小猪分类面积阈值 |

### 光流法参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FARNCBACK_PYR_SCALE` | 0.5 | 金字塔缩放比例 |
| `FARNCBACK_LEVELS` | 3 | 金字塔层数 |
| `FARNCBACK_WINSIZE` | 15 | 窗口大小 |

### 跟踪参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TRACK_BUFFER` | 30 | 跟踪缓冲帧数 |
| `MATCH_THRESHOLD` | 0.3 | IOU匹配阈值 |

### 行为分析参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FARROWING_MOTION_THRESHOLD` | 5.0 | 分娩运动阈值 |
| `FARROWING_SCORE_THRESHOLD` | 0.6 | 分娩评分阈值 |
| `CRUSH_DURATION_THRESHOLD` | 3.0 | 压猪持续时间阈值 |

## 注意事项

1. **摄像头位置**：建议将摄像头安装在猪舍上方，确保能清晰拍摄到整个猪舍
2. **光线条件**：确保猪舍内光线充足，避免过暗或过亮的环境
3. **模型精度**：YOLOv8模型在默认情况下可能无法区分母猪和小猪，需要根据实际情况调整面积阈值
4. **误报处理**：系统可能会产生误报，需要根据实际情况调整阈值参数
5. **GPU加速**：如有NVIDIA GPU，设置`YOLO_DEVICE = 'cuda:0'`可大幅提升检测速度

## 后续改进

1. **模型优化**：使用专门针对猪只的数据集训练模型，提高检测精度
2. **ByteTrack集成**：集成更先进的ByteTrack跟踪算法
3. **远程监控**：添加网络传输功能，实现远程监控和报警
4. **智能预测**：基于历史数据预测分娩时间，提前做好准备
5. **多猪舍管理**：支持同时监控多个猪舍
6. **数据分析**：收集数据进行分析，优化养殖管理

## 许可证

本项目采用 MIT 许可证

## 联系方式

如有问题或建议，请联系项目维护者
