import time
import pyttsx3
import logging
from config.config import config

class AlarmSystem:
    def __init__(self):
        # 初始化语音引擎
        if config.ALARM_SOUND:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)  # 设置语速
                self.engine.setProperty('volume', 1.0)  # 设置音量
            except Exception as e:
                logging.error(f"Failed to initialize text-to-speech engine: {e}")
                self.engine = None
        else:
            self.engine = None
        
        # 初始化报警状态
        self.last_alarm_time = 0
        self.alarm_active = False
    
    def sound_alarm(self, message):
        """
        触发声音报警
        
        Args:
            message: 报警消息
        """
        if self.engine:
            try:
                self.engine.say(message)
                self.engine.runAndWait()
            except Exception as e:
                logging.error(f"Failed to play alarm sound: {e}")
        
        # 记录报警
        logging.warning(f"ALARM: {message}")
    
    def trigger_alarm(self, alarm_type):
        """
        触发报警
        
        Args:
            alarm_type: 报警类型，'farrowing' 或 'crush'
        """
        # 检查是否在报警冷却期
        current_time = time.time()
        if current_time - self.last_alarm_time < config.ALARM_DURATION:
            return
        
        if alarm_type == 'farrowing':
            message = "注意，母猪开始分娩"
        elif alarm_type == 'crush':
            message = "危险，母猪压到小猪"
        else:
            message = "检测到异常情况"
        
        # 触发报警
        self.sound_alarm(message)
        
        # 更新报警时间
        self.last_alarm_time = current_time
        self.alarm_active = True
    
    def reset_alarm(self):
        """
        重置报警状态
        """
        self.alarm_active = False
