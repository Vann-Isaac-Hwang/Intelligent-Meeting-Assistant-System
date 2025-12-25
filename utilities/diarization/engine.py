import numpy as np
# [关键] 导入数据库单例 (在 main.py 中已经初始化，这里重新实例化引用同一个DB文件即可)
# 或者更好的方式是 main.py 传进来，但为了接口兼容，我们这里直接实例化
from .speaker_db import SpeakerDB

class SpeakerEngine:
    def __init__(self):
        # 直接使用 SpeakerDB，它内部包含了模型 (self.classifier)
        # 这样避免了双重加载模型，节省显存
        self.db = SpeakerDB()

    def diarize(self, audio_np, sr=16000, window_sec=1.5, step_sec=0.75):
        """
        对音频进行滑窗识别，并利用数据库匹配说话人信息。
        
        Args:
            audio_np: 音频数据 (numpy array)
            sr: 采样率
            window_sec: 窗口大小
            step_sec: 步长
            
        Returns:
            timeline: List[Dict] -> [{'start':0, 'end':1.5, 'speaker':'Name (Title)'}, ...]
        """
        if len(audio_np) == 0:
            return []

        # 计算窗口参数
        window_samples = int(window_sec * sr)
        step_samples = int(step_sec * sr)
        total_samples = len(audio_np)
        
        segments = []
        
        # 1. 滑动窗口遍历
        for i in range(0, total_samples - window_samples + 1, step_samples):
            # 获取当前片段
            chunk = audio_np[i : i + window_samples]
            
            # 提取声纹 (委托给 DB)
            embedding = self.db.extract_embedding_from_memory(chunk)
            
            # 数据库匹配
            name, title = self.db.match_speaker(embedding, threshold=0.30)
            
            # 格式化显示名称
            if name != "Unknown":
                if title:
                    display_name = f"{name} ({title})"
                else:
                    display_name = name
            else:
                display_name = "Unknown"
            
            start_t = i / sr
            end_t = (i + window_samples) / sr
            
            segments.append({
                "start": float(f"{start_t:.2f}"),
                "end": float(f"{end_t:.2f}"),
                "speaker": display_name
            })

        # 2. 合并连续的相同说话人 (Smoothing)
        if not segments:
            return []
            
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            # 如果说话人相同，且时间连续（允许一点点重叠或间隙），则合并
            if next_seg['speaker'] == current['speaker']:
                current['end'] = next_seg['end']
            else:
                merged.append(current)
                current = next_seg
        merged.append(current)
        
        return merged