import numpy as np
from .speaker_db import SpeakerDB

class SpeakerEngine:
    def __init__(self):
        # 实例化 DB 时会自动加载模型 (init -> load_model)
        self.db = SpeakerDB()

    def diarize(self, audio_np, sr=16000, window_sec=1.5, step_sec=0.75):
        """
        对音频进行滑窗识别，并在结束后释放显存。
        """
        if len(audio_np) == 0:
            self.db.unload_model() # 安全起见
            return []

        window_samples = int(window_sec * sr)
        step_samples = int(step_sec * sr)
        total_samples = len(audio_np)
        
        segments = []
        
        try:
            # 1. 滑动窗口遍历
            for i in range(0, total_samples - window_samples + 1, step_samples):
                chunk = audio_np[i : i + window_samples]
                
                # 提取声纹 (如果模型被卸载，这里会自动重载)
                embedding = self.db.extract_embedding_from_memory(chunk)
                
                # 数据库匹配 (纯 CPU)
                name, title = self.db.match_speaker(embedding, threshold=0.30)
                
                if name != "Unknown":
                    display_name = f"{name} ({title})" if title else name
                else:
                    display_name = "Unknown"
                
                start_t = i / sr
                end_t = (i + window_samples) / sr
                
                segments.append({
                    "start": float(f"{start_t:.2f}"),
                    "end": float(f"{end_t:.2f}"),
                    "speaker": display_name
                })
        finally:
            # [关键] 任务完成或出错后，立即卸载模型，释放 GPU
            # print("[SpeakerEngine] Task done, unloading model...")
            self.db.unload_model()

        # 2. 合并连续的相同说话人
        if not segments:
            return []
            
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            if next_seg['speaker'] == current['speaker']:
                current['end'] = next_seg['end']
            else:
                merged.append(current)
                current = next_seg
        merged.append(current)
        
        return merged