import numpy as np
import webrtcvad

class VADHandler:
    def __init__(self, aggressiveness=3, sr=16000):
        # 灵敏度 0-3，3表示对静音最敏感，仅保留最清晰的人声 
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sr = sr

    def extract_speech(self, audio_np):
        """
        输入降噪后的 numpy 数组，返回过滤掉静音后的音频
        """
        # 将 float32 转换为 16-bit PCM 
        pcm_data = (audio_np * 32767).astype(np.int16).tobytes()
        
        # VAD 窗口通常为 30ms [cite: 32]
        frame_duration = 30 
        frame_size = int(self.sr * frame_duration / 1000) * 2 # 2 bytes per sample
        
        voiced_frames = []
        for i in range(0, len(pcm_data) - frame_size, frame_size):
            frame = pcm_data[i : i + frame_size]
            if self.vad.is_speech(frame, self.sr):
                voiced_frames.append(audio_np[i // 2 : (i + frame_size) // 2])
        
        return np.concatenate(voiced_frames) if voiced_frames else np.array([])