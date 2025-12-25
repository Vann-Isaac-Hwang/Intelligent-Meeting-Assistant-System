import os
import numpy as np
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment

class AudioEnhancer:
    def __init__(self, sr=16000):
        self.sr = sr

    def reduce_noise(self, audio_data):
        """
        [新增] 直接对 numpy 数组进行降噪处理，符合项目缓解复杂环境噪声的策略 [cite: 31, 32]
        """
        # 使用 noisereduce 库处理平稳噪声 
        reduced_audio = nr.reduce_noise(y=audio_data, sr=self.sr)
        return reduced_audio
    
    def process_file(self, input_path, output_path):
        """
        处理离线文件：读取 -> 降噪 -> 增益增强 -> 保存
        """
        # 1. 读取音频 (符合 FR1)
        audio, sr = sf.read(input_path)
        
        # 确保是单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # 2. 降噪处理 (符合 Mitigation 32)
        print(f"正在为 {input_path} 进行降噪...")
        reduced_audio = nr.reduce_noise(y=audio, sr=self.sr)

        # 3. 增益增强 (提高后续 ASR 的准确度)
        # 将数据转换回 int16 以便 pydub 处理音量平衡
        audio_int16 = (reduced_audio * 32767).astype(np.int16)
        segment = AudioSegment(audio_int16.tobytes(), frame_rate=self.sr, sample_width=2, channels=1)
        
        # 归一化到 -20dBFS
        normalized_segment = segment.normalize(headroom=0.1)
        
        # 4. 保存结果到 resource 文件夹
        enhanced_audio = np.array(normalized_segment.get_array_of_samples()).astype(np.float32) / 32767.0
        sf.write(output_path, enhanced_audio, self.sr)
        print(f"处理完成！已保存至: {output_path}")