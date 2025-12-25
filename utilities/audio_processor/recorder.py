import os
import wave
import threading
import pyaudio
import webrtcvad
from datetime import datetime

class RealTimeAudioProvider:
    def __init__(self, sr=16000, chunk_ms=30, resource_path="resource"):
        self.sr = sr
        self.chunk_size = int(sr * chunk_ms / 1000)
        self.vad = webrtcvad.Vad(3)
        self.is_running = False
        self.all_frames = []
        self.resource_path = resource_path
        self.custom_filename = None

    def _record_loop(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.sr,
                        input=True, frames_per_buffer=self.chunk_size)
        
        while self.is_running:
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            self.all_frames.append(data)
                
        stream.stop_stream()
        stream.close()
        p.terminate()
        self._save_to_file()

    def _save_to_file(self):
        if not self.all_frames:
            return
            
        # --- 修改开始: 确保保存到 resource/raw 目录 ---
        raw_dir = os.path.join(self.resource_path, "raw")
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
            
        if self.custom_filename:
            file_name = self.custom_filename if self.custom_filename.endswith('.wav') else f"{self.custom_filename}.wav"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"meeting_{timestamp}.wav"
            
        full_path = os.path.join(raw_dir, file_name)
        # --- 修改结束 ---
        
        with wave.open(full_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            wf.writeframes(b''.join(self.all_frames))
        
        print(f"\n>>> 录音已妥善保存至: {full_path} ")
        self.all_frames = []
        self.custom_filename = None 

    def start(self, filename=None):
        self.is_running = True
        self.custom_filename = filename
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join()