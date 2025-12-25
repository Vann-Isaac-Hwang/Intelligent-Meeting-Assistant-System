import os
import shutil
import datetime
import time
import soundfile as sf
import numpy as np

# --- 导入底层模块 ---
from utilities.audio_processor.recorder import RealTimeAudioProvider
from utilities.audio_processor.enhancer import AudioEnhancer
from utilities.diarization.engine import SpeakerEngine
from utilities.ASR.whisper_engine import AsyncWhisperEngine

# VAD Check
try:
    import webrtcvad
    class WebRTCVADWrapper:
        def __init__(self, aggressiveness=3, sr=16000):
            self.vad = webrtcvad.Vad(aggressiveness)
            self.sr = sr
        def process(self, audio_np, sr=16000):
            pcm_data = (audio_np * 32767).astype(np.int16).tobytes()
            frame_ms = 30
            n_samples = int(sr * frame_ms / 1000)
            n_bytes = n_samples * 2
            voiced = []
            for i in range(0, len(pcm_data) - n_bytes, n_bytes):
                chunk = pcm_data[i:i+n_bytes]
                if self.vad.is_speech(chunk, sr):
                    idx_start = i // 2
                    idx_end = idx_start + n_samples
                    voiced.append(audio_np[idx_start:idx_end])
            return np.concatenate(voiced) if voiced else np.array([])
    AdvancedVAD = WebRTCVADWrapper
except ImportError:
    AdvancedVAD = None

# Fallback VAD
class SimpleEnergyVAD:
    def __init__(self, threshold=0.01): self.threshold = threshold
    def process(self, audio_np, sr=16000):
        frame_len = int(sr * 0.03)
        n_frames = len(audio_np) // frame_len
        speech = []
        for i in range(n_frames):
            frame = audio_np[i*frame_len : (i+1)*frame_len]
            if np.mean(frame**2) > self.threshold: speech.append(frame)
        return np.concatenate(speech) if speech else np.array([])

# LLM Check
try:
    from utilities.meeting_extractor import meeting_extractor as llm_local
    from utilities.meeting_extractor import meeting_extractor_ol as llm_online
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

# --- Base Class ---
class NodeProcessor:
    def process(self, context, config, log_cb): raise NotImplementedError

# --- Implementations ---
class SourceProcessor(NodeProcessor):
    def __init__(self, resource_dir):
        self.raw_dir = os.path.join(resource_dir, "raw")
        self.recorder = RealTimeAudioProvider(resource_path=resource_dir)
        os.makedirs(self.raw_dir, exist_ok=True)

    def process(self, context, config, log_cb):
        # [修复] 统一从 config 获取路径
        # main.py 保证了无论是录音还是文件，路径都会传到 config['file_path']
        path = config.get('file_path')
        
        if not path:
             # 双重保险：如果 config 没拿到，看看 context 有没有
             path = context.get('audio_path')
        
        if not path:
            raise ValueError("Audio path not provided in config or context.")

        # 确保文件存在
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        # 将文件统一归档到 raw 目录 (如果是外部加载的文件)
        target = os.path.join(self.raw_dir, os.path.basename(path))
        
        # 如果源文件和目标路径不一样，执行复制
        if os.path.abspath(path) != os.path.abspath(target):
            try:
                shutil.copy2(path, target)
            except shutil.SameFileError:
                pass
            except Exception as e:
                log_cb(f"[Source] Warning: Copy failed ({e}), using original.")
                target = path
        
        # 写入上下文，供后续节点使用
        context['audio_path'] = target
        
        mode_label = "Recorded" if config.get('mode') == 'mic' else "Loaded"
        log_cb(f"[Source] {mode_label}: {os.path.basename(target)}")
        
        return context

class EnhancerProcessor(NodeProcessor):
    def process(self, context, config, log_cb):
        if not config.get('enable', True): 
            log_cb("[Enhancer] Skipped")
            return context
        path = context['audio_path']
        log_cb("[Enhancer] Denoising...")
        audio, sr = sf.read(path)
        if len(audio.shape) > 1: audio = np.mean(audio, axis=1)
        clean = AudioEnhancer(sr=sr).reduce_noise(audio)
        out_path = path.replace(".wav", "_clean.wav")
        sf.write(out_path, clean, sr)
        context['audio_path'] = out_path
        return context

class VADProcessor(NodeProcessor):
    def process(self, context, config, log_cb):
        path = context['audio_path']
        agg = int(config.get('aggressiveness', 3))
        log_cb(f"[VAD] Processing (Agg={agg})...")
        audio, sr = sf.read(path)
        if len(audio.shape) > 1: audio = np.mean(audio, axis=1)

        vad = AdvancedVAD(aggressiveness=agg, sr=sr) if AdvancedVAD else SimpleEnergyVAD(0.005 * (agg + 1))
        
        try: clean_speech = vad.process(audio, sr=sr)
        except Exception as e:
            log_cb(f"[VAD] Error: {e}"); return context

        if len(clean_speech) == 0:
            log_cb("[VAD] Warning: All silence. Keeping original."); return context
        
        out_path = path.replace(".wav", "_vad.wav")
        sf.write(out_path, clean_speech, sr)
        context['audio_path'] = out_path
        return context

class SpeakerIDProcessor(NodeProcessor):
    def process(self, context, config, log_cb):
        log_cb("[SpeakerID] Analyzing...")
        audio, sr = sf.read(context['audio_path'])
        timeline = SpeakerEngine().diarize(audio, sr=sr, 
                                         window_sec=config.get('window', 1.5),
                                         step_sec=config.get('step', 0.75))
        context['timeline'] = timeline if timeline else []
        log_cb(f"[SpeakerID] Segments: {len(context['timeline'])}")
        return context

class ASRProcessor(NodeProcessor):
    def __init__(self, res_dir):
        self.temp_dir = os.path.join(res_dir, "temp_segments")
        os.makedirs(self.temp_dir, exist_ok=True)

    def process(self, context, config, log_cb):
        log_cb("[ASR] Transcribing...")
        engine = AsyncWhisperEngine(model_size=config.get('model', 'small'))
        audio, sr = sf.read(context['audio_path'])
        timeline = context.get('timeline', [])
        tasks = []

        if not timeline:
            tid = engine.submit_task(context['audio_path'])
            tasks.append({'id': tid, 'info': {'start':0,'end':0,'speaker':'?'}, 'path':None})
        else:
            for i, seg in enumerate(timeline):
                s, e = int(seg['start']*sr), int(seg['end']*sr)
                if e <= s: continue
                chunk_path = os.path.join(self.temp_dir, f"chunk_{i}.wav")
                sf.write(chunk_path, audio[s:e], sr)
                tid = engine.submit_task(chunk_path)
                tasks.append({'id': tid, 'info': seg, 'path': chunk_path})

        results = []
        while True:
            done = sum(1 for t in tasks if engine.get_task_status(t['id'])['status'] in ['COMPLETED', 'FAILED'])
            if done == len(tasks): break
            time.sleep(0.5)

        for t in tasks:
            res = engine.get_task_status(t['id'])
            if res['status'] == 'COMPLETED':
                line = f"[{t['info']['start']:.1f}s] {t['info']['speaker']}: {res['result'].strip()}"
                results.append(line)
                log_cb(line, is_result=True)
            if t['path']: 
                try: os.remove(t['path'])
                except: pass

        full_text = "\n".join(results)
        context['transcript'] = full_text
        
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(os.path.dirname(self.temp_dir), "meeting_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"Log_{ts}.txt")
        with open(log_path, 'w', encoding='utf-8') as f: f.write(full_text)
        context['log_path'] = log_path
        return context

class LLMProcessor(NodeProcessor):
    def process(self, context, config, log_cb):
        if not config.get('enable', False) or not HAS_LLM: return context
        log_cb("[LLM] Summarizing...")
        backend = config.get('backend', 'Local')
        Cls = llm_online.RobustMeetingExtractor if 'Online' in backend else llm_local.RobustMeetingExtractor
        data = Cls().process(context['log_path'])
        if 'error' not in data:
            # 报告已经在 meeting_extractor 里生成并保存了 MD 文件
            # 这里我们需要拿到生成的报告内容来显示
            # meeting_extractor_ol.py/local.py 的 save_results 返回的是 json path
            # 但我们可以通过重新生成或者直接读取 data 中的内容来显示
            
            # 由于 Cls 是临时实例化的，我们调用静态方法或重新生成以便显示
            report = Cls().generate_readable_report(data)
            
            # 显示结果
            log_cb(report, is_result=True)
        else:
            log_cb(f"!!! LLM Error: {data['error']}")
        return context