import os
import shutil
import datetime
import time
import soundfile as sf
import numpy as np
import gc
# 导入 torch 用于清理显存
try:
    import torch
except ImportError:
    torch = None

# --- 导入底层模块 ---
from utilities.audio_processor.recorder import RealTimeAudioProvider
from utilities.audio_processor.enhancer import AudioEnhancer
from utilities.diarization.engine import SpeakerEngine
from utilities.ASR.whisper_engine import AsyncWhisperEngine

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

try:
    from utilities.meeting_extractor import meeting_extractor as llm_local
    from utilities.meeting_extractor import meeting_extractor_ol as llm_online
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

class NodeProcessor:
    def process(self, context, config, log_cb): raise NotImplementedError

class SourceProcessor(NodeProcessor):
    def __init__(self, resource_dir):
        self.raw_dir = os.path.join(resource_dir, "raw")
        self.recorder = RealTimeAudioProvider(resource_path=resource_dir)
        os.makedirs(self.raw_dir, exist_ok=True)

    def process(self, context, config, log_cb):
        path = config.get('file_path')
        if not path: path = context.get('audio_path')
        if not path: raise ValueError("Audio path not provided in config or context.")
        if not os.path.exists(path): raise FileNotFoundError(f"Audio file not found: {path}")

        target = os.path.join(self.raw_dir, os.path.basename(path))
        if os.path.abspath(path) != os.path.abspath(target):
            try: shutil.copy2(path, target)
            except: target = path
        
        context['audio_path'] = target
        context['orig_audio_path'] = target 
        
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
        except Exception as e: log_cb(f"[VAD] Error: {e}"); return context
        if len(clean_speech) == 0: log_cb("[VAD] Warning: All silence. Keeping original."); return context
        out_path = path.replace(".wav", "_vad.wav")
        sf.write(out_path, clean_speech, sr)
        context['audio_path'] = out_path
        return context

class SpeakerIDProcessor(NodeProcessor):
    def process(self, context, config, log_cb):
        log_cb("[SpeakerID] Analyzing...")
        audio, sr = sf.read(context['audio_path'])
        
        # Engine 内部现在会在 diarize 结束后自动 unload_model
        timeline = SpeakerEngine().diarize(audio, sr=sr, 
                                         window_sec=config.get('window', 1.5),
                                         step_sec=config.get('step', 0.75))
        context['timeline'] = timeline if timeline else []
        log_cb(f"[SpeakerID] Segments: {len(context['timeline'])}")
        
        # 再次确保垃圾回收
        gc.collect()
        if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
        return context

class ASRProcessor(NodeProcessor):
    def __init__(self, res_dir):
        self.temp_dir = os.path.join(res_dir, "temp_segments")
        os.makedirs(self.temp_dir, exist_ok=True)

    def process(self, context, config, log_cb):
        model_size = config.get('model', 'small')
        full_correction = config.get('full_text_correction', False)
        enhanced_opt = config.get('enhanced_audio', False)
        
        log_cb(f"[ASR] Transcribing ({model_size})...")
        engine = AsyncWhisperEngine(model_size=model_size)
        
        try:
            # --- 1. 处理 Segmented Audio ---
            input_path = context['audio_path']
            if enhanced_opt and "_clean" not in os.path.basename(input_path):
                log_cb("[ASR] Enhancing segmented input...")
                audio_data, rate = sf.read(input_path)
                if len(audio_data.shape) > 1: audio_data = np.mean(audio_data, axis=1)
                clean_audio = AudioEnhancer(sr=rate).reduce_noise(audio_data)
                clean_path = input_path.replace(".wav", "_asr_clean.wav")
                sf.write(clean_path, clean_audio, rate)
                input_path = clean_path
            
            audio, sr = sf.read(input_path)
            timeline = context.get('timeline', [])
            tasks = []

            if timeline and len(timeline) > 0:
                for i, seg in enumerate(timeline):
                    s, e = int(seg['start']*sr), int(seg['end']*sr)
                    if e <= s: continue
                    chunk_path = os.path.join(self.temp_dir, f"chunk_{i}.wav")
                    sf.write(chunk_path, audio[s:e], sr)
                    tid = engine.submit_task(chunk_path)
                    tasks.append({'id': tid, 'type': 'segment', 'info': seg, 'path': chunk_path})
            else:
                log_cb("[ASR] No timeline. Forcing full transcription.")
                tid = engine.submit_task(input_path)
                tasks.append({'id': tid, 'type': 'segment', 'info': {'start':0,'end':len(audio)/sr,'speaker':'?'}, 'path':None})

            # --- 2. 处理 Full Text Audio ---
            full_text_tid = None
            full_text_result = ""
            
            if full_correction:
                original_input = context.get('orig_audio_path', context['audio_path'])
                if enhanced_opt and "_clean" not in os.path.basename(original_input):
                    if os.path.basename(original_input) == os.path.basename(context['audio_path']) and "_asr_clean" in input_path:
                         log_cb("[ASR] Reusing enhanced audio for full text.")
                         original_input = input_path
                    else:
                        log_cb("[ASR] Enhancing full input...")
                        orig_data, orig_rate = sf.read(original_input)
                        if len(orig_data.shape) > 1: orig_data = np.mean(orig_data, axis=1)
                        clean_orig = AudioEnhancer(sr=orig_rate).reduce_noise(orig_data)
                        clean_orig_path = original_input.replace(".wav", "_full_clean.wav")
                        sf.write(clean_orig_path, clean_orig, orig_rate)
                        original_input = clean_orig_path

                log_cb(f"[ASR] + Full Correction: {os.path.basename(original_input)}")
                full_text_tid = engine.submit_task(original_input)
                tasks.append({'id': full_text_tid, 'type': 'full', 'info': None, 'path': None})

            # 3. 等待
            results_text = []
            while True:
                done = sum(1 for t in tasks if engine.get_task_status(t['id'])['status'] in ['COMPLETED', 'FAILED'])
                if done == len(tasks): break
                time.sleep(0.5)

            # 4. 收集
            for t in tasks:
                res = engine.get_task_status(t['id'])
                if res['status'] == 'COMPLETED':
                    text = res['result'].strip()
                    if t['type'] == 'segment':
                        line = f"[{t['info']['start']:.1f}s] {t['info']['speaker']}: {text}"
                        results_text.append(line)
                        log_cb(line, is_result=True)
                    elif t['type'] == 'full':
                        full_text_result = text
                if t['path']: 
                    try: os.remove(t['path'])
                    except: pass

            final_log_content = "=== Segmented Transcript (Speaker Diarized) ===\n"
            final_log_content += "\n".join(results_text)
            
            if full_correction and full_text_result:
                final_log_content += "\n\n=== Full Text Reference (Continuous Audio) ===\n"
                final_log_content += full_text_result
                log_cb("[ASR] Full text reference added.")

            context['transcript'] = final_log_content
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(os.path.dirname(self.temp_dir), "meeting_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"Log_{ts}.txt")
            with open(log_path, 'w', encoding='utf-8') as f: f.write(final_log_content)
            context['log_path'] = log_path
            
        finally:
            # [核心修改] 任务结束后强制销毁 Whisper 引擎并清理显存
            del engine
            gc.collect()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            # log_cb("[ASR] Memory cleared.")
            
        return context

class LLMProcessor(NodeProcessor):
    def process(self, context, config, log_cb):
        if not config.get('enable', False) or not HAS_LLM: return context
        log_cb("[LLM] Summarizing...")
        backend = config.get('backend', 'Local')
        Cls = llm_online.RobustMeetingExtractor if 'Online' in backend else llm_local.RobustMeetingExtractor
        data = Cls().process(context['log_path'])
        if 'error' not in data:
            report = Cls().generate_readable_report(data)
            log_cb("√ Summary Ready!", is_result=True)
            log_cb(report, is_result=True)
        else:
            log_cb(f"!!! LLM Error: {data['error']}")
        return context