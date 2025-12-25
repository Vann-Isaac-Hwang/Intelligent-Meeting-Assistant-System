import sqlite3
import numpy as np
import os
import datetime
import io
import gc

# 尝试导入声纹提取模型 (SpeechBrain)
try:
    from speechbrain.inference.speaker import EncoderClassifier
    import torch
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    print("[Warning] SpeechBrain not found. Voiceprint extraction will be simulated.")

class SpeakerDB:
    def __init__(self, db_path="resource/speakers.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
        self.classifier = None
        self._load_model() # 初始化时加载，保证 UI 响应快

    def _init_db(self):
        """初始化 SQLite 数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS speakers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                title TEXT,
                embedding BLOB NOT NULL,
                created_at TEXT
            )
        ''')
        # 自动迁移
        cursor.execute("PRAGMA table_info(speakers)")
        columns = [info[1] for info in cursor.fetchall()]
        if "title" not in columns:
            try:
                cursor.execute("ALTER TABLE speakers ADD COLUMN title TEXT")
                conn.commit()
            except Exception as e:
                print(f"[Error] Migration failed: {e}")
        conn.commit()
        conn.close()

    def _load_model(self):
        """加载声纹提取模型"""
        if self.classifier is not None:
            return # 避免重复加载

        if HAS_MODEL:
            try:
                # print("[SpeakerDB] Loading model to GPU...")
                self.classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="resource/models/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
                )
            except Exception as e:
                print(f"Error loading speaker model: {e}")

    def unload_model(self):
        """[新增] 卸载模型并释放显存"""
        if self.classifier:
            del self.classifier
            self.classifier = None
        
        if HAS_MODEL:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # print("[SpeakerDB] Model unloaded and GPU memory cleared.")

    def _ensure_model(self):
        """确保模型已加载（懒加载机制）"""
        if self.classifier is None:
            self._load_model()

    def extract_embedding_from_memory(self, audio_np):
        self._ensure_model() # 自动重载
        if not self.classifier:
            return np.random.rand(192).astype(np.float32)
        
        signal = torch.from_numpy(audio_np).float().unsqueeze(0)
        # 如果模型在 GPU，确保输入也在 GPU (SpeechBrain通常自动处理，但显式转换更稳)
        if torch.cuda.is_available():
            signal = signal.to("cuda")
            
        embedding = self.classifier.encode_batch(signal)
        return embedding.squeeze().cpu().numpy()

    def extract_embedding(self, audio_path):
        self._ensure_model()
        if not self.classifier:
            return np.random.rand(192).astype(np.float32)
        signal = self.classifier.load_audio(audio_path)
        embedding = self.classifier.encode_batch(signal)
        return embedding.squeeze().cpu().numpy()

    def add_speaker(self, name, title, audio_path):
        try:
            vector = self.extract_embedding(audio_path)
            vector_bytes = vector.tobytes()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO speakers (name, title, embedding, created_at) VALUES (?, ?, ?, ?)",
                           (name, title, vector_bytes, datetime.datetime.now().isoformat()))
            conn.commit()
            conn.close()
            return True, "Success"
        except sqlite3.IntegrityError:
            return False, f"Name '{name}' already exists."
        except Exception as e:
            return False, str(e)

    def delete_speaker(self, name):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM speakers WHERE name=?", (name,))
        conn.commit()
        conn.close()

    def update_speaker_info(self, current_name, new_name=None, new_title=None):
        if not new_name and not new_title: return False, "Nothing to update."
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            if new_name and new_title:
                cursor.execute("UPDATE speakers SET name=?, title=? WHERE name=?", (new_name, new_title, current_name))
            elif new_name:
                cursor.execute("UPDATE speakers SET name=? WHERE name=?", (new_name, current_name))
            elif new_title:
                cursor.execute("UPDATE speakers SET title=? WHERE name=?", (new_title, current_name))
            conn.commit()
            return True, "Success"
        except sqlite3.IntegrityError:
            return False, f"Name '{new_name}' already exists."
        except Exception as e:
            return False, str(e)
        finally:
            conn.close()

    def get_all_speakers(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, title, created_at FROM speakers ORDER BY created_at DESC")
        data = cursor.fetchall()
        conn.close()
        return data

    def match_speaker(self, input_embedding, threshold=0.25):
        # 匹配不需要加载模型，纯 CPU 计算
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, title, embedding FROM speakers")
        rows = cursor.fetchall()
        conn.close()

        best_score = -1.0
        best_info = ("Unknown", "")

        norm_input = np.linalg.norm(input_embedding)
        
        for name, title, blob in rows:
            target_emb = np.frombuffer(blob, dtype=np.float32)
            score = np.dot(input_embedding, target_emb) / (norm_input * np.linalg.norm(target_emb))
            if score > best_score:
                best_score = score
                safe_title = title if title else ""
                best_info = (name, safe_title)
        
        if best_score > threshold:
            return best_info
        return ("Unknown", "")