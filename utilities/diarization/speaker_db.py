import sqlite3
import numpy as np
import os
import datetime
import io

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
        self._load_model()

    def _init_db(self):
        """初始化 SQLite 数据库 (包含自动迁移逻辑)"""
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
        
        # 自动迁移: 检查是否缺少 title 字段
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
        self.classifier = None
        if HAS_MODEL:
            try:
                self.classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="resource/models/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
                )
            except Exception as e:
                print(f"Error loading speaker model: {e}")

    # --- [核心修改] 支持从内存(Numpy Array)提取 ---
    def extract_embedding_from_memory(self, audio_np):
        """
        从 Numpy 数组提取声纹。
        audio_np: shape (N,) 采样率应为 16000
        """
        if not self.classifier:
            return np.random.rand(192).astype(np.float32)
        
        # 转换为 Tensor: (batch=1, time)
        signal = torch.from_numpy(audio_np).float().unsqueeze(0)
        
        # 移至 GPU (如果模型在 GPU 上)
        if self.classifier.mods.embedding_model.training: # 简单判断设备
             pass # SpeechBrain 处理设备比较自动，通常不需要手动 move，除非明确指定
        
        # SpeechBrain 要求输入是 Tensor
        embedding = self.classifier.encode_batch(signal)
        return embedding.squeeze().cpu().numpy()

    def extract_embedding(self, audio_path):
        """从文件提取 (兼容旧代码)"""
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

    # --- [核心修改] 返回 (name, title) ---
    def match_speaker(self, input_embedding, threshold=0.25):
        """
        匹配数据库声纹。
        返回: (name, title) 或 ("Unknown", "")
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # [修改] 同时查询 name 和 title
        cursor.execute("SELECT name, title, embedding FROM speakers")
        rows = cursor.fetchall()
        conn.close()

        best_score = -1.0
        best_info = ("Unknown", "") # name, title

        norm_input = np.linalg.norm(input_embedding)
        
        for name, title, blob in rows:
            target_emb = np.frombuffer(blob, dtype=np.float32)
            score = np.dot(input_embedding, target_emb) / (norm_input * np.linalg.norm(target_emb))
            
            if score > best_score:
                best_score = score
                # 确保 title 不是 None
                safe_title = title if title else ""
                best_info = (name, safe_title)
        
        if best_score > threshold:
            return best_info
        
        return ("Unknown", "")