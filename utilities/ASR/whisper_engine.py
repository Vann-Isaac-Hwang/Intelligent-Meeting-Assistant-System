import whisper
import torch
import threading
import queue
import uuid
import time
import os
import warnings

# 过滤显存警告
warnings.filterwarnings("ignore")

class AsyncWhisperEngine:
    def __init__(self, model_size="small"):
        """
        初始化 Whisper 引擎
        :param model_size: 模型大小 (tiny, base, small, medium, large)
                           RTX 4060 推荐使用 'small' 或 'medium'
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Engine] 正在加载 Whisper 模型 ({model_size}) 到 {self.device.upper()}...")
        
        # 加载模型 (耗时操作，只做一次)
        try:
            self.model = whisper.load_model(model_size, device=self.device)
            print("[Engine] 模型加载完成，后台线程准备就绪。")
        except Exception as e:
            print(f"[Engine] 模型加载失败: {e}")
            raise e

        # 任务队列
        self.task_queue = queue.Queue()
        # 任务结果/状态存储: { "task_id": { "status": str, "result": str, "error": str } }
        self.tasks = {}
        
        # 启动后台处理线程
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        """
        后台工作线程：不断从队列取任务并执行推理
        """
        while True:
            # 阻塞等待任务
            task_id, file_path = self.task_queue.get()
            
            try:
                # 更新状态为进行中
                self.tasks[task_id]["status"] = "PROCESSING"
                print(f"[Worker] 开始处理任务: {task_id} | 文件: {os.path.basename(file_path)}")

                # 执行推理 (核心耗时步骤)
                # fp16=True 在 GPU 上更快
                result = self.model.transcribe(file_path, fp16=(self.device == "cuda"))
                text = result["text"].strip()

                # 更新结果
                self.tasks[task_id]["status"] = "COMPLETED"
                self.tasks[task_id]["result"] = text
                print(f"[Worker] 任务完成: {task_id}")

            except Exception as e:
                self.tasks[task_id]["status"] = "FAILED"
                self.tasks[task_id]["error"] = str(e)
                print(f"[Worker] 任务失败: {task_id} | 原因: {e}")
            
            finally:
                # 标记队列任务完成
                self.task_queue.task_done()

    def submit_task(self, audio_file_path):
        """
        提交一个音频文件进行转录
        :return: task_id (str) 用于后续查询
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"文件未找到: {audio_file_path}")

        task_id = str(uuid.uuid4())[:8] # 生成简短 ID
        
        # 初始化任务状态
        self.tasks[task_id] = {
            "status": "QUEUED",
            "file": audio_file_path,
            "result": None,
            "error": None
        }
        
        # 放入队列
        self.task_queue.put((task_id, audio_file_path))
        return task_id

    def get_task_status(self, task_id):
        """
        查询任务状态
        :return: dict 包含 status, result, error
        """
        return self.tasks.get(task_id, None)

    def is_completed(self, task_id):
        """
        简便方法：检查是否完成
        """
        task = self.tasks.get(task_id)
        if task and task["status"] == "COMPLETED":
            return True
        return False

# --- 模拟主程序调用 (Example Usage) ---
if __name__ == "__main__":
    # 1. 实例化引擎 (这一步会加载模型，稍微花点时间)
    engine = AsyncWhisperEngine(model_size="small")

    # 2. 模拟音频文件 (这里为了演示创建一个假文件，实际请传入真实路径)
    # 在实际使用中，你直接传入你的 wav/mp3 路径即可
    import soundfile as sf
    import numpy as np
    
    test_file = "test_audio_async.wav"
    # 生成一段静音仅作测试，防止报错
    sf.write(test_file, np.random.uniform(-0.1, 0.1, 16000*2), 16000) 

    print("\n--- 主程序继续运行 ---")
    
    # 3. 提交任务 (非阻塞，立即返回 ID)
    print(">>> 提交任务...")
    t_id = engine.submit_task(test_file)
    print(f">>> 任务已提交，ID: {t_id}")

    # 4. 模拟主程序做其他事情，同时轮询状态
    start_time = time.time()
    while True:
        status_info = engine.get_task_status(t_id)
        status = status_info["status"]
        
        print(f"主程序正在忙... 任务状态: {status}")
        
        if status == "COMPLETED":
            print("\n✅ 转换成功!")
            print(f"识别内容: {status_info['result']}")
            break
        elif status == "FAILED":
            print("\n❌ 转换失败")
            print(f"错误信息: {status_info['error']}")
            break
        
        time.sleep(0.5) # 模拟做其他操作

    # 清理测试文件
    if os.path.exists(test_file):
        os.remove(test_file)