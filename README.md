这是一个包含了**模块详细接口说明**的完整 `README.md` 更新版本。我根据您提供的最新代码（v19.0，包含 SQLite 数据库、职位管理、Markdown 生成等功能）对文档进行了全面修订。

---

# 智能会议辅助系统 (IMA) - v19.0

**Intelligent Meeting Assistant (IMA)** 是一个基于 Python 的全流程会议记录与分析系统。它集成了实时音频处理、声纹识别（带职位信息）、语音转写以及 LLM 智能总结功能，并通过可视化的 **Dear PyGui** 界面进行管理。

---

## ✨ 核心特性 (Key Features)

1. **全流程自动化**: 从录音采集 -> 降噪 -> 声纹识别 -> 语音转写 -> 会议纪要生成，一键完成。
2. **声纹数据库管理 (New)**:
* 基于 **SQLite** 存储声纹特征，取代旧版的文件存储。
* 支持录入**姓名**与**职位 (Job Title)**，生成的纪要可直接引用“产品经理 Alice 说了...”。
* 提供专门的 **Speaker Manager** 界面，支持录音/文件导入、重命名和删除操作。


3. **可视化管道设计 (Pipeline Designer)**: 采用节点编辑器 (Node Editor) 自由编排处理流程，支持热插拔模块（如开关降噪、切换 LLM 后端）。
4. **智能会议纪要**:
* 支持 **DeepSeek (Online)** 和 **Ollama (Local)** 双后端。
* 自动生成结构化 **Markdown 报告**，包含金色标题、高亮列表等富文本渲染。


5. **双视图仪表盘**: 实时查看 **Live Transcript** (流式转写) 和渲染后的 **Meeting Minutes** (Markdown 纪要)。

---

## 📂 项目目录结构

```text
IMA_System/
├── config/
│   └── default_config.json       # 默认管道连线配置
├── core/                         # 核心系统逻辑
│   ├── executor.py               # 图执行引擎 (GraphExecutor)
│   ├── processors.py             # 各个节点的具体处理逻辑 (Source, ASR, LLM等)
│   └── ui_utils.py               # UI 组件与字体管理器
├── resource/                     # 数据存储目录
│   ├── raw/                      # 原始录音文件 (.wav)
│   ├── meeting_logs/             # ASR 转写文本 (.txt)
│   ├── meeting_summaries/        # LLM 提取的原始 JSON 数据
│   ├── meeting_sum_md/           # 最终生成的 Markdown 报告 (.md)
│   └── speakers.db               # SQLite 声纹数据库
├── utilities/                    # 算法模块
│   ├── ASR/                      # Whisper 语音转写
│   ├── audio_processor/          # 录音、降噪、VAD
│   ├── diarization/              # 声纹提取与识别引擎
│   └── meeting_extractor/        # LLM 摘要生成 (Local/Online)
└── main.py                       # 程序入口 (GUI)

```

---

## 📚 模块详细接口说明 (API Reference)

### 1. 音频处理层 (Audio Processor)

#### 🎙️ 录音模块 (Recorder)

* **路径**: `utilities/audio_processor/recorder.py`
* **类**: `RealTimeAudioProvider`
* **功能**: 负责通过麦克风采集音频并保存为 WAV 格式，默认采样率 16000Hz 以适配声纹模型。

| 方法 | 参数 | 描述 |
| --- | --- | --- |
| `start` | `filename=None` | 启动后台录音线程。若未指定文件名，自动生成时间戳文件名。 |
| `stop` | 无 | 停止录音并将缓冲区数据写入 `resource/raw` 目录。 |

#### 🎧 增强模块 (Enhancer)

* **路径**: `utilities/audio_processor/enhancer.py`
* **类**: `AudioEnhancer`
* **功能**: 使用 `noisereduce` 库直接处理内存中的 Numpy 数组进行降噪。

| 方法 | 参数 | 描述 |
| --- | --- | --- |
| `reduce_noise` | `audio_data` (numpy array) | **核心方法**。对输入的音频数组进行频谱减法降噪。 |
| `process_file` | `input_path`, `output_path` | 处理本地文件：读取 -> 降噪 -> 增益归一化 -> 保存。 |

#### 🔇 静音检测 (VAD)

* **路径**: `utilities/audio_processor/vad_handler.py`
* **类**: `VADHandler`
* **功能**: 基于 `webrtcvad` 移除静音片段，仅保留有效人声，提升后续识别率。

| 方法 | 参数 | 描述 |
| --- | --- | --- |
| `extract_speech` | `audio_np` (numpy array) | 输入音频数组，返回拼接好的纯人声数组。 |

---

### 2. 声纹识别层 (Diarization)

#### 🗄️ 声纹数据库 (Speaker DB)

* **路径**: `utilities/diarization/speaker_db.py`
* **类**: `SpeakerDB`
* **功能**: 封装 SQLite 操作与声纹模型调用，负责声纹的增删改查与特征提取。

| 方法 | 参数 | 描述 |
| --- | --- | --- |
| `add_speaker` | `name`, `title`, `audio_path` | 提取音频特征，将姓名、职位和声纹(BLOB)存入数据库。 |
| `update_speaker_info` | `current_name`, `new_name`, `new_title` | 更新现有说话人的姓名或职位信息。 |
| `extract_embedding_from_memory` | `audio_np` | **核心方法**。从内存数组直接提取 192维 Embedding 向量。 |
| `match_speaker` | `input_embedding`, `threshold` | 将输入向量与数据库对比，返回 `(Name, Title)` 或 `("Unknown", "")`。 |

#### 🗣️ 识别引擎 (Speaker Engine)

* **路径**: `utilities/diarization/engine.py`
* **类**: `SpeakerEngine`
* **功能**: 结合滑动窗口算法与 `SpeakerDB`，实现长音频的说话人切分。

| 方法 | 参数 | 描述 |
| --- | --- | --- |
| `diarize` | `audio_np`, `window_sec`, `step_sec` | 对音频进行滑窗分析。调用 DB 的 `extract` 和 `match` 方法，返回包含 `{start, end, speaker}` 的时间轴列表。 |

---

### 3. 转写与摘要层 (ASR & LLM)

#### 📝 语音转写 (ASR Engine)

* **路径**: `utilities/ASR/whisper_engine.py`
* **类**: `AsyncWhisperEngine`
* **功能**: 异步多线程转写引擎，基于 OpenAI Whisper 模型。

| 方法 | 参数 | 描述 |
| --- | --- | --- |
| `submit_task` | `audio_file_path` | 提交转写任务，返回任务 ID (非阻塞)。 |
| `get_task_status` | `task_id` | 查询任务状态，返回 `{"status": "COMPLETED", "result": "..."}`。 |

#### 🤖 会议摘要 (Meeting Extractor)

* **路径**: `utilities/meeting_extractor/meeting_extractor.py` (Local) / `_ol.py` (Online)
* **类**: `RobustMeetingExtractor`
* **功能**: 调用 LLM (Ollama/DeepSeek) 生成结构化会议纪要，并转换为 Markdown。

| 方法 | 参数 | 描述 |
| --- | --- | --- |
| `process` | `input_file` (txt log) | 执行全流程：读取文本 -> LLM 提取 JSON -> 生成 Markdown -> 保存文件。 |
| `save_results` | `data` (dict), `input_filename` | 将 JSON 存入 `meeting_summaries/`，将 Markdown 存入 `meeting_sum_md/`。 |

---

## 🚀 快速启动 (Quick Start)

1. **安装依赖**:
```bash
pip install dearpygui torch numpy soundfile speechbrain openai httpx ollama webrtcvad noisereduce pydub

```


2. **配置 LLM**:
* **本地版 (Local)**: 确保已安装 Ollama 并拉取模型 (默认 `qwen3-vl:8b`)。
* **在线版 (Online)**: 在 `utilities/meeting_extractor/meeting_extractor_ol.py` 中填入你的 DeepSeek API Key。


3. **运行系统**:
```bash
python main.py

```


4. **操作流程**:
* 进入 **Speaker Manager** 录入您的声纹和职位。
* 进入 **Dashboard** 点击 Start Recording 开始会议。
* 会议结束后，系统将自动生成 Markdown 纪要并弹窗显示。