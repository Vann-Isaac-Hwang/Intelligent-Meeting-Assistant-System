import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

# [新增] 导入 httpx 用于手动创建客户端
try:
    import httpx
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class RobustMeetingExtractor:
    """
    稳健会议纪要提取器 (Online版 - DeepSeek API)
    已配置为自动分类保存文件：
    - JSON -> resource/meeting_summaries
    - MD   -> resource/meeting_sum_md
    """
    
    def __init__(self, api_key: str = "sk-578656dcadf24b72b523460eb9c8dfb3", model_name: str = "deepseek-chat"):
        if not HAS_OPENAI:
            print("!!! 错误: 未检测到 openai 库。请运行: pip install openai httpx")
            raise ImportError("Missing openai dependency")

        self.api_key = api_key
        self.model_name = model_name
        
        # 手动创建 http_client 以解决 httpx 版本兼容性报错
        try:
            custom_http_client = httpx.Client()
        except Exception as e:
            print(f"Warning: Manual httpx client creation failed ({e}), using default.")
            custom_http_client = None

        # 初始化客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
            http_client=custom_http_client
        )
        
    def load_transcript(self, file_path: str) -> str:
        """加载会议记录文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def create_successful_prompt(self, transcript: str) -> str:
        """
        [更新] 适配 Full Text Correction 的提示词
        指导 LLM 利用全文参考来修复分段记录中的错误
        """
        return f"""你是一个专业的会议秘书。我将提供一份会议记录文件，其中可能包含两个部分：
1. "Segmented Transcript": 按发言人分段的记录，包含 [时间] 和 [姓名]。但由于切片原因，句子末尾可能不完整。
2. "Full Text Reference" (可选): 对同一段音频的连续转写，文字内容更准确，但没有发言人信息。

请结合这两部分信息（如果存在第二部分），生成一份准确的结构化会议纪要。
请利用 "Full Text Reference" 来修复 "Segmented Transcript" 中可能存在的断句或错词，但必须保留 "Segmented Transcript" 中的发言人归属。

以下是会议记录内容：
---------------------
{transcript}
---------------------

输出必须严格遵守以下 JSON 结构：
{{
  "会议主题": "主题名称",
  "参与人员": [
    {{ "姓名": "姓名", "职位": "职位(可选)" }}
  ],
  "重要决定": [
    "决定1", "决定2"
  ],
  "行动项": [
    {{ "任务": "任务描述", "负责人": "负责人", "截止时间": "截止时间" }}
  ],
  "问题与风险": [
    "风险1", "风险2"
  ],
  "会议总结": "简要总结"
}}

只输出 JSON 字符串，不要输出 Markdown 代码块标记（如 ```json），也不要任何其他解释性文字。"""
    
    def clean_response_text(self, text: str) -> str:
        """使用正则强力提取 JSON 部分"""
        text = re.sub(r'```(?:json)?', '', text)
        text = re.sub(r'```', '', text)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return text.strip()
    
    def fix_json_format(self, text: str) -> str:
        """尝试修复常见的 JSON 格式错误"""
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]
        
        fixes = [
            (r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":'),
            (r"'([^']*)'", r'"\1"'),
            (r',\s*}', '}'),
            (r',\s*]', ']'),
        ]
        for pattern, replacement in fixes:
            text = re.sub(pattern, replacement, text)
        return text
    
    def extract_to_json(self, transcript: str) -> Dict[str, Any]:
        """DeepSeek API 调用"""
        prompt = self.create_successful_prompt(transcript)
        
        print(f">>> [Online] 正在调用 DeepSeek API ({self.model_name}) 分析会议记录...")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的会议秘书助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000,
                stream=False
            )
            
            result_text = response.choices[0].message.content.strip()
            print(f"✓ 收到 API 响应，长度: {len(result_text)} 字符")
            
            cleaned_text = self.clean_response_text(result_text)
            try:
                data = json.loads(cleaned_text)
                return data
            except json.JSONDecodeError:
                print("⚠️ 初次解析失败，尝试自动修复格式...")
                fixed_text = self.fix_json_format(cleaned_text)
                try:
                    data = json.loads(fixed_text)
                    return data
                except:
                    print("⚠️ JSON 解析最终失败，切换至纯文本兜底模式")
                    return {
                        "会议主题": "（自动提取失败）",
                        "会议总结": result_text,
                        "is_raw_fallback": True
                    }

        except Exception as e:
            error_msg = str(e)
            print(f"✗ DeepSeek API 调用错误: {error_msg}")
            return {"error": f"API请求失败: {error_msg}"}
    
    def enhance_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        enhanced = data.copy()
        enhanced["提取时间"] = datetime.now().isoformat()
        enhanced["模型来源"] = f"DeepSeek API ({self.model_name})"
        return enhanced
    
    def generate_readable_report(self, data: Dict[str, Any]) -> str:
        """
        生成 Markdown 格式的报告
        适配 main.py 中的 render_markdown 函数
        """
        report = []
        
        # 1. 兜底模式 (Raw Text)
        if data.get("is_raw_fallback"):
            report.append("# 会议纪要 (原始输出)")
            report.append("⚠️ 自动结构化失败，以下为原始内容：")
            report.append("")
            report.append(data.get("会议总结", ""))
            return "\n".join(report)

        # 2. 正常模式 (Markdown构建)
        
        # 标题 (H1)
        topic = data.get("会议主题", "未命名会议")
        report.append(f"# {topic}")
        
        # 摘要 (H2)
        if "会议总结" in data:
            report.append("## 会议摘要")
            report.append(data['会议总结'])
        
        # 参与人员 (H2)
        if "参与人员" in data and data["参与人员"]:
            report.append("## 参与人员")
            for person in data["参与人员"]:
                if isinstance(person, dict):
                    name = person.get("姓名", "未知")
                    title = person.get("职位", "")
                    info = f"{name} ({title})" if title else name
                    report.append(f"- {info}")
                else:
                    report.append(f"- {person}")
        
        # 重要决定 (H3)
        if "重要决定" in data and data["重要决定"]:
            report.append("### ✅ 重要决定")
            for decision in data["重要决定"]:
                report.append(f"- {decision}")
        
        # 行动项 (H3)
        if "行动项" in data and data["行动项"]:
            report.append("### 📋 后续行动 (Action Items)")
            for action in data["行动项"]:
                if isinstance(action, dict):
                    task = action.get("任务", "")
                    who = action.get("负责人", "待定")
                    ddl = action.get("截止时间", "")
                    line = f"{task}"
                    if who: line += f" **负责人**: {who}"
                    if ddl: line += f" (截止: {ddl})"
                    report.append(f"- {line}")
                else:
                    report.append(f"- {action}")

        # 问题与风险 (H3)
        if "问题与风险" in data and data["问题与风险"]:
            report.append("### ⚠️ 问题与风险")
            for issue in data["问题与风险"]:
                report.append(f"- {issue}")

        # 添加页脚信息
        report.append("")
        report.append(f"Generated by IMA System | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        return "\n".join(report)
    
    def save_results(self, data: Dict[str, Any], input_filename: str):
        """
        保存结果，自动分类：
        - JSON -> resource/meeting_summaries
        - MD   -> resource/meeting_sum_md  <-- [修改] 新路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(input_filename).replace('.txt', '')
        
        # 1. 路径推导
        # input_filename = .../resource/meeting_logs/Log_xxx.txt
        log_dir = os.path.dirname(input_filename) # -> .../resource/meeting_logs
        resource_dir = os.path.dirname(log_dir)   # -> .../resource
        
        summary_dir = os.path.join(resource_dir, "meeting_summaries")
        md_dir = os.path.join(resource_dir, "meeting_sum_md") # [新增]
        
        os.makedirs(summary_dir, exist_ok=True)
        os.makedirs(md_dir, exist_ok=True) # [新增]
        
        # 2. 保存 JSON
        json_file_name = f"{base_name}_extracted_{timestamp}.json"
        json_full_path = os.path.join(summary_dir, json_file_name)
        with open(json_full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ JSON数据已保存: {json_full_path}")
        
        # 3. 保存 Markdown 报告
        if "error" not in data:
            if not data.get("is_raw_fallback"):
                data = self.enhance_extracted_data(data)
            
            report = self.generate_readable_report(data)
            
            report_file_name = f"{base_name}_report_{timestamp}.md"
            # [修改] 使用 md_dir
            report_full_path = os.path.join(md_dir, report_file_name)
            
            with open(report_full_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✓ Markdown报告已保存: {report_full_path}")
            
        return json_full_path
    
    def process(self, input_file: str) -> Dict[str, Any]:
        try:
            transcript = self.load_transcript(input_file)
            data = self.extract_to_json(transcript)
            self.save_results(data, input_file)
            return data
        except Exception as e:
            return {"error": str(e)}