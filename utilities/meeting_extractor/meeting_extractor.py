import ollama
import json
import re
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

class RobustMeetingExtractor:
    """
    稳健会议纪要提取器 (本地 Ollama 版)
    支持双模式输入 (分段 + 全文)
    """
    
    def __init__(self, model_name: str = "qwen3-vl:8b"):
        self.model_name = model_name
        
    def load_transcript(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def create_successful_prompt(self, transcript: str) -> str:
        """
        [修改] 适配双输入模式的 Prompt
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

只输出 JSON 字符串，不要输出 Markdown 代码块标记，也不要其他解释。"""
    
    def clean_response_text(self, text: str) -> str:
        text = re.sub(r'```(?:json)?', '', text)
        text = re.sub(r'```', '', text)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return text.strip()
    
    def fix_json_format(self, text: str) -> str:
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
        prompt = self.create_successful_prompt(transcript)
        print(">>> [Local] Calling Ollama...")
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个JSON格式输出助手。"},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.1, "num_predict": 4000}
            )
            result_text = response['message']['content'].strip()
            print(f"✓ 收到响应 ({len(result_text)} chars)")
            
            cleaned_text = self.clean_response_text(result_text)
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                print("⚠️ JSON 解析失败，尝试修复...")
                fixed_text = self.fix_json_format(cleaned_text)
                try:
                    return json.loads(fixed_text)
                except:
                    print("⚠️ 修复失败，使用兜底。")
                    return {
                        "会议主题": "（自动结构化失败）",
                        "会议总结": result_text,
                        "is_raw_fallback": True
                    }
        except Exception as e:
            print(f"✗ Error: {e}")
            return {"error": str(e)}
    
    def enhance_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        enhanced = data.copy()
        enhanced["提取时间"] = datetime.now().isoformat()
        return enhanced
    
    def generate_readable_report(self, data: Dict[str, Any]) -> str:
        report = []
        if data.get("is_raw_fallback"):
            report.append("# 会议纪要 (原始输出)")
            report.append("⚠️ 自动结构化失败，以下为原始内容：")
            report.append("")
            report.append(data.get("会议总结", ""))
            return "\n".join(report)

        topic = data.get("会议主题", "未命名会议")
        report.append(f"# {topic}")
        
        if "会议总结" in data:
            report.append("## 会议摘要")
            report.append(data['会议总结'])
        
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
        
        if "重要决定" in data and data["重要决定"]:
            report.append("### ✅ 重要决定")
            for decision in data["重要决定"]:
                report.append(f"- {decision}")
        
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

        report.append("")
        report.append(f"Generated by Local Ollama | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        return "\n".join(report)
    
    def save_results(self, data: Dict[str, Any], input_filename: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(input_filename).replace('.txt', '')
        
        log_dir = os.path.dirname(input_filename)
        resource_dir = os.path.dirname(log_dir)
        summary_dir = os.path.join(resource_dir, "meeting_summaries")
        md_dir = os.path.join(resource_dir, "meeting_sum_md")
        
        os.makedirs(summary_dir, exist_ok=True)
        os.makedirs(md_dir, exist_ok=True)
        
        json_file = f"{base_name}_extracted_{timestamp}.json"
        with open(os.path.join(summary_dir, json_file), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        if "error" not in data:
            if not data.get("is_raw_fallback"):
                data = self.enhance_extracted_data(data)
            report = self.generate_readable_report(data)
            report_file = f"{base_name}_report_{timestamp}.md"
            with open(os.path.join(md_dir, report_file), 'w', encoding='utf-8') as f:
                f.write(report)
        return json_file
    
    def process(self, input_file: str) -> Dict[str, Any]:
        try:
            transcript = self.load_transcript(input_file)
            data = self.extract_to_json(transcript)
            self.save_results(data, input_file)
            return data
        except Exception as e:
            return {"error": str(e)}