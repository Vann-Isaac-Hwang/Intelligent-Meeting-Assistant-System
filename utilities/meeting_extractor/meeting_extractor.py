import ollama
import json
import re
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

class RobustMeetingExtractor:
    """
    稳健会议纪要提取器 (本地 Ollama 版)
    已配置为自动分类保存文件：
    - JSON -> resource/meeting_summaries
    - MD   -> resource/meeting_sum_md  <-- [修改] 新路径
    """
    
    def __init__(self, model_name: str = "qwen3-vl:8b"):
        self.model_name = model_name
        
    def load_transcript(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def create_successful_prompt(self, transcript: str) -> str:
        return f"""请将以下会议记录转换为 JSON 格式：

{transcript}

输出格式必须严格遵守以下 JSON 结构：
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
        print(f">>> [Local] 正在调用 Ollama ({self.model_name})...")
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个JSON格式输出助手。只输出JSON格式的数据。"},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.1, "num_predict": 4000}
            )
            result_text = response['message']['content'].strip()
            print(f"✓ 收到模型响应，长度: {len(result_text)} 字符")
            
            cleaned_text = self.clean_response_text(result_text)
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                print("⚠️ 初次解析失败，尝试修复...")
                fixed_text = self.fix_json_format(cleaned_text)
                try:
                    return json.loads(fixed_text)
                except:
                    print("⚠️ JSON 解析失败，切换兜底模式")
                    return {
                        "会议主题": "（自动结构化失败）",
                        "会议总结": result_text,
                        "is_raw_fallback": True
                    }
        except Exception as e:
            print(f"✗ 模型错误: {e}")
            return {"error": str(e)}
    
    def enhance_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        enhanced = data.copy()
        enhanced["提取时间"] = datetime.now().isoformat()
        enhanced["模型来源"] = f"Local Ollama ({self.model_name})"
        stats = {
            "参与人数": len(data.get("参与人员", [])),
            "决定数量": len(data.get("重要决定", [])),
            "行动项数量": len(data.get("行动项", [])),
        }
        enhanced["统计信息"] = stats
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

        if "问题与风险" in data and data["问题与风险"]:
            report.append("### ⚠️ 问题与风险")
            for issue in data["问题与风险"]:
                report.append(f"- {issue}")

        report.append("")
        report.append(f"Generated by Local Ollama | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        return "\n".join(report)
    
    def save_results(self, data: Dict[str, Any], input_filename: str):
        """
        保存结果，自动分类：
        - JSON -> resource/meeting_summaries
        - MD   -> resource/meeting_sum_md  <-- [修改]
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(input_filename).replace('.txt', '')
        
        log_dir = os.path.dirname(input_filename)
        resource_dir = os.path.dirname(log_dir)
        
        summary_dir = os.path.join(resource_dir, "meeting_summaries")
        md_dir = os.path.join(resource_dir, "meeting_sum_md") # [新增]
        
        os.makedirs(summary_dir, exist_ok=True)
        os.makedirs(md_dir, exist_ok=True) # [新增]
        
        # 保存 JSON
        json_file_name = f"{base_name}_extracted_{timestamp}.json"
        json_full_path = os.path.join(summary_dir, json_file_name)
        with open(json_full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ JSON数据已保存: {json_full_path}")
        
        # 保存 Markdown
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