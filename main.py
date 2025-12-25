import dearpygui.dearpygui as dpg
import os
import sys
import time
import threading
import datetime
import traceback
import json
import tkinter as tk
from tkinter import filedialog

# --- 导入 Core 包 ---
from core.constants import *
from core.ui_utils import create_node, FontManager 
from core.executor import GraphExecutor
from utilities.diarization.speaker_db import SpeakerDB

# ==========================================
# 1. 全局状态
# ==========================================
executor = GraphExecutor("resource")
LINK_DB = {}
DEFAULT_CONFIG_FILE = os.path.join("config", "default_config.json")
GLOBAL_SUMMARY_CACHE = ""
FONTS = {} 

speaker_db = SpeakerDB()

# ==========================================
# 2. Markdown 渲染 (保持不变)
# ==========================================
def render_markdown(container_tag, markdown_text):
    dpg.delete_item(container_tag, children_only=True)
    global GLOBAL_SUMMARY_CACHE
    GLOBAL_SUMMARY_CACHE = markdown_text 
    lines = markdown_text.split('\n')
    with dpg.group(parent=container_tag):
        dpg.add_spacer(height=10)
        for line in lines:
            line = line.strip()
            if not line: dpg.add_spacer(height=8); continue
            if line.startswith("# "):
                txt = dpg.add_text(line[2:], color=(255, 215, 0), wrap=450)
                if "h1" in FONTS: dpg.bind_item_font(txt, FONTS["h1"])
                dpg.add_separator(); dpg.add_spacer(height=5)
            elif line.startswith("## "):
                dpg.add_spacer(height=5)
                txt = dpg.add_text(line[3:], color=(255, 165, 0), wrap=450)
                if "h2" in FONTS: dpg.bind_item_font(txt, FONTS["h2"])
                dpg.add_spacer(height=2)
            elif line.startswith("### "):
                txt = dpg.add_text(line[4:], color=(135, 206, 235), wrap=450)
                if "bold" in FONTS: dpg.bind_item_font(txt, FONTS["bold"])
            elif line.startswith("- ") or line.startswith("* ") or line.startswith("· "):
                dpg.add_text(f"• {line[2:]}", indent=20, wrap=430, color=(230, 230, 230))
            elif line[0].isdigit() and line[1:3] == ". ":
                dpg.add_text(line, indent=20, wrap=430, color=(230, 230, 230))
            elif "**" in line and ":" in line:
                clean_line = line.replace("**", "")
                txt = dpg.add_text(clean_line, wrap=450)
                if "bold" in FONTS: dpg.bind_item_font(txt, FONTS["bold"])
            else:
                dpg.add_text(line, wrap=450, color=(200, 200, 200))

# ==========================================
# 3. 日志系统
# ==========================================
def log(msg, is_result=False):
    t = datetime.datetime.now().strftime("%H:%M:%S")
    if dpg.does_item_exist("LogBox"):
        dpg.add_text(f"[{t}] {msg}", parent="LogBox")
        if dpg.does_item_exist("LogWindow"): dpg.set_y_scroll("LogWindow", 99999)
    if is_result:
        is_summary = msg.startswith("# ") or "会议纪要" in msg or "会议总结" in msg
        if is_summary:
            if dpg.does_item_exist("SummaryContainer"):
                render_markdown("SummaryContainer", msg)
                dpg.set_value("ResultTabs", "tab_summary")
        elif msg.startswith("[") or msg.startswith("chunk"):
             if dpg.does_item_exist("TranscriptBox"):
                dpg.add_text(msg, parent="TranscriptBox", color=(150, 255, 150), wrap=380)
                if dpg.does_item_exist("TranscriptWindow"): dpg.set_y_scroll("TranscriptWindow", 99999)
        else:
             if dpg.does_item_exist("TranscriptBox"):
                dpg.add_text(f">>> {msg}", parent="TranscriptBox", color=(255, 255, 0))

# ==========================================
# 4. 节点工厂
# ==========================================
def build_enhancer_ui(nid): dpg.add_checkbox(label="Enable", default_value=True, tag=f"chk_enhance_{nid}")
def build_vad_ui(nid): dpg.add_slider_int(label="Aggressiveness", default_value=3, min_value=0, max_value=3, width=120, tag=f"vad_agg_{nid}")
def build_spk_ui(nid): dpg.add_drag_float(label="Win(s)", default_value=1.5, width=60, tag=f"win_{nid}"); dpg.add_drag_float(label="Step(s)", default_value=0.75, width=60, tag=f"step_{nid}")
def build_asr_ui(nid): dpg.add_combo(["tiny","small","base","medium"], default_value="small", width=100, tag=f"model_{nid}")
def build_llm_ui(nid): dpg.add_checkbox(label="Gen Summary", default_value=True, tag=f"chk_llm_{nid}"); dpg.add_radio_button(["Local","Online"], default_value="Local", tag=f"back_{nid}")

NODE_FACTORY = {
    "Audio Source":   {"ins": [], "outs": [("Audio Out", TYPE_AUDIO, COLOR_AUDIO)], "ui": None},
    "Audio Enhancer": {"ins": [("Audio In", TYPE_AUDIO, COLOR_AUDIO)], "outs": [("Audio Out", TYPE_AUDIO, COLOR_AUDIO)], "ui": build_enhancer_ui},
    "VAD Detector":   {"ins": [("Audio In", TYPE_AUDIO, COLOR_AUDIO)], "outs": [("Audio Out", TYPE_AUDIO, COLOR_AUDIO)], "ui": build_vad_ui},
    "Speaker ID":     {"ins": [("Audio In", TYPE_AUDIO, COLOR_AUDIO)], "outs": [("Timeline Out", TYPE_TIMELINE, COLOR_TIMELINE)], "ui": build_spk_ui},
    "Whisper ASR":    {"ins": [("Audio In", TYPE_AUDIO, COLOR_AUDIO), ("Timeline", TYPE_TIMELINE, COLOR_TIMELINE)], "outs": [("Text Out", TYPE_TEXT, COLOR_TEXT)], "ui": build_asr_ui},
    "LLM Summary":    {"ins": [("Text In", TYPE_TEXT, COLOR_TEXT)], "outs": [("Report", TYPE_TEXT, COLOR_TEXT)], "ui": build_llm_ui},
}

def get_current_state():
    state = {"nodes": [], "links": []}
    node_ids = dpg.get_item_children(TAG_NODE_EDITOR, 1) or []
    id_map = {nid: i for i, nid in enumerate(node_ids)}
    for nid in node_ids:
        label = dpg.get_item_label(nid)
        pos = dpg.get_item_pos(nid)
        cfg = {}
        try:
            if label == "Audio Enhancer": cfg['enable'] = dpg.get_value(f"chk_enhance_{nid}")
            elif label == "VAD Detector": cfg['aggressiveness'] = dpg.get_value(f"vad_agg_{nid}")
            elif label == "Speaker ID": cfg['window'] = dpg.get_value(f"win_{nid}"); cfg['step'] = dpg.get_value(f"step_{nid}")
            elif label == "Whisper ASR": cfg['model'] = dpg.get_value(f"model_{nid}")
            elif label == "LLM Summary": cfg['enable'] = dpg.get_value(f"chk_llm_{nid}"); cfg['backend'] = dpg.get_value(f"back_{nid}")
        except: pass
        state["nodes"].append({"label": label, "pos": pos, "config": cfg})
    for link_id, (attr1, attr2) in LINK_DB.items():
        try:
            n1, n2 = dpg.get_item_parent(attr1), dpg.get_item_parent(attr2)
            if n1 in id_map and n2 in id_map:
                state["links"].append({
                    "src_node_idx": id_map[n1],
                    "src_attr_idx": dpg.get_item_children(n1, 1).index(attr1),
                    "dst_node_idx": id_map[n2],
                    "dst_attr_idx": dpg.get_item_children(n2, 1).index(attr2)
                })
        except: pass
    return state

def load_state(state):
    try:
        dpg.delete_item(TAG_NODE_EDITOR, children_only=True)
        LINK_DB.clear()
        new_node_ids = []
        for n_data in state["nodes"]:
            label = n_data["label"]
            pos = n_data["pos"]
            cfg = n_data.get("config", {})
            spec = NODE_FACTORY.get(label)
            if spec:
                nid = create_node(label, pos, spec["ins"], spec["outs"], spec["ui"])
                new_node_ids.append(nid)
                try:
                    if label == "Audio Enhancer": dpg.set_value(f"chk_enhance_{nid}", cfg.get('enable', True))
                    elif label == "VAD Detector": dpg.set_value(f"vad_agg_{nid}", cfg.get('aggressiveness', 3))
                    elif label == "Speaker ID": dpg.set_value(f"win_{nid}", cfg.get('window', 1.5)); dpg.set_value(f"step_{nid}", cfg.get('step', 0.75))
                    elif label == "Whisper ASR": dpg.set_value(f"model_{nid}", cfg.get('model', 'small'))
                    elif label == "LLM Summary": dpg.set_value(f"chk_llm_{nid}", cfg.get('enable', True)); dpg.set_value(f"back_{nid}", cfg.get('backend', 'Local'))
                except: pass
        for l_data in state["links"]:
            try:
                n1 = new_node_ids[l_data["src_node_idx"]]
                n2 = new_node_ids[l_data["dst_node_idx"]]
                attr1 = dpg.get_item_children(n1, 1)[l_data["src_attr_idx"]]
                attr2 = dpg.get_item_children(n2, 1)[l_data["dst_attr_idx"]]
                link_cb(TAG_NODE_EDITOR, (attr1, attr2))
            except: pass
        log("Config loaded.")
    except Exception as e: log(f"Load error: {e}")

# ==========================================
# 5. 声纹管理逻辑 (新增职位)
# ==========================================
def refresh_speaker_list():
    # 数据格式: [(name, title, date), ...]
    speakers = speaker_db.get_all_speakers()
    items = []
    for s in speakers:
        name = s[0]
        title = s[1] if s[1] else "No Title" # 处理空值
        date = s[2][:10] if s[2] else "?"
        # [显示格式] "Name (Job Title) | Date"
        items.append(f"{name} ({title})  |  {date}")
    
    dpg.configure_item("SpeakerList", items=items)

def get_selected_speaker_name():
    selection = dpg.get_value("SpeakerList")
    if not selection: return None
    # "Name (Title)  |  Date" -> 提取 Name
    # 逻辑：取第一个 " (" 之前的内容
    part1 = selection.split("  |  ")[0] # Name (Title)
    if " (" in part1:
        return part1.split(" (")[0]
    return part1

def spk_btn_add_file():
    name = dpg.get_value("SpeakerNameInput").strip()
    title = dpg.get_value("SpeakerTitleInput").strip() # [新增]
    
    if not name:
        log("⚠️ Error: Please enter a NAME first!", is_result=False)
        return

    try:
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost',True)
        p = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.m4a")])
        root.destroy()
        
        if p:
            log(f"Extracting voiceprint from: {os.path.basename(p)}")
            # [修改] 传入 title
            success, msg = speaker_db.add_speaker(name, title, p)
            if success:
                log(f"✅ Speaker '{name}' ({title}) added!")
                refresh_speaker_list()
                dpg.set_value("SpeakerNameInput", "") 
                dpg.set_value("SpeakerTitleInput", "") 
            else:
                log(f"❌ Failed: {msg}")
    except Exception as e:
        log(f"Error dialog: {e}")

def spk_btn_record_add(s):
    name = dpg.get_value("SpeakerNameInput").strip()
    title = dpg.get_value("SpeakerTitleInput").strip() # [新增]
    
    current_label = dpg.get_item_label(s)
    
    if "Start" in current_label:
        if not name:
            log("⚠️ Error: Enter NAME before recording!")
            return

        dpg.set_item_label(s, "Stop & Save")
        dpg.bind_item_theme(s, "theme_red")
        
        f_name = f"voiceprint_{int(time.time())}"
        executor.recorder.start(f_name)
        
        full_path = os.path.join("resource", "raw", f_name + ".wav")
        dpg.set_item_user_data(s, full_path)
        
        log(f"🎙️ Recording '{name}'... Click Stop to save.")
        
    else:
        dpg.set_item_label(s, "Start Recording")
        dpg.bind_item_theme(s, "theme_green")
        
        executor.recorder.stop()
        log("Processing voiceprint...")
        
        full_path = dpg.get_item_user_data(s)
        time.sleep(0.5) 
        
        if full_path and os.path.exists(full_path):
            # [修改] 传入 title
            success, msg = speaker_db.add_speaker(name, title, full_path)
            if success:
                log(f"✅ Speaker '{name}' added!")
                refresh_speaker_list()
                dpg.set_value("SpeakerNameInput", "")
                dpg.set_value("SpeakerTitleInput", "")
            else:
                log(f"❌ Failed: {msg}")
        else:
            log("Error: Recording not found.")

def spk_btn_delete():
    name = get_selected_speaker_name()
    if not name: return
    speaker_db.delete_speaker(name)
    log(f"Speaker '{name}' deleted.")
    refresh_speaker_list()

def spk_btn_update():
    # 1. 获取当前选中的人 (旧名字)
    old_name = get_selected_speaker_name()
    if not old_name: 
        log("⚠️ Please select a speaker from the list first.")
        return

    # 2. 获取输入框的内容
    new_name = dpg.get_value("SpeakerNameInput").strip()
    new_title = dpg.get_value("SpeakerTitleInput").strip()
    
    # 3. 校验
    if not new_name and not new_title:
        log("⚠️ Please enter a new Name OR a new Title to update.")
        return
    
    # 4. 调用数据库更新
    # 注意：如果输入框是空的，传给数据库 None
    success, msg = speaker_db.update_speaker_info(
        current_name=old_name, 
        new_name=new_name if new_name else None, 
        new_title=new_title if new_title else None
    )
    
    if success:
        log_msg = f"Updated '{old_name}': "
        if new_name: log_msg += f"Name->{new_name} "
        if new_title: log_msg += f"Title->{new_title}"
        
        log(f"✅ {log_msg}")
        refresh_speaker_list()
        
        # 清空输入框，方便下一次操作
        dpg.set_value("SpeakerNameInput", "")
        dpg.set_value("SpeakerTitleInput", "")
    else:
        log(f"❌ Update failed: {msg}")

# ==========================================
# 6. 回调 & 线程
# ==========================================
def update_progress(val):
    if dpg.does_item_exist("ProgressBar"): dpg.set_value("ProgressBar", val)
def link_cb(s, d):
    uid = dpg.generate_uuid(); dpg.add_node_link(d[0], d[1], parent=s, tag=uid); LINK_DB[uid] = d
def delink_cb(s, d):
    dpg.delete_item(d); 
    if d in LINK_DB: del LINK_DB[d]
def btn_export_config():
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost',True)
    f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
    root.destroy()
    if f:
        with open(f, "w") as file: json.dump(get_current_state(), file, indent=2)
        log(f"Exported: {os.path.basename(f)}")
def btn_import_config():
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost',True)
    f = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
    root.destroy()
    if f:
        with open(f, "r") as file: load_state(json.load(file))
def btn_set_default():
    folder = os.path.dirname(DEFAULT_CONFIG_FILE)
    if folder and not os.path.exists(folder): os.makedirs(folder)
    with open(DEFAULT_CONFIG_FILE, "w") as file: json.dump(get_current_state(), file, indent=2)
    log("Saved as default.")
def btn_export_summary():
    if not GLOBAL_SUMMARY_CACHE: return log("No summary to export.")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    f = f"Meeting_Minutes_{ts}.md"
    with open(f, "w", encoding="utf-8") as file: file.write(GLOBAL_SUMMARY_CACHE)
    log(f"Exported: {f}"); os.startfile(os.path.abspath(f))

def start_processing_thread(path, mode):
    nodes = {}
    ids = dpg.get_item_children(TAG_NODE_EDITOR, 1) or []
    for nid in ids:
        lbl = dpg.get_item_label(nid)
        cfg = {}
        try:
            if lbl == "Audio Enhancer": cfg['enable'] = dpg.get_value(f"chk_enhance_{nid}")
            elif lbl == "VAD Detector": cfg['aggressiveness'] = dpg.get_value(f"vad_agg_{nid}")
            elif lbl == "Speaker ID": 
                cfg['window'] = dpg.get_value(f"win_{nid}"); cfg['step'] = dpg.get_value(f"step_{nid}")
            elif lbl == "Whisper ASR": cfg['model'] = dpg.get_value(f"model_{nid}")
            elif lbl == "LLM Summary": 
                cfg['enable'] = dpg.get_value(f"chk_llm_{nid}"); cfg['backend'] = dpg.get_value(f"back_{nid}")
        except: pass
        ins, outs = [], []
        for c in dpg.get_item_children(nid, 1):
            ud = dpg.get_item_user_data(c)
            if ud:
                ctype = dpg.get_item_configuration(c)['attribute_type']
                if ctype == dpg.mvNode_Attr_Input: ins.append(c)
                else: outs.append(c)
        nodes[nid] = {'label': lbl, 'config': cfg, 'inputs': ins, 'outputs': outs}

    start_id = next((nid for nid, d in nodes.items() if d['label']=="Audio Source"), None)
    if not start_id: log("Error: No Source Node!"); return
    nodes[start_id]['config'] = {'mode': mode, 'file_path': path}
    
    if dpg.does_item_exist("TranscriptBox"): dpg.delete_item("TranscriptBox", children_only=True)
    if dpg.does_item_exist("SummaryContainer"): dpg.delete_item("SummaryContainer", children_only=True)
    dpg.set_value("ResultTabs", "tab_transcript")
    threading.Thread(target=executor.execute, args=(start_id, nodes, LINK_DB.copy(), {}, log, update_progress), daemon=True).start()

def btn_rec_click(s):
    if "Start" in dpg.get_item_label(s):
        dpg.set_item_label(s, "Stop & Process"); dpg.bind_item_theme(s, "theme_red")
        f = f"rec_{int(time.time())}"; executor.recorder.start(f)
        dpg.set_item_user_data(s, os.path.join(executor.res_dir, "raw", f+".wav"))
        log(">>> Recording...")
    else:
        dpg.set_item_label(s, "Start Recording"); dpg.bind_item_theme(s, "theme_green")
        executor.recorder.stop(); log(">>> Processing...")
        time.sleep(0.5); start_processing_thread(dpg.get_item_user_data(s), 'mic')

def btn_stop_click(): executor.stop(); log(">>> Stopping...")
def btn_load_click():
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost',True)
    p = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3")])
    root.destroy()
    if p: log(f"Selected: {os.path.basename(p)}"); start_processing_thread(p, 'file')

# ==========================================
# 7. Build GUI
# ==========================================
def build_gui():
    dpg.create_context()
    fm = FontManager()
    global FONTS
    FONTS = fm.setup_fonts()
    
    with dpg.theme(tag="theme_red"):
        with dpg.theme_component(dpg.mvButton): dpg.add_theme_color(dpg.mvThemeCol_Button, (200,50,50))
    with dpg.theme(tag="theme_green"):
        with dpg.theme_component(dpg.mvButton): dpg.add_theme_color(dpg.mvThemeCol_Button, (50,150,50))
    with dpg.theme(tag="theme_orange"):
        with dpg.theme_component(dpg.mvButton): dpg.add_theme_color(dpg.mvThemeCol_Button, (200,120,50))

    with dpg.window(tag="Primary Window"):
        with dpg.tab_bar():
            # Tab 1: Dashboard
            with dpg.tab(label="  Dashboard  "):
                dpg.add_spacer(height=15)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=20)
                    dpg.add_button(label="Start Recording", width=180, height=50, tag="btn_rec", callback=btn_rec_click)
                    dpg.bind_item_theme("btn_rec", "theme_green")
                    dpg.add_spacer(width=10)
                    dpg.add_button(label="Load File", width=120, height=50, callback=btn_load_click)
                    dpg.add_spacer(width=10)
                    dpg.add_button(label="STOP", width=100, height=50, callback=btn_stop_click)
                    dpg.bind_item_theme(dpg.last_item(), "theme_orange")
                    dpg.add_spacer(width=30)
                    dpg.add_button(label="Export Summary", width=150, height=50, callback=btn_export_summary)

                dpg.add_spacer(height=10)
                dpg.add_progress_bar(tag="ProgressBar", width=-20, default_value=0.0)
                dpg.add_separator()
                
                with dpg.group(horizontal=True):
                    with dpg.child_window(width=350, border=False):
                        dpg.add_text("System Logs")
                        dpg.add_separator()
                        with dpg.child_window(tag="LogWindow"): dpg.add_group(tag="LogBox")
                    
                    with dpg.child_window(width=-1, border=False):
                        with dpg.tab_bar(tag="ResultTabs"):
                            with dpg.tab(label="Live Transcript", tag="tab_transcript"):
                                with dpg.child_window(tag="TranscriptWindow", border=False):
                                    dpg.add_group(tag="TranscriptBox")
                            with dpg.tab(label="Meeting Minutes (Markdown)", tag="tab_summary"):
                                with dpg.child_window(tag="SummaryWindow", border=False):
                                    dpg.add_group(tag="SummaryContainer")
            
            # Tab 2: Speaker Manager
            with dpg.tab(label="  Speaker Manager  "):
                dpg.add_spacer(height=10)
                with dpg.group(horizontal=True):
                    with dpg.child_window(width=400):
                        dpg.add_text("Registered Speakers", color=(100,255,100))
                        dpg.add_listbox(tag="SpeakerList", width=-1, num_items=20)
                    
                    with dpg.child_window(width=-1):
                        dpg.add_text("Manage Speaker")
                        dpg.add_separator()
                        dpg.add_spacer(height=10)
                        
                        dpg.add_text("Name:")
                        dpg.add_input_text(tag="SpeakerNameInput", width=300, hint="e.g. Alice")
                        dpg.add_spacer(height=5)
                        
                        # [新增] 职位输入框
                        dpg.add_text("Job Title / Position:")
                        dpg.add_input_text(tag="SpeakerTitleInput", width=300, hint="e.g. Project Manager")
                        
                        dpg.add_spacer(height=20)
                        dpg.add_text("Add New Voiceprint:")
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Start Recording", width=140, height=40, tag="btn_spk_rec", callback=spk_btn_record_add)
                            dpg.bind_item_theme("btn_spk_rec", "theme_green")
                            dpg.add_spacer(width=20)
                            dpg.add_button(label="Import File", width=140, height=40, callback=spk_btn_add_file)
                        
                        dpg.add_spacer(height=40)
                        dpg.add_separator()
                        dpg.add_spacer(height=10)
                        
                        dpg.add_text("Actions on Selected:")
                        with dpg.group(horizontal=True):
                            # [修改] 按钮文字改为 Update Info，回调改为 spk_btn_update
                            dpg.add_button(label="Update Info", width=100, callback=spk_btn_update)
                            
                            dpg.add_spacer(width=20)
                            dpg.add_button(label="Delete", width=100, callback=spk_btn_delete)
                            dpg.bind_item_theme(dpg.last_item(), "theme_red")

                refresh_speaker_list()

            with dpg.tab(label="  Pipeline Designer  "):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Import Config", callback=btn_import_config)
                    dpg.add_button(label="Export Config", callback=btn_export_config)
                    dpg.add_spacer(width=20)
                    dpg.add_button(label="Set as Default", callback=btn_set_default)
                dpg.add_separator()
                with dpg.node_editor(callback=link_cb, delink_callback=delink_cb, tag=TAG_NODE_EDITOR):
                    loaded = False
                    if os.path.exists(DEFAULT_CONFIG_FILE):
                        try:
                            with open(DEFAULT_CONFIG_FILE, 'r') as f: load_state(json.load(f))
                            loaded = True
                        except: pass
                    if not loaded:
                         for name, data in NODE_FACTORY.items():
                             create_node(name, [100,100], data["ins"], data["outs"], data["ui"])
                         # Auto link (simplified)
                         ids = dpg.get_item_children(TAG_NODE_EDITOR, 1) or []
                         lbl_map = {dpg.get_item_label(i): i for i in ids}
                         link_list = [("Audio Source",0,"Audio Enhancer",0), ("Audio Enhancer",0,"VAD Detector",0), ("VAD Detector",0,"Speaker ID",0), ("VAD Detector",0,"Whisper ASR",0), ("Speaker ID",0,"Whisper ASR",1), ("Whisper ASR",0,"LLM Summary",0)]
                         for s,si,d,di in link_list:
                             if s in lbl_map and d in lbl_map:
                                 n1, n2 = lbl_map[s], lbl_map[d]
                                 outs = [c for c in dpg.get_item_children(n1,1) if dpg.get_item_configuration(c)['attribute_type']==dpg.mvNode_Attr_Output]
                                 ins = [c for c in dpg.get_item_children(n2,1) if dpg.get_item_configuration(c)['attribute_type']==dpg.mvNode_Attr_Input]
                                 if len(outs)>si and len(ins)>di: link_cb(TAG_NODE_EDITOR, (outs[si], ins[di]))

    dpg.create_viewport(title="IMA System v19.0 (Job Titles)", width=1280, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Primary Window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    build_gui()