import dearpygui.dearpygui as dpg
import time
import traceback
from .processors import *
from .ui_utils import NodeThemeManager

class GraphExecutor:
    def __init__(self, resource_dir):
        self.res_dir = resource_dir
        self.processors = {
            "Audio Source": SourceProcessor(resource_dir),
            "Audio Enhancer": EnhancerProcessor(),
            "VAD Detector": VADProcessor(),
            "Speaker ID": SpeakerIDProcessor(),
            "Whisper ASR": ASRProcessor(resource_dir),
            "LLM Summary": LLMProcessor()
        }
        self.recorder = self.processors["Audio Source"].recorder
        self.theme_mgr = None 
        
        # [新增] 中断控制标志
        self.stop_flag = False

    def stop(self):
        """外部调用此方法来中断执行"""
        self.stop_flag = True

    def execute(self, start_id, nodes, links, context, log_cb, prog_cb):
        if not self.theme_mgr: self.theme_mgr = NodeThemeManager()
        
        # [新增] 开始执行前重置标志
        self.stop_flag = False
        
        curr_id = start_id
        link_map = {l[0]: l[1] for l in links.values()}
        
        step = 0
        while curr_id and step < 15:
            # [新增] 每一轮循环开始前，检查是否需要停止
            if self.stop_flag:
                log_cb(">>> 🛑 Process Interrupted by User.", is_result=True)
                # 将当前节点状态设回 idle，防止卡在 running 绿色状态
                dpg.split_frame()
                self.theme_mgr.set_status(curr_id, 'idle')
                break

            node = nodes[curr_id]
            label = node['label']
            
            # Visual Feedback
            dpg.split_frame()
            self.theme_mgr.set_status(curr_id, 'running')
            
            # Execute
            proc = self.processors.get(label)
            if proc:
                try:
                    # [优化] 如果是耗时操作，理论上 Processor 内部也应该支持检查 stop_flag
                    # 这里暂时只支持"节点级"中断（即做完当前节点后停止）
                    context = proc.process(context, node['config'], log_cb)
                    self.theme_mgr.set_status(curr_id, 'idle')
                except Exception as e:
                    log_cb(f"!!! Error in {label}: {e}")
                    traceback.print_exc()
                    self.theme_mgr.set_status(curr_id, 'error')
                    break
            
            # Find Next
            output_ids = node['outputs']
            next_id = None
            if output_ids and output_ids[0] in link_map:
                target_in = link_map[output_ids[0]]
                for nid, ndata in nodes.items():
                    if target_in in ndata['inputs']:
                        next_id = nid
                        break
            
            curr_id = next_id
            step += 1
            prog_cb(step/6.0)
        
        # 结束或中断后，进度条归位
        if self.stop_flag:
            prog_cb(0.0)
        else:
            prog_cb(1.0)