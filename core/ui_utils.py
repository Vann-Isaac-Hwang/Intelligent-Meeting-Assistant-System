import dearpygui.dearpygui as dpg
import os
from .constants import TAG_NODE_EDITOR

# ==========================================
# 1. 高级字体管理器 (支持多字号 + 特殊符号)
# ==========================================
class FontManager:
    def __init__(self):
        self.font_registry = None
        self.font_regular = None # 正文 (18px)
        self.font_h1 = None      # 一级标题 (30px)
        self.font_h2 = None      # 二级标题 (24px)
        self.font_bold = None    # 强调/粗体 (20px) - 用稍大一点代替粗体
    
    def setup_fonts(self):
        """
        加载字体，并返回一个包含各级字体的字典。
        """
        # 1. 寻找字体路径
        font_path = None
        local_font = os.path.join("resource", "font.ttf")
        # Windows 常用字体 (微软雅黑)
        win_fonts = ["C:\\Windows\\Fonts\\msyh.ttc", "C:\\Windows\\Fonts\\simhei.ttf"] 
        # Mac/Linux 备选
        unix_fonts = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/System/Library/Fonts/PingFang.ttc"]

        if os.path.exists(local_font): font_path = local_font
        else:
            for f in win_fonts + unix_fonts:
                if os.path.exists(f):
                    font_path = f
                    break
        
        if not font_path:
            print("[Warning] No Chinese font found. UI might be ugly.")
            return None

        print(f"[System] Loading fonts from: {font_path}")
        
        # 2. 注册字体 (加载多次以实现不同字号)
        with dpg.font_registry() as registry:
            self.font_registry = registry
            
            # --- 定义需要加载的字符范围 ---
            # 这是一个帮助函数，给每个字体都加上必要的字符集
            def add_chars(font_id):
                # 1. 基础 ASCII
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default, parent=font_id)
                
                # 2. 中文常用汉字 (0x4E00 - 0x9FA5)
                dpg.add_font_range(0x4e00, 0x9fa5, parent=font_id)
                
                # 3. [关键修复] 广义标点符号 (0x2000 - 0x206F)
                # 这包含了 '·' (Bullet), '—' (Dash), '“' (Quotes) 等
                dpg.add_font_range(0x2000, 0x206F, parent=font_id)
                
                # 4. 全角标点 (0x3000 - 0x303F)
                dpg.add_font_range(0x3000, 0x303F, parent=font_id)
                
                # 5. 全角字母数字 (0xFF00 - 0xFFEF)
                dpg.add_font_range(0xFF00, 0xFFEF, parent=font_id)
                
                # 6. 补充特定的点 '·' (Middle Dot 0x00B7)
                dpg.add_font_chars([0x00B7], parent=font_id)

            # --- A. 正文 (18px) ---
            self.font_regular = dpg.add_font(font_path, 18)
            add_chars(self.font_regular)
            
            # --- B. H1 大标题 (32px) ---
            self.font_h1 = dpg.add_font(font_path, 32)
            add_chars(self.font_h1)

            # --- C. H2 中标题 (24px) ---
            self.font_h2 = dpg.add_font(font_path, 24)
            add_chars(self.font_h2)
            
            # --- D. 伪粗体/重点 (20px) ---
            self.font_bold = dpg.add_font(font_path, 20)
            add_chars(self.font_bold)

            # 绑定默认字体
            dpg.bind_font(self.font_regular)
            
        return {
            "regular": self.font_regular,
            "h1": self.font_h1,
            "h2": self.font_h2,
            "bold": self.font_bold
        }

# ==========================================
# 2. 主题管理器 (保持不变)
# ==========================================
class NodeThemeManager:
    def __init__(self):
        self.themes = {}
        self._create_themes()
    
    def _create_themes(self):
        with dpg.theme() as t_running:
            with dpg.theme_component(dpg.mvNode):
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, (50, 255, 50), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvNodeStyleVar_NodeBorderThickness, 4.0, category=dpg.mvThemeCat_Core)
        self.themes['running'] = t_running

        with dpg.theme() as t_error:
            with dpg.theme_component(dpg.mvNode):
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, (255, 50, 50), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvNodeStyleVar_NodeBorderThickness, 4.0, category=dpg.mvThemeCat_Core)
        self.themes['error'] = t_error

        with dpg.theme() as t_idle:
            with dpg.theme_component(dpg.mvNode):
                pass 
        self.themes['idle'] = t_idle

    def set_status(self, node_id, status):
        if status in self.themes:
            dpg.bind_item_theme(node_id, self.themes[status])

# ==========================================
# 3. 节点创建辅助 (保持不变)
# ==========================================
def create_node(label, pos, inputs=[], outputs=[], widget_builder=None):
    with dpg.node(label=label, pos=pos, parent=TAG_NODE_EDITOR) as node_id:
        for i_name, i_type, i_color in inputs:
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, user_data=i_type):
                dpg.add_text(i_name, color=i_color)
        
        if widget_builder:
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                widget_builder(node_id)
                
        for o_name, o_type, o_color in outputs:
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, user_data=o_type):
                dpg.add_text(o_name, color=o_color)
    return node_id