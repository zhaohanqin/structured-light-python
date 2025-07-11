import sys
import os
from pathlib import Path
import numpy as np
import json

# 设置Matplotlib使用PySide6后端，必须在导入matplotlib前设置
import matplotlib
matplotlib.use('QtAgg')  # 使用Qt后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import open3d as o3d

from PySide6.QtCore import Qt, QEasingCurve, QPropertyAnimation, QRect, Property, QSize
from PySide6.QtGui import QColor, QPainter, QPainterPath, QFont, QIcon, QPixmap, QLinearGradient, QBrush
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QLineEdit, QFileDialog, QCheckBox, 
    QDoubleSpinBox, QFrame, QSplitter, QStackedWidget, QProgressBar,
    QScrollArea, QMessageBox
)

# 导入3D重建功能 - 使用importlib处理文件名中的连字符
import importlib
reconstruction_module = importlib.import_module("three-dimensional_reconstruction")

# 从导入的模块中获取函数
load_camera_params = reconstruction_module.load_camera_params
load_projector_params = reconstruction_module.load_projector_params
load_extrinsics = reconstruction_module.load_extrinsics
load_unwrapped_phases = reconstruction_module.load_unwrapped_phases
create_mask = reconstruction_module.create_mask
phase_to_pointcloud = reconstruction_module.phase_to_pointcloud
create_open3d_pointcloud = reconstruction_module.create_open3d_pointcloud
visualize_pointcloud = reconstruction_module.visualize_pointcloud
create_mesh_from_pointcloud = reconstruction_module.create_mesh_from_pointcloud
reconstruct_3d_scene = reconstruction_module.reconstruct_3d_scene

# 定义颜色调色板
COLOR_PRIMARY = "#4F94CD"       # 主色调：蓝色
COLOR_SECONDARY = "#87CEFA"     # 次要色调：浅蓝色
COLOR_ACCENT = "#1E90FF"        # 强调色：鲜亮蓝色
COLOR_BACKGROUND = "#F8FAFF"    # 背景色：浅灰蓝色
COLOR_CARD_BG = "#FFFFFF"       # 卡片背景：白色
COLOR_TEXT_PRIMARY = "#333333"  # 主要文本：深灰色
COLOR_TEXT_SECONDARY = "#666666"# 次要文本：中灰色
COLOR_ERROR = "#E57373"         # 错误提示：红色
COLOR_SUCCESS = "#81C784"       # 成功提示：绿色
COLOR_DISABLED = "#BDBDBD"      # 禁用状态：浅灰色


class StyledPushButton(QPushButton):
    """自定义样式的按钮"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.setFixedHeight(40)
        self.setCursor(Qt.PointingHandCursor)
        
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_ACCENT};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:disabled {{
                background-color: {COLOR_DISABLED};
                color: #999999;
            }}
        """)


class CardWidget(QFrame):
    """带有阴影和圆角的卡片小部件"""
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setStyleSheet("""
            #card {
                background-color: #FFFFFF;
                border-radius: 12px;
            }
        """)
        
        # 创建阴影效果
        self.setGraphicsEffect(None)  # 我们将使用CSS代替图形效果
        
        # 主布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(12)
        
        # 卡片标题
        if title:
            title_label = QLabel(title)
            title_label.setStyleSheet(f"""
                font-size: 16px;
                font-weight: bold;
                color: {COLOR_TEXT_PRIMARY};
                margin-bottom: 8px;
            """)
            self.layout.addWidget(title_label)
            
    def paintEvent(self, event):
        """绘制卡片的圆角矩形和阴影"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # 启用抗锯齿
        
        # 创建圆角矩形路径
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 12, 12)
        
        # 绘制阴影（使用渐变）
        gradient = QLinearGradient(0, 0, 0, 12)
        gradient.setColorAt(0, QColor(0, 0, 0, 25))  # 顶部半透明黑色
        gradient.setColorAt(1, QColor(0, 0, 0, 0))   # 底部透明
        painter.fillPath(path, QBrush(gradient))
        
        # 绘制卡片背景
        painter.fillPath(path, QColor(COLOR_CARD_BG))
        
        super().paintEvent(event)


class FileInputWidget(QWidget):
    """文件输入选择小部件，带有浏览按钮"""
    
    def __init__(self, label_text, file_filter="All Files (*.*)", tooltip_text="", parent=None):
        super().__init__(parent)
        self.file_filter = file_filter
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 标签
        self.label = QLabel(label_text)
        self.label.setMinimumWidth(150)
        self.label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; font-weight: bold;")
        
        # 输入字段
        self.file_path = QLineEdit()
        self.file_path.setMinimumHeight(36)
        self.file_path.setStyleSheet(f"""
            QLineEdit {{
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 4px 10px;
                background-color: white;
                color: {COLOR_TEXT_PRIMARY};
            }}
            QLineEdit:focus {{
                border: 1px solid {COLOR_ACCENT};
            }}
        """)
        
        # 浏览按钮
        self.browse_btn = StyledPushButton("浏览")
        self.browse_btn.clicked.connect(self.browse_file)
        
        # 设置提示
        if tooltip_text:
            self.file_path.setToolTip(tooltip_text)
            self.browse_btn.setToolTip(tooltip_text)
            self.label.setToolTip(tooltip_text)

        layout.addWidget(self.label)
        layout.addWidget(self.file_path, 1)
        layout.addWidget(self.browse_btn)
        
    def browse_file(self):
        """打开文件对话框选择文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "", self.file_filter
        )
        if file_path:
            self.file_path.setText(file_path)
            
    def get_file_path(self):
        """获取选择的文件路径"""
        return self.file_path.text()
    
    def set_file_path(self, path):
        """设置文件路径"""
        self.file_path.setText(path)


class OutputDirWidget(QWidget):
    """输出目录选择小部件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 标签
        self.label = QLabel("输出目录:")
        self.label.setMinimumWidth(150)
        self.label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; font-weight: bold;")
        
        # 输入字段
        self.dir_path = QLineEdit("reconstruction_output")
        self.dir_path.setMinimumHeight(36)
        self.dir_path.setStyleSheet(f"""
            QLineEdit {{
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 4px 10px;
                background-color: white;
                color: {COLOR_TEXT_PRIMARY};
            }}
            QLineEdit:focus {{
                border: 1px solid {COLOR_ACCENT};
            }}
        """)
        
        # 浏览按钮
        self.browse_btn = StyledPushButton("浏览")
        self.browse_btn.clicked.connect(self.browse_dir)
        
        # 设置提示
        tooltip = "选择一个文件夹用于保存最终的点云、网格模型和其他结果文件。"
        self.label.setToolTip(tooltip)
        self.dir_path.setToolTip(tooltip)
        self.browse_btn.setToolTip(tooltip)

        layout.addWidget(self.label)
        layout.addWidget(self.dir_path, 1)
        layout.addWidget(self.browse_btn)
        
    def browse_dir(self):
        """打开目录选择对话框"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择输出目录"
        )
        if dir_path:
            self.dir_path.setText(dir_path)
            
    def get_dir_path(self):
        """获取选择的目录路径"""
        return self.dir_path.text()


# 替代matplotlib图表显示的占位类
class PlaceholderWidget(QWidget):
    """图表占位符，用于替代matplotlib"""
    
    def __init__(self, title="图表占位符", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        
        layout = QVBoxLayout(self)
        
        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(f"""
            color: {COLOR_TEXT_SECONDARY};
            font-size: 16px;
            font-weight: bold;
            background-color: white;
            border: 1px dashed #E0E0E0;
            border-radius: 8px;
            padding: 40px;
        """)
        
        layout.addWidget(label)
    
    def draw(self):
        """模拟matplotlib的draw方法"""
        pass


class StructuredLightUI(QMainWindow):
    """结构光3D重建应用程序的主UI窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 设置窗口
        self.setWindowTitle("结构光3D重建")
        self.setMinimumSize(1200, 800)  # 设置最小窗口大小
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {COLOR_BACKGROUND};
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
            }}
            QLabel {{
                color: {COLOR_TEXT_PRIMARY};
            }}
            QProgressBar {{
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                background-color: white;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {COLOR_PRIMARY};
                border-radius: 3px;
            }}
        """)
        
        # 中央部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # 创建分割器用于主内容区域
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)  # 设置分割条宽度
        splitter.setChildrenCollapsible(False)  # 防止子部件被完全折叠
        
        # 左侧面板（输入参数）
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)
        
        # 页眉
        logo_label = QLabel("3D重建")
        logo_label.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            color: #333333;
            margin-bottom: 5px;
        """)
        left_layout.addWidget(logo_label)
        
        # 输入参数卡片
        input_card = CardWidget("输入参数")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(16)
        
        # 文件输入
        self.camera_params_input = FileInputWidget(
            "相机参数:", "参数文件 (*.npy *.json)",
            "选择包含相机内参矩阵和畸变系数的标定文件。"
        )
        self.projector_params_input = FileInputWidget(
            "投影仪参数:", "参数文件 (*.json *.npy)",
            "选择包含投影仪内参矩阵、宽度和高度的标定文件。"
        )
        self.extrinsics_input = FileInputWidget(
            "外参:", "参数文件 (*.npy *.json)",
            "选择包含相机和投影仪之间旋转(R)和平移(T)关系的外参文件。"
        )
        self.phase_x_input = FileInputWidget(
            "X方向相位:", "相位文件 (*.npy *.png *.jpg *.jpeg *.bmp *.tiff *.tif)",
            "选择由相位解包裹程序生成的X方向（水平）解包裹相位图。"
        )
        self.phase_y_input = FileInputWidget(
            "Y方向相位:", "相位文件 (*.npy *.png *.jpg *.jpeg *.bmp *.tiff *.tif)",
            "选择由相位解包裹程序生成的Y方向（垂直）解包裹相位图。"
        )
        self.output_dir_input = OutputDirWidget()
        
        # 选项
        options_layout = QHBoxLayout()
        
        # 掩码百分位
        mask_layout = QHBoxLayout()
        mask_label = QLabel("掩码百分位:")
        mask_label.setMinimumWidth(150)
        mask_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; font-weight: bold;")
        self.mask_percentile = QDoubleSpinBox()
        self.mask_percentile.setToolTip(
            "用于生成有效区域掩码的相位梯度阈值。\n"
            "值越高，保留的区域越多，可能包含噪声；值越低，结果越干净，但可能丢失细节。"
        )
        self.mask_percentile.setRange(1.0, 99.9)  # 设置范围
        self.mask_percentile.setValue(98.0)  # 默认值
        self.mask_percentile.setDecimals(1)  # 小数点位数
        self.mask_percentile.setSingleStep(0.1)  # 单步值
        self.mask_percentile.setMinimumHeight(36)
        self.mask_percentile.setStyleSheet("""
            QDoubleSpinBox {
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 4px 10px;
            }
            QDoubleSpinBox:focus {
                border: 1px solid #1E90FF;
            }
        """)
        mask_layout.addWidget(mask_label)
        mask_layout.addWidget(self.mask_percentile)
        
        # 创建网格选项
        self.create_mesh = QCheckBox("从点云创建网格")
        self.create_mesh.setChecked(True)  # 默认选中
        self.create_mesh.setToolTip("如果选中，程序将在生成点云后，自动进行网格化处理，生成一个连续的3D表面模型。")
        self.create_mesh.setStyleSheet(f"""
            QCheckBox {{
                color: {COLOR_TEXT_PRIMARY};
                font-weight: bold;
            }}
            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLOR_PRIMARY};
                image: url(checkmark.png);
            }}
        """)
        
        options_layout.addLayout(mask_layout)
        options_layout.addWidget(self.create_mesh, alignment=Qt.AlignRight)
        
        # 将所有输入添加到布局
        input_layout.addWidget(self.camera_params_input)
        input_layout.addWidget(self.projector_params_input)
        input_layout.addWidget(self.extrinsics_input)
        input_layout.addWidget(self.phase_x_input)
        input_layout.addWidget(self.phase_y_input)
        input_layout.addWidget(self.output_dir_input)
        input_layout.addLayout(options_layout)
        
        # 重建按钮
        self.reconstruct_btn = StyledPushButton("重建3D场景")
        self.reconstruct_btn.setToolTip("加载所有参数并开始完整的三维重建流程。")
        self.reconstruct_btn.setMinimumHeight(50)
        self.reconstruct_btn.clicked.connect(self.start_reconstruction)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(8)
        self.progress_bar.setTextVisible(False)  # 不显示文本
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # 初始时隐藏
        
        input_layout.addWidget(self.reconstruct_btn)
        input_layout.addWidget(self.progress_bar)
        
        input_card.layout.addLayout(input_layout)
        left_layout.addWidget(input_card)
        
        # 状态卡片
        status_card = CardWidget("状态")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("就绪")
        self.status_label.setWordWrap(True)  # 允许文本换行
        self.status_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-weight: bold;")
        
        status_layout.addWidget(self.status_label)
        status_card.layout.addLayout(status_layout)
        left_layout.addWidget(status_card)
        
        left_layout.addStretch()  # 添加弹性空间，将内容推到顶部
        
        # 为左侧面板创建滚动区域
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)  # 允许小部件调整大小
        left_scroll.setFrameShape(QFrame.NoFrame)  # 无边框
        
        # 右侧面板（可视化）
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(15)
        
        # 可视化卡片
        visualization_card = CardWidget("可视化")
        viz_layout = QVBoxLayout()
        
        # 不同可视化的标签页
        self.viz_stack = QStackedWidget()
        
        # 标签页按钮
        tab_layout = QHBoxLayout()
        tab_layout.setSpacing(8)
        
        self.phase_tab_btn = QPushButton("解包裹相位")
        self.mask_tab_btn = QPushButton("掩码")
        self.pointcloud_tab_btn = QPushButton("点云")
        self.mesh_tab_btn = QPushButton("网格")
        
        # 配置所有标签页按钮
        for btn in [self.phase_tab_btn, self.mask_tab_btn, self.pointcloud_tab_btn, self.mesh_tab_btn]:
            btn.setCheckable(True)  # 使按钮可选中
            btn.setFixedHeight(36)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    border: none;
                    border-bottom: 3px solid transparent;
                    background-color: transparent;
                    color: {COLOR_TEXT_SECONDARY};
                    padding: 8px 16px;
                    font-weight: bold;
                }}
                QPushButton:checked {{
                    border-bottom: 3px solid {COLOR_PRIMARY};
                    color: {COLOR_PRIMARY};
                }}
                QPushButton:hover:!checked {{
                    color: {COLOR_TEXT_PRIMARY};
                }}
            """)
            tab_layout.addWidget(btn)
        
        tab_layout.addStretch()  # 添加弹性空间，将按钮推到左侧
        
        # 连接标签页按钮
        self.phase_tab_btn.clicked.connect(lambda: self.viz_stack.setCurrentIndex(0))
        self.mask_tab_btn.clicked.connect(lambda: self.viz_stack.setCurrentIndex(1))
        self.pointcloud_tab_btn.clicked.connect(lambda: self.viz_stack.setCurrentIndex(2))
        self.mesh_tab_btn.clicked.connect(lambda: self.viz_stack.setCurrentIndex(3))
        
        # 相位可视化小部件
        self.phase_viz = QWidget()
        phase_layout = QVBoxLayout(self.phase_viz)
        # 使用占位符替代MatplotlibCanvas
        self.phase_canvas = PlaceholderWidget("X和Y方向解包裹相位图（重建后显示）")
        phase_layout.addWidget(self.phase_canvas)
        
        # 掩码可视化小部件
        self.mask_viz = QWidget()
        mask_layout = QVBoxLayout(self.mask_viz)
        # 使用占位符替代MatplotlibCanvas
        self.mask_canvas = PlaceholderWidget("掩码图（重建后显示）")
        mask_layout.addWidget(self.mask_canvas)
        
        # 点云和网格的占位符（将在重建后替换为实际可视化）
        self.pointcloud_viz = QWidget()
        pointcloud_layout = QVBoxLayout(self.pointcloud_viz)
        pointcloud_placeholder = QLabel("重建完成后将在此处显示点云")
        pointcloud_placeholder.setAlignment(Qt.AlignCenter)
        pointcloud_placeholder.setStyleSheet("font-weight: bold;")
        pointcloud_layout.addWidget(pointcloud_placeholder)
        
        self.mesh_viz = QWidget()
        mesh_layout = QVBoxLayout(self.mesh_viz)
        mesh_placeholder = QLabel("重建完成后将在此处显示网格")
        mesh_placeholder.setAlignment(Qt.AlignCenter)
        mesh_placeholder.setStyleSheet("font-weight: bold;")
        mesh_layout.addWidget(mesh_placeholder)
        
        # 将小部件添加到堆栈中
        self.viz_stack.addWidget(self.phase_viz)
        self.viz_stack.addWidget(self.mask_viz)
        self.viz_stack.addWidget(self.pointcloud_viz)
        self.viz_stack.addWidget(self.mesh_viz)
        
        # 设置默认标签页
        self.phase_tab_btn.setChecked(True)
        self.viz_stack.setCurrentIndex(0)
        
        viz_layout.addLayout(tab_layout)
        viz_layout.addWidget(self.viz_stack)
        
        visualization_card.layout.addLayout(viz_layout)
        right_layout.addWidget(visualization_card)
        
        # 将面板添加到分割器
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        
        # 设置初始大小（左侧40%，右侧60%）
        splitter.setSizes([400, 800])
        
        # 将分割器添加到主布局
        main_layout.addWidget(splitter)
        
        # 初始化状态变量
        self.camera_matrix = None         # 相机内参矩阵
        self.projector_matrix = None      # 投影仪内参矩阵
        self.projector_width = 1280       # 投影仪宽度
        self.projector_height = 800       # 投影仪高度
        self.R = None                     # 旋转矩阵
        self.T = None                     # 平移向量
        self.unwrapped_phase_x = None     # X方向解包裹相位
        self.unwrapped_phase_y = None     # Y方向解包裹相位
        self.mask = None                  # 掩码
        self.points = None                # 点云坐标
        self.colors = None                # 点云颜色
        self.pcd = None                   # Open3D点云对象
        self.mesh = None                  # Open3D网格对象
    
    def update_status(self, message):
        """更新状态消息"""
        self.status_label.setText(message)
        QApplication.processEvents()  # 刷新UI
    
    def load_parameters(self):
        """加载输入字段中的所有参数"""
        try:
            # 相机参数
            camera_path = self.camera_params_input.get_file_path()
            if not camera_path or not os.path.exists(camera_path):
                self.show_error("未找到相机参数文件")
                return False
            self.camera_matrix = load_camera_params(camera_path)
            if self.camera_matrix is None:
                self.show_error("加载相机参数失败")
                return False
                
            # 投影仪参数
            projector_path = self.projector_params_input.get_file_path()
            if not projector_path or not os.path.exists(projector_path):
                self.show_error("未找到投影仪参数文件")
                return False
            self.projector_matrix, self.projector_width, self.projector_height = load_projector_params(projector_path)
            if self.projector_matrix is None:
                self.show_error("加载投影仪参数失败")
                return False
                
            # 外参
            extrinsics_path = self.extrinsics_input.get_file_path()
            if not extrinsics_path or not os.path.exists(extrinsics_path):
                self.show_error("未找到外参文件")
                return False
            self.R, self.T = load_extrinsics(extrinsics_path)
            if self.R is None or self.T is None:
                self.show_error("加载外参失败")
                return False
                
            # 相位数据
            phase_x_path = self.phase_x_input.get_file_path()
            phase_y_path = self.phase_y_input.get_file_path()
            if not phase_x_path or not phase_y_path or not os.path.exists(phase_x_path) or not os.path.exists(phase_y_path):
                self.show_error("未找到相位数据文件")
                return False
            self.unwrapped_phase_x, self.unwrapped_phase_y = load_unwrapped_phases(phase_x_path, phase_y_path)
            if self.unwrapped_phase_x is None or self.unwrapped_phase_y is None:
                self.show_error("加载相位数据失败")
                return False
                
            return True
            
        except Exception as e:
            self.show_error(f"加载参数时出错: {str(e)}")
            return False
    
    def show_error(self, message):
        """显示错误消息对话框"""
        QMessageBox.critical(self, "错误", message)
        self.update_status(f"错误: {message}")
    
    def visualize_phases(self):
        """可视化解包裹相位"""
        # 在占位符上显示相位信息
        self.phase_canvas = PlaceholderWidget("已加载X和Y方向解包裹相位图")
        phase_layout = QVBoxLayout()
        phase_layout.addWidget(self.phase_canvas)
        
        # 清空现有布局并添加新的占位符
        layout = self.phase_viz.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
        layout.addWidget(self.phase_canvas)
    
    def visualize_mask(self):
        """可视化掩码"""
        # 在占位符上显示掩码信息
        self.mask_canvas = PlaceholderWidget(f"已生成掩码 (百分位: {self.mask_percentile.value()}%)")
        mask_layout = QVBoxLayout()
        mask_layout.addWidget(self.mask_canvas)
        
        # 清空现有布局并添加新的占位符
        layout = self.mask_viz.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
        layout.addWidget(self.mask_canvas)
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        QApplication.processEvents()  # 刷新UI
    
    def start_reconstruction(self):
        """开始3D重建过程"""
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.reconstruct_btn.setEnabled(False)  # 禁用重建按钮
        
        try:
            # 步骤1: 加载参数
            self.update_status("正在加载参数...")
            self.update_progress(10)
            if not self.load_parameters():
                self.reconstruct_btn.setEnabled(True)
                self.progress_bar.setVisible(False)
                return
                
            # 步骤2: 创建输出目录
            self.update_status("正在创建输出目录...")
            self.update_progress(20)
            output_dir = self.output_dir_input.get_dir_path()
            os.makedirs(output_dir, exist_ok=True)
            
            # 步骤3: 创建掩码
            self.update_status("正在创建掩码...")
            self.update_progress(30)
            mask_percentile = self.mask_percentile.value()
            self.mask = create_mask(self.unwrapped_phase_x, self.unwrapped_phase_y, mask_percentile)
            
            # 步骤4: 可视化相位和掩码
            self.update_status("正在生成可视化...")
            self.update_progress(40)
            self.visualize_phases()
            self.visualize_mask()
            
            # 步骤5: 生成点云
            self.update_status("正在生成点云...")
            self.update_progress(50)
            self.points, self.colors = phase_to_pointcloud(
                self.unwrapped_phase_x, self.unwrapped_phase_y, self.mask,
                self.camera_matrix, self.projector_matrix, self.R, self.T,
                self.projector_width, self.projector_height
            )
            
            if len(self.points) == 0:
                self.show_error("生成的点云为空")
                self.reconstruct_btn.setEnabled(True)
                self.progress_bar.setVisible(False)
                return
                
            # 步骤6: 创建Open3D点云
            self.update_status("正在创建和处理点云...")
            self.update_progress(60)
            self.pcd = create_open3d_pointcloud(self.points, self.colors)
            
            # 移除噪声点
            self.pcd, _ = self.pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # 保存点云
            output_ply_file = os.path.join(output_dir, 'reconstructed_pointcloud.ply')
            o3d.io.write_point_cloud(output_ply_file, self.pcd)
            
            # 步骤7: 如果需要，创建网格
            if self.create_mesh.isChecked():
                self.update_status("正在从点云创建网格...")
                self.update_progress(80)
                try:
                    self.mesh = create_mesh_from_pointcloud(self.pcd, voxel_size=0.01, depth=9)
                    
                    # 保存网格
                    output_mesh_file = os.path.join(output_dir, 'reconstructed_mesh.ply')
                    o3d.io.write_triangle_mesh(output_mesh_file, self.mesh)
                except Exception as e:
                    self.show_error(f"网格创建失败: {str(e)}")
            
            # 步骤8: 更新状态
            self.update_progress(100)
            self.update_status(f"重建完成。结果已保存到 {output_dir}")
            
            # 更新点云和网格标签
            self.update_pointcloud_label(f"点云已生成并保存到 {output_ply_file}")
            if self.create_mesh.isChecked() and self.mesh:
                self.update_mesh_label(f"网格已生成并保存到 {output_mesh_file}")
            
        except Exception as e:
            self.show_error(f"重建错误: {str(e)}")
        finally:
            self.reconstruct_btn.setEnabled(True)  # 重新启用重建按钮
    
    def update_pointcloud_label(self, text):
        """更新点云标签"""
        layout = self.pointcloud_viz.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(f"""
            color: {COLOR_SUCCESS};
            font-size: 16px;
            font-weight: bold;
            background-color: white;
            border: 1px solid {COLOR_SUCCESS};
            border-radius: 8px;
            padding: 20px;
        """)
        
        view_button = StyledPushButton("查看点云")
        view_button.clicked.connect(self.visualize_open3d_pointcloud)
        
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(view_button)
        
        self.pointcloud_viz.setLayout(layout)
    
    def update_mesh_label(self, text):
        """更新网格标签"""
        layout = self.mesh_viz.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(f"""
            color: {COLOR_SUCCESS};
            font-size: 16px;
            font-weight: bold;
            background-color: white;
            border: 1px solid {COLOR_SUCCESS};
            border-radius: 8px;
            padding: 20px;
        """)
        
        view_button = StyledPushButton("查看网格")
        view_button.clicked.connect(self.visualize_open3d_mesh)
        
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(view_button)
        
        self.mesh_viz.setLayout(layout)
    
    def visualize_open3d_pointcloud(self):
        """启动Open3D点云可视化"""
        if self.pcd:
            # 创建坐标系参考
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([self.pcd, coordinate_frame], window_name="3D点云")
    
    def visualize_open3d_mesh(self):
        """启动Open3D网格可视化"""
        if self.mesh:
            o3d.visualization.draw_geometries([self.mesh], window_name="3D网格")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序风格
    app.setStyle("Fusion")
    
    # 创建并显示主窗口
    window = StructuredLightUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 