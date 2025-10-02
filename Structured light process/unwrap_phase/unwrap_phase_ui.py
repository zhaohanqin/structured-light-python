#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
相位解包裹程序UI界面

基于PySide6构建的界面，用于便捷地进行相位解包裹操作。
支持水平和垂直方向的相位解包裹，以及三种不同的解包裹算法。
"""

import sys
import os
import numpy as np
import cv2
from typing import List, Optional, Tuple
from enum import Enum
import matplotlib
# 设置Matplotlib不使用GUI后端，避免线程问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QComboBox, QGroupBox, 
    QRadioButton, QButtonGroup, QMessageBox, QProgressBar,
    QScrollArea, QSplitter, QFrame, QTabWidget, QStackedWidget,
    QSpinBox, QSlider, QDoubleSpinBox
)
from PySide6.QtGui import QPixmap, QImage, QColor, QPalette, QFont, QPainter, QPen
from PySide6.QtCore import Qt, QSize, Signal, Slot, QThread

# 导入相位解包裹模块（使用修改版，不在线程中显示图形）
import unwrap_phase as unwrap_phase


class UnwrapDirection(Enum):
    """解包裹方向枚举"""
    HORIZONTAL = 0    # 水平方向
    VERTICAL = 1      # 垂直方向
    BOTH = 2          # 两个方向


class UnwrappingWorker(QThread):
    """相位解包裹处理线程"""
    # 定义信号
    progress_updated = Signal(int)
    processing_done = Signal(dict)
    error_occurred = Signal(str)
    
    def __init__(self, 
                horizontal_images: Optional[List[str]] = None,
                vertical_images: Optional[List[str]] = None,
                unwrap_direction: UnwrapDirection = UnwrapDirection.BOTH,
                unwrap_method: str = "quality_guided",
                output_dir: str = "output",
                mask_method: str = "otsu",
                mask_confidence: float = 0.5):
        super().__init__()
        self.horizontal_images = horizontal_images
        self.vertical_images = vertical_images
        self.unwrap_direction = unwrap_direction
        self.unwrap_method = unwrap_method
        self.output_dir = output_dir
        self.mask_method = mask_method
        self.mask_confidence = mask_confidence
        
    def run(self):
        try:
            result = {}
            
            # 确保输出目录存在
            os.makedirs(self.output_dir, exist_ok=True)

            # 根据选择的方向进行解包裹
            if self.unwrap_direction in [UnwrapDirection.HORIZONTAL, UnwrapDirection.BOTH] and self.horizontal_images:
                self.progress_updated.emit(10)
                # 水平方向解包裹（垂直条纹图像）
                horizontal_dir = os.path.join(self.output_dir, "horizontal")
                os.makedirs(horizontal_dir, exist_ok=True)
                
                # 处理水平方向的图像
                processed_data = unwrap_phase.process_single_frequency_images(
                    self.horizontal_images, 
                    output_dir=horizontal_dir, 
                    method=self.unwrap_method,
                    show_plots=False,  # 不显示图形，只保存
                    use_mask=True,
                    mask_method=self.mask_method,
                    mask_confidence=self.mask_confidence
                )
                if processed_data:
                    result["horizontal"] = {
                        "unwrapped_phase": processed_data["unwrapped_phase"],
                        "wrapped_phase": processed_data["wrapped_phase"],
                        "output_dir": horizontal_dir
                    }
                
                self.progress_updated.emit(50)
            
            if self.unwrap_direction in [UnwrapDirection.VERTICAL, UnwrapDirection.BOTH] and self.vertical_images:
                # 垂直方向解包裹（水平条纹图像）
                vertical_dir = os.path.join(self.output_dir, "vertical")
                os.makedirs(vertical_dir, exist_ok=True)
                
                # 处理垂直方向的图像
                processed_data = unwrap_phase.process_single_frequency_images(
                    self.vertical_images, 
                    output_dir=vertical_dir, 
                    method=self.unwrap_method,
                    show_plots=False,  # 不显示图形，只保存
                    use_mask=True,
                    mask_method=self.mask_method,
                    mask_confidence=self.mask_confidence
                )
                if processed_data:
                    result["vertical"] = {
                        "unwrapped_phase": processed_data["unwrapped_phase"],
                        "wrapped_phase": processed_data["wrapped_phase"],
                        "output_dir": vertical_dir
                    }
                
                self.progress_updated.emit(90)
            
            # 如果两个方向都处理了，生成组合相位图
            if ("horizontal" in result and "vertical" in result and
                result["horizontal"]["unwrapped_phase"] is not None and
                result["vertical"]["unwrapped_phase"] is not None):
                
                self.progress_updated.emit(95)
                # 生成组合相位图 (此函数可能需要更新以处理字典)
                h_unwrapped = result["horizontal"]["unwrapped_phase"]
                v_unwrapped = result["vertical"]["unwrapped_phase"]
                
                combined_path = os.path.join(self.output_dir, "combined_phase.png")
                unwrap_phase.generate_combined_phase(
                    h_unwrapped,
                    v_unwrapped,
                    "水平和垂直方向相位组合图",
                    combined_path,
                    show_plots=False
                )
                
                # 保存组合相位数据
                np.save(os.path.join(self.output_dir, "combined_phase_h.npy"), h_unwrapped)
                np.save(os.path.join(self.output_dir, "combined_phase_v.npy"), v_unwrapped)
                
                # 添加到结果
                result["combined"] = {
                    "image_path": combined_path
                }
            
            self.progress_updated.emit(100)
            
            self.processing_done.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class InteractivePhaseViewer(QLabel):
    """
    一个可交互的相位图像查看器。
    - 支持鼠标悬停显示十字准星。
    - 实时显示鼠标位置的坐标和相位值。
    - 存储原始相位数据以便精确查找。
    """
    # 信号：当鼠标移动并需要更新外部信息标签时发出
    info_updated = Signal(str)
    
    def __init__(self, title: str = "图像查看器"):
        super().__init__()
        self.title = title
        self.phase_data_h = None  # 水平方向的原始相位数据
        self.phase_data_v = None  # 垂直方向的原始相位数据
        self.wrapped_phase_data_h = None # 水平包裹相位
        self.wrapped_phase_data_v = None # 垂直包裹相位
        self.pixmap = None
        self.mouse_pos = None
        self.is_combined = False
        
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd;")
        
        self.setText(f"{self.title}\n\n(暂无图像)")

    def set_phase_data(self, phase_data, title, is_combined=False, phase_data_v=None, wrapped_phase=None, wrapped_phase_v=None):
        """设置并显示相位数据。"""
        self.title = title
        self.is_combined = is_combined

        img_color = None
        if self.is_combined:
            self.phase_data_h = phase_data
            self.phase_data_v = phase_data_v
            self.wrapped_phase_data_h = wrapped_phase
            self.wrapped_phase_data_v = wrapped_phase_v
            
            if self.phase_data_h is not None and self.phase_data_v is not None:
                # 创建一个组合的伪彩色图像 (H -> Red, V -> Green)
                h_norm = cv2.normalize(self.phase_data_h, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                v_norm = cv2.normalize(self.phase_data_v, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # 创建一个RGB图像
                img_color = np.zeros((*self.phase_data_h.shape, 3), dtype=np.uint8)
                img_color[:,:,0] = h_norm
                img_color[:,:,1] = v_norm
        else:
            self.phase_data_h = phase_data
            self.wrapped_phase_data_h = wrapped_phase
            self.phase_data_v = None
            self.wrapped_phase_data_v = None
            
            if phase_data is not None:
                img_normalized = cv2.normalize(phase_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img_color = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
                img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

        if img_color is not None:
            height, width, channel = img_color.shape
            bytes_per_line = channel * width
            q_image = QImage(img_color.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(q_image)
            self.update()
        else:
            self.reset()


    def paintEvent(self, event):
        """重写绘制事件以添加十字准星和文本。"""
        super().paintEvent(event)
        if not self.pixmap:
            self.setText(f"{self.title}\n\n(暂无图像)")
            return

        painter = QPainter(self)
        
        # 计算图像在Label中的实际显示区域（保持宽高比）
        pixmap_size = self.pixmap.size()
        label_size = self.size()
        scaled_pixmap = self.pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x_offset = (label_size.width() - scaled_pixmap.width()) / 2
        y_offset = (label_size.height() - scaled_pixmap.height()) / 2
        
        # 绘制图像
        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
        
        # 绘制标题
        if self.title:
            painter.setPen(QColor(0, 0, 0))
            painter.setFont(QFont("Arial", 12, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignHCenter | Qt.AlignTop, self.title)

        # 当鼠标离开时，mouse_pos为None，不执行后续操作
        if self.mouse_pos is None:
            return

        # 将Label坐标转换为图像坐标
        img_x = self.mouse_pos.x() - x_offset
        img_y = self.mouse_pos.y() - y_offset
        
        # 检查坐标是否在图像区域内
        if 0 <= img_x < scaled_pixmap.width() and 0 <= img_y < scaled_pixmap.height():
            # 绘制十字准星
            painter.setPen(QPen(QColor(255, 255, 0, 180), 1, Qt.DashLine))
            painter.drawLine(x_offset, self.mouse_pos.y(), x_offset + scaled_pixmap.width(), self.mouse_pos.y())
            painter.drawLine(self.mouse_pos.x(), y_offset, self.mouse_pos.x(), y_offset + scaled_pixmap.height())

    def mouseMoveEvent(self, event):
        """处理鼠标移动事件。"""
        self.mouse_pos = event.position().toPoint()
        self.update() # 触发-paintEvent

        # 计算图像在Label中的实际显示区域
        if not self.pixmap:
            return
        pixmap_size = self.pixmap.size()
        label_size = self.size()
        scaled_pixmap = self.pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x_offset = (label_size.width() - scaled_pixmap.width()) / 2
        y_offset = (label_size.height() - scaled_pixmap.height()) / 2

        # 将Label坐标转换为图像坐标
        img_x = self.mouse_pos.x() - x_offset
        img_y = self.mouse_pos.y() - y_offset

        # 检查坐标是否在图像区域内
        if 0 <= img_x < scaled_pixmap.width() and 0 <= img_y < scaled_pixmap.height():
            # 将缩放后的图像坐标转换回原始图像坐标
            orig_x = int(img_x * (pixmap_size.width() / scaled_pixmap.width()))
            orig_y = int(img_y * (pixmap_size.height() / scaled_pixmap.height()))
            
            info_text = f"坐标: ({orig_x}, {orig_y})"
            
            if self.is_combined:
                if self.phase_data_h is not None and self.phase_data_v is not None:
                    phase_h = self.phase_data_h[orig_y, orig_x]
                    phase_v = self.phase_data_v[orig_y, orig_x]
                    info_text += f" | 水平相位: {phase_h:.3f}"
                    info_text += f" | 垂直相位: {phase_v:.3f}"
                    # 使用Cantor配对在给定精度下产生唯一组合相位ID
                    try:
                        from unwrap_phase import combine_pair_scalar
                        unique_id = combine_pair_scalar(float(phase_h), float(phase_v), precision=1e-3)
                        info_text += f" | 组合相位ID: {unique_id}"
                    except Exception:
                        pass

                    if self.wrapped_phase_data_h is not None and self.wrapped_phase_data_v is not None:
                        wrapped_h = self.wrapped_phase_data_h[orig_y, orig_x]
                        wrapped_v = self.wrapped_phase_data_v[orig_y, orig_x]
                        k_h = np.round((phase_h - wrapped_h) / (2 * np.pi))
                        k_v = np.round((phase_v - wrapped_v) / (2 * np.pi))
                        # 确保周期值为非负数
                        k_h = int(abs(k_h))
                        k_v = int(abs(k_v))
                        info_text += f" | 周期(H): {k_h}, (V): {k_v}"

            else:
                if self.phase_data_h is not None:
                    phase = self.phase_data_h[orig_y, orig_x]
                    info_text += f" | 相位值: {phase:.3f}"

                    if self.wrapped_phase_data_h is not None:
                        wrapped = self.wrapped_phase_data_h[orig_y, orig_x]
                        k = np.round((phase - wrapped) / (2 * np.pi))
                        # 确保周期值为非负数
                        k = int(abs(k))
                        info_text += f" | 周期数 k: {k}"
            
            self.info_updated.emit(info_text)
        else:
            self.info_updated.emit("")


    def leaveEvent(self, event):
        """处理鼠标离开事件。"""
        self.mouse_pos = None
        self.info_updated.emit("") # 清空信息
        self.update()
    
    def reset(self):
        """重置视图。"""
        self.phase_data_h = None
        self.phase_data_v = None
        self.wrapped_phase_data_h = None
        self.wrapped_phase_data_v = None
        self.pixmap = None
        self.mouse_pos = None
        self.setText(f"{self.title}\n\n(暂无图像)")
        self.update()


class CombinedViewerWindow(QWidget):
    """一个用于显示交互式组合相位图的新窗口"""
    def __init__(self, h_phase, v_phase, h_wrapped, v_wrapped, parent=None):
        super().__init__(parent)
        self.setWindowTitle("组合相位图 (H-红, V-绿)")
        self.setMinimumSize(600, 500)

        layout = QVBoxLayout(self)

        viewer = InteractivePhaseViewer("")
        viewer.set_phase_data(
            phase_data=h_phase,
            title="",
            is_combined=True,
            phase_data_v=v_phase,
            wrapped_phase=h_wrapped,
            wrapped_phase_v=v_wrapped
        )
        
        info_label = QLabel("将鼠标悬停在图像上以查看详细信息")
        info_label.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

        viewer.info_updated.connect(lambda text: info_label.setText(text or "将鼠标悬停在图像上以查看详细信息"))

        layout.addWidget(viewer, 1)
        layout.addWidget(info_label, 0)

        self.setAttribute(Qt.WA_DeleteOnClose)


class PhaseUnwrapperUI(QMainWindow):
    """相位解包裹程序主界面"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("相位解包裹程序")
        self.setMinimumSize(1200, 800)
        
        # 初始化数据
        self.horizontal_images = []  # 水平方向（垂直条纹）图像
        self.vertical_images = []    # 垂直方向（水平条纹）图像
        self.unwrap_direction = UnwrapDirection.BOTH
        self.unwrap_method = "improved_quality_guided"  # 默认使用改进的质量引导法
        self.mask_method = "otsu"  # 仅使用Otsu方法
        self.mask_confidence = 0.6  # 掩膜置信度，范围0.1-0.9
        self.output_dir = "phase_unwrap_results"
        self.combined_viewer_window = None # 用于持有对新窗口的引用
        
        # 设置应用样式
        self.set_application_style()
        
        # 初始化UI
        self.init_ui()
    
    def set_application_style(self):
        """设置应用程序样式"""
        # 设置应用程序调色板
        palette = QPalette()
        
        # 背景色（浅灰色）
        palette.setColor(QPalette.Window, QColor(245, 245, 245))
        
        # 窗口文本颜色（深灰色）
        palette.setColor(QPalette.WindowText, QColor(70, 70, 70))
        
        # 按钮颜色（淡蓝色）
        palette.setColor(QPalette.Button, QColor(210, 230, 255))
        
        # 按钮文本颜色（深蓝色）
        palette.setColor(QPalette.ButtonText, QColor(40, 80, 140))
        
        # 基础控件背景（白色）
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        
        # 设置应用程序调色板
        QApplication.setPalette(palette)
        
        # 设置全局样式表
        QApplication.setStyle("Fusion")
        
        style_sheet = """
        QMainWindow, QWidget {
            background-color: #f7f7f7;
        }
        
        QPushButton {
            background-color: #d5e8f8;
            border: 1px solid #a0c0e0;
            border-radius: 4px;
            padding: 6px 12px;
            color: #2c3e50;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #bbd5f1;
        }
        
        QPushButton:pressed {
            background-color: #a0c0e0;
        }
        
        QComboBox, QLineEdit {
            border: 1px solid #a0c0e0;
            border-radius: 4px;
            padding: 4px;
            background-color: white;
        }
        
        QGroupBox {
            border: 1px solid #a0c0e0;
            border-radius: 6px;
            margin-top: 12px;
            font-weight: bold;
            color: #2c3e50;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
            background-color: #f7f7f7;
        }
        
        QRadioButton {
            color: #2c3e50;
        }
        
        QLabel {
            color: #2c3e50;
        }
        
        QProgressBar {
            border: 1px solid #a0c0e0;
            border-radius: 4px;
            text-align: center;
            background-color: white;
        }
        
        QProgressBar::chunk {
            background-color: #3498db;
            width: 1px;
        }
        """
        
        QApplication.instance().setStyleSheet(style_sheet)
    
    def init_ui(self):
        """初始化UI界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建控制面板
        control_panel = self.create_control_panel()
        main_layout.addLayout(control_panel)
        
        # 创建状态面板
        status_panel = self.create_status_panel()
        main_layout.addLayout(status_panel)
        
        # 创建图像显示区域
        image_display = self.create_image_display()
        main_layout.addWidget(image_display, 1)
        
        # 创建底部信息栏
        self.info_label = QLabel("将鼠标悬停在图像上以查看详细信息")
        self.info_label.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        main_layout.addWidget(self.info_label)
    
    def create_control_panel(self):
        """创建控制面板"""
        control_layout = QHBoxLayout()
        
        # 图像选择组
        image_selection_group = QGroupBox("图像选择")
        image_selection_layout = QVBoxLayout()
        
        # 选择文件夹按钮
        self.select_folder_btn = QPushButton("选择图像文件夹")
        self.select_folder_btn.clicked.connect(self.select_image_folder)
        self.select_folder_btn.setToolTip("选择一个主文件夹，程序将根据下方设置的\"相移步数 (N)\"自动加载图像。\n"
                                        "文件夹内应包含至少 2*N 张按顺序命名的图像。\n"
                                        "例如：当 N=4 时，I1-I4 用于水平方向，I5-I8 用于垂直方向。")
        image_selection_layout.addWidget(self.select_folder_btn)
        
        # 选择水平图像按钮 (保留，但主要用于手动选择)
        select_horizontal_btn = QPushButton("手动选择水平图像 (0)")
        select_horizontal_btn.clicked.connect(lambda: self.select_images("horizontal"))
        select_horizontal_btn.setToolTip("手动选择N张用于水平方向解包裹的图像。")
        self.select_horizontal_btn = select_horizontal_btn
        image_selection_layout.addWidget(select_horizontal_btn)
        
        # 选择垂直图像按钮 (保留，但主要用于手动选择)
        select_vertical_btn = QPushButton("手动选择垂直图像 (0)")
        select_vertical_btn.clicked.connect(lambda: self.select_images("vertical"))
        select_vertical_btn.setToolTip("手动选择N张用于垂直方向解包裹的图像。")
        self.select_vertical_btn = select_vertical_btn
        image_selection_layout.addWidget(select_vertical_btn)
        
        image_selection_group.setLayout(image_selection_layout)
        
        # 解包裹方向选择组
        direction_group = QGroupBox("解包裹方向")
        direction_layout = QVBoxLayout()
        
        # 创建单选按钮
        horizontal_radio = QRadioButton("仅水平方向")
        vertical_radio = QRadioButton("仅垂直方向")
        both_radio = QRadioButton("两个方向")
        both_radio.setChecked(True)
        
        # 添加到布局
        direction_layout.addWidget(horizontal_radio)
        direction_layout.addWidget(vertical_radio)
        direction_layout.addWidget(both_radio)
        
        # 创建按钮组
        direction_button_group = QButtonGroup(self)
        direction_button_group.addButton(horizontal_radio, 0)  # HORIZONTAL
        direction_button_group.addButton(vertical_radio, 1)    # VERTICAL
        direction_button_group.addButton(both_radio, 2)        # BOTH
        direction_button_group.idClicked.connect(self.update_unwrap_direction)
        
        direction_group.setLayout(direction_layout)
        
        # 解包裹方法选择组
        method_group = QGroupBox("解包裹方法与参数")
        method_layout = QVBoxLayout()
        
        # 相移步数
        n_steps_layout = QHBoxLayout()
        self.n_steps_spin = QSpinBox()
        self.n_steps_spin.setRange(3, 16)
        self.n_steps_spin.setValue(4)
        self.n_steps_spin.setToolTip("设置相移的步数 (N)。\n当使用\"选择图像文件夹\"功能时，程序将需要 2*N 张图像。")
        n_steps_layout.addWidget(QLabel("相移步数 (N):"))
        n_steps_layout.addWidget(self.n_steps_spin)
        method_layout.addLayout(n_steps_layout)
        
        # 创建下拉菜单
        self.method_combo = QComboBox()
        self.method_combo.addItem("质量引导法 (原始)", "quality_guided")
        self.method_combo.addItem("改进质量引导法 (推荐)", "improved_quality_guided")
        self.method_combo.addItem("鲁棒解包裹法", "robust")
        self.method_combo.setCurrentIndex(1)  # 默认选择改进质量引导法
        self.method_combo.currentIndexChanged.connect(self.update_unwrap_method)
        self.method_combo.setToolTip("选择用于执行相位解包裹的核心算法。\n"
                                   "• 质量引导法(原始): 基础的质量引导解包裹\n"
                                   "• 改进质量引导法(推荐): 结合相位梯度的改进算法\n"
                                   "• 鲁棒解包裹法: 结合相位跳跃的鲁棒算法")
        
        method_layout.addWidget(QLabel("选择解包裹算法:"))
        method_layout.addWidget(self.method_combo)

        # 掩膜生成方法说明（固定使用Otsu方法）
        mask_info_label = QLabel("掩膜生成方法: Otsu自适应阈值")
        mask_info_label.setStyleSheet("color: #2c3e50; font-weight: bold;")
        method_layout.addWidget(mask_info_label)
        
        # 掩膜置信度（仅数值输入）
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("掩膜置信度:"))
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.1, 0.9)
        self.confidence_spinbox.setSingleStep(0.1)
        self.confidence_spinbox.setDecimals(1)
        self.confidence_spinbox.setValue(0.6)
        self.confidence_spinbox.valueChanged.connect(self.update_mask_confidence_from_spinbox)
        self.confidence_spinbox.setToolTip("输入 0.1-0.9 的数值以设置掩膜置信度\n\n"
                                         "• Otsu方法：基于Otsu算法的自动阈值化\n"
                                         "• 置信度对Otsu方法影响较小")
        confidence_layout.addWidget(self.confidence_spinbox)
        method_layout.addLayout(confidence_layout)
        
        # 置信度说明标签
        self.confidence_info_label = QLabel("推荐范围: 0.4-0.6 (平衡掩膜质量)")
        self.confidence_info_label.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        method_layout.addWidget(self.confidence_info_label)
        
        # 初始化置信度信息显示
        self.update_confidence_info()
        
        # 输出目录设置
        method_layout.addWidget(QLabel("输出目录:"))
        output_dir_layout = QHBoxLayout()
        self.output_dir_label = QLabel(self.output_dir)
        self.output_dir_label.setToolTip("显示当前用于保存结果的输出文件夹路径。")
        output_dir_layout.addWidget(self.output_dir_label)
        
        select_output_dir_btn = QPushButton("选择...")
        select_output_dir_btn.clicked.connect(self.select_output_dir)
        select_output_dir_btn.setToolTip("点击以选择一个新的文件夹来保存输出结果。")
        output_dir_layout.addWidget(select_output_dir_btn)
        
        method_layout.addLayout(output_dir_layout)
        
        method_group.setLayout(method_layout)
        
        # 操作按钮组
        operation_group = QGroupBox("操作")
        operation_layout = QVBoxLayout()

        # 开始处理按钮
        start_btn = QPushButton("开始处理")
        start_btn.setMinimumHeight(40)
        start_btn.clicked.connect(self.start_processing)
        start_btn.setToolTip("开始执行相位解包裹处理流程。")
        operation_layout.addWidget(start_btn)
        
        # 查看结果按钮
        view_results_btn = QPushButton("查看结果文件夹")
        view_results_btn.clicked.connect(self.open_result_folder)
        view_results_btn.setToolTip("在文件管理器中打开当前设置的输出文件夹。")
        operation_layout.addWidget(view_results_btn)

        # 重置按钮
        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self.reset_ui)
        reset_btn.setToolTip("清空所有选择的图像和结果，将界面恢复到初始状态。")
        operation_layout.addWidget(reset_btn)
        
        operation_group.setLayout(operation_layout)

        # 顶部三列布局，缓解拥挤
        left_column = QVBoxLayout()
        left_column.addWidget(image_selection_group)
        left_column.addWidget(direction_group)
        left_widget = QWidget()
        left_widget.setLayout(left_column)

        # 通过伸缩因子让三列随着窗口宽度自适应填满
        control_layout.addWidget(left_widget, 2)
        control_layout.addWidget(method_group, 3)
        control_layout.addWidget(operation_group, 1)
        
        return control_layout
    
    def create_status_panel(self):
        """创建状态面板"""
        status_layout = QHBoxLayout()
        
        # 状态标签
        self.status_label = QLabel("就绪")
        status_layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        
        return status_layout
    
    def create_viewer_tabs(self, title_2d, title_3d):
        """创建一个包含2D和3D视图的Tab小部件"""
        tabs = QTabWidget()
        
        # 2D 视图
        viewer_2d = InteractivePhaseViewer(title_2d)
        tabs.addTab(viewer_2d, "2D 相位图")
        
        # 3D 视图 (静态图像)
        viewer_3d = QLabel(f"{title_3d}\n\n(暂无图像)")
        viewer_3d.setAlignment(Qt.AlignCenter)
        viewer_3d.setMinimumSize(400, 300)
        viewer_3d.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd;")
        tabs.addTab(viewer_3d, "3D 表面图")

        # 连接交互式查看器的信号到底部信息栏
        viewer_2d.info_updated.connect(self.update_info_label)
        
        return tabs, viewer_2d, viewer_3d

    def create_image_display(self):
        """创建图像显示区域"""
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 创建水平方向的Tab视图
        self.h_tabs, self.horizontal_viewer_2d, self.horizontal_viewer_3d = self.create_viewer_tabs(
            "水平方向解包裹相位", "水平方向 3D 表面"
        )
        splitter.addWidget(self.h_tabs)
        
        # 创建垂直方向的Tab视图
        self.v_tabs, self.vertical_viewer_2d, self.vertical_viewer_3d = self.create_viewer_tabs(
            "垂直方向解包裹相位", "垂直方向 3D 表面"
        )
        splitter.addWidget(self.v_tabs)
        
        # 设置分割器的初始大小
        splitter.setSizes([600, 600])
        
        return splitter

    @Slot(str)
    def update_info_label(self, text: str):
        """更新底部信息标签的内容"""
        self.info_label.setText(text or "将鼠标悬停在图像上以查看详细信息")

    @Slot()
    def select_image_folder(self):
        """选择图像文件夹，并根据N步相移自动分配图像"""
        folder = QFileDialog.getExistingDirectory(self, "选择包含相移图像的主文件夹")
        if not folder:
            return
            
        n_steps = self.n_steps_spin.value()
        required_images = 2 * n_steps
            
        # 查找文件夹中的图像文件
        image_files = []
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(tuple(valid_extensions)):
                image_files.append(os.path.join(folder, f))
        
        if len(image_files) < n_steps:
            QMessageBox.warning(self, "图像不足", f"文件夹中仅找到 {len(image_files)} 张图像，无法进行最基本的 {n_steps}-步相移计算。")
            return
            
        # 根据图像数量分配
        if len(image_files) < required_images:
            QMessageBox.warning(self, "图像不足", f"检测到 {len(image_files)} 张图像，不足以进行双方向解包裹（需要 {required_images} 张）。\n"
                                                  f"将仅加载前 {n_steps} 张图像用于水平方向处理。")
            self.horizontal_images = image_files[:n_steps]
            self.vertical_images = []
        else:
            if len(image_files) > required_images:
                 QMessageBox.information(self, "提示", f"文件夹中包含 {len(image_files)} 张图像。\n"
                                                    f"将使用前 {n_steps} 张作为水平方向图像，\n"
                                                    f"接下来 {n_steps} 张作为垂直方向图像。")
            self.horizontal_images = image_files[:n_steps]
            self.vertical_images = image_files[n_steps:required_images]

        # 更新UI
        self.select_horizontal_btn.setText(f"水平方向图像 ({len(self.horizontal_images)})")
        self.select_vertical_btn.setText(f"垂直方向图像 ({len(self.vertical_images)})")
        self.status_label.setText(f"已从文件夹加载 {len(self.horizontal_images) + len(self.vertical_images)} 张图像")
    
    @Slot(str)
    def select_images(self, direction: str):
        """选择图像"""
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            f"选择{'水平' if direction == 'horizontal' else '垂直'}方向的{self.n_steps_spin.value()}步相移图像",
            "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        
        if not files:
            return
            
        n_steps = self.n_steps_spin.value()
        if len(files) < n_steps:
            QMessageBox.warning(self, "警告", f"请至少选择 {n_steps} 张相移图像。")
            return
            
        # 只取前N张
        selected_files = files[:n_steps]
        
        if direction == "horizontal":
            self.horizontal_images = selected_files
            self.select_horizontal_btn.setText(f"水平方向图像 ({len(self.horizontal_images)})")
        else:
            self.vertical_images = selected_files
            self.select_vertical_btn.setText(f"垂直方向图像 ({len(self.vertical_images)})")
        
        self.status_label.setText(f"已选择 {len(selected_files)} 张{('水平' if direction == 'horizontal' else '垂直')}方向图像")
    
    @Slot(int)
    def update_unwrap_direction(self, direction_id: int):
        """更新解包裹方向"""
        self.unwrap_direction = UnwrapDirection(direction_id)
    
    @Slot(int)
    def update_unwrap_method(self, index: int):
        """更新解包裹方法"""
        self.unwrap_method = self.method_combo.itemData(index)

    @Slot(float)
    def update_mask_confidence_from_spinbox(self, value: float):
        """从数值框更新掩膜置信度"""
        self.mask_confidence = value
        self.update_confidence_info()
    
    def update_confidence_info(self):
        """更新置信度说明信息（仅Otsu方法）"""
        confidence = self.mask_confidence
        
        # Otsu方法的说明
        if confidence < 0.4:
            info = f"当前值: {confidence:.1f} (Otsu方法，置信度对此方法影响较小)"
            color = "#868e96"  # 灰色
        elif confidence <= 0.6:
            info = f"当前值: {confidence:.1f} (Otsu方法，传统自动阈值化)"
            color = "#51cf66"  # 绿色
        else:
            info = f"当前值: {confidence:.1f} (Otsu方法，置信度对此方法影响较小)"
            color = "#868e96"  # 灰色
        
        self.confidence_info_label.setText(info)
        self.confidence_info_label.setStyleSheet(f"color: {color}; font-size: 11px; font-style: italic;")
    
    @Slot()
    def select_output_dir(self):
        """选择输出目录"""
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if folder:
            self.output_dir = folder
            self.output_dir_label.setText(self.output_dir)
    
    @Slot()
    def start_processing(self):
        """开始处理"""
        n_steps = self.n_steps_spin.value()
        
        # 检查是否有足够的图像
        if self.unwrap_direction in [UnwrapDirection.HORIZONTAL, UnwrapDirection.BOTH] and len(self.horizontal_images) < n_steps:
            QMessageBox.warning(self, "警告", f"请先选择至少 {n_steps} 张水平方向的相移图像")
            return
            
        if self.unwrap_direction in [UnwrapDirection.VERTICAL, UnwrapDirection.BOTH] and len(self.vertical_images) < n_steps:
            QMessageBox.warning(self, "警告", f"请先选择至少 {n_steps} 张垂直方向的相移图像")
            return

        # 更新UI状态
        self.status_label.setText("正在处理...")
        self.progress_bar.setValue(0)
        
        # 创建并启动工作线程
        self.worker = UnwrappingWorker(
            horizontal_images=self.horizontal_images if self.unwrap_direction in [UnwrapDirection.HORIZONTAL, UnwrapDirection.BOTH] else None,
            vertical_images=self.vertical_images if self.unwrap_direction in [UnwrapDirection.VERTICAL, UnwrapDirection.BOTH] else None,
            unwrap_direction=self.unwrap_direction,
            unwrap_method=self.unwrap_method,
            output_dir=self.output_dir,
            mask_method=self.mask_method,
            mask_confidence=self.mask_confidence
        )
        
        # 连接信号
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.processing_done.connect(self.processing_finished)
        self.worker.error_occurred.connect(self.handle_error)
        
        # 启动线程
        self.worker.start()
    
    @Slot(int)
    def update_progress(self, value: int):
        """更新进度条"""
        self.progress_bar.setValue(value)

    @Slot(dict)
    def processing_finished(self, result: dict):
        """处理完成后的回调"""
        self.status_label.setText("处理完成")
        self.reset_viewers() # 先重置所有视图

        # 显示水平结果
        if "horizontal" in result:
            h_data = result["horizontal"]
            if h_data and h_data.get("unwrapped_phase") is not None:
                self.horizontal_viewer_2d.set_phase_data(
                    h_data.get("unwrapped_phase"), 
                    "水平方向解包裹相位",
                    wrapped_phase=h_data.get("wrapped_phase")
                )
                # 加载3D图像
                h_3d_path = os.path.join(h_data["output_dir"], "unwrapped_phase_3d.png")
                if os.path.exists(h_3d_path):
                    pixmap = QPixmap(h_3d_path)
                    self.horizontal_viewer_3d.setPixmap(pixmap.scaled(self.horizontal_viewer_3d.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # 显示垂直结果
        if "vertical" in result:
            v_data = result["vertical"]
            if v_data and v_data.get("unwrapped_phase") is not None:
                self.vertical_viewer_2d.set_phase_data(
                    v_data.get("unwrapped_phase"),
                    "垂直方向解包裹相位",
                    wrapped_phase=v_data.get("wrapped_phase")
                )
                # 加载3D图像
                v_3d_path = os.path.join(v_data["output_dir"], "unwrapped_phase_3d.png")
                if os.path.exists(v_3d_path):
                    pixmap = QPixmap(v_3d_path)
                    self.vertical_viewer_3d.setPixmap(pixmap.scaled(self.vertical_viewer_3d.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 如果两个方向都处理完成，则在单独的窗口中显示组合图并触发3D表面图
        if "horizontal" in result and "vertical" in result:
            h_data = result.get("horizontal", {})
            v_data = result.get("vertical", {})
            h_phase = h_data.get("unwrapped_phase")
            v_phase = v_data.get("unwrapped_phase")

            if h_phase is not None and v_phase is not None:
                # 在新窗口中显示交互式2D组合图
                # 将窗口引用存储到self中，以防止它被垃圾回收
                self.combined_viewer_window = CombinedViewerWindow(
                    h_phase=h_phase,
                    v_phase=v_phase,
                    h_wrapped=h_data.get("wrapped_phase"),
                    v_wrapped=v_data.get("wrapped_phase")
                )
                self.combined_viewer_window.show()

                # 提示用户即将显示3D窗口
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setText("即将显示组合三维表面图。")
                msg_box.setInformativeText("关闭该图后程序将继续响应。")
                msg_box.setWindowTitle("组合视图")
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.exec()
                
                # 在新窗口中显示组合3D图
                unwrap_phase.visualize_combined_3d_surface(
                    h_phase,
                    v_phase,
                    title="组合相位 3D 表面",
                    show_plots=True  # 这将调用 plt.show()
                )
        
        # 显示成功消息
        QMessageBox.information(self, "成功", "相位解包裹处理完成")

    @Slot(str)
    def handle_error(self, error_msg: str):
        """处理错误"""
        self.status_label.setText(f"错误: {error_msg}")
        QMessageBox.critical(self, "错误", f"处理过程中发生错误:\n{error_msg}")
    
    @Slot()
    def open_result_folder(self):
        """打开结果文件夹"""
        if not os.path.exists(self.output_dir):
            QMessageBox.warning(self, "警告", "输出目录不存在")
            return
            
        # 使用系统默认程序打开文件夹
        import subprocess
        import platform
        
        if platform.system() == "Windows":
            os.startfile(self.output_dir)
        elif platform.system() == "Darwin":  # macOS
            subprocess.call(["open", self.output_dir])
        else:  # Linux
            subprocess.call(["xdg-open", self.output_dir])
    
    def reset_viewers(self):
        """重置所有视图到初始状态"""
        self.horizontal_viewer_2d.reset()
        self.horizontal_viewer_3d.setText("水平方向 3D 表面\n\n(暂无图像)")
        self.horizontal_viewer_3d.setPixmap(QPixmap()) # 清空
        
        self.vertical_viewer_2d.reset()
        self.vertical_viewer_3d.setText("垂直方向 3D 表面\n\n(暂无图像)")
        self.vertical_viewer_3d.setPixmap(QPixmap())
        
    @Slot()
    def reset_ui(self):
        """重置UI状态"""
        self.horizontal_images = []
        self.vertical_images = []
        self.select_horizontal_btn.setText("手动选择水平图像 (0)")
        self.select_vertical_btn.setText("垂直方向图像 (0)")
        self.reset_viewers()
        self.progress_bar.setValue(0)
        self.status_label.setText("就绪")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用图标
    #app.setWindowIcon(QIcon("icon.png"))
    
    window = PhaseUnwrapperUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 