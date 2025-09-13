#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多频外差法相位解包裹程序UI界面

基于PySide6构建的界面，用于便捷地进行多频外差法相位解包裹操作。
支持水平和垂直方向的相位解包裹。
"""

import sys
import os
import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict
from enum import Enum
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob # Added for folder scanning
import re # Added for folder scanning

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QGroupBox, 
    QRadioButton, QButtonGroup, QMessageBox, QProgressBar,
    QScrollArea, QSplitter, QFrame, QTabWidget,
    QSpinBox, QFormLayout
)
from PySide6.QtGui import QPixmap, QImage, QColor, QPalette, QFont, QPainter, QPen
from PySide6.QtCore import Qt, Signal, Slot, QThread

# 导入多频相位解包裹模块
import multi_freq_unwrap_phase as unwrap_phase


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
                output_dir: str = "output",
                multi_freq_h_data: Optional[Dict[int, List[str]]] = None,
                multi_freq_v_data: Optional[Dict[int, List[str]]] = None,
                filter_kernel_size: int = 9):
        super().__init__()
        self.output_dir = output_dir
        self.multi_freq_h_data = multi_freq_h_data
        self.multi_freq_v_data = multi_freq_v_data
        self.filter_kernel_size = filter_kernel_size
        
    def run(self):
        try:
            result = {}
            os.makedirs(self.output_dir, exist_ok=True)

            # --- 多频处理逻辑 (now supports dual direction) ---
            self.progress_updated.emit(10)
            if not self.multi_freq_h_data and not self.multi_freq_v_data:
                raise ValueError("没有提供多频模式的数据。")
            
            result = unwrap_phase.process_multi_frequency_dual_direction(
                h_freq_data=self.multi_freq_h_data,
                v_freq_data=self.multi_freq_v_data,
                output_dir=self.output_dir,
                method="multi_freq", # Hardcoded
                show_plots=False,
                filter_kernel_size=self.filter_kernel_size
            )
            self.progress_updated.emit(100)

            self.processing_done.emit(result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))


class InteractiveImageLabel(QLabel):
    """一个带有十字线交互的图像标签"""
    mouse_moved = Signal(object) # 发出鼠标移动事件
    mouse_left = Signal()      # 发出鼠标离开事件

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.crosshair_pos = None

    def mouseMoveEvent(self, event):
        self.crosshair_pos = event.pos()
        self.mouse_moved.emit(event)
        self.update() 

    def leaveEvent(self, event):
        self.crosshair_pos = None
        self.mouse_left.emit()
        self.update() 

    def paintEvent(self, event):
        super().paintEvent(event)
        
        pixmap = self.pixmap()
        if pixmap and not pixmap.isNull() and self.crosshair_pos:
            label_size = self.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            pixmap_w, pixmap_h = scaled_pixmap.width(), scaled_pixmap.height()
            offset_x = (label_size.width() - pixmap_w) / 2
            offset_y = (label_size.height() - pixmap_h) / 2
            
            x = self.crosshair_pos.x()
            y = self.crosshair_pos.y()

            if offset_x <= x < offset_x + pixmap_w and offset_y <= y < offset_y + pixmap_h:
                painter = QPainter(self)
                pen = QPen(QColor(220, 220, 220, 180)) 
                pen.setStyle(Qt.DashLine)
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawLine(int(offset_x), y, int(offset_x + pixmap_w), y)
                painter.drawLine(x, int(offset_y), x, int(offset_y + pixmap_h))


class PhaseImageViewer(QWidget):
    """相位图像查看器组件"""
    mouse_hover_info = Signal(str)
    
    def __init__(self, title: str = "图像查看器"):
        super().__init__()
        self.title = title
        self.phase_data = None
        self.k_map_data = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        self.image_label = InteractiveImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd;")
        
        self.image_label.mouse_moved.connect(self._on_mouse_move)
        self.image_label.mouse_left.connect(self._on_mouse_leave)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll)
        self.setLayout(layout)
    
    def set_image(self, image_path: str):
        if not os.path.exists(image_path):
            self.image_label.setText(f"图像不存在: {image_path}")
            return
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_label.setText(f"无法加载图像: {image_path}")
            return
        pixmap = pixmap.scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    def set_interactive_image(self, phase_data: np.ndarray, k_map: Optional[np.ndarray]):
        self.phase_data = phase_data
        self.k_map_data = k_map
        self.set_numpy_image(phase_data)

    def _on_mouse_leave(self):
        self.mouse_hover_info.emit("")

    def _on_mouse_move(self, event):
        if self.phase_data is None: return
        pixmap = self.image_label.pixmap()
        if pixmap is None or pixmap.isNull(): return
        
        label_pos = event.pos()
        label_w, label_h = self.image_label.width(), self.image_label.height()
        pixmap_w, pixmap_h = pixmap.width(), pixmap.height()
        pixmap_x_offset = (label_w - pixmap_w) / 2
        pixmap_y_offset = (label_h - pixmap_h) / 2
        pixmap_x = label_pos.x() - pixmap_x_offset
        pixmap_y = label_pos.y() - pixmap_y_offset

        if not (0 <= pixmap_x < pixmap_w and 0 <= pixmap_y < pixmap_h):
            self.mouse_hover_info.emit("")
            return

        data_h, data_w = self.phase_data.shape
        data_ix = int(pixmap_x * data_w / pixmap_w)
        data_iy = int(pixmap_y * data_h / pixmap_h)

        if not (0 <= data_ix < data_w and 0 <= data_iy < data_h):
            self.mouse_hover_info.emit("")
            return

        phase_value = self.phase_data[data_iy, data_ix]
        if self.k_map_data is not None:
            k_value = self.k_map_data[data_iy, data_ix]
            info_str = f"坐标: ({data_ix}, {data_iy})   相位: {phase_value:.4f} rad   周期数: {int(k_value)}"
        else:
            info_str = f"坐标: ({data_ix}, {data_iy})   相位: {phase_value:.4f} rad"
        self.mouse_hover_info.emit(info_str)

    def set_numpy_image(self, image: np.ndarray, colormap=cv2.COLORMAP_JET):
        if image is None:
            self.image_label.setText("图像数据为空")
            return
        
        if image.dtype != np.uint8:
            img_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            img_normalized = img_normalized.astype(np.uint8)
        else:
            img_normalized = image
        
        if len(img_normalized.shape) == 2:
            img_color = cv2.applyColorMap(img_normalized, colormap)
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        else:
            img_color = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
        
        height, width, channel = img_color.shape
        bytes_per_line = channel * width
        q_image = QImage(img_color.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)


class PhaseViewerContainer(QWidget):
    """相位查看器容器，包含2D和3D视图"""
    def __init__(self, title: str = "相位数据"):
        super().__init__()
        self.title = title
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        self.tab_widget = QTabWidget()
        self.viewer_2d = PhaseImageViewer("2D 视图")
        self.tab_widget.addTab(self.viewer_2d, "2D 视图")
        self.viewer_3d = PhaseImageViewer("3D 视图")
        self.tab_widget.addTab(self.viewer_3d, "3D 视图")
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
    
    def set_interactive_2d_image(self, phase_data: np.ndarray, k_map: Optional[np.ndarray]):
        self.viewer_2d.set_interactive_image(phase_data, k_map)

    def set_3d_image(self, image_path: str):
        if os.path.exists(image_path):
            self.viewer_3d.set_image(image_path)
            self.tab_widget.setTabVisible(1, True)
        else:
            self.tab_widget.setTabVisible(1, False)
    
    def reset(self):
        self.viewer_2d.image_label.setText("暂无图像")
        self.viewer_3d.image_label.setText("暂无图像")
        self.tab_widget.setTabVisible(1, False)


class MultiFreqPhaseUnwrapperUI(QMainWindow):
    """多频外差法相位解包裹程序主界面"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多频外差法相位解包裹程序")
        self.setMinimumSize(1200, 800)
        
        self.freq_widgets = []
        self.unwrap_direction = UnwrapDirection.BOTH
        self.output_dir = "multi_freq_phase_unwrap_results"
        self.n_steps = 4
        self.permanent_status_message = "就绪"
        
        self.set_application_style()
        self.init_ui()
        self.update_multi_freq_widgets(3) # Initialize with 3 frequencies
    
    def set_application_style(self):
        QApplication.setStyle("Fusion")
        style_sheet = """
        QMainWindow, QWidget { background-color: #f7f7f7; }
        QPushButton {
            background-color: #d5e8f8; border: 1px solid #a0c0e0;
            border-radius: 4px; padding: 6px 12px;
            color: #2c3e50; font-weight: bold;
        }
        QPushButton:hover { background-color: #bbd5f1; }
        QPushButton:pressed { background-color: #a0c0e0; }
        QComboBox, QLineEdit, QSpinBox {
            border: 1px solid #a0c0e0; border-radius: 4px;
            padding: 4px; background-color: white;
        }
        QGroupBox {
            border: 1px solid #a0c0e0; border-radius: 6px;
            margin-top: 12px; font-weight: bold; color: #2c3e50;
        }
        QGroupBox::title {
            subcontrol-origin: margin; subcontrol-position: top center;
            padding: 0 5px; background-color: #f7f7f7;
        }
        QRadioButton, QLabel { color: #2c3e50; }
        QProgressBar {
            border: 1px solid #a0c0e0; border-radius: 4px;
            text-align: center; background-color: white;
        }
        QProgressBar::chunk { background-color: #3498db; width: 1px; }
        """
        QApplication.instance().setStyleSheet(style_sheet)
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        control_panel = self.create_control_panel()
        main_layout.addLayout(control_panel)
        
        status_panel = self.create_status_panel()
        main_layout.addLayout(status_panel)
        
        image_display = self.create_image_display()
        main_layout.addWidget(image_display, 1)
    
    def create_control_panel(self):
        control_layout = QHBoxLayout()
        control_layout.setSpacing(20)

        # --- 左侧：参数设置 ---
        settings_widget = self.create_settings_panel()
        control_layout.addWidget(settings_widget, 2)

        # --- 右侧：操作和方向 ---
        right_panel_widget = self.create_operation_panel()
        control_layout.addWidget(right_panel_widget, 1)
        
        return control_layout

    def create_settings_panel(self):
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)

        general_group = QGroupBox("通用设置")
        general_layout = QFormLayout(general_group)
        self.n_steps_spinbox = QSpinBox()
        self.n_steps_spinbox.setRange(3, 20)
        self.n_steps_spinbox.setValue(self.n_steps)
        self.n_steps_spinbox.setToolTip("设置相移的步数(N)。\n程序会根据此设置寻找并加载指定数量的图像。")
        self.n_steps_spinbox.valueChanged.connect(self.update_n_steps)
        general_layout.addRow("相移步数 (N):", self.n_steps_spinbox)
        settings_layout.addWidget(general_group)

        self.multi_freq_settings_group = QGroupBox("多频设置")
        multi_freq_form_layout = QFormLayout(self.multi_freq_settings_group)
        self.num_freq_spinbox = QSpinBox()
        self.num_freq_spinbox.setRange(2, 8)
        self.num_freq_spinbox.setValue(3)
        self.num_freq_spinbox.setToolTip("设置要使用的不同条纹频率的数量。")
        self.num_freq_spinbox.valueChanged.connect(self.update_multi_freq_widgets)
        multi_freq_form_layout.addRow("频率数量:", self.num_freq_spinbox)

        self.filter_size_spinbox = QSpinBox()
        self.filter_size_spinbox.setRange(3, 21)
        self.filter_size_spinbox.setSingleStep(2)
        self.filter_size_spinbox.setValue(9)
        self.filter_size_spinbox.setToolTip("设置用于平滑条纹阶数的2D中值滤波器的边长。")
        multi_freq_form_layout.addRow("滤波器尺寸:", self.filter_size_spinbox)

        self.multi_freq_widgets_container = QWidget()
        self.multi_freq_widgets_layout = QVBoxLayout(self.multi_freq_widgets_container)
        multi_freq_form_layout.addRow(self.multi_freq_widgets_container)
        
        settings_layout.addWidget(self.multi_freq_settings_group)
        settings_layout.addStretch()
        return settings_widget

    def create_operation_panel(self):
        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)

        operation_group = QGroupBox("操作")
        operation_layout = QVBoxLayout(operation_group)
        operation_layout.setSpacing(15)
        start_btn = QPushButton("开始处理")
        start_btn.setToolTip("根据当前设置开始执行相位解包裹。")
        start_btn.setMinimumHeight(40)
        start_btn.clicked.connect(self.start_processing)
        view_results_btn = QPushButton("查看结果文件夹")
        view_results_btn.clicked.connect(self.open_result_folder)
        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self.reset_ui)
        operation_layout.addWidget(start_btn)
        operation_layout.addWidget(view_results_btn)
        operation_layout.addWidget(reset_btn)

        self.direction_group = QGroupBox("解包裹方向")
        direction_layout = QVBoxLayout(self.direction_group)
        self.horizontal_radio = QRadioButton("仅水平方向")
        self.vertical_radio = QRadioButton("仅垂直方向")
        self.both_radio = QRadioButton("两个方向")
        self.both_radio.setChecked(True)
        direction_layout.addWidget(self.horizontal_radio)
        direction_layout.addWidget(self.vertical_radio)
        direction_layout.addWidget(self.both_radio)
        self.direction_button_group = QButtonGroup(self)
        self.direction_button_group.addButton(self.horizontal_radio, 0)
        self.direction_button_group.addButton(self.vertical_radio, 1)
        self.direction_button_group.addButton(self.both_radio, 2)
        self.direction_button_group.idClicked.connect(self.update_unwrap_direction)

        output_group = QGroupBox("输出设置")
        output_layout = QHBoxLayout(output_group)
        self.output_dir_label = QLabel(self.output_dir)
        self.output_dir_label.setWordWrap(True)
        select_output_dir_btn = QPushButton("选择...")
        select_output_dir_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.output_dir_label)
        output_layout.addWidget(select_output_dir_btn)

        right_panel_layout.addWidget(operation_group)
        right_panel_layout.addWidget(self.direction_group)
        right_panel_layout.addWidget(output_group)
        right_panel_layout.addStretch()
        return right_panel_widget
    
    def create_status_panel(self):
        status_layout = QHBoxLayout()
        self.status_label = QLabel(self.permanent_status_message)
        status_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        return status_layout
    
    def create_image_display(self):
        splitter = QSplitter(Qt.Horizontal)
        self.horizontal_viewer = PhaseViewerContainer("水平方向解包裹相位")
        self.horizontal_viewer.viewer_2d.mouse_hover_info.connect(self._update_hover_info)
        splitter.addWidget(self.horizontal_viewer)
        
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        splitter.addWidget(line)
        
        self.vertical_viewer = PhaseViewerContainer("垂直方向解包裹相位")
        self.vertical_viewer.viewer_2d.mouse_hover_info.connect(self._update_hover_info)
        splitter.addWidget(self.vertical_viewer)
        
        splitter.setSizes([500, 10, 500])
        return splitter

    def update_multi_freq_widgets(self, num_freqs):
        for i in reversed(range(self.multi_freq_widgets_layout.count())): 
            self.multi_freq_widgets_layout.itemAt(i).widget().setParent(None)
        self.freq_widgets.clear()

        for i in range(num_freqs):
            group = QGroupBox(f"频率 {i+1}")
            main_row_layout = QHBoxLayout(group)
            freq_spinbox = QSpinBox()
            freq_spinbox.setRange(1, 100)
            freq_spinbox.setValue([1, 8, 64, 32, 64, 1, 1, 1][i])
            freq_spinbox.setToolTip(f"设置第 {i+1} 组条纹图的频率（周期数）。")
            main_row_layout.addWidget(QLabel("F:"))
            main_row_layout.addWidget(freq_spinbox)
            btn_folder = QPushButton("选择文件夹 (0)")
            btn_folder.setToolTip(f"为频率 {i+1} 选择一个包含N或2N张图像的文件夹。")
            btn_folder.clicked.connect(lambda checked=False, index=i: self.select_multi_freq_folder(index))
            main_row_layout.addWidget(btn_folder, 1)
            self.multi_freq_widgets_layout.addWidget(group)
            self.freq_widgets.append({"btn": btn_folder, "spinbox": freq_spinbox, "h_paths": [], "v_paths": []})
    
    @Slot(int)
    def select_multi_freq_folder(self, index):
        folder = QFileDialog.getExistingDirectory(self, f"为频率 {index+1} 选择图像文件夹", "")
        if not folder: return

        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            image_files.extend(
                glob.glob(os.path.join(folder, f"*{ext}"), recursive=False) + 
                glob.glob(os.path.join(folder, f"*{ext.upper()}"), recursive=False)
            )
        image_files = sorted(list(set(image_files)))
        
        n = self.n_steps
        def get_img_num(path):
            match = re.search(r'[iI](\d+)\.', os.path.basename(path))
            return int(match.group(1)) if match else -1
        all_imgs_map = {get_img_num(p): p for p in image_files if get_img_num(p) != -1}

        h_paths = [all_imgs_map.get(i) for i in range(1, n + 1) if all_imgs_map.get(i)]
        v_paths = [all_imgs_map.get(i) for i in range(n + 1, 2 * n + 1) if all_imgs_map.get(i)]

        found_h = len(h_paths) == n
        found_v = len(v_paths) == n

        self.freq_widgets[index]["h_paths"] = h_paths if found_h else []
        self.freq_widgets[index]["v_paths"] = v_paths if found_v else []
        
        if found_h and found_v:
            btn_text, info_text = f"水平+垂直 ({2*n}张)", f"频率{index+1}: 已加载水平和垂直图像。"
        elif found_h:
            btn_text, info_text = f"仅水平 ({n}张)", f"频率{index+1}: 已加载水平图像。"
        elif found_v:
            btn_text, info_text = f"仅垂直 ({n}张)", f"频率{index+1}: 已加载垂直图像。"
        else:
            btn_text, info_text = "选择文件夹 (0)", f"频率{index+1}: 未找到符合命名规则(I1-I{2*n})的完整图像集。"
            QMessageBox.warning(self, "未找到图像", info_text)

        self.freq_widgets[index]["btn"].setText(btn_text)
        self.status_label.setText(info_text)

    @Slot(int)
    def update_unwrap_direction(self, direction_id: int):
        self.unwrap_direction = UnwrapDirection(direction_id)

    @Slot(int)
    def update_n_steps(self, value: int):
        if self.n_steps != value:
            self.n_steps = value
            self.reset_ui()
            QMessageBox.information(self, "提示", f"相移步数已更新为 {self.n_steps}。\n图像选择已重置，请重新加载。")
    
    @Slot()
    def select_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if folder:
            self.output_dir = folder
            self.output_dir_label.setText(self.output_dir)
    
    def start_processing(self):
        h_freq_data, v_freq_data = {}, {}
        for widget_set in self.freq_widgets:
            freq = widget_set["spinbox"].value()
            if widget_set["h_paths"]: h_freq_data[freq] = widget_set["h_paths"]
            if widget_set["v_paths"]: v_freq_data[freq] = widget_set["v_paths"]

        can_process_h, can_process_v = len(h_freq_data) >= 2, len(v_freq_data) >= 2
        final_h_data, final_v_data = (h_freq_data if can_process_h else {}), (v_freq_data if can_process_v else {})

        if self.unwrap_direction == UnwrapDirection.HORIZONTAL:
            if not can_process_h:
                QMessageBox.warning(self, "数据不足", "水平方向处理需要至少为2个频率加载图像。"); return
            final_v_data = {}
        elif self.unwrap_direction == UnwrapDirection.VERTICAL:
            if not can_process_v:
                QMessageBox.warning(self, "数据不足", "垂直方向处理需要至少为2个频率加载图像。"); return
            final_h_data = {}
        elif self.unwrap_direction == UnwrapDirection.BOTH:
            if not can_process_h and not can_process_v:
                QMessageBox.warning(self, "数据不足", "需要至少为一个方向加载2个频率以上的图像。"); return
            msg = ""
            if not can_process_h: msg += "缺少水平数据，仅处理垂直方向。\n"
            if not can_process_v: msg += "缺少垂直数据，仅处理水平方向。\n"
            if msg: QMessageBox.information(self, "处理调整", msg.strip())

        if not final_h_data and not final_v_data:
            QMessageBox.warning(self, "无有效数据", "没有足够内容可以处理。"); return
        
        worker_params = {
            "output_dir": self.output_dir,
            "multi_freq_h_data": final_h_data,
            "multi_freq_v_data": final_v_data,
            "filter_kernel_size": self.filter_size_spinbox.value()
        }
        self.worker = UnwrappingWorker(**worker_params)
        self.worker.processing_done.connect(self.handle_processing_finished)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.error_occurred.connect(self.handle_error)
        
        self.permanent_status_message = "正在处理..."
        self.status_label.setText(self.permanent_status_message)
        self.progress_bar.setValue(0)
        self.worker.start()
    
    @Slot(int)
    def update_progress(self, value: int):
        self.progress_bar.setValue(value)
    
    def handle_processing_finished(self, result: dict):
        self.permanent_status_message = "多频处理完成"
        self.status_label.setText(self.permanent_status_message)

        if "horizontal" in result and result["horizontal"].get("unwrapped_phase") is not None:
            res_data = result["horizontal"]
            d3_path = os.path.join(res_data["output_dir"], "unwrapped_phase_final_3d.png")
            self.horizontal_viewer.set_interactive_2d_image(res_data["unwrapped_phase"], res_data.get("k_map"))
            self.horizontal_viewer.set_3d_image(d3_path)
        else:
            self.horizontal_viewer.reset()

        if "vertical" in result and result["vertical"].get("unwrapped_phase") is not None:
            res_data = result["vertical"]
            d3_path = os.path.join(res_data["output_dir"], "unwrapped_phase_final_3d.png")
            self.vertical_viewer.set_interactive_2d_image(res_data["unwrapped_phase"], res_data.get("k_map"))
            self.vertical_viewer.set_3d_image(d3_path)
        else:
            self.vertical_viewer.reset()

        QMessageBox.information(self, "成功", "多频相位解包裹处理完成")

    @Slot(str)
    def handle_error(self, error_msg: str):
        self.permanent_status_message = f"错误: {error_msg}"
        self.status_label.setText(self.permanent_status_message)
        QMessageBox.critical(self, "错误", f"处理过程中发生错误:\n{error_msg}")
    
    @Slot()
    def open_result_folder(self):
        if not os.path.exists(self.output_dir):
            QMessageBox.warning(self, "警告", "输出目录不存在"); return
        import platform, subprocess
        if platform.system() == "Windows": os.startfile(self.output_dir)
        elif platform.system() == "Darwin": subprocess.call(["open", self.output_dir])
        else: subprocess.call(["xdg-open", self.output_dir])
    
    @Slot(str)
    def _update_hover_info(self, info: str):
        if info: self.status_label.setText(info)
        else: self.status_label.setText(self.permanent_status_message)

    @Slot()
    def reset_ui(self):
        for widget_set in self.freq_widgets:
            widget_set["h_paths"], widget_set["v_paths"] = [], []
            widget_set["btn"].setText("选择文件夹 (0)")
        self.horizontal_viewer.reset()
        self.vertical_viewer.reset()
        self.progress_bar.setValue(0)
        self.permanent_status_message = "就绪"
        self.status_label.setText(self.permanent_status_message)


def main():
    app = QApplication(sys.argv)
    window = MultiFreqPhaseUnwrapperUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 