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
    QScrollArea, QSplitter, QFrame, QTabWidget, QStackedWidget
)
from PySide6.QtGui import QPixmap, QImage, QColor, QPalette, QFont
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
                output_dir: str = "output"):
        super().__init__()
        self.horizontal_images = horizontal_images
        self.vertical_images = vertical_images
        self.unwrap_direction = unwrap_direction
        self.unwrap_method = unwrap_method
        self.output_dir = output_dir
        
    def run(self):
        try:
            result = {}
            
            # 确保输出目录存在
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 如果同时处理两个方向且都有图像，使用process_dual_direction_images
            if (self.unwrap_direction == UnwrapDirection.BOTH and 
                self.horizontal_images and self.vertical_images):
                self.progress_updated.emit(10)
                
                # 使用新函数同时处理两个方向
                unwrap_results = unwrap_phase.process_dual_direction_images(
                    h_image_paths=self.horizontal_images,
                    v_image_paths=self.vertical_images,
                    output_dir=self.output_dir,
                    method=self.unwrap_method,
                    show_plots=False  # 不显示图形，只保存
                )
                
                # 添加到结果
                if "horizontal" in unwrap_results:
                    result["horizontal"] = {
                        "unwrapped_phase": unwrap_results["horizontal"],
                        "output_dir": os.path.join(self.output_dir, "horizontal")
                    }
                    
                    # 添加水平方向的3D图像
                    h_3d_path = os.path.join(self.output_dir, "horizontal_3d.png")
                    if os.path.exists(h_3d_path):
                        result["horizontal"]["3d_path"] = h_3d_path
                
                if "vertical" in unwrap_results:
                    result["vertical"] = {
                        "unwrapped_phase": unwrap_results["vertical"],
                        "output_dir": os.path.join(self.output_dir, "vertical")
                    }
                    
                    # 添加垂直方向的3D图像
                    v_3d_path = os.path.join(self.output_dir, "vertical_3d.png")
                    if os.path.exists(v_3d_path):
                        result["vertical"]["3d_path"] = v_3d_path
                
                # 添加组合相位图
                combined_path = os.path.join(self.output_dir, "combined_phase.png")
                if os.path.exists(combined_path):
                    result["combined"] = {
                        "image_path": combined_path
                    }
                
                self.progress_updated.emit(100)
            else:
                # 原来的分别处理方式
                # 根据选择的方向进行解包裹
                if self.unwrap_direction in [UnwrapDirection.HORIZONTAL, UnwrapDirection.BOTH] and self.horizontal_images:
                    self.progress_updated.emit(10)
                    # 水平方向解包裹（垂直条纹图像）
                    horizontal_dir = os.path.join(self.output_dir, "horizontal")
                    os.makedirs(horizontal_dir, exist_ok=True)
                    
                    # 处理水平方向的图像
                    horizontal_unwrapped = unwrap_phase.process_four_step_images(
                        self.horizontal_images, 
                        output_dir=horizontal_dir, 
                        method=self.unwrap_method,
                        show_plots=False  # 不显示图形，只保存
                    )
                    result["horizontal"] = {
                        "unwrapped_phase": horizontal_unwrapped,
                        "output_dir": horizontal_dir
                    }
                    
                    # 添加水平方向的3D图像
                    h_3d_path = os.path.join(horizontal_dir, "unwrapped_phase_3d.png")
                    if os.path.exists(h_3d_path):
                        result["horizontal"]["3d_path"] = h_3d_path
                    
                    self.progress_updated.emit(50)
                
                if self.unwrap_direction in [UnwrapDirection.VERTICAL, UnwrapDirection.BOTH] and self.vertical_images:
                    # 垂直方向解包裹（水平条纹图像）
                    vertical_dir = os.path.join(self.output_dir, "vertical")
                    os.makedirs(vertical_dir, exist_ok=True)
                    
                    # 处理垂直方向的图像
                    vertical_unwrapped = unwrap_phase.process_four_step_images(
                        self.vertical_images, 
                        output_dir=vertical_dir, 
                        method=self.unwrap_method,
                        show_plots=False  # 不显示图形，只保存
                    )
                    result["vertical"] = {
                        "unwrapped_phase": vertical_unwrapped,
                        "output_dir": vertical_dir
                    }
                    
                    # 添加垂直方向的3D图像
                    v_3d_path = os.path.join(vertical_dir, "unwrapped_phase_3d.png")
                    if os.path.exists(v_3d_path):
                        result["vertical"]["3d_path"] = v_3d_path
                    
                    self.progress_updated.emit(90)
                
                # 如果两个方向都处理了，生成组合相位图
                if ("horizontal" in result and "vertical" in result and
                    result["horizontal"]["unwrapped_phase"] is not None and
                    result["vertical"]["unwrapped_phase"] is not None):
                    
                    self.progress_updated.emit(95)
                    # 生成组合相位图
                    combined_path = os.path.join(self.output_dir, "combined_phase.png")
                    combined_rgb = unwrap_phase.generate_combined_phase(
                        result["horizontal"]["unwrapped_phase"],
                        result["vertical"]["unwrapped_phase"],
                        "水平和垂直方向相位组合图",
                        combined_path,
                        show_plots=False
                    )
                    
                    # 保存组合相位数据
                    np.save(os.path.join(self.output_dir, "combined_phase.npy"), combined_rgb)
                    
                    # 添加到结果
                    result["combined"] = {
                        "image_path": combined_path
                    }
                    
                    # 生成并保存3D可视化
                    h_3d_path = os.path.join(self.output_dir, "horizontal_3d.png")
                    unwrap_phase.visualize_3d_surface(
                        result["horizontal"]["unwrapped_phase"],
                        "水平方向解包裹相位 3D 表面",
                        'viridis',
                        h_3d_path,
                        show_plots=False
                    )
                    result["horizontal"]["3d_path"] = h_3d_path
                    
                    v_3d_path = os.path.join(self.output_dir, "vertical_3d.png")
                    unwrap_phase.visualize_3d_surface(
                        result["vertical"]["unwrapped_phase"],
                        "垂直方向解包裹相位 3D 表面",
                        'plasma',
                        v_3d_path,
                        show_plots=False
                    )
                    result["vertical"]["3d_path"] = v_3d_path
                
                self.progress_updated.emit(100)
            
            self.processing_done.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class PhaseImageViewer(QWidget):
    """相位图像查看器组件"""
    
    def __init__(self, title: str = "图像查看器"):
        super().__init__()
        self.title = title
        self.init_ui()
    
    def init_ui(self):
        # 创建布局
        layout = QVBoxLayout()
        
        # 标题标签
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        # 图像标签
        self.image_label = QLabel("暂无图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd;")
        
        # 使用滚动区域包装图像标签
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)
        
        layout.addWidget(scroll)
        
        # 设置布局
        self.setLayout(layout)
    
    def set_image(self, image_path: str):
        """设置图像"""
        if not os.path.exists(image_path):
            self.image_label.setText(f"图像不存在: {image_path}")
            return
        
        # 读取图像
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_label.setText(f"无法加载图像: {image_path}")
            return
        
        # 调整图像大小以适应标签
        pixmap = pixmap.scaled(
            self.image_label.width(), 
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # 显示图像
        self.image_label.setPixmap(pixmap)
    
    def set_numpy_image(self, image: np.ndarray, colormap=cv2.COLORMAP_JET):
        """显示NumPy数组图像"""
        if image is None:
            self.image_label.setText("图像数据为空")
            return
        
        # 归一化到0-255
        if image.dtype != np.uint8:
            img_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            img_normalized = img_normalized.astype(np.uint8)
        else:
            img_normalized = image
        
        # 应用颜色映射
        if len(img_normalized.shape) == 2:  # 灰度图
            img_color = cv2.applyColorMap(img_normalized, colormap)
            # 转换BGR到RGB
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        else:  # 彩色图
            img_color = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
        
        # 创建QImage
        height, width, channel = img_color.shape
        bytes_per_line = channel * width
        q_image = QImage(img_color.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # 创建QPixmap并显示
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(
            self.image_label.width(), 
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.image_label.setPixmap(pixmap)


class PhaseViewerContainer(QWidget):
    """
    相位查看器容器，包含2D和3D视图
    """
    def __init__(self, title: str = "相位数据"):
        super().__init__()
        self.title = title
        self.init_ui()
    
    def init_ui(self):
        # 创建布局
        layout = QVBoxLayout()
        
        # 标题标签
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        # 创建标签页小部件
        self.tab_widget = QTabWidget()
        
        # 创建2D视图
        self.viewer_2d = PhaseImageViewer("2D 视图")
        self.tab_widget.addTab(self.viewer_2d, "2D 视图")
        
        # 创建3D视图
        self.viewer_3d = PhaseImageViewer("3D 视图")
        self.tab_widget.addTab(self.viewer_3d, "3D 视图")
        
        layout.addWidget(self.tab_widget)
        
        # 设置布局
        self.setLayout(layout)
    
    def set_2d_image(self, image_path: str):
        """设置2D图像"""
        self.viewer_2d.set_image(image_path)
    
    def set_3d_image(self, image_path: str):
        """设置3D图像"""
        if os.path.exists(image_path):
            self.viewer_3d.set_image(image_path)
            self.tab_widget.setTabVisible(1, True)
        else:
            self.tab_widget.setTabVisible(1, False)
    
    def reset(self):
        """重置视图"""
        self.viewer_2d.image_label.setText("暂无图像")
        self.viewer_3d.image_label.setText("暂无图像")
        self.tab_widget.setTabVisible(1, False)


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
        self.unwrap_method = "quality_guided"
        self.output_dir = "phase_unwrap_results"
        
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
    
    def create_control_panel(self):
        """创建控制面板"""
        control_layout = QHBoxLayout()
        
        # 图像选择组
        image_selection_group = QGroupBox("图像选择")
        image_selection_layout = QVBoxLayout()
        
        # 选择文件夹按钮
        select_folder_btn = QPushButton("选择图像文件夹")
        select_folder_btn.clicked.connect(self.select_image_folder)
        image_selection_layout.addWidget(select_folder_btn)
        
        # 选择水平图像按钮
        select_horizontal_btn = QPushButton("选择水平方向图像 (0)")
        select_horizontal_btn.clicked.connect(lambda: self.select_images("horizontal"))
        self.select_horizontal_btn = select_horizontal_btn
        image_selection_layout.addWidget(select_horizontal_btn)
        
        # 选择垂直图像按钮
        select_vertical_btn = QPushButton("选择垂直方向图像 (0)")
        select_vertical_btn.clicked.connect(lambda: self.select_images("vertical"))
        self.select_vertical_btn = select_vertical_btn
        image_selection_layout.addWidget(select_vertical_btn)
        
        image_selection_group.setLayout(image_selection_layout)
        control_layout.addWidget(image_selection_group)
        
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
        control_layout.addWidget(direction_group)
        
        # 解包裹方法选择组
        method_group = QGroupBox("解包裹方法")
        method_layout = QVBoxLayout()
        
        # 创建下拉菜单
        method_combo = QComboBox()
        method_combo.addItem("质量引导法", "quality_guided")
        method_combo.addItem("多频率法", "multi_freq")
        method_combo.addItem("时序相位解包裹", "temporal")
        method_combo.currentIndexChanged.connect(self.update_unwrap_method)
        
        method_layout.addWidget(QLabel("选择解包裹算法:"))
        method_layout.addWidget(method_combo)
        
        # 输出目录设置
        method_layout.addWidget(QLabel("输出目录:"))
        output_dir_layout = QHBoxLayout()
        self.output_dir_label = QLabel(self.output_dir)
        output_dir_layout.addWidget(self.output_dir_label)
        
        select_output_dir_btn = QPushButton("选择...")
        select_output_dir_btn.clicked.connect(self.select_output_dir)
        output_dir_layout.addWidget(select_output_dir_btn)
        
        method_layout.addLayout(output_dir_layout)
        
        method_group.setLayout(method_layout)
        control_layout.addWidget(method_group)
        
        # 操作按钮组
        operation_group = QGroupBox("操作")
        operation_layout = QVBoxLayout()
        
        # 开始处理按钮
        start_btn = QPushButton("开始处理")
        start_btn.setMinimumHeight(40)
        start_btn.clicked.connect(self.start_processing)
        operation_layout.addWidget(start_btn)
        
        # 查看结果按钮
        view_results_btn = QPushButton("查看结果文件夹")
        view_results_btn.clicked.connect(self.open_result_folder)
        operation_layout.addWidget(view_results_btn)
        
        # 重置按钮
        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self.reset_ui)
        operation_layout.addWidget(reset_btn)
        
        operation_group.setLayout(operation_layout)
        control_layout.addWidget(operation_group)
        
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
    
    def create_image_display(self):
        """创建图像显示区域"""
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 创建水平方向图像查看器
        self.horizontal_viewer = PhaseViewerContainer("水平方向解包裹相位")
        splitter.addWidget(self.horizontal_viewer)
        
        # 添加分隔线
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        splitter.addWidget(line)
        
        # 创建垂直方向图像查看器
        self.vertical_viewer = PhaseViewerContainer("垂直方向解包裹相位")
        splitter.addWidget(self.vertical_viewer)
        
        # 创建组合相位图像查看器（默认隐藏）
        self.combined_viewer = PhaseImageViewer("水平和垂直方向相位组合图")
        self.combined_viewer.hide()
        
        # 创建垂直布局来包含分割器和组合图像查看器
        layout = QVBoxLayout()
        layout.addWidget(splitter, 2)  # 分割器占2/3高度
        layout.addWidget(self.combined_viewer, 1)  # 组合图像查看器占1/3高度
        
        # 创建容器小部件
        container = QWidget()
        container.setLayout(layout)
        
        # 设置分割器的初始大小
        splitter.setSizes([500, 10, 500])
        
        return container
    
    @Slot()
    def select_image_folder(self):
        """选择图像文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if not folder:
            return
            
        # 查找文件夹中的图像文件
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            image_files.extend([os.path.join(folder, f) for f in os.listdir(folder) 
                               if f.lower().endswith(ext)])
        
        image_files.sort()  # 按文件名排序
        
        if len(image_files) < 4:
            QMessageBox.warning(self, "警告", "文件夹中图像数量不足四张，无法进行四步相移计算")
            return
            
        # 前四张作为水平方向图像
        self.horizontal_images = image_files[:4]
        self.select_horizontal_btn.setText(f"选择水平方向图像 ({len(self.horizontal_images)})")
        
        # 如果有八张或更多图像，则后四张作为垂直方向图像
        if len(image_files) >= 8:
            self.vertical_images = image_files[4:8]
            self.select_vertical_btn.setText(f"选择垂直方向图像 ({len(self.vertical_images)})")
        
        self.status_label.setText(f"已从文件夹加载 {len(image_files)} 张图像")
    
    @Slot(str)
    def select_images(self, direction: str):
        """选择图像"""
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            f"选择{'水平' if direction == 'horizontal' else '垂直'}方向的四步相移图像",
            "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        
        if not files:
            return
            
        if len(files) < 4:
            QMessageBox.warning(self, "警告", "请选择四张相移图像")
            return
            
        # 只取前四张
        selected_files = files[:4]
        
        if direction == "horizontal":
            self.horizontal_images = selected_files
            self.select_horizontal_btn.setText(f"选择水平方向图像 ({len(self.horizontal_images)})")
        else:
            self.vertical_images = selected_files
            self.select_vertical_btn.setText(f"选择垂直方向图像 ({len(self.vertical_images)})")
        
        self.status_label.setText(f"已选择 {len(selected_files)} 张{('水平' if direction == 'horizontal' else '垂直')}方向图像")
    
    @Slot(int)
    def update_unwrap_direction(self, direction_id: int):
        """更新解包裹方向"""
        self.unwrap_direction = UnwrapDirection(direction_id)
    
    @Slot(int)
    def update_unwrap_method(self, index: int):
        """更新解包裹方法"""
        sender = self.sender()
        self.unwrap_method = sender.itemData(index)
    
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
        # 检查是否有足够的图像
        if self.unwrap_direction in [UnwrapDirection.HORIZONTAL, UnwrapDirection.BOTH] and len(self.horizontal_images) < 4:
            QMessageBox.warning(self, "警告", "请先选择四张水平方向的相移图像")
            return
            
        if self.unwrap_direction in [UnwrapDirection.VERTICAL, UnwrapDirection.BOTH] and len(self.vertical_images) < 4:
            QMessageBox.warning(self, "警告", "请先选择四张垂直方向的相移图像")
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
            output_dir=self.output_dir
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
        
        # 显示结果
        if "horizontal" in result:
            horizontal_dir = result["horizontal"]["output_dir"]
            unwrapped_path = os.path.join(horizontal_dir, "unwrapped_phase.png")
            self.horizontal_viewer.set_2d_image(unwrapped_path)
            
            # 如果有3D图像，显示它
            if "3d_path" in result["horizontal"]:
                self.horizontal_viewer.set_3d_image(result["horizontal"]["3d_path"])
                
            self.horizontal_viewer.show()
        else:
            self.horizontal_viewer.hide()
        
        if "vertical" in result:
            vertical_dir = result["vertical"]["output_dir"]
            unwrapped_path = os.path.join(vertical_dir, "unwrapped_phase.png")
            self.vertical_viewer.set_2d_image(unwrapped_path)
            
            # 如果有3D图像，显示它
            if "3d_path" in result["vertical"]:
                self.vertical_viewer.set_3d_image(result["vertical"]["3d_path"])
                
            self.vertical_viewer.show()
        else:
            self.vertical_viewer.hide()
        
        # 如果有组合相位图，显示它
        if "combined" in result:
            combined_path = result["combined"]["image_path"]
            self.combined_viewer.set_image(combined_path)
            self.combined_viewer.show()
        else:
            self.combined_viewer.hide()
        
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
    
    @Slot()
    def reset_ui(self):
        """重置UI状态"""
        self.horizontal_images = []
        self.vertical_images = []
        self.select_horizontal_btn.setText("选择水平方向图像 (0)")
        self.select_vertical_btn.setText("选择垂直方向图像 (0)")
        self.horizontal_viewer.reset()
        self.vertical_viewer.reset()
        self.combined_viewer.image_label.setText("暂无图像")
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