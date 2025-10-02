#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机标定 UI 程序

该程序提供基于PySide6的图形用户界面，用于相机标定流程。
允许用户选择标定图像、配置标定板参数、执行标定并查看结果。

作者: [Your Name]
日期: [Date]
"""

import os
import sys
import numpy as np
import cv2
import glob
import json
from datetime import datetime
import matplotlib
# 设置后端为Agg，这是一个非交互式后端，可以在没有GUI的情况下工作
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
# 移除对FigureCanvasQTAgg的导入
import io
from PySide6.QtCore import (Qt, QSize, QTimer, QPropertyAnimation, 
                           QEasingCurve, QPoint, QRect, Signal, Slot, Property)
from PySide6.QtGui import (QIcon, QPixmap, QImage, QColor, QPainter, 
                          QPalette, QBrush, QLinearGradient, QFont, QPen, QPainterPath)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                              QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
                              QLineEdit, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox,
                              QCheckBox, QTabWidget, QScrollArea, QFrame, QGroupBox, QSizePolicy,
                              QSlider, QProgressBar, QMessageBox, QListWidget, QListWidgetItem,
                              QToolBar, QStatusBar, QDialog, QGraphicsDropShadowEffect)

# 导入相机标定功能模块
import camera_calibration as cam_calib

# 自定义卡片式容器组件
class CardWidget(QFrame):
    """现代化卡片式容器，带有阴影和圆角"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cardWidget")
        
        # 设置圆角和阴影效果
        self.setStyleSheet("""
            #cardWidget {
                background-color: #ffffff;
                border-radius: 10px;
            }
        """)
        
        # 创建阴影效果
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(15)
        self.shadow.setColor(QColor(0, 0, 0, 30))
        self.shadow.setOffset(0, 2)
        self.setGraphicsEffect(self.shadow)
        
        # 创建内部布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)

# 自定义风格按钮
class StylishButton(QPushButton):
    """现代化风格按钮，带有悬停效果"""
    def __init__(self, text="", parent=None, primary=True):
        super().__init__(text, parent)
        self.primary = primary
        self.setObjectName("primaryButton" if primary else "secondaryButton")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(36)
        
        # 设置风格
        self._update_style()
        
    def _update_style(self):
        if self.primary:
            self.setStyleSheet("""
                #primaryButton {
                    background-color: #2196f3;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                #primaryButton:hover {
                    background-color: #1976d2;
                }
                #primaryButton:pressed {
                    background-color: #0d47a1;
                }
                #primaryButton:disabled {
                    background-color: #bbdefb;
                    color: #e3f2fd;
                }
            """)
        else:
            self.setStyleSheet("""
                #secondaryButton {
                    background-color: #e0e0e0;
                    color: #424242;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                #secondaryButton:hover {
                    background-color: #bdbdbd;
                }
                #secondaryButton:pressed {
                    background-color: #9e9e9e;
                }
                #secondaryButton:disabled {
                    background-color: #f5f5f5;
                    color: #e0e0e0;
                }
            """)

# 自定义图像预览组件
class ImagePreview(QLabel):
    """图像预览组件，支持拖放和点击"""
    clicked = Signal()  # 定义点击信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 2px dashed #bdbdbd;
                border-radius: 5px;
            }
        """)
        self.setText("点击或拖放图像以预览")
        self.pixmap = None
        
    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        self.clicked.emit()
        super().mousePressEvent(event)

    def setImage(self, image_data):
        """设置预览图像"""
        if isinstance(image_data, str) and os.path.isfile(image_data):
            # 从文件加载
            self.pixmap = QPixmap(image_data)
        elif isinstance(image_data, np.ndarray):
            # 从OpenCV图像转换
            height, width, channel = image_data.shape
            bytes_per_line = 3 * width
            q_img = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.pixmap = QPixmap.fromImage(q_img)
        else:
            return
            
        # 根据控件大小缩放图像
        if self.pixmap and not self.pixmap.isNull():
            self.pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(self.pixmap)

    def resizeEvent(self, event):
        """处理大小变化事件"""
        if self.pixmap and not self.pixmap.isNull():
            scaled_pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
        super().resizeEvent(event)

# 结果查看对话框
class CalibrationResultDialog(QDialog):
    """显示标定结果的对话框"""
    def __init__(self, result_dict, parent=None):
        super().__init__(parent)
        self.result_dict = result_dict
        
        self.setWindowTitle("相机标定结果")
        self.setMinimumSize(800, 600)
        
        # 创建选项卡
        tabs = QTabWidget()
        
        # 相机矩阵选项卡
        matrix_tab = QWidget()
        matrix_layout = QVBoxLayout(matrix_tab)
        
        # 使用表格形式显示相机矩阵
        matrix_label = QLabel(f"""
        <h3>相机内参矩阵</h3>
        <pre>
        [{result_dict['camera_matrix'][0,0]:.2f}  {result_dict['camera_matrix'][0,1]:.2f}  {result_dict['camera_matrix'][0,2]:.2f}]
        [{result_dict['camera_matrix'][1,0]:.2f}  {result_dict['camera_matrix'][1,1]:.2f}  {result_dict['camera_matrix'][1,2]:.2f}]
        [{result_dict['camera_matrix'][2,0]:.2f}  {result_dict['camera_matrix'][2,1]:.2f}  {result_dict['camera_matrix'][2,2]:.2f}]
        </pre>
        
        <h3>畸变系数</h3>
        <pre>
        {', '.join([f"{x:.4f}" for x in result_dict['dist_coeffs'].flatten()])}
        </pre>
        
        <h3>标定质量</h3>
        <p>平均重投影误差: {result_dict['reprojection_error']:.4f} 像素</p>
        <p>成功标定的图像: {len(result_dict.get('successful_images', []))}/{result_dict.get('total_images', 0)}</p>
        </pre>
        """)
        
        matrix_label.setTextFormat(Qt.RichText)
        matrix_layout.addWidget(matrix_label)
        
        # 误差可视化选项卡
        error_tab = QWidget()
        error_layout = QVBoxLayout(error_tab)
        
        # 创建一个QLabel来显示误差图
        error_image_label = QLabel()
        error_image_label.setAlignment(Qt.AlignCenter)
        
        # 绘制每个图像的重投影误差
        if 'per_view_errors' in result_dict:
            # 使用matplotlib创建图表
            plt.figure(figsize=(8, 6), dpi=100)
            errors = result_dict['per_view_errors']
            plt.bar(range(len(errors)), errors)
            plt.xlabel('图像索引')
            plt.ylabel('重投影误差 (像素)')
            plt.title('每张图像的重投影误差')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 将图表保存到内存缓冲区
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            
            # 将图像数据转换为QPixmap
            buf.seek(0)
            image_data = buf.getvalue()
            qimage = QImage.fromData(image_data)
            pixmap = QPixmap.fromImage(qimage)
            
            # 设置到QLabel
            error_image_label.setPixmap(pixmap)
        else:
            error_image_label.setText("没有可用的误差数据")
        
        # 添加到布局
        error_layout.addWidget(error_image_label)
        
        # 失真校正示例选项卡
        if 'undistorted_example' in result_dict:
            undistort_tab = QWidget()
            undistort_layout = QVBoxLayout(undistort_tab)
            undistort_label = QLabel()
            undistort_label.setAlignment(Qt.AlignCenter)
            undistort_label.setPixmap(QPixmap.fromImage(result_dict['undistorted_example']).scaled(
                760, 540, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            undistort_layout.addWidget(undistort_label)
            tabs.addTab(undistort_tab, "畸变校正示例")
        
        tabs.addTab(matrix_tab, "标定参数")
        tabs.addTab(error_tab, "误差分析")
        
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(tabs)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        save_button = QPushButton("保存结果")
        save_button.clicked.connect(self._save_results)
        button_layout.addWidget(save_button)
        button_layout.addWidget(close_button)
        
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
    
    def _save_results(self):
        """保存标定结果到文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存标定结果", "", "JSON 文件 (*.json);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                # 提取需要保存的结果
                save_dict = {
                    'camera_matrix': self.result_dict['camera_matrix'].tolist(),
                    'dist_coeffs': self.result_dict['dist_coeffs'].tolist(),
                    'image_size': self.result_dict.get('image_size', [0, 0]),
                    'reprojection_error': self.result_dict.get('reprojection_error', 0),
                    'board_info': self.result_dict.get('board_info', {}),
                    'calibration_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # 保存为JSON
                with open(file_path, 'w') as f:
                    json.dump(save_dict, f, indent=4)
                    
                QMessageBox.information(self, "成功", f"标定结果已保存到 {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存标定结果失败：{str(e)}")

# 相机标定主窗口
class CameraCalibrationWindow(QMainWindow):
    """相机标定主窗口"""
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        """初始化UI"""
        # 设置窗口属性
        self.setWindowTitle("相机标定工具")
        self.setMinimumSize(1200, 750)
        
        # 设置应用主题
        self._set_application_style()
        
        # 创建主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局（垂直）
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 10, 15, 10)
        main_layout.setSpacing(10)
        
        # 创建标题区域
        title_label = QLabel("相机标定工具")
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #1976d2;")
        title_label.setAlignment(Qt.AlignCenter)
        
        description_label = QLabel("通过标定板图像计算相机内参矩阵和畸变系数")
        description_label.setStyleSheet("font-size: 13px; color: #616161;")
        description_label.setAlignment(Qt.AlignCenter)
        
        main_layout.addWidget(title_label)
        main_layout.addWidget(description_label)
        
        # ===== 主工作区：左右分栏布局 =====
        work_layout = QHBoxLayout()
        work_layout.setSpacing(15)
        
        # ----- 左侧：参数配置区 -----
        left_panel = CardWidget()
        left_panel.setMinimumWidth(450)
        left_panel.setMaximumWidth(550)
        
        # 1. 图像文件夹选择
        folder_group = QGroupBox("标定图像")
        folder_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #2196f3;
                border: 2px solid #e3f2fd;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        folder_layout = QVBoxLayout(folder_group)
        
        folder_select_layout = QHBoxLayout()
        self.images_folder_edit = QLineEdit()
        self.images_folder_edit.setPlaceholderText("选择包含标定图像的文件夹")
        browse_button = StylishButton("浏览", primary=False)
        browse_button.setMaximumWidth(80)
        browse_button.clicked.connect(self._browse_images_folder)
        folder_select_layout.addWidget(self.images_folder_edit)
        folder_select_layout.addWidget(browse_button)
        folder_layout.addLayout(folder_select_layout)
        
        left_panel.layout.addWidget(folder_group)
        
        # 2. 标定板配置
        board_group = QGroupBox("标定板配置")
        board_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #2196f3;
                border: 2px solid #e3f2fd;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        board_layout = QVBoxLayout(board_group)
        
        # 标定板类型
        board_type_layout = QHBoxLayout()
        board_type_layout.addWidget(QLabel("类型:"))
        self.board_type_combo = QComboBox()
        self.board_type_combo.addItem("棋盘格", "chessboard")
        self.board_type_combo.addItem("圆形 (白底黑圆)", "circles")
        self.board_type_combo.addItem("空心圆环", "ring_circles")
        self.board_type_combo.currentIndexChanged.connect(self._update_board_tips)
        board_type_layout.addWidget(self.board_type_combo, 1)
        board_layout.addLayout(board_type_layout)
        
        # 标定板尺寸 - 使用网格布局
        size_grid = QGridLayout()
        size_grid.setColumnStretch(1, 1)
        size_grid.setColumnStretch(3, 1)
        
        self.board_width = QSpinBox()
        self.board_width.setRange(2, 20)
        self.board_width.setValue(9)
        self.board_width.setToolTip("水平方向的内角点或圆心数量")
        
        self.board_height = QSpinBox()
        self.board_height.setRange(2, 20)
        self.board_height.setValue(6)
        self.board_height.setToolTip("垂直方向的内角点或圆心数量")
        
        size_grid.addWidget(QLabel("水平点数:"), 0, 0)
        size_grid.addWidget(self.board_width, 0, 1)
        size_grid.addWidget(QLabel("垂直点数:"), 0, 2)
        size_grid.addWidget(self.board_height, 0, 3)
        
        board_layout.addLayout(size_grid)
        
        # 方格尺寸
        square_layout = QHBoxLayout()
        square_layout.addWidget(QLabel("方格尺寸:"))
        self.square_size = QDoubleSpinBox()
        self.square_size.setRange(1, 1000)
        self.square_size.setValue(20.0)
        self.square_size.setSuffix(" mm")
        self.square_size.setToolTip("单个方格边长或相邻圆心距离")
        square_layout.addWidget(self.square_size, 1)
        board_layout.addLayout(square_layout)
        
        # 标定板提示
        self.board_tips_label = QLabel()
        self.board_tips_label.setWordWrap(True)
        self.board_tips_label.setStyleSheet("""
            background-color: #e3f2fd; 
            padding: 8px; 
            border-radius: 4px;
            font-size: 11px;
            color: #1565c0;
        """)
        board_layout.addWidget(self.board_tips_label)
        
        left_panel.layout.addWidget(board_group)
        
        # 3. 标定选项
        options_group = QGroupBox("标定选项")
        options_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #2196f3;
                border: 2px solid #e3f2fd;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        options_layout = QVBoxLayout(options_group)
        
        self.show_corners_check = QCheckBox("实时显示检测到的角点")
        self.show_corners_check.setChecked(True)
        options_layout.addWidget(self.show_corners_check)
        
        left_panel.layout.addWidget(options_group)
        
        # 4. 执行标定区域
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        left_panel.layout.addWidget(self.progress_bar)
        
        # 按钮布局
        buttons_layout = QHBoxLayout()
        self.calibrate_button = StylishButton("开始标定")
        self.calibrate_button.setMinimumHeight(42)
        self.calibrate_button.clicked.connect(self._run_calibration)
        
        self.test_button = StylishButton("测试校正", primary=False)
        self.test_button.setMinimumHeight(42)
        self.test_button.clicked.connect(self._test_undistortion)
        self.test_button.setEnabled(False)
        
        buttons_layout.addWidget(self.calibrate_button, 2)
        buttons_layout.addWidget(self.test_button, 1)
        left_panel.layout.addLayout(buttons_layout)
        
        # 5. 标定结果显示
        self.result_label = QLabel()
        self.result_label.setVisible(False)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("""
            background-color: #e8f5e9; 
            padding: 12px; 
            border-radius: 6px;
            border: 1px solid #4caf50;
            font-size: 12px;
            color: #2e7d32;
        """)
        left_panel.layout.addWidget(self.result_label)
        
        # 添加弹性空间
        left_panel.layout.addStretch()
        
        # ----- 右侧：图像预览区 -----
        right_panel = CardWidget()
        
        # 图像列表标题
        list_title_layout = QHBoxLayout()
        list_title = QLabel("标定图像列表")
        list_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #2196f3;")
        self.image_count_label = QLabel("(0 张)")
        self.image_count_label.setStyleSheet("font-size: 12px; color: #757575;")
        list_title_layout.addWidget(list_title)
        list_title_layout.addWidget(self.image_count_label)
        list_title_layout.addStretch()
        right_panel.layout.addLayout(list_title_layout)
        
        # 图像列表（横向显示，较矮）
        self.images_list = QListWidget()
        self.images_list.setMaximumHeight(120)
        self.images_list.setFlow(QListWidget.LeftToRight)
        self.images_list.setWrapping(True)
        self.images_list.setResizeMode(QListWidget.Adjust)
        self.images_list.setIconSize(QSize(80, 60))
        self.images_list.itemClicked.connect(self._on_image_clicked)
        self.images_list.setStyleSheet("""
            QListWidget::item {
                border: 2px solid transparent;
                border-radius: 4px;
                padding: 2px;
                margin: 2px;
            }
            QListWidget::item:selected {
                border: 2px solid #2196f3;
                background-color: #e3f2fd;
            }
            QListWidget::item:hover {
                background-color: #f5f5f5;
            }
        """)
        right_panel.layout.addWidget(self.images_list)
        
        # 图像预览区域
        preview_title = QLabel("图像预览")
        preview_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #2196f3; margin-top: 5px;")
        right_panel.layout.addWidget(preview_title)
        
        self.image_preview = ImagePreview()
        self.image_preview.clicked.connect(lambda: self._browse_images_folder(True))
        self.image_preview.setMinimumHeight(400)
        right_panel.layout.addWidget(self.image_preview, 1)
        
        # 添加左右面板到工作区
        work_layout.addWidget(left_panel)
        work_layout.addWidget(right_panel, 1)
        
        main_layout.addLayout(work_layout, 1)
        
        # 状态栏
        self.statusBar().showMessage("就绪 - 请选择标定图像文件夹开始")
        
        # 初始化变量
        self.calibration_result = None
        
        # 更新标定板提示
        self._update_board_tips()
        
    def _set_application_style(self):
        """设置应用样式"""
        # 设置全局样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QLabel {
                color: #212121;
            }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                padding: 6px;
                border: 1px solid #bdbdbd;
                border-radius: 4px;
                background-color: white;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
                border: 1px solid #2196f3;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #bdbdbd;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #2196f3;
                border-radius: 3px;
                background-color: #2196f3;
                image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxOCIgaGVpZ2h0PSIxOCIgdmlld0JveD0iMCAwIDE4IDE4Ij48cGF0aCBmaWxsPSIjRkZGRkZGIiBkPSJNNi42MSAxMS44OUwyLjUgNy43OUwxLjA5IDkuMjFMNi42MSAxNC43MUwxNy4zNSA0TDE1Ljk0IDIuNTlaIi8+PC9zdmc+);
            }
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #e0e0e0;
                height: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2196f3;
                border-radius: 4px;
            }
            QListWidget {
                border: 1px solid #bdbdbd;
                border-radius: 4px;
                background-color: white;
                padding: 5px;
            }
            QListWidget::item {
                border-radius: 3px;
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
                color: #1976d2;
            }
        """)
    
    def _update_board_tips(self):
        """根据选择的标定板类型更新提示"""
        board_type = self.board_type_combo.currentData()
        
        if board_type == "chessboard":
            tips = """<b>提示：</b>确保打印精确无变形，拍摄时覆盖视场不同区域（特别是边角），以不同角度拍摄至少10张图像，避免强反光。"""
        elif board_type == "circles":
            tips = """<b>提示：</b>适用于光照不理想场景，圆点大小应适中，避免极端视角，确保光照均匀，减少阴影。"""
        else:  # ring_circles
            tips = """<b>提示：</b>适合强光或反光条件，圆环厚度应适中，保持形状规则，特别适合玻璃表面或反光场景。"""
        
        self.board_tips_label.setText(tips)
    
    def _browse_images_folder(self, single_file=False):
        """浏览并选择标定图像文件夹"""
        if single_file:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择标定图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*.*)"
            )
            if file_path:
                folder_path = os.path.dirname(file_path)
                self.images_folder_edit.setText(folder_path)
                self._load_images(folder_path)
                
                # 选择刚刚打开的图像
                for i in range(self.images_list.count()):
                    item = self.images_list.item(i)
                    if os.path.basename(file_path) == item.text():
                        self.images_list.setCurrentItem(item)
                        break
        else:
            folder_path = QFileDialog.getExistingDirectory(
                self, "选择标定图像文件夹"
            )
            if folder_path:
                self.images_folder_edit.setText(folder_path)
                self._load_images(folder_path)
    
    def _load_images(self, folder):
        """加载文件夹中的图像到列表"""
        # 清空列表
        self.images_list.clear()
        
        # 查找所有图像文件
        image_files = glob.glob(os.path.join(folder, '*.jpg'))
        image_files.extend(glob.glob(os.path.join(folder, '*.jpeg')))
        image_files.extend(glob.glob(os.path.join(folder, '*.png')))
        image_files.extend(glob.glob(os.path.join(folder, '*.bmp')))
        
        # 如果没有找到图像
        if not image_files:
            QMessageBox.warning(self, "警告", "所选文件夹中未找到图像文件")
            self.image_count_label.setText("(0 张)")
            return
            
        # 按文件名排序
        image_files.sort()
        
        # 更新图像计数
        self.image_count_label.setText(f"({len(image_files)} 张)")
        
        # 添加到列表（带缩略图）
        for img_file in image_files:
            # 创建缩略图
            pixmap = QPixmap(img_file)
            if not pixmap.isNull():
                icon = QIcon(pixmap.scaled(80, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                icon = QIcon()
            
            item = QListWidgetItem(icon, "")  # 不显示文件名文本
            item.setData(Qt.UserRole, img_file)  # 存储完整路径
            item.setToolTip(os.path.basename(img_file))  # 悬停显示文件名
            self.images_list.addItem(item)
            
        # 显示第一张图像
        if self.images_list.count() > 0:
            self.images_list.setCurrentRow(0)
            self._on_image_clicked(self.images_list.item(0))
    
    def _on_image_clicked(self, item):
        """点击图像列表项时显示预览"""
        if item:
            img_path = item.data(Qt.UserRole)
            self.image_preview.setImage(img_path)
    
    def _run_calibration(self):
        """执行相机标定"""
        # 获取参数
        images_folder = self.images_folder_edit.text()
        if not images_folder or not os.path.isdir(images_folder):
            QMessageBox.warning(self, "错误", "请选择有效的图像文件夹")
            return
            
        board_type = self.board_type_combo.currentData()
        board_size = (self.board_width.value(), self.board_height.value())
        square_size = self.square_size.value()
        visualize = self.show_corners_check.isChecked()
        
        # 禁用界面
        self.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 设置为不确定状态
        self.statusBar().showMessage("正在执行标定...")
        
        try:
            # 创建临时输出文件夹
            output_folder = os.path.join(images_folder, "calibration_results")
            os.makedirs(output_folder, exist_ok=True)
            
            # 执行标定
            try:
                # calibrate_camera返回的是一个元组，而不是字典
                camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error, img_shape, successful_images = cam_calib.calibrate_camera(
                    images_folder, 
                    board_type=board_type,
                    board_size=board_size, 
                    square_size=square_size,
                    visualize=visualize,
                    delay=100  # 可视化显示每张图像的时间(毫秒)
                )
            except TypeError as e:
                if "indices must be integers or slices" in str(e):
                    QMessageBox.warning(self, "标定失败", 
                                       f"标定板类型与图像不匹配或标定板尺寸设置错误。\n\n错误详情: {str(e)}\n\n请检查:\n1. 标定板类型选择是否正确\n2. 水平点数和垂直点数设置是否与实际标定板匹配")
                    return
                else:
                    raise
        
            # 构建结果字典，供UI使用
            result = {
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs,
                'rvecs': rvecs,
                'tvecs': tvecs,
                'reprojection_error': reprojection_error,
                'image_size': img_shape,
                'successful_images': successful_images,
                'total_images': len(glob.glob(os.path.join(images_folder, '*.jpg'))) + 
                               len(glob.glob(os.path.join(images_folder, '*.jpeg'))) + 
                               len(glob.glob(os.path.join(images_folder, '*.png'))) + 
                               len(glob.glob(os.path.join(images_folder, '*.bmp'))),
                'image_points': [], # 添加空列表，以便后续代码可以访问
                'per_view_errors': [reprojection_error] * len(successful_images),  # 使用平均误差作为每个视图的误差
                'board_info': {
                    'type': board_type,
                    'size': board_size,
                    'square_size': square_size
                }
            }
            
            # 保存标定结果到文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            npy_file = os.path.join(output_folder, f"camera_calibration_{timestamp}.npy")
            json_file = os.path.join(output_folder, f"camera_calibration_{timestamp}.json")
            default_json_file = os.path.join(output_folder, "camera_calibration_latest.json")
            
            # 保存NumPy格式
            calibration_data_npy = {
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs
            }
            np.save(npy_file, calibration_data_npy)
            
            # 保存JSON格式
            calibration_data_json = {
                'camera_matrix': camera_matrix.tolist(),
                'dist_coeffs': dist_coeffs.tolist(),
                'image_size': img_shape,
                'reprojection_error': float(reprojection_error),
                'board_type': board_type,
                'board_size': board_size,
                'square_size': square_size,
                'calibration_time': timestamp,
                'successful_images': successful_images
            }
            
            with open(json_file, 'w') as f:
                json.dump(calibration_data_json, f, indent=4)
                
            # 保存最新结果文件
            with open(default_json_file, 'w') as f:
                json.dump(calibration_data_json, f, indent=4)
                
            # 显示结果
            self.result_label.setText(
                f"标定成功! 平均重投影误差: {reprojection_error:.4f} 像素\n"
                f"成功标定的图像: {len(successful_images)}/{result['total_images']}\n"
                f"标定结果已保存至:\n"
                f"- {output_folder}"
            )
            self.result_label.setVisible(True)
            
            # 存储结果
            self.calibration_result = result
            
            # 启用测试按钮
            self.test_button.setEnabled(True)
            
            # 显示结果对话框
            result_dialog = CalibrationResultDialog(result, self)
            result_dialog.exec()
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"标定过程中出错: {str(e)}")
            
        finally:
            # 重新启用界面
            self.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.statusBar().showMessage("标定完成")
    
    def _test_undistortion(self):
        """测试畸变校正"""
        if not self.calibration_result:
            QMessageBox.warning(self, "Error", "Please run calibration first")
            return
            
        # 选择测试图像
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Test Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            # 获取标定结果
            camera_matrix = self.calibration_result['camera_matrix']
            dist_coeffs = self.calibration_result['dist_coeffs']
            
            # 读取原始图像
            original_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if original_img is None:
                # 如果彩色模式失败，尝试以灰度模式读取
                original_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if original_img is not None:
                    # 将灰度图像转换为3通道以保持一致性
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
            
            if original_img is None:
                raise ValueError(f"Cannot read image: {file_path}")
            
            # 确保original_img是numpy数组
            if not isinstance(original_img, np.ndarray):
                raise ValueError(f"Image data is not a numpy array: {type(original_img)}")
                
            # 执行畸变校正
            undistorted_img = cam_calib.test_undistortion(
                file_path, camera_matrix, dist_coeffs, output_folder=None
            )
            
            # 确保undistorted_img是numpy数组
            if undistorted_img is not None and not isinstance(undistorted_img, np.ndarray):
                raise ValueError(f"Undistorted image data is not a numpy array: {type(undistorted_img)}")
            
            if undistorted_img is not None:
                # 设置固定显示尺寸
                display_width, display_height = 1080, 720
                
                # 调整原始图像和校正后图像到固定显示尺寸
                original_img_resized = cv2.resize(original_img, (display_width, display_height))
                undistorted_img_resized = cv2.resize(undistorted_img, (display_width, display_height))
                
                # 创建并排比较图像
                comparison = np.zeros((display_height, display_width*2, 3), dtype=np.uint8)
                comparison[:, :display_width] = original_img_resized
                comparison[:, display_width:] = undistorted_img_resized
                
                # 添加标签
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(comparison, "Original", (50, 50), font, 1.5, (0, 255, 0), 3)
                cv2.putText(comparison, "Undistorted", (display_width+50, 50), font, 1.5, (0, 255, 0), 3)
                
                # 显示结果
                cv2.imshow("Distortion Correction Result", comparison)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Distortion correction failed: {str(e)}")

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("相机标定工具")
    app.setOrganizationName("结构光3D扫描")
    
    # 创建并显示主窗口
    window = CameraCalibrationWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 