import matplotlib
import json
matplotlib.use('TkAgg')
# from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# 实现默认的matplotlib按键绑定
from matplotlib.backend_bases import key_press_handler


from matplotlib.figure import Figure

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
    from tkinter import ttk

import numpy as np
import cv2


def MinMaxProjectorCalibration(patterns, cameras, projector):
    '''
    投影仪最小/最大亮度校准工具
    
    参数:
        patterns: 用于测试的投影图案
        cameras: 用于捕获反馈的相机列表
        projector: 要校准的投影仪对象
    '''
    end = False
    pattern_num = 0

    def quit(root):
        nonlocal end
        end = True
        root.quit()     # 停止主循环
        root.destroy()  # 在Windows上防止致命Python错误：PyEval_RestoreThread: NULL tstate

    # 创建Tkinter根窗口
    root = Tk.Tk()
    root.wm_title("投影仪校准")

    # 显示下一个图案的回调函数
    def next_pattern(root):
        nonlocal pattern_num
        pattern_num = pattern_num + 1
        if pattern_num == len(patterns):
            pattern_num = 0

    # 创建matplotlib图形
    f = Figure(figsize=(5, 4), dpi=100)

    # 创建Tk绘图区域
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    # 添加matplotlib导航工具栏
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    # 添加退出和下一个按钮
    button = Tk.Button(master=root, text='退出', command=lambda: quit(root))
    button.pack(side=Tk.BOTTOM)
    button = Tk.Button(master=root, text='下一个', command=lambda: next_pattern(root))
    button.pack(side=Tk.BOTTOM)

    # 亮度滑动条变化回调函数
    def slider_max_changed(event):
        projector.max_image_brightness = current_max_brightness.get()

    def slider_min_changed(event):
        projector.min_image_brightness = current_min_brightness.get()

    # 创建亮度调整滑动条
    current_max_brightness = Tk.DoubleVar(value=projector.max_image_brightness)
    current_min_brightness = Tk.DoubleVar(value=projector.min_image_brightness)
    scale_max = Tk.Scale(root, orient="horizontal",
                            from_=0, to=1.0,
                            digits=4,
                            resolution=0.01,
                            variable=current_max_brightness,
                            command=slider_max_changed,
                            length=300)
    scale_min = Tk.Scale(root, orient="horizontal",
                            from_=0, to=1.0, 
                            digits=4,
                            resolution=0.01,
                            variable=current_min_brightness,
                            command=slider_min_changed,
                            length=300)
    
    # 添加标签
    label_1 = Tk.Label(root, text="最大亮度")
    label_2 = Tk.Label(root, text="最小亮度")

    # 打包UI元素
    label_1.pack(side=Tk.TOP)
    scale_max.pack(side=Tk.TOP)
    label_2.pack(side=Tk.TOP)
    scale_min.pack(side=Tk.TOP)

    # 设置投影仪窗口
    projector.set_up_window()

    # 主循环
    while True:
        # 投影当前图案
        projector.project_pattern(patterns[pattern_num][0])
        
        # 对于网络摄像头，先清除缓冲
        if cameras[0].type == 'web':
            _1 = cameras[0].get_image()
        if cameras[1].type == 'web':
            _2 = cameras[1].get_image()
        
        # 捕获图像
        frame_1 = cameras[0].get_image()
        frame_2 = cameras[1].get_image()

        # 转换为灰度图像
        if cameras[0].type == 'web':
            frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
        if cameras[1].type == 'web':
            frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
        
        # 定义感兴趣区域
        delta_height = 50
        ROI = slice(int(frame_1.shape[0] / 2 - delta_height), int(frame_1.shape[0] / 2 + delta_height))

        # 绘制第一个相机的亮度分布和图像
        a1 = f.add_subplot(221)
        a1.plot(np.mean(frame_1[ROI, :], axis=0))
        # a1.set_ylim((0, 4096))
        b1 = f.add_subplot(222)
        b1.imshow(frame_1)

        # 绘制第二个相机的亮度分布和图像
        a2 = f.add_subplot(223)
        a2.plot(np.mean(frame_2[ROI, :], axis=0))
        # a2.set_ylim((0, 4096))
        b2 = f.add_subplot(224)
        b2.imshow(frame_2)
        root.update()

        # 如果用户点击退出
        if (end):
            # 保存校准结果
            with open('config.json') as f:
                data = json.load(f)

            data['projector']["min_brightness"] = projector.min_image_brightness
            data['projector']["max_brightness"] = projector.max_image_brightness

            with open('config.json', 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            # 关闭投影仪窗口
            projector.close_window()

            break
        # 更新画布
        canvas.draw()
        # 清除子图
        f.delaxes(a1)
        f.delaxes(b1)
        f.delaxes(a2)
        f.delaxes(b2)