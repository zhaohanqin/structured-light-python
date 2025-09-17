import cv2
import numpy as np
import os

def generate_projection_mask(image_path, smooth=True, feather_size=30, display=True, save=True):
    """
    根据投影仪的全白投影图像生成掩码
    :param image_path: 输入图像路径（全白投影图）
    :param smooth: 是否生成带平滑过渡的掩码
    :param feather_size: 羽化程度（数值越大，边缘越平滑）
    :param display: 是否显示结果
    :param save: 是否保存结果
    :return: 掩码 mask (float32, 0~1)
    """
    # 读取灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(img, (15, 15), 0)

    # 使用 Otsu 阈值分割
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学操作（保证区域完整）
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 取最大连通域（只保留投影矩形区域）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.uint8(labels == largest) * 255

    if smooth:
        # 平滑掩码：羽化边缘
        mask = mask.astype(np.float32) / 255.0
        mask = cv2.GaussianBlur(mask, (feather_size, feather_size), 0)
        mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)
    else:
        # 硬边掩码
        mask = (mask > 0).astype(np.float32)

    # 可视化
    if display:
        cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Input", 1080, 720)
        cv2.imshow("Input", img)

        cv2.namedWindow("Projection Mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Projection Mask", 1080, 720)
        cv2.imshow("Projection Mask", (mask * 255).astype(np.uint8))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 保存结果
    if save:
        base, ext = os.path.splitext(image_path)
        save_path = base + ("_mask_soft.png" if smooth else "_mask_hard.png")
        cv2.imwrite(save_path, (mask * 255).astype(np.uint8))
        print(f"掩码已保存: {save_path}")

    return mask


if __name__ == "__main__":
    # 输入一张全白投影图像
    image_path = "Mean Intensity.png"  # ← 这里改成你的文件路径

    # 生成硬边掩码
    generate_projection_mask(image_path, smooth=False, feather_size=30)

    # 生成平滑掩码
    generate_projection_mask(image_path, smooth=True, feather_size=51)
