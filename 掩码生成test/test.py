import cv2
import numpy as np
from skimage import morphology

def compute_amplitude_modulation(I0, I1, I2, I3):
    """
    输入: 四步相移图像 (灰度图)
    输出: 振幅A, 调制度M, 平均亮度I_mean
    """
    Ik = np.stack([I0, I1, I2, I3], axis=2).astype(np.float32)
    I_max = np.max(Ik, axis=2)
    I_min = np.min(Ik, axis=2)
    I_mean = np.mean(Ik, axis=2)

    # 振幅（Amplitude）
    A = (I_max - I_min) / 2.0
    # 调制度（Modulation）
    M = (I_max - I_min) / (I_max + I_min + 1e-9)

    return A, M, I_mean

def build_fused_mask(A, M, I_mean, min_area=2000):
    """
    输入: 振幅A, 调制度M, 平均亮度I_mean
    输出: 可靠掩膜mask
    """
    # 归一化
    A_norm = cv2.normalize(A, None, 0, 1, cv2.NORM_MINMAX)
    M_norm = cv2.normalize(M, None, 0, 1, cv2.NORM_MINMAX)
    I_norm = cv2.normalize(I_mean, None, 0, 1, cv2.NORM_MINMAX)

    # 1. 振幅阈值 (Otsu)
    _, mask_A = cv2.threshold((A_norm*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 2. 调制度阈值 (Otsu)
    _, mask_M = cv2.threshold((M_norm*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 3. 亮度阈值 (保留亮度高于50%分位的区域)
    th_I = np.percentile(I_norm, 50)
    mask_I = (I_norm > th_I).astype(np.uint8) * 255

    # 4. 多条件融合 (振幅 OR 调制度)，再 AND 亮度
    mask = np.logical_or(mask_A > 0, mask_M > 0)
    mask = np.logical_and(mask, mask_I > 0)

    # 5. 形态学处理
    mask = morphology.binary_closing(mask, morphology.disk(7))
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    mask = morphology.remove_small_holes(mask, area_threshold=min_area)

    # 6. 保留最大连通区域
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if num_labels > 1:
        areas = [(labels == i).sum() for i in range(1, num_labels)]
        keep = np.argmax(areas) + 1
        mask = (labels == keep)

    # 膨胀一点，避免边缘缺失
    mask = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

    return mask.astype(np.uint8)

# ---------------- 示例 ----------------
if __name__ == "__main__":
    # 读取四步相移图像
    I0 = cv2.imread("I1.png", cv2.IMREAD_GRAYSCALE)
    I1 = cv2.imread("I2.png", cv2.IMREAD_GRAYSCALE)
    I2 = cv2.imread("I3.png", cv2.IMREAD_GRAYSCALE)
    I3 = cv2.imread("I4.png", cv2.IMREAD_GRAYSCALE)

    # 计算特征
    A, M, I_mean = compute_amplitude_modulation(I0, I1, I2, I3)

    # 生成掩膜
    mask = build_fused_mask(A, M, I_mean)

    # 可视化 (窗口大小统一设置为1080x720)
cv2.namedWindow("Amplitude", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Amplitude", 1080, 720)
cv2.imshow("Amplitude", (A/A.max()*255).astype(np.uint8))

cv2.namedWindow("Modulation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Modulation", 1080, 720)
cv2.imshow("Modulation", (M/M.max()*255).astype(np.uint8))

cv2.namedWindow("Mean Intensity", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mean Intensity", 1080, 720)
cv2.imshow("Mean Intensity", (I_mean/I_mean.max()*255).astype(np.uint8))

cv2.namedWindow("Final Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Final Mask", 1080, 720)
cv2.imshow("Final Mask", mask*255)

cv2.waitKey(0)
cv2.destroyAllWindows()

