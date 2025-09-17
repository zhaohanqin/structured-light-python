import cv2
import numpy as np
from skimage import morphology

def compute_amplitude_from_4step(I0, I1, I2, I3):
    # 4-step PSP -> S and C
    # S = 2/N * sum( I_k * sin(2πk/N) ), C = 2/N * sum( I_k * cos(...) )
    N = 4
    ks = np.arange(N)
    angles = 2*np.pi*ks/N
    Ik = np.stack([I0, I1, I2, I3], axis=2).astype(np.float32)
    C = (2.0/N) * np.sum(Ik * np.cos(angles)[None, None, :], axis=2)
    S = (2.0/N) * np.sum(Ik * np.sin(angles)[None, None, :], axis=2)
    A = np.sqrt(C**2 + S**2)  # 振幅（调制度的 proxy）
    return A, C, S

def make_mask_from_amplitude(A, method='otsu', thresh_rel=None, min_area=500):
    # A: 振幅图（float）
    import skimage.filters as filters
    a = A.copy()
    a_norm = (a - a.min())/(a.max()-a.min()+1e-9)
    a8 = (a_norm*255).astype(np.uint8)

    if method == 'otsu':
        th = filters.threshold_otsu(a8)
        mask = a8 >= th
    elif method == 'adaptive':
        mask = cv2.adaptiveThreshold(a8,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,51,-5) > 0
    elif method == 'relative':
        # relative percentile threshold: keep top thresh_rel fraction
        if thresh_rel is None: thresh_rel = 0.2
        th = np.percentile(a8, 100*(1-thresh_rel))
        mask = a8 >= th
    else:
        raise ValueError(method)

    # morphological cleanup
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    mask = morphology.remove_small_holes(mask, area_threshold=min_area)
    mask = morphology.binary_closing(mask, morphology.disk(5))
    mask = mask.astype(np.uint8)

    # keep largest connected component (optional)
    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels > 1:
        area = [(labels==i).sum() for i in range(1, num_labels)]
        if len(area)>0:
            keep = np.argmax(area)+1
            mask = (labels==keep).astype(np.uint8)

    return mask

# 使用示例（读取你的 4 帧图）
I0 = cv2.imread('I1.png', cv2.IMREAD_UNCHANGED)
I1 = cv2.imread('I2.png', cv2.IMREAD_UNCHANGED)
I2 = cv2.imread('I3.png', cv2.IMREAD_UNCHANGED)
I3 = cv2.imread('I4.png', cv2.IMREAD_UNCHANGED)

A, C, S = compute_amplitude_from_4step(I0, I1, I2, I3)
mask = make_mask_from_amplitude(A, method='otsu', min_area=2000)

# 可视化
cv2.namedWindow('amplitude', cv2.WINDOW_NORMAL)
cv2.resizeWindow('amplitude', 1080, 720)
cv2.imshow('amplitude', (A/A.max()*255).astype(np.uint8))

cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.resizeWindow('mask', 1080, 720)
cv2.imshow('mask', mask*255)

cv2.waitKey(0)
cv2.destroyAllWindows()

