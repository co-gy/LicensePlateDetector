from torch.utils.data import *
import numpy as np
import cv2

def gray1_to_gray3(gray_img):
    """
    如：将(24,94)变为(24,94,3)
    """
    gray_rgb_img = np.repeat(gray_img[:, :, np.newaxis],3,axis=2)
    gray_rgb_img = np.clip(gray_rgb_img, 0, 255).astype(np.uint8)
    return gray_rgb_img

def Clahe3(img):
    """
    clahe灰度均衡
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,5))    # clipLimit越大，越容易抗干扰，tileGridSize越大，越精细？
    img = clahe.apply(gray_img)     # (24,94)
    img = gray1_to_gray3(img)
    
    return img


