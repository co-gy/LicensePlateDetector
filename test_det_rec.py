import unittest
import torch
import cv2
import os
from det_rec import det_rec
from utils.clahe import Clahe3
import time


class TestLicensePlateRecognition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ 类级别的初始化，适用于所有测试函数 """
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 根据是否有GPU选择设备
        print(f"Testing on device: {cls.device}")

    def test_image_shape(self):
        """ 测试输入图像的形状 """
        img = cv2.imread(r"imgs/CAR/1.jpg")
        # assert img
        self.assertIsNotNone(img, "Failed to load image!")
        print(f"Image shape: {img.shape}")
        self.assertEqual(len(img.shape), 3, "Image should have 3 dimensions (height, width, channels)")
    
    def test_detection(self):
        """ 测试车牌检测 """
        img = cv2.imread(r"imgs/CAR/1.jpg")
        pred, crop_img = det_rec(img, device=self.device)
        
        self.assertIsNotNone(crop_img, "Failed to detect license plate!")
        print("Detected crop image shape:", crop_img.shape)
        self.assertGreater(crop_img.shape[0], 0, "Detected crop image has invalid height!")
        self.assertGreater(crop_img.shape[1], 0, "Detected crop image has invalid width!")

    def test_recognition(self):
        """ 测试车牌识别 """
        img = cv2.imread(r"imgs/CAR/1.jpg")
        pred, crop_img = det_rec(img, device=self.device)
        
        self.assertIsInstance(pred, str, "Prediction should be a string!")
        print("Predicted license plate:", pred)
        self.assertGreater(len(pred), 0, "Prediction should not be empty!")
    
    def test_clahe_processing(self):
        """ 测试CLAHE预处理 """
        img = cv2.imread(r"imgs/CAR/1.jpg")
        pred, crop_img = det_rec(img, device=self.device)
        clahe_img = Clahe3(crop_img)
        
        self.assertIsNotNone(clahe_img, "CLAHE processing failed!")
        print(f"CLAHE image shape: {clahe_img.shape}")
    
    def test_performance(self):
        """ 测试性能（处理时间） """
        img = cv2.imread(r"imgs/CAR/1.jpg")
        start_time = time.time()
        pred, crop_img = det_rec(img, device=self.device)
        end_time = time.time()
        
        print(f"Prediction: {pred}")
        print(f"Total time taken for detection and recognition: {end_time - start_time:.4f} seconds")
        
        # 假设您希望每次的执行时间都小于某个阈值，例如 2秒
        self.assertLess(end_time - start_time, 2, "Total time exceeded threshold!")

    def test_device_switch(self):
        """ 测试GPU与CPU之间的切换 """
        cpu_device = 'cpu'
        gpu_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 测试GPU
        pred_gpu, _ = det_rec(cv2.imread(r"imgs/CAR/1.jpg"), device=gpu_device)
        print(f"Prediction on GPU: {pred_gpu}")
        
        # 测试CPU
        pred_cpu, _ = det_rec(cv2.imread(r"imgs/CAR/1.jpg"), device=cpu_device)
        print(f"Prediction on CPU: {pred_cpu}")

        self.assertNotEqual(pred_gpu, "", "Prediction on GPU is empty!")
        self.assertNotEqual(pred_cpu, "", "Prediction on CPU is empty!")
        self.assertEqual(pred_gpu, pred_cpu, "Predictions from GPU and CPU should be the same!")

if __name__ == '__main__':
    unittest.main()
