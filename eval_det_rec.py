import os
import time
import cv2
import torch
import csv
from utils.clahe import Clahe3
from utils.lp_utils import return_ccpd_info
from det_rec import det_rec

def evaluate(img_path, device='cpu'):
    """
    根据给定图片路径，检测并识别车牌，判断识别结果与实际车牌是否一致
    """
    # 读取输入图像
    img = cv2.imread(img_path)
    pred, crop_img = det_rec(img, device)

    # 获取实际车牌号
    img_name = os.path.basename(img_path).split('.')[0]
    actual_plate = return_ccpd_info(img_name)

    # 判断车牌是否成功检测到
    detected = 1 if pred else 0
    
    # 输出真实车牌和预测车牌
    print(f"Image: {img_name}")
    print(f"Actual Plate: {actual_plate}, Predicted Plate: {pred}")

    # 返回结果
    return {
        "image_name": img_name,
        "actual_plate": actual_plate,
        "detected": detected,
        "predicted_plate": pred
    }


def save_results_to_csv(results, output_csv):
    """
    将检测结果保存到CSV文件
    """
    fieldnames = ["image_name", "actual_plate", "detected", "predicted_plate"]
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def evaluate_and_save_to_csv(img_path_or_dir, device='cpu', output_csv="results.csv"):
    results = []
    correct_predictions = 0
    total_images = 0

    # 检查路径类型：单张图片还是目录
    if os.path.isdir(img_path_or_dir):
        # 如果是目录路径，遍历目录中的所有图片
        for img_name in os.listdir(img_path_or_dir):
            img_path = os.path.join(img_path_or_dir, img_name)
            if os.path.isfile(img_path) and img_path.lower().endswith(('jpg', 'jpeg', 'png')):
                result = evaluate(img_path, device)
                results.append(result)
                # 判断车牌是否预测正确
                if result["actual_plate"] == result["predicted_plate"]:
                    correct_predictions += 1
                total_images += 1
    else:
        # 如果是单张图片路径
        result = evaluate(img_path_or_dir, device)
        results.append(result)
        if result["actual_plate"] == result["predicted_plate"]:
            correct_predictions += 1
        total_images += 1

    # 计算准确率
    if total_images > 0:
        accuracy = correct_predictions / total_images * 100
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        accuracy = 0
        print("No images to evaluate.")

    # 将结果保存到CSV文件
    save_results_to_csv(results, output_csv)


if __name__ == "__main__":
    device = 'cuda'
    device = device if torch.cuda.is_available() else 'cpu'
    print("device:", device)

    # 输入单张图片路径
    # img_file = r"imgs/CAR/1.jpg"
    # evaluate_and_save_to_csv(img_file, device)

    # 如果是目录路径
    img_dir = r"S:\Learn\DATASET\CCPD2019_CAR\images\val\weather"
    evaluate_and_save_to_csv(img_dir, device, "val_weather.csv")
