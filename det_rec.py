import torch
import cv2
import numpy as np
from ultralytics import YOLO
from model.LPRNET import LPRNet
from model.STN import STNet
from deblur import Debluror
from utils.clahe import Clahe3
import time

class Recognizer(object):
    """
    输入图像，返回车牌号
    """
    CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
             '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
             '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
             '新',
             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
             'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
             'W', 'X', 'Y', 'Z', 'I', 'O', '-'
            ]

    def __init__(self, weights_path, device):
        self.device = device
        self.lprnet = LPRNet(class_num=len(self.CHARS)).to(self.device)
        self.lprnet.load_state_dict(torch.load(weights_path, map_location=self.device)["lprnet_state_dict"])
        self.stnet = STNet().to(self.device)
        self.stnet.load_state_dict(torch.load(weights_path, map_location=self.device)["stn_state_dict"])
        self.lprnet.eval()
        self.stnet.eval()
        self.image_size = (94, 24)

    @classmethod
    def list2string(cls, label):
        return ''.join(list(map(lambda n: cls.CHARS[n], label)))

    def recognize(self, image):
        height, width, _ = image.shape
        if height != self.image_size[1] or width != self.image_size[0]:
            image = cv2.resize(image, self.image_size)

        # normalization
        image = image.astype('float32')
        image -= 127.5
        image *= 0.0078125
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image).unsqueeze(0).to(self.device)  # add batch dimension and move to device

        # forward
        image = self.stnet(image)
        prebs = self.lprnet(image)

        # greedy decode
        prebs = prebs.detach().cpu().numpy()
        preb = prebs[0, :, :]

        preb_label = list()
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))

        # dropout repeat label and blank label
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(self.CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:
            if (pre_c == c) or (c == len(self.CHARS) - 1):
                if c == len(self.CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c

        return self.list2string(no_repeat_blank_label)


class Detector(object):
    """
    输入图像，返回车牌图像（只有一个车牌）
    """
    def __init__(self, weights_path, device):
        self.device = device
        self.model = YOLO(model=weights_path).to(self.device)  # 将YOLO模型移到设备上

    def detect(self, img, save=False):
        results = self.model(img)  # 如果模型在GPU上，输入图像应为GPU张量
        for result in results:
            boxes = result.boxes
            if len(boxes):
                x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy()  # 将GPU上的结果转换为CPU
                plate_image = img[int(y1):int(y2), int(x1):int(x2)]  # 车牌图像
            else:
                print("No plate detected!")
            return plate_image


def det_rec(img, device='cpu'):
    det_weights = r"weights/yolov11.pt"  # 车牌检测模型权重
    detector = Detector(det_weights, device)
    rec_weights = r"weights/lpr_stn_clahe.pt"  # 车牌识别模型权重
    recognizer = Recognizer(rec_weights, device)

    crop_img = detector.detect(img)  # 车牌图像
    clahe_img = Clahe3(crop_img)
    pred = recognizer.recognize(clahe_img)  # 车牌号
    return pred, crop_img


def main(device='cpu'):
    img_file = r"S:\Learn\DATASET\CCPD2019_CAR\images\train\base\01-90_85-274&482_457&544-456&545_270&533_261&470_447&482-0_0_16_24_31_28_31-146-27.jpg"  # 输入图片路径
    IF_DETECT = True
    IF_SAVE_CROP = True
    save_crop = r"crop.jpg"
    IF_CLAHE = True
    IF_SAVE_CLAHE = True
    save_clahe = r"clahe.jpg"

    # 加载模型
    det_weights = r"weights/yolov11.pt"  # 车牌检测模型权重
    detector = Detector(det_weights, device)
    rec_weights = r"weights/lpr_stn_clahe.pt"  # 车牌识别模型权重
    recognizer = Recognizer(rec_weights, device)

    # 检测识别
    img = cv2.imread(img_file)
    if IF_DETECT:
        img = detector.detect(img)  # 车牌图像
    if IF_SAVE_CROP:
        cv2.imwrite(save_crop, img)
    if IF_CLAHE:
        img = Clahe3(img)
        if IF_SAVE_CLAHE:
            cv2.imwrite(save_clahe, img)

    pred = recognizer.recognize(img)  # 车牌号

    print(pred)

if __name__ == "__main__":

    device = 'cpu'
    device = device if torch.cuda.is_available() else 'cpu'
    print("device:", device)

    # main(device)
    start_time = time.time()
    img = cv2.imread(r"imgs/CAR/1.jpg")
    print("shape:", img.shape)
    pred, crop_img = det_rec(img, device)
    print("result:", pred)
    # cv2.imwrite("read.jpg", img)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.4f} seconds")
