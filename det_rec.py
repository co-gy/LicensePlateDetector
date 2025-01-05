import torch
import cv2
import numpy as np
from ultralytics import YOLO
from model.LPRNET import LPRNet
from model.STN import STNet
from deblur import Debluror

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

    def __init__(self, weights_path):
        self.lprnet = LPRNet(class_num=len(self.CHARS))
        self.lprnet.load_state_dict(torch.load(weights_path)["lprnet_state_dict"])
        self.stnet = STNet()
        self.stnet.load_state_dict(torch.load(weights_path)["stn_state_dict"])
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
        image = torch.tensor(image).unsqueeze(0)  # add batch dimension

        # forward
        image = self.stnet(image)
        prebs = self.lprnet(image)

        # greedy decode
        prebs = prebs.detach().numpy()
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
    def __init__(self, weights_path):
        self.model = YOLO(model=weights_path)
    def detect(self, img, save = False):
        results = self.model(img)
        for result in results:
            boxes = result.boxes
            if len(boxes):
                x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy()         
                plate_image = img[int(y1):int(y2), int(x1):int(x2)]    # 车牌图像

            else:
                print("No plate detected!")
            return plate_image

def det_rec(img):
    det_weights = r"weights/yolov11.pt"  # 车牌检测模型权重
    detector = Detector(det_weights)
    rec_weights = r"weights/lpr_stn.pt"  # 车牌识别模型权重
    recognizer = Recognizer(rec_weights)
    crop_img = detector.detect(img)                       # 车牌图像
    pred = recognizer.recognize(crop_img)                     # 车牌号
    return pred, crop_img
    
def main():
    img_file = r"8a677ceb-0bd0-4c6c-a65a-f46b8e433e6a240713_004450.jpeg"  # 输入图片路径
    IF_DETECT = True
    IF_SAVE_CROP = False;  save_crop = r"crop.jpg"
    IF_SAVE_DEBLUR = False; save_deblur = r"deblur.jpg"  # 保存车牌图像路径

    # 加载模型
    det_weights = r"weights/yolov11.pt"  # 车牌检测模型权重
    detector = Detector(det_weights)
    rec_weights = r"weights/LPRNet_model_Init.pth"  # 车牌识别模型权重
    recognizer = Recognizer(r"weights/LPRNet_model_Init.pth", r"weights/Final_STN_model.pth")
    # debluror = Debluror()

    # 检测识别
    img = cv2.imread(img_file)
    if IF_DETECT:
        img = detector.detect(img)                       # 车牌图像
    if IF_SAVE_CROP:
        cv2.imwrite(save_crop, img)
    pred = recognizer.recognize(img)                     # 车牌号

        
    print(pred)

if __name__ == "__main__":
    # main()
    img = cv2.imread(r"imgs/CAR/1.jpg")
    print("shape:", img.shape)
    pred, crop_img = det_rec(img)
    print("result:", pred)
    # cv2.imwrite("read.jpg", img)