import torch
import cv2
import os
import numpy as np
from imutils import paths
import shutil
from model.LPRNET import LPRNet
from model.STN import STNet



class Recognizer(object):
    """
    输入图像（不是图像路径），返回车牌号
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

    def __init__(self,weights_path):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lprnet = LPRNet(class_num=len(self.CHARS))
        self.lprnet.load_state_dict(torch.load(weights_path)["lprnet_state_dict"])
        self.stnet = STNet()
        self.stnet.load_state_dict(torch.load(weights_path)["stn_state_dict"])
        # self.lprnet.to(device)
        # self.stnet.to(device)
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
        image = image.astype('float32')             # 数组元素类型转换为浮点数  # numpy.ndarray
        image -= 127.5                              # [0,255] → [-127.5,127.5]
        image *= 0.0078125                          # [-1,1]
        image = np.transpose(image, (2, 0, 1))      # 3×24×94
        image = torch.tensor(image).unsqueeze(0)    # numpy → tensor
                                                    # 1×3×24×94：在第0维前增加一个维度, 作为批次维度，存储批大小


        # forward
        image = self.stnet(image)
        prebs = self.lprnet(image)                  # 1×68×18
        
        # greedy decode
        prebs = prebs.detach().numpy()
        preb = prebs[0, :, :]                       # 一个样本的概率分布 (68×18)
        # print(preb.shape)

        preb_label = list()                                     # len=18            
        for j in range(preb.shape[1]):                          # 取一个位置的所有字符的概率分布
            preb_label.append(np.argmax(preb[:, j], axis=0))    # 某个位置最大的概率的字符  
        
        # print("解码前:", preb_label)
        # print("解码前:", self.list2string(preb_label))

        # dropout repeat label and blank label
        no_repeat_blank_label = list()      
        pre_c = preb_label[0]                       # 第一个字符
        if pre_c != len(self.CHARS) - 1:            # 如果不是‘-’
            no_repeat_blank_label.append(pre_c)     
        for c in preb_label:  
            if (pre_c == c) or (c == len(self.CHARS) - 1):  # 如果重复或者为"-", 则跳过; 否则添加到no_repeat_blank_label中
                if c == len(self.CHARS) - 1:                
                    pre_c = c                       
                continue                                  
            no_repeat_blank_label.append(c)
            pre_c = c
        
        # print("解码后:", no_repeat_blank_label)
        # print("解码后:", self.list2string(no_repeat_blank_label))
        
        return self.list2string(no_repeat_blank_label)


def main():
    test_path = r"imgs/LP"               # 测试数据集路径
    weights_path = r"S:\Learn\EXP\LPRNet\exp2\best.ckpt"    # 权重路径
    IF_SAVE = True  # 是否保存图像

    recognizer = Recognizer(weights_path)

    # 获取目录中的所有图片文件
    all_files = [el for el in paths.list_images(test_path)]
    print(f"Total images: {len(all_files)}")

    for filename in all_files:
        img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)   # 读取图片
        pred = recognizer.recognize(img)                                # 识别车牌
        print(pred)


if __name__ == "__main__":
    main()
