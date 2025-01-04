import os
import time
import datetime
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)						# 为CPU设置随机种子		# 设置PyTorch的随机种子
    torch.cuda.manual_seed_all(seed)			# 为所有GPU设置随机种子		# 设置所有CUDA设备的随机种子
    torch.cuda.manual_seed(seed)				# 为当前GPU设置随机种子		# 设置CUDA的随机种子
    cudnn.deterministic = True					# 确保每次卷积算法选择都是确定的
    # cudnn.enabled = False
    # cudnn.benchmark = False					# 关闭cuDNN自动优化功能，确保结果可复现
    # cudnn.benchmark = True 					# for accelerating the running

def auto_increment_path(input_path, create=True):
    """
    :param input_path: 文件或目录
    :param create: {Bool} 是否创建文件或目录
    :return path: 路径（自动增加后缀）
    """
    """
    Example1: input_path = 'runs/logs/log.txt'，如果runs/logs/log.txt文件不存在，则创建runs/logs/log.txt文件
    Example2: input_path = 'runs/logs/log.txt'，但runs/logs/log.txt文件已存在，则创建runs/logs/log_1.txt文件
    Example3: input_path = 'runs/logs/log.txt'，但runs/logs/log.txt和runs/logs/log_1.txt文件已存在，则创建runs/logs/log_2.txt文件
    Example4: input_path = 'runs/logs'，但runs/logs文件夹不存在，则创建runs/logs文件夹
    Example5: input_path = 'runs/logs'，但runs/logs文件夹已存在，则创建runs/logs_1文件夹
    """
    base_name, extension = os.path.splitext(input_path)     # runs/logs/log    .txt
    counter = 0
    while True:
        if counter != 0:
            input_path = f"{base_name}_{counter}{extension}"
        input_path = os.path.normpath(input_path)           # 新路径
        if not os.path.exists(input_path):                  # 新路径不存在
            if create:
                if extension == '':
                    os.makedirs(input_path)
                else:
                    with open(input_path, 'a') as file:
                        file.write('')  # 创建一个空文件
                    print(f"Created: {input_path}...")
            break
        else:
            counter += 1
    return input_path

# if __name__ == '__main__':
#     LOG_DIR = "runs/run"
#     auto_increment_path(LOG_DIR)


def get_now_time_formatted():
    """
    :return: 当前时间格式化字符串
    """
    now = datetime.datetime.now()  # 获取当前时间
    formatted_time = now.strftime("%Y%m%d_%H%M%S")  # 格式化时间为所需的字符串格式
    return formatted_time

def load_pretrained_model(model, pretrained_model_path, mode = "ckpt"):
    if mode == "ckpt":
        model.load_state_dict(torch.load(pretrained_model_path)['model_state_dict'])
    else:
        model.load_state_dict(torch.load(pretrained_model_path))
    return model

class Logger(object):
    """
    用于记录日志
    self.log_path: 日志文件路径
    """
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, 'a', encoding='gbk') as f:
            pass
    def add_info(self, info):
        """
        添加日志信息
        :param info: 日志信息
        """
        with open(self.log_path, 'a', encoding='gbk') as f:
            f.write(info + '\n')
        f.close()

class Timer(object):
    """
    计时器
    """
    def __init__(self):
        self.start_time = None
        self.since_time = None
        self.elapsed_time = None
        self.epoch_time = None
        self.end_time = None
    def timer_start(self, time_format = "%Y%m%d_%H%M%S"):
        """
        开始计时
        Returns:
        """
        self.start_time = time.time()
        self.since_time = time.time()
        start_time = datetime.datetime.fromtimestamp(self.start_time).strftime(time_format)
        return start_time

    def timer_recode(self):
        """
        记录时间
        Returns:
        """
        elapsed_time = self.format_time(time.time() - self.start_time)
        epoch_time = self.format_time(time.time()-self.since_time)
        self.since_time = time.time()
        return elapsed_time, epoch_time

    @staticmethod
    def format_time(spend_time):
        """
        格式化时间为 hh:mm:ss
        """
        elapsed_rounded = int(round((spend_time)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def timer_end(self, time_format = "%Y%m%d_%H%M%S"):
        self.end_time = time.time()
        end_time = datetime.fromtimestamp(self.end_time).strftime(time_format)
        elapsed_time = self.format_time(self.end_time - self.start_time)
        return end_time, elapsed_time










