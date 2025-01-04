"""
用法：
1. 将训练日志文件train_log.txt和验证日志文件val_log.txt放到./data目录下，直接运行即可

测试：
Epoch 254/399, Iters: 011700, validation_accuracy: 0.6093
Epoch 273/399, Iters: 012600, validation_accuracy: 0.5913
Epoch 293/399, Iters: 013500, validation_accuracy: 0.6000
Epoch 313/399, Iters: 014400, validation_accuracy: 0.5486
Epoch 332/399, Iters: 015300, validation_accuracy: 0.6360
Epoch 352/399, Iters: 016200, validation_accuracy: 0.5985
Epoch 371/399, Iters: 017100, validation_accuracy: 0.6093
Epoch 391/399, Iters: 018000, validation_accuracy: 0.6278


r'Epoch .*?, Iters: (\d+), validation_accuracy: (\d+\.\d+)'

测试_2:
Epoch 10/499, Iters: 000500, validation_accuracy: 0.9215, base_acc: 0.9318, dim_acc: 0.7311
r'Epoch .*?, Iters: (\d+),.*?validation_accuracy: (\d+\.\d+), base_acc: (\d+\.\d+), dim_acc: (\d+\.\d+)'


测试_mode3:
Epoch 0/49999, Iters: 000100, best_Iters: 000100, validation_accuracy: 0.0000, base_acc: 0.0000, dim_acc: 0.0000, dim10_acc: 0.0000, dim20_acc: 0.0000, dim30_acc: 0.0000, dim40_acc: 0.0000, dim50_acc: 0.0000
r'Epoch .*?, Iters: (\d+),.*?validation_accuracy: (\d+\.\d+), base_acc: (\d+\.\d+), dim_acc: (\d+\.\d+), dim10_acc: (\d+\.\d+), dim20_acc: (\d+\.\d+), dim30_acc: (\d+\.\d+), dim40_acc: (\d+\.\d+), dim50_acc: (\d+\.\d+)'




训练1：
Epoch 30/399, Iters: 001400, loss: 0.4050, train_accuracy: 0.5000, time: 0.03 s/iter, learning rate: 0.001
r'Epoch .*?, Iters: (\d+), loss: (\d+\.\d+), train_accuracy: (\d+\.\d+), time: .*? s/iter, learning rate: (\d+\.\d+)'

训练2:（通用）
Epoch 13/499, Iters: 000600, loss: 0.1467, train_accuracy: 0.8203, time: 0.05 s/iter, learning rate: 1e-05
r'Epoch .*?, Iters: (\d+), loss: (\d+\.\d+), train_accuracy: (\d+\.\d+), .*?, learning rate: .*?'



"""

import matplotlib.pyplot as plt
import re
import os
import shutil
def get_val_Iter_acc(val_log_file):
    Iter = []
    Acc = []
    with open(val_log_file, 'r') as file:
        for line in file:
            line = line.strip()
            line.split()

            pattern = r'Epoch .*?, Iters: (\d+), validation_accuracy: (\d+\.\d+)'
            matchs = re.match(pattern, line)
            if matchs:
                # print(matchs.group(1), matchs.group(2))
                Iter.append(int(matchs.group(1)))
                Acc.append(float(matchs.group(2)))
    return Iter, Acc

def get_val_Iter_acc_2(val_log_file):
    Iter = []
    ACC = []
    base_acc = []
    dim_acc = []

    with open(val_log_file, 'r') as file:
        for line in file:
            line = line.strip()
            line.split()

            pattern = r'Epoch .*?, Iters: (\d+),.*?validation_accuracy: (\d+\.\d+), base_acc: (\d+\.\d+), dim_acc: (\d+\.\d+)'


            matchs = re.match(pattern, line)
            if matchs:
                # print(matchs.group(1), matchs.group(2))
                Iter.append(int(matchs.group(1)))
                ACC.append(float(matchs.group(2)))
                base_acc.append(float(matchs.group(3)))
                dim_acc.append(float(matchs.group(4)))


    return Iter, ACC, base_acc, dim_acc

def get_val_Iter_acc_3(val_log_file):
    Iter = []
    ACC = []
    base_acc = []
    dim_acc = []
    dim10_acc = []
    dim20_acc = []
    dim30_acc = []
    dim40_acc = []
    dim50_acc = []
    with open(val_log_file, 'r') as file:
        for line in file:
            line = line.strip()
            line.split()

            pattern = r'Epoch .*?, Iters: (\d+),.*?validation_accuracy: (\d+\.\d+), base_acc: (\d+\.\d+), dim_acc: (\d+\.\d+), dim10_acc: (\d+\.\d+), dim20_acc: (\d+\.\d+), dim30_acc: (\d+\.\d+), dim40_acc: (\d+\.\d+), dim50_acc: (\d+\.\d+)'
            matchs = re.match(pattern, line)
            if matchs:
                # print(matchs.group(1), matchs.group(2))
                Iter.append(int(matchs.group(1)))
                ACC.append(float(matchs.group(2)))
                base_acc.append(float(matchs.group(3)))
                dim_acc.append(float(matchs.group(4)))
                dim10_acc.append(float(matchs.group(5)))
                dim20_acc.append(float(matchs.group(6)))
                dim30_acc.append(float(matchs.group(7)))
                dim40_acc.append(float(matchs.group(8)))
                dim50_acc.append(float(matchs.group(9)))
    return Iter, ACC, base_acc, dim_acc, dim10_acc, dim20_acc, dim30_acc, dim40_acc, dim50_acc

def get_train_Iter_acc(train_log_file,mode):
    Iter = []
    Acc = []
    Loss = []
    with open(train_log_file, 'r') as file:
        for line in file:
            line = line.strip()
            line.split()
            if mode == 1:
                pattern = r'Epoch .*?, Iters: (\d+), loss: (\d+\.\d+), train_accuracy: (\d+\.\d+),.*?, learning rate: (\d+\.\d+)'
            if mode == 2:
                pattern = r'Epoch .*?, Iters: (\d+), loss: (\d+\.\d+), train_accuracy: (\d+\.\d+),.*?, learning rate: .*?'
            matchs = re.match(pattern, line)
            if matchs:
                Iter.append(int(matchs.group(1)))
                Loss.append(float(matchs.group(2)))
                Acc.append(float(matchs.group(3)))
    return Iter, Acc, Loss


def draw_begin_end(label,xs,ys,begin_iter,end_iter):
    """
    绘制(xs,ys)，其中xs在begin_iter到end_iter之间
    """
    if end_iter == -1:
        end_iter = 1e50
    xs_filted = [x for x in xs if begin_iter <=int(x)<=end_iter]
    ys_filted = [y for x,y in zip(xs,ys) if begin_iter <=int(x)<=end_iter]
    print(len(xs_filted), len(ys_filted))
    plt.plot(xs_filted, ys_filted, label=label)


if __name__ == '__main__':
    begin_iter = 0
    end_iter = 80000
    end_iter = -1
    train_log_file = r"repeart_ccpd_lpr_train_logging.txt"
    val_log_file = r"repeart_ccpd_lpr_validation_logging.txt"

    # train_log_file = r"L:\_LOG\lp_ccpd_color_train_logging.txt"
    # val_log_file = r"L:\_LOG\lp_ccpd_color_validation_logging.txt"

    # train_log_file = "L:\_LOG\seed=130_2_train_logging.txt"
    # val_log_file = "L:\_LOG\seed=130_2_validation_logging.txt"

    # train_log_file = "L:\_LOG\seed=130_2_train_logging.txt"
    # val_log_file = "L:\_LOG\seed=130_2_validation_logging.txt"
    #
    # train_log_file = r"L:\_LOG\20w_lr0.0001_train_logging.txt"
    # val_log_file = r"L:\_LOG\20w_lr0.0001_validation_logging.txt"


    # train_log_file = "test_train_logging.txt"
    # val_log_file = "test_validation_logging.txt"


    val_mode = 2
    train_mode = 2

    """
    acc曲线
    """

    # 训练acc曲线
    Iter_train, Acc_train, Loss_train = get_train_Iter_acc(train_log_file, train_mode)
    draw_begin_end('Train Accuracy', Iter_train, Acc_train, begin_iter, end_iter)
    # 验证acc曲线
    if val_mode == 1:
        Iter_val, Acc_val = get_val_Iter_acc(val_log_file)
        # plt.plot(Iter_val[begin:end], Acc_val[begin:end], label='Test Accuracy')
        draw_begin_end('Test Accuracy',Iter_val,Acc_val,begin_iter,end_iter)
    elif val_mode == 2:
        Iter_val, Acc_val, base_acc, dim_acc = get_val_Iter_acc_2(val_log_file)
        draw_begin_end('Test Accuracy', Iter_val, Acc_val, begin_iter, end_iter)
        draw_begin_end('Test base Accuracy', Iter_val, base_acc, begin_iter, end_iter)
        draw_begin_end('Test dim Accuracy', Iter_val, dim_acc, begin_iter, end_iter)
        # plt.plot(Iter_val[begin:end], Acc_val[begin:end], label='Test Accuracy')
        # plt.plot(Iter_val[begin:end], base_acc[begin:end], label='Test base Accuracy')
        # plt.plot(Iter_val[begin:end], dim_acc[begin:end], label='Test dim Accuracy')
    elif val_mode == 3:
        Iter_val, Acc_val, base_acc, dim_acc, dim10_acc, dim20_acc, dim30_acc, dim40_acc, dim50_acc = get_val_Iter_acc_3(val_log_file)
        print(Iter_val)
        print("test_acc:",Acc_val)
        print("base_acc:",base_acc)
        print("dim_acc:",dim_acc)
        print("dim10_acc:",dim10_acc)
        print("dim20_acc:",dim20_acc)
        print("dim30_acc:",dim30_acc)
        print("dim40_acc:",dim40_acc)
        print("dim50_acc:",dim50_acc)
        # draw_begin_end('Test Accuracy', Iter_val, Acc_val, begin_iter, end_iter)
        # draw_begin_end('Test base Accuracy', Iter_val, base_acc, begin_iter, end_iter)
        draw_begin_end('Test dim Accuracy', Iter_val, dim_acc, begin_iter, end_iter)
        draw_begin_end('Test dim10 Accuracy', Iter_val, dim10_acc, begin_iter, end_iter)
        draw_begin_end('Test dim20 Accuracy', Iter_val, dim20_acc, begin_iter, end_iter)
        draw_begin_end('Test dim30 Accuracy', Iter_val, dim30_acc, begin_iter, end_iter)
        draw_begin_end('Test dim40 Accuracy', Iter_val, dim40_acc, begin_iter, end_iter)
        draw_begin_end('Test dim50 Accuracy', Iter_val, dim50_acc, begin_iter, end_iter)

    plt.legend(loc='center left')
    plt.title('Test and Train Accuracy vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)      # 纵坐标0到1
    plt.grid(True)
    # plt.show()


    """
    训练 loss曲线
    """
    # plt.plot(Iter_train[begin:end], Loss_train[begin:end], label='Train Loss')
    draw_begin_end('Train Loss', Iter_train, Loss_train, begin_iter, end_iter)
    plt.legend(loc='lower right',bbox_to_anchor=(1, 0.1))       # label框的右下角的百分坐标
    plt.title('Train Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()