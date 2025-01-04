from LPDGAN_test import test
from model.LPDGAN import create_model
import argparse
from PIL import Image
import numpy as np
from data import aug
from torchvision import transforms
import matplotlib.pyplot as plt
# from util import util, html
import cv2

class Debluror(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', default='LPDGAN', type=str)
        parser.add_argument('--dataroot', type=str, default=r'../DATASET/LPBlur')
        parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
        parser.add_argument('--gpu_ids', type=str, default='0')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints2')            # 权重路径
        parser.add_argument('--input_nc', type=int, default=3)
        parser.add_argument('--output_nc', type=int, default=3)
        parser.add_argument('--ndf', type=int, default=64)

        # Train
        parser.add_argument('--batch_size', type=int, default=7)
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--load_iter', type=int, default=155)                               # 权重路径
        parser.add_argument('--epoch', type=int, default=200)
        parser.add_argument('--print_freq', type=int, default=10400)
        parser.add_argument('--num_worker', type=int, default=0)
        parser.add_argument('--save_latest_freq', type=int, default=5000)
        parser.add_argument('--save_epoch_freq', type=int, default=5)
        parser.add_argument('--save_by_iter', action='store_true')
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100,
                            help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--gan_mode', type=str, default='wgangp')
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')


        # Test
        parser.add_argument('--results_dir', type=str, default='./results2/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--num_test', type=int, default=1000, help='how many test images to run')


        # For display
        parser.add_argument('--display_freq', type=int, default=10400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=3,
                            help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000,
                            help='frequency of saving training results to html')
        parser.add_argument('--no_html', action='store_true',
                            help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')


        self.transform_fn = aug.get_transforms_fortest(size=(112, 224))
        self.transform_fn1 = aug.get_transforms_fortest(size=(56, 112))
        self.transform_fn2 = aug.get_transforms_fortest(size=(28, 56))
        self.transform_fn3 = aug.get_transforms_fortest(size=(14, 28))
        self.normalize_fn = aug.get_normalize()


        self.args = parser.parse_args()
        self.model = create_model(self.args)
        self.model.setup(self.args)

    def deblur(self, img):
        blur_image = np.array(img)
        sharp_image = np.array(img)

        blur_image, sharp_image = self.transform_fn(blur_image, sharp_image)
        blur_image1, sharp_image1 = self.transform_fn1(blur_image, sharp_image)
        blur_image2, sharp_image2 = self.transform_fn2(blur_image, sharp_image)
        blur_image3, sharp_image3 = self.transform_fn3(blur_image, sharp_image)

        blur_image, sharp_image = self.normalize_fn(blur_image, sharp_image)
        blur_image1, sharp_image1 = self.normalize_fn(blur_image1, sharp_image1)
        blur_image2, sharp_image2 = self.normalize_fn(blur_image2, sharp_image2)
        blur_image3, sharp_image3 = self.normalize_fn(blur_image3, sharp_image3)

        blur_image = transforms.ToTensor()(blur_image)
        sharp_image = transforms.ToTensor()(sharp_image)
        blur_image1 = transforms.ToTensor()(blur_image1)
        sharp_image1 = transforms.ToTensor()(sharp_image1)
        blur_image2 = transforms.ToTensor()(blur_image2)
        sharp_image2 = transforms.ToTensor()(sharp_image2)
        blur_image3 = transforms.ToTensor()(blur_image3)
        sharp_image3 = transforms.ToTensor()(sharp_image3)

        data = {'A': blur_image, 'B': sharp_image, 
                'A_paths': "", 'B_paths': "",
                'A1': blur_image1, 'B1': sharp_image1, 'A2': blur_image2, 'B2': sharp_image2, 'A3': blur_image3,
                'B3': sharp_image3}

        self.model.set_input(data)
        self.model.test()
        visuals = self.model.get_current_visuals()
        fake_B = visuals['fake_B']
        fake_B = util.tensor2im(fake_B)
        # fake_B = Image.fromarray(fake_B)

        return fake_B

        # test(self.args)

if __name__=='__main__':
    debluror = Debluror()
    img = Image.open(r"S:\Learn\DATASET\CCPD2019_LP\base\川A1B02L_0369336685824-96_77-273&510_568&644-578&646_283&597_262&502_557&551-22_0_25_1_24_26_10-84-55.jpg")
    deblurred_img = debluror.deblur(img)
    # img.show()
    # deblurred_img.show()
    # 保存
    deblurred_img = np.array(deblurred_img)
    deblurred_img = Image.fromarray(deblurred_img)
    deblurred_img.save('deblur.jpg')
    img = np.array(img)
    img = Image.fromarray(img)
    img.save('original.jpg')

    # image_pil = Image.fromarray(img)
    

    


