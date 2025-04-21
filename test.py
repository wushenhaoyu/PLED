

import time
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob

import cv2
import argparse
from model.model import MyModel
parser = argparse.ArgumentParser(description='Demo Low-light Image Enhancement')
parser.add_argument('--input_dir', default='/home/ubuntu/wushen/dark/dataset/LOLv1/eval/low', type=str, help='Input images')
parser.add_argument('--result_dir', default='/home/ubuntu/wushen/dark/code/ZZ/results/AllStage', type=str, help='Directory for results')
parser.add_argument('--weights',
                    default='/home/ubuntu/wushen/dark/code/ZZ/checkpoints/PSNR23.8_dark/model_bestPSNR.pth', type=str,
                    help='Path to weights')

args = parser.parse_args()


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))




inp_dir = args.input_dir
out_dir = args.result_dir

os.makedirs(out_dir, exist_ok=True)

files = natsorted(#glob(os.path.join(inp_dir, '*.jpg')) +
                  glob(os.path.join(inp_dir, '*.JPG'))
                  + glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.PNG')))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")



model = MyModel().cuda()
model.load_state_dict(torch.load(args.weights)['state_dict'])
print(torch.load(args.weights)['best_psnr'])
model.eval()

print('restoring images......')

mul = 16
index = 0
psnr_val_rgb = []
for file_ in files:
    img = Image.open(file_).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()
    # Pad the input if not_multiple_of 16
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    with torch.no_grad():
        restored_list = model(input_)
    for i , restored in enumerate(restored_list):
        restored = torch.clamp(restored, 0, 1)
        print(restored.shape)
        restored = restored[:, :, :h, :w]
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])
        f = os.path.splitext(os.path.split(file_)[-1])[0]
        save_img((os.path.join(out_dir, f + f'_{i}.png')), restored)
    index += 1
    print('%d/%d' % (index, len(files)))
print(f"Files saved at {out_dir}")
print('finish !')
