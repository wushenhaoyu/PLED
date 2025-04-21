import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import numpy as np
from math import exp
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim
import torchvision.transforms as T
from torch.nn import L1Loss, MSELoss
from pytorch_msssim import SSIM, MS_SSIM
from torchvision.models import vgg16
def l1_loss(predicted, target):
    return F.l1_loss(predicted, target)

def l2_loss(predicted, target):
    return F.mse_loss(predicted, target)


class VGG19(torch.nn.Module): # VGG19的网络
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.L1Loss(reduction='sum') # 求和
        self.criterion2 = nn.L1Loss()# 求平均
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0] # 各个slice的输出权重

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach()) # 不同slice的loss的加权和
        return loss

    def forward2(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion2(x_vgg[i], y_vgg[i].detach())
        return loss


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim_module = SSIM()

    def forward(self, predicted, target):
        return 1 - self.ssim_module(predicted, target)

def adversarial_loss(logits, is_real=True):
    if is_real:
        return F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    else:
        return F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self._init_sobel_weights()

    def _init_sobel_weights(self):
        sobel_kernel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        self.sobel_x.weight.data = sobel_kernel_x
        self.sobel_y.weight.data = sobel_kernel_y

    def forward(self, predicted, target):
        edge_pred = self.sobel_x(predicted) + self.sobel_y(predicted)
        edge_target = self.sobel_x(target) + self.sobel_y(target)
        return F.l1_loss(edge_pred, edge_target)

def color_consistency_loss(predicted, target):
    hist_pred = torch.histc(predicted, bins=256, min=0, max=1)
    hist_target = torch.histc(target, bins=256, min=0, max=1)
    return F.l1_loss(hist_pred, hist_target)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=True).features[:16]  # Until block3_conv3
        self.loss_model = vgg.to(device).eval()
        for param in self.loss_model.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        y_true, y_pred = y_true.to(next(self.loss_model.parameters()).device), y_pred.to(next(self.loss_model.parameters()).device)
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))


def color_loss(y_true, y_pred):
    return torch.mean(torch.abs(torch.mean(y_true, dim=[1, 2, 3]) - torch.mean(y_pred, dim=[1, 2, 3])))

def psnr_loss(y_true, y_pred):
    mse = F.mse_loss(y_true, y_pred)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return 40.0 - torch.mean(psnr)

def smooth_l1_loss(y_true, y_pred):
    return F.smooth_l1_loss(y_true, y_pred)

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0, power_factors=[0.5, 0.5]):
    return 1.0 - ms_ssim(y_true, y_pred, data_range=max_val, size_average=True)

def gaussian_kernel(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

def histogram_loss(y_true, y_pred, bins=256, sigma=0.01):
    
    bin_edges = torch.linspace(0.0, 1.0, bins, device=y_true.device)

    y_true_hist = torch.sum(gaussian_kernel(y_true.unsqueeze(-1), bin_edges, sigma), dim=0)
    y_pred_hist = torch.sum(gaussian_kernel(y_pred.unsqueeze(-1), bin_edges, sigma), dim=0)
    
    y_true_hist /= y_true_hist.sum()
    y_pred_hist /= y_pred_hist.sum()

    hist_distance = torch.mean(torch.abs(y_true_hist - y_pred_hist))
    return hist_distance

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss_model = VGGPerceptualLoss(device)
        self.alpha1 = 1.00
        self.alpha2 = 0.06
        self.alpha3 = 0.05
        self.alpha4 = 0.5
        self.alpha5 = 0.0083
        self.alpha6 = 0.25

    def forward(self, y_true, y_pred):
        smooth_l1_l = smooth_l1_loss(y_true, y_pred)
        #ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
        perc_l = self.perceptual_loss_model(y_true, y_pred)
        hist_l = histogram_loss(y_true, y_pred)
        psnr_l = psnr_loss(y_true, y_pred)
        color_l = color_loss(y_true, y_pred)

        total_loss = (self.alpha1 * smooth_l1_l + self.alpha2 * perc_l + 
                      self.alpha3 * hist_l + self.alpha5 * psnr_l + 
                      self.alpha6 * color_l)

        return torch.mean(total_loss)
    

def compute_gradient(img):
    gradx = img[..., 1:, :] - img[..., :-1, :]
    grady = img[..., 1:] - img[..., :-1]
    return gradx, grady


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_gradient(predict)
        target_gradx, target_grady = compute_gradient(target)

        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)


class SSIMLoss(nn.Module):
    def __init__(self, channels):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)

    def forward(self, output, target):
        ssim_loss = 1 - self.ssim(output, target)
        return ssim_loss


class SSIML1Loss(nn.Module):
    def __init__(self, channels):
        super(SSIML1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)
        self.alpha = 1.4

    def forward(self, output, target):
        l1_loss = self.l1_loss_func(output, target)
        ssim_loss = 1 - self.ssim(output, target)
        total_loss = l1_loss + self.alpha * ssim_loss
        return total_loss


class GradSSIML1Loss(nn.Module):
    def __init__(self, channels):
        super(GradSSIML1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)
        self.grad_loss_func = GradientLoss()
        self.alpha = 1.4

    def forward(self, output, target):
        l1_loss = self.l1_loss_func(output, target)
        ssim_loss = 1 - self.ssim(output, target)
        grad_loss = self.grad_loss_func(output, target)
        total_loss = l1_loss + self.alpha * ssim_loss + 0.2 * grad_loss
        return total_loss


class SSIML2Loss(nn.Module):
    def __init__(self, channels):
        super(SSIML2Loss, self).__init__()
        self.l2_loss_func = nn.MSELoss()
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)
        self.alpha = 1.

    def forward(self, output, target):
        l2_loss = self.l2_loss_func(output, target)
        ssim_loss = 1 - self.ssim(output, target)
        total_loss = l2_loss + self.alpha * ssim_loss
        return total_loss


class MSSSIML1Loss(nn.Module):
    def __init__(self, channels):
        super(MSSSIML1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ms_ssim = MS_SSIM(data_range=1., size_average=True, channel=channels)
        self.alpha = 1.0

    def forward(self, output, target):
        ms_ssim_loss = 1 - self.ms_ssim(output, target)
        l1_loss = self.l1_loss_func(output, target)
        total_loss = l1_loss + self.alpha * ms_ssim_loss
        return total_loss


class MSSSIML2Loss(nn.Module):
    def __init__(self, channels):
        super(MSSSIML2Loss, self).__init__()
        self.l2_loss_func = nn.MSELoss()
        self.ms_ssim = MS_SSIM(data_range=1., size_average=True, channel=channels)
        # self.alpha = 0.84
        self.alpha = 1.2

    def forward(self, output, target):
        l2_loss = self.l2_loss_func(output, target)
        ms_ssim_loss = 1 - self.ms_ssim(output, target)
        total_loss = l2_loss + self.alpha * ms_ssim_loss
        return total_loss


class PerLoss(torch.nn.Module):
    def __init__(self):
        super(PerLoss, self).__init__()
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.to('cuda')
        for param in vgg_model.parameters():
            param.requires_grad = False

        self.vgg_layers = vgg_model

        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, data, gt):
        loss = []
        if data.shape[1] == 1:
            data = data.repeat(1, 3, 1, 1)
            gt = gt.repeat(1, 3, 1, 1)

        dehaze_features = self.output_features(data)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss) / len(loss)


class PerL1Loss(torch.nn.Module):
    def __init__(self):
        super(PerL1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.per_loss_func = PerLoss().to('cuda')

    def forward(self, output, target):
        l1_loss = self.l1_loss_func(output, target)
        per_loss = self.per_loss_func(output, target)
        # total_loss = l1_loss + 0.04 * per_loss
        total_loss = l1_loss + 0.2 * per_loss
        return total_loss


class MSPerL1Loss(torch.nn.Module):
    def __init__(self, channels):
        super(MSPerL1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ms_ssim = MS_SSIM(data_range=1., size_average=True, channel=channels)
        self.per_loss_func = PerLoss().to('cuda')

    def forward(self, output, target):
        ms_ssim_loss = 1 - self.ms_ssim(output, target)
        l1_loss = self.l1_loss_func(output, target)
        per_loss = self.per_loss_func(output, target)
        total_loss = l1_loss + 1.2 * ms_ssim_loss + 0.04 * per_loss
        return total_loss


class MSPerL2Loss(torch.nn.Module):
    def __init__(self):
        super(MSPerL2Loss, self).__init__()
        self.l2_loss_func = nn.MSELoss()
        self.ms_ssim = MS_SSIM(data_range=1., size_average=True, channel=3)
        self.per_loss_func = PerLoss().to('cuda')

    def forward(self, output, target):
        ms_ssim_loss = 1 - self.ms_ssim(output, target)
        l2_loss = self.l2_loss_func(output, target)
        per_loss = self.per_loss_func(output, target)
        total_loss = l2_loss + 0.16 * ms_ssim_loss + 0.2 * per_loss
        return total_loss


class TVLoss(torch.nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, data):
        w_variance = torch.sum(torch.pow(data[:, :, :, :-1] - data[:, :, :, 1:], 2))
        h_variance = torch.sum(torch.pow(data[:, :, :-1, :] - data[:, :, 1:, :], 2))

        count_h = self._tensor_size(data[:, :, 1:, :])
        count_w = self._tensor_size(data[:, :, :, 1:])

        tv_loss = h_variance / count_h + w_variance / count_w
        return tv_loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def safe_div(a, b, eps=1e-2):
    return a / torch.clamp_min(b, eps)


class WTVLoss(torch.nn.Module):
    def __init__(self):
        super(WTVLoss, self).__init__()
        self.eps = 1e-2

    def forward(self, data, aux):
        data_dw = data[:, :, :, :-1] - data[:, :, :, 1:]
        data_dh = data[:, :, :-1, :] - data[:, :, 1:, :]
        aux_dw = torch.abs(aux[:, :, :, :-1] - aux[:, :, :, 1:])
        aux_dh = torch.abs(aux[:, :, :-1, :] - aux[:, :, 1:, :])

        w_variance = torch.sum(torch.pow(safe_div(data_dw, aux_dw, self.eps), 2))
        h_variance = torch.sum(torch.pow(safe_div(data_dh, aux_dh, self.eps), 2))

        count_h = self._tensor_size(data[:, :, 1:, :])
        count_w = self._tensor_size(data[:, :, :, 1:])

        tv_loss = h_variance / count_h + w_variance / count_w
        return tv_loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class WTVLoss2(torch.nn.Module):
    def __init__(self):
        super(WTVLoss2, self).__init__()
        self.eps = 1e-2
        self.criterion = nn.MSELoss()

    def forward(self, data, aux):
        N, C, H, W = data.shape

        data_dw = F.pad(torch.abs(data[:, :, :, :-1] - data[:, :, :, 1:]), (1, 0, 0, 0))
        data_dh = F.pad(torch.abs(data[:, :, :-1, :] - data[:, :, 1:, :]), (0, 0, 1, 0))
        aux_dw = F.pad(torch.abs(aux[:, :, :, :-1] - aux[:, :, :, 1:]), (1, 0, 0, 0))
        aux_dh = F.pad(torch.abs(aux[:, :, :-1, :] - aux[:, :, 1:, :]), (0, 0, 1, 0))

        data_d = data_dw + data_dh
        aux_d = aux_dw + aux_dh

        loss1 = self.criterion(data_d, aux_d)
        # loss2 = torch.norm(data_d / (aux_d + self.eps), p=1) / (C * H * W)
        loss2 = torch.norm(data_d / (aux_d + self.eps)) / (C * H * W)
        return loss1 * 0.5 + loss2 * 4.0


class MSTVPerL1Loss(torch.nn.Module):
    def __init__(self):
        super(MSTVPerL1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ms_ssim = MS_SSIM(data_range=1., size_average=True, channel=3)
        self.per_loss_func = PerLoss().to('cuda')
        self.tv_loss_func = TVLoss()
        self.color_loss_func = color_loss
        self.l2_loss_func = nn.MSELoss()

    def forward(self, output, target):
        ms_ssim_loss = 1 - self.ms_ssim(output, target)
        l1_loss = self.l1_loss_func(output, target)
        per_loss = self.per_loss_func(output, target)
        tv_loss = self.tv_loss_func(output)
        color_loss = self.color_loss_func(output, target)
        total_loss = l1_loss + 1.2 * ms_ssim_loss + 0.04 * per_loss + 1e-7 * tv_loss+ 0.2 *  color_loss 
        return total_loss
