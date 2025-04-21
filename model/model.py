import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import *
from .DEconv import *

class MyModel(nn.Module):
    def __init__(self,k = 1):
        super(MyModel, self).__init__()
        self.stage = k
        self.illumination_enhance = Illumination_enhance()
        self.loss_cal = MSTVPerL1Loss()
    def forward(self, x):
        for i in range(self.stage):
            x = self.illumination_enhance(x)

        return x
    
    def loss_(self, x, y):
        x = self.forward(x)
 
        loss = self.loss_cal(x, y)
        return loss 
class Illumination_enhance(nn.Module):
    def __init__(self):
        super(Illumination_enhance, self).__init__()

        

        self.MSRB_1 = Multi_scale_Residual_Block(3, 3)
        self.MSRB_2 = Multi_scale_Residual_Block(3, 3)
        self.MSRB_3 = Multi_scale_Residual_Block(3, 3)
        self.DPAEM_1 = DPAEM(3,denoise=True)

    
    def forward(self, x):
        x1 = self.MSRB_1(x)
        x1_a , x1_b = self.DPAEM_1(x, x1, denoise=True) 
        x2 = self.MSRB_2(x1_b+x1_a ) 
        x2_a , x2_b = self.DPAEM_1(x, x2, denoise=True)
        x3 = self.MSRB_3(x2_b+x2_a )


        return x3

    

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()
        

        self.channel_attention = SEBlock(channels, reduction)
        

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.apply(self.weights_init)
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:  
                m.bias.data.zero_() 

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)
            if m.bias is not None:  
                m.bias.data.zero_()

    def forward(self, x):
        x = self.channel_attention(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.spatial_attention(spatial_attn)
        
        return x * spatial_attn


class Multi_scale_Residual_Block(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(Multi_scale_Residual_Block, self).__init__()
        self.conv1x1 = nn.Conv2d(channels_in, channels_out, 1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(channels_in, channels_out, 5, padding=2, bias=False)
        self.conv7x7 = nn.Conv2d(channels_in, channels_out, 7, padding=3, bias=False)
        
        self.Sobel = Sobelxy(channels_in, channels_out)
        self.Lapla = laplacian(channels_in, channels_out)
        self.color_mix = nn.Conv2d(channels_in*7, channels_in*7, 1, groups=3) 
        self.se_block = SEBlock(channels_in * 7)
        
        self.local_conv = nn.Conv2d(channels_in * 7, channels_out, 3, padding=1, bias=False)
        self.global_conv = nn.Conv2d(channels_in * 7, channels_out, 7, padding=3, bias=False)
        
        self.conv_out = nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False)
        self.cbam_block = CBAMBlock(channels_out)
        
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        input = x
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x4 = self.Sobel(x)
        x5 = self.Lapla(x)
        x6 = self.conv7x7(x)
        x = torch.cat((x, x1, x2, x3, x4, x5, x6), 1)
        x = self.color_mix(x)
        x = self.se_block(x)   
        
        x_local = self.local_conv(x)
        x_local = self.leakyrelu(x_local)
        x_global = self.global_conv(x)
        x_global = self.leakyrelu(x_global)
        
        x = x_local + x_global  
        x = self.cbam_block(x)
        x = self.conv_out(x)  
        x = x + input  
        x = torch.clamp(x, min=0, max=1)
        return x



class laplacian(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(laplacian, self).__init__()
        laplacian_filter = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]])
        self.conv_x = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=3, padding=1)
        self.conv_x.weight.data.copy_(torch.from_numpy(laplacian_filter))
        self.conv_y = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=3, padding=1)
        self.conv_y.weight.data.copy_(torch.from_numpy(laplacian_filter.T))
    def forward(self, x):
        laplacianx = self.conv_x(x)
        laplaciany = self.conv_y(x)
        x = torch.abs(laplacianx) + torch.abs(laplaciany)
        return x

class Sobelxy(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.conv_x = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=3, padding=1)
        self.conv_x.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.conv_y = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=3, padding=1)
        self.conv_y.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.conv_x(x)
        sobely = self.conv_y(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x
    

class NoiseReductionModule(nn.Module):
    def __init__(self, in_channels, num_layers):
        super(NoiseReductionModule, self).__init__()
        self.hidden_channels = 16
        

        self.convin = nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1, bias=False)
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(self.hidden_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.dncnn = nn.Sequential(*layers)
        
        self.final_conv = nn.Conv2d(self.hidden_channels, in_channels, kernel_size=3, padding=1, bias=False)

        self.apply(self.weights_init)
    
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x , att):

        x = self.convin(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        residual = self.dncnn(x)
        
        noise = self.final_conv(residual)

        noise = noise * att 
        
        return noise





class DPAEM(nn.Module):
    def __init__(self, in_channels,denoise=True):
        super().__init__()
        self.detail_enhance = DEConv(in_channels)
        self.noise_reduction = NoiseReductionModule(in_channels, num_layers=10)
        self.noise_attention = NoiseAttetion()
        self.denoise = denoise
    
    def forward(self, input, enhance, denoise):
        a = self.detail_enhance(input)
        if denoise:
            L = input / (enhance + 1e-4)
            L = torch.clamp(L, 0, 1)
            snr , att = self.noise_attention(L)
            b = self.noise_reduction(L,att)
            return a  , enhance - b
        else:
            return a  , enhance

class NoiseAttetion(nn.Module):
    def __init__(self, blur_kernel_size=5, in_channels=1):
        super(NoiseAttetion, self).__init__()
        self.blur_kernel_size = blur_kernel_size
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1) 

    def forward(self, img):
        dark = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
        dark = dark.unsqueeze(1)  
        light = self.mean_blur(dark, kernel_size=self.blur_kernel_size)
        noise = torch.abs(dark - light)
        snr = light / (noise + 1e-4)  
        spatial_attention = self.spatial_attention(snr)
        return snr, spatial_attention

    def mean_blur(self, image, kernel_size=5):
        weight = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        weight = weight.to(image.device)
        blurred_image = F.conv2d(image, weight, padding=kernel_size // 2, groups=image.shape[1])
        return blurred_image

    def spatial_attention(self, snr):
        max_pool = F.max_pool2d(snr, kernel_size=3, stride=1, padding=1)
        avg_pool = F.avg_pool2d(snr, kernel_size=3, stride=1, padding=1)
        spatial_attention = torch.sigmoid(max_pool + avg_pool)
        spatial_attention = self.conv1(spatial_attention)
        spatial_attention = torch.sigmoid(spatial_attention)

        return spatial_attention