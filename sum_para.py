# -- coding: utf-8 --
import torch
import torchvision
from thop import profile
from model.model import MyModel

print('==> Building model..')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)

dummy_input = torch.randn(1, 3, 256, 256).to(device)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable parameters: {trainable_params}')
for name, param in model.named_parameters():
    param_count = param.numel()  
    print(f'{name}: requires_grad={param.requires_grad}, numel={param_count}')

