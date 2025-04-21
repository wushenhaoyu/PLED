# ====== External Libraries (installed via pip) ======
import argparse
import logging
import os
import torch
import torch.nn as nn
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import cv2
import numpy as np


# ====== Custom Libraries ======
import utils.losses
from utils.scheduler import GradualWarmupScheduler
from DataProcess.data_RGB import get_training_data, get_validation_data2
from model.model import MyModel



train_dir = '/home/ubuntu/wushen/dark/dataset/LOLv1/train'
val_dir = '/home/ubuntu/wushen/dark/dataset/LOLv1/eval'
dir_checkpoint = Path('./checkpoints/')
Load = True
GT_mean = False
TrainTime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def train_model(
        model,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
):

    ## GPU
    x= [0]
    gpus = ','.join([str(i) for i in x])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    if len(device_ids) > 1:
        model= nn.DataParallel(model, device_ids=device_ids)


    ## Optimizer
    start_epoch = 1
    LR_MIN = 1e-6
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    ## Scheduler (Strategy)
    warmup_epochs = 10
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs - warmup_epochs,
                                                            eta_min=float(LR_MIN))
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    ## Loss
    #criterion = nn.SmoothL1Loss()
    ## DataLoaders
    print('==> Loading datasets')
    patch_size = 164
    train_dataset = get_training_data(train_dir, {'patch_size': patch_size})
    print(f'Train dataset size: {len(train_dataset)}')

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=32, drop_last=False)
    val_dataset = get_validation_data2(val_dir, {'patch_size': patch_size})
    print(f'Val dataset size: {len(val_dataset)}')
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
                            drop_last=False)



    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Device:          {device.type}
    ''')

    
    # Start training!
    print('==> Training start: ')
    best_psnr = 0
    best_ssim = 0
    best_epoch_psnr = 0
    best_epoch_ssim = 0
    total_start_time = time.time()

    if Load:
        try:
            path_chk_rest = utils.get_last_path("checkpoints", '_bestPSNR.pth')
            utils.load_checkpoint(model, path_chk_rest)
            start_epoch = utils.load_start_epoch(path_chk_rest) + 1
            utils.load_optim(optimizer, path_chk_rest)

            best_epoch_psnr = utils.load_best_psnr_epoch(path_chk_rest)
            best_epoch_ssim = utils.load_best_ssim_epoch(path_chk_rest)
            best_psnr = utils.load_best_psnr(path_chk_rest)
            best_ssim = utils.load_best_ssim(path_chk_rest)
            for i in range(1, start_epoch):
                scheduler.step()
            new_lr = scheduler.get_lr()[0]
            print('------------------------------------------------------------------')
            print("==> Loading  Resuming Training with learning rate:", new_lr)
            print(best_psnr)
            print(best_ssim)
            print('------------------------------------------------------------------')
        except Exception as e:
            print(e)
            print("未找到当前的保存的model")

    ## Log
    log_dir = "log/"
    utils.mkdir(log_dir)
    print(1)
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        epoch_loss = 0

        model.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            # Forward propagation
            for param in model.parameters():
                param.grad = None
            target = data[0].cuda()
            input_ = data[1].cuda()

            total_loss = model.loss_(input_,target)


            # Back propagation
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        # Evaluation (Validation)
        if epoch % 1== 0:
            model.eval()
            psnr_val_rgb = []
            ssim_val_rgb = []
            for ii, data_val in enumerate(val_loader, 0):
                try:
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    h, w = target.shape[2], target.shape[3]
                    with torch.no_grad():
                        restored = model(input_)
                        restored = restored[:, :, :h, :w]

                    for res, tar in zip(restored, target):
                        if GT_mean:
                                res = res.permute(1, 2, 0).cpu().numpy().astype(np.float32)
                                tar = tar.permute(1, 2, 0).cpu().numpy().astype(np.float32)
                                mean_res = cv2.cvtColor(res.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                                mean_tar = cv2.cvtColor(tar.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                                # 将 RGB 图像转换为灰度图像，并计算均值 
                                mean_res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY).mean()
                                mean_tar = cv2.cvtColor(tar, cv2.COLOR_RGB2GRAY).mean()

                                res = np.clip(res * (mean_tar / mean_res), 0, 1)
                                res = torch.from_numpy(res).permute(2, 0, 1).cuda()
                                tar = torch.from_numpy(tar).permute(2, 0, 1).cuda()
                                # print(res.shape,tar.shape)
                        # psnr_val_rgb.append(utils.torchPSNR(res, tar))


                        psnr_val_rgb.append(utils.calculate_psnr(res, tar))
                        # ssim_val_rgb.append(utils.torchSSIM(restored, target))
                        ssim_val_rgb.append(utils.torchSSIM(res, tar))
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"显存溢出！跳过 batch {ii}，继续处理下一个输入。")
                        torch.cuda.empty_cache()  # 清理缓存，释放显存
                    else:
                        raise e  # 其他错误抛出

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
            ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

            # Save the best PSNR model of validation
            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch_psnr = epoch
                torch.save({'epoch': epoch,
                            "best_epoch_psnr":best_epoch_psnr,
                            "best_psnr":best_psnr,
                            "best_epoch_ssim":best_epoch_ssim,
                            "best_ssim":best_ssim,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(dir_checkpoint, "model_bestPSNR.pth"))

            logging.info(
                "[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
                epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))


            # Save the best SSIM model of validation
            if ssim_val_rgb > best_ssim:
                best_ssim = ssim_val_rgb
                best_epoch_ssim = epoch
                torch.save({'epoch': epoch,
                            "best_epoch_psnr":best_epoch_psnr,
                            "best_psnr":best_psnr,
                            "best_epoch_ssim":best_epoch_ssim,
                            "best_ssim":best_ssim,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(dir_checkpoint, "model_bestSSIM.pth"))
            logging.info(
                "[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
                epoch, ssim_val_rgb, best_epoch_ssim, best_ssim)
            )

            

        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.10f}".format(epoch, time.time() - epoch_start_time,
                                                                                epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")
        logging.info(f"Epoch: {epoch}\tTime: {time.time() - epoch_start_time:.4f}\tLoss: {epoch_loss:.4f}\tLearningRate: {scheduler.get_lr()[0]:.10f}")
        # Save the last model
        torch.save({'epoch': epoch,
                    "best_epoch_psnr":best_epoch_psnr,
                    "best_psnr":best_psnr,
                    "best_epoch_ssim":best_epoch_ssim,
                    "best_ssim":best_ssim,

                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(dir_checkpoint, "model_latest.pth"))

        # logging.info(f'train/loss: {epoch_loss}, epoch: {epoch}')
        # logging.info(f'train/lr: {scheduler.get_lr()[0]}, epoch: {epoch}')


    total_finish_time = (time.time() - total_start_time)  # seconds
    print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))



def create_logger(logger_file_path):

    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(logger_file_path, log_name)

    logger = logging.getLogger()  # 设定日志对象
    logger.setLevel(logging.INFO)  # 设定日志等级

    file_handler = logging.FileHandler(final_log_file)  # 文件输出
    console_handler = logging.StreamHandler()  # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)  # 设置文件输出格式
    console_handler.setFormatter(formatter)  # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger



def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=50, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logger = create_logger("log")
    logger.info('------Begin Training Model------')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    model = MyModel()
    print(model)
    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)
    
    train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
