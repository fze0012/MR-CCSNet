from utils import *
import time
import cv2
from torchvision.transforms import ToPILImage
import torchvision


def train(train_loader, model, criterion, optimizer, epoch):
    print('Epoch: %d' % (epoch + 1))
    model.train()
    sum_loss = 0
    for inputs, _ in train_loader:
        inputs = rgb_to_ycbcr(inputs.cuda())[:, 0, :, :].unsqueeze(1) / 255.
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[0], inputs) + criterion(outputs[1], inputs)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    return sum_loss


def valid_bsds(valid_loader, model, criterion):
    sum_psnr = 0
    sum_ssim = 0
    _ssim = SSIM().cuda()
    model.eval()
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            inputs = rgb_to_ycbcr(inputs.cuda())[:, 0, :, :].unsqueeze(1) / 255.
            outputs = model(inputs)
            mse = F.mse_loss(outputs[0], inputs)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr
            sum_ssim += ssim(outputs[0], inputs)

    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)


def valid_set(valid_loader, model, criterion):
    sum_psnr = 0
    sum_ssim = 0
    _ssim = SSIM().cuda()
    model.eval()
    time_sum = 0
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            inputs = rgb_to_ycbcr(inputs.cuda())[:, 0, :, :].unsqueeze(1) / 255.
            outputs = model(inputs)
            mse = F.mse_loss(outputs[0], inputs)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr
            sum_ssim += ssim(outputs[0], inputs)
    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)
